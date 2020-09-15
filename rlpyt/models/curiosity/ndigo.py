
import torch
from torch import nn

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead

GruState = namedarraytuple("GruState", ["c"])  # For downstream namedarraytuples to work

class NdigoForward(nn.Module):
    """Frame predictor MLP for NDIGO curiosity algorithm"""
    def __init__(self,
                 feature_size,
                 action_size, # usually multi-action sequence
                 output_size # observation size
                 ):
        super(NdigoForward, self).__init__()

        self.model = nn.Sequential(nn.Linear(feature_size + action_size, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, output_size),
                                   nn.ReLU())

    def forward(self, belief, action_seq):
        predicted_state = self.model(torch.cat([belief, action_seq], 2))
        return predicted_state

class NDIGO(torch.nn.Module):
    """Curiosity model for intrinsically motivated agents: a convolutional network 
    into an FC layer into an LSTM into an MLP which outputs forward predictions on
    future states, and computes an intrinsic reward using the error in these predictions.
    """

    def __init__(
            self,
            image_shape,
            action_size,
            horizon,
            feature_encoding='idf_maze',
            gru_size=128,
            batch_norm=False,
            obs_stats=None
            ):
        """Instantiate neural net module according to inputs."""
        super(NDIGO, self).__init__()

        self.horizon = horizon
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        if self.feature_encoding != 'none':
            if self.feature_encoding == 'idf':
                self.feature_size = 288
                self.encoder = UniverseHead(image_shape=image_shape, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_burda':
                self.feature_size = 512
                self.encoder = BurdaHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_maze':
                self.feature_size = 256
                self.encoder = MazeHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)

        self.gru_size = gru_size
        self.gru = torch.nn.GRU(self.feature_size + (action_size * horizon), self.gru_size)

        self.forward_model = NdigoForward(feature_size=self.gru_size, 
                                          action_size=action_size * horizon, 
                                          output_size=image_shape[0]*image_shape[1]*image_shape[2])


    def forward(self, obs1, obs2, action_seq, gru_state):

        img1 = obs1.type(torch.float)
        img2 = obs2.type(torch.float) # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs1, 3)

        # encode image
        z_t = self.encoder(img1.view(T * B, *img_shape)).view(T, B, -1)

        # pass through GRU
        gru_state = None if gru_state is None else gru_state.c
        gru_input = torch.cat([z_t, action_seq.view(T, B, -1)], dim=2)
        b_t, gru_state_next = self.gru(gru_input, gru_state) # CURRENTLY USING a_t not a_{t-1}

        return self.forward_model(b_t, action_seq.view(T, B, -1)), gru_state_next
        

    def compute_bonus(self, obs, action_seq, next_obs, last_loss, gru_state):
        predicted_obs, gru_state_next = self.forward(obs, next_obs, action_seq, gru_state)
        next_obs = torch.max(next_obs.view(-1, *predicted_obs.shape[2:]), 1)[1]
        loss = nn.functional.cross_entropy(predicted_obs.view(-1, *predicted_obs.shape[2:]), next_obs.detach())
        r_int = last_loss - loss
        return r_int.squeeze(), GruState(gru_state_next), loss


    def compute_loss(self, obs, action_seq, next_obs, gru_state):
        #------------------------------------------------------------#
        # hacky dimension add for when you have only one environment
        if action_seq.dim() == 2: 
            action_seq = action_seq.unsqueeze(1)
        #------------------------------------------------------------#
        predicted_obs, gru_state_next = self.forward(obs, next_obs, action_seq, gru_state)
        next_obs = torch.max(next_obs.view(-1, *predicted_obs.shape[2:]), 1)[1]
        loss = nn.functional.cross_entropy(predicted_obs.view(-1, *predicted_obs.shape[2:]), next_obs.detach())
        return loss



