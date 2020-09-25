
from PIL import Image
import os

import numpy as np
import torch
from torch import nn
torch.set_printoptions(threshold=500*2*100)

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

    def forward(self, belief_states, action_seqs):
        predicted_states = self.model(torch.cat([belief_states, action_seqs], 2))
        return predicted_states

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
            obs_stats=None,
            num_predictors=10
            ):
        """Instantiate neural net module according to inputs."""
        super(NDIGO, self).__init__()

        assert num_predictors >= horizon

        self.action_size = action_size
        self.horizon = horizon
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
        self.num_predictors = num_predictors
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
        self.gru = torch.nn.GRU(self.feature_size + action_size, self.gru_size)
        self.gru_states = None # state output of last batch - (1, B, gru_size) or None

        self.forward_model = []
        for k in range(1, self.num_predictors+1):
            self.forward_model.append(NdigoForward(feature_size=self.gru_size, 
                                                   action_size=action_size * k, 
                                                   output_size=image_shape[0]*image_shape[1]*image_shape[2]))


    def forward(self, observations, prev_actions, actions):

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)

        # encode batch
        images = observations.type(torch.float)
        encoded_states = self.encoder(images.view(T * B, *img_shape)).view(T, B, -1)

        # pass encoded batch through GRU
        # gru_state = None if gru_state is None else gru_state.c
        gru_inputs = torch.cat([encoded_states, prev_actions], dim=2)
        belief_states, gru_output_states = self.gru(gru_inputs, self.gru_states)

        return belief_states, gru_output_states
        

    def compute_bonus(self, observations, prev_actions, actions):
        #------------------------------------------------------------#
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        # hacky dimension add for when you have only one environment
        if prev_actions.dim() == 1: 
            prev_actions = prev_actions.view(1, 1, -1)
        if actions.dim() == 1:
            actions = actions.view(1, 1, -1)
        #------------------------------------------------------------#

        # generate belief states
        belief_states, gru_output_states = self.forward(observations, prev_actions, actions)
        self.gru_states = None # only bc we're processing exactly 1 episode per batch

        # slice beliefs and actions
        belief_states = belief_states[:T-self.horizon] # slice off last timesteps
        
        # action_seqs = torch.zeros((T-self.horizon, B, self.horizon*self.action_size)) # placeholder
        # for i in range(len(actions)-self.horizon):
        #     action_seq = actions[i:i+self.horizon]
        #     action_seq = torch.transpose(action_seq, 0, 1)
        #     action_seq = torch.reshape(action_seq, (action_seq.shape[0], -1))
        #     action_seqs[i] = action_seq
        
        # make forward model predictions
        predicted_states = self.forward_model[self.horizon-1](belief_states, actions[:T-self.horizon].detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-k, B, 75)
        predicted_states = torch.sigmoid(predicted_states)
        true_obs = observations[self.horizon:].view(-1, *predicted_states.shape[1:])

        # DEBUGGING
        path = '/curiosity_baselines/results/ppo_Deepmind5Room-v0/run_96/images'
        if not os.path.isdir(path):
            os.mkdir(path)
        ep_num = len(os.listdir(path))
        os.mkdir(path + '/ep_{}'.format(ep_num))
        pred = predicted_states.detach().clone().data.numpy()
        true = true_obs.detach().clone().data.numpy()
        pred = np.reshape(pred[10, 0], (3, 5, 5))
        true = np.reshape(true[10, 0], (3, 5, 5))

        print(pred)
        print('-'*100)
        print(true)
        print('#'*100)
        for i in range(3):
            pred_img = Image.fromarray((pred[i]*500).astype(np.uint8), 'L')
            true_img = Image.fromarray((true[i]*500).astype(np.uint8), 'L')
            pred_img.save(path + '/ep_{}/pred_{}.jpg'.format(ep_num, i))
            true_img.save(path + '/ep_{}/true_{}.jpg'.format(ep_num, i))

        # generate losses
        losses = nn.functional.binary_cross_entropy(predicted_states, true_obs.detach(), reduction='none')
        losses = torch.sum(losses, dim=-1)/losses.shape[-1] # average of each feature for each environment at each timestep (T, B, ave_loss_over_feature)
        
        # subtract losses to get rewards
        r_int = torch.zeros((T, B))
        for i in range(1, len(losses)):
            r_int[i+self.horizon-1] = losses[i-1] - losses[i]

        return r_int


    def compute_loss(self, observations, prev_actions, actions):
        #------------------------------------------------------------#
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        # hacky dimension add for when you have only one environment
        if prev_actions.dim() == 2: 
            prev_actions = prev_actions.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        #------------------------------------------------------------#

        # generate belief states
        belief_states, gru_output_states = self.forward(observations, prev_actions, actions)
        self.gru_states = None # only bc we're processing exactly 1 episode per batch

        # generate loss for each forward predictor
        loss = torch.tensor(0.0)
        for k in range(1, self.num_predictors+1):
            # action_seqs = torch.zeros((T-k, B, k*self.action_size)) # placeholder
            # for i in range(len(actions)-k):
            #     action_seq = actions[i:i+k]
            #     action_seq = torch.transpose(action_seq, 0, 1)
            #     action_seq = torch.reshape(action_seq, (action_seq.shape[0], -1))
            #     action_seqs[i] = action_seq

            # make forward model predictions for this predinorctor
            predicted_states = self.forward_model[k-1](belief_states[:T-k], actions[:T-k].detach()).view(-1, img_shape[0]*img_shape[1]*img_shape[2]) # (T-k, B, 75)
            predicted_states = torch.sigmoid(predicted_states)

            # generate losses for this predictor
            true_obs = observations[k:].view(-1, *predicted_states.shape[1:]).detach()

            loss += nn.functional.binary_cross_entropy(predicted_states, true_obs.detach(), reduction='mean')

        return loss



