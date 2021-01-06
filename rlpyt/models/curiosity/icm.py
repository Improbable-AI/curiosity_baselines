
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead

class ResBlock(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size):
        super(ResBlock, self).__init__()

        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.lin_2 = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, x, action):
        res = nn.functional.leaky_relu(self.lin_1(torch.cat([x, action], 2)))
        res = self.lin_2(torch.cat([res, action], 2))
        return res + x

class ResForward(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size):
        super(ResForward, self).__init__()

        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.res_block_1 = ResBlock(feature_size, action_size)
        self.res_block_2 = ResBlock(feature_size, action_size)
        self.res_block_3 = ResBlock(feature_size, action_size)
        self.res_block_4 = ResBlock(feature_size, action_size)
        self.lin_last = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, phi1, action):
        x = nn.functional.leaky_relu(self.lin_1(torch.cat([phi1, action], 2)))
        x = self.res_block_1(x, action)
        x = self.res_block_2(x, action)
        x = self.res_block_3(x, action)
        x = self.res_block_4(x, action)
        x = self.lin_last(torch.cat([x, action], 2))
        return x

class ICM(nn.Module):
    """ICM curiosity agent: two neural networks, one
    forward model that predicts the next state, and one inverse model that predicts 
    the action given two states. The forward model uses the prediction error to
    compute an intrinsic reward. The inverse model trains features that are invariant
    to distracting environment stochasticity.
    """

    def __init__(
            self, 
            image_shape, 
            action_size,
            feature_encoding='idf', 
            batch_norm=False,
            prediction_beta=1.0,
            obs_stats=None
            ):
        super(ICM, self).__init__()

        self.prediction_beta = prediction_beta
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

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_size * 2, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, action_size)
            )

        # 2019 ICM paper (Burda et al.)
        self.forward_model = ResForward(feature_size=self.feature_size, action_size=action_size)

        # 2017 ICM paper (Pathak et al.)
        # self.forward_model = nn.Sequential(
        #     nn.Linear(self.feature_size + action_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.feature_size)
        #     )


    def forward(self, obs1, obs2, action):

        if self.obs_stats is not None:
            img1 = (obs1 - self.obs_mean) / self.obs_std
            img2 = (obs2 - self.obs_mean) / self.obs_std

        img1 = obs1.type(torch.float)
        img2 = obs2.type(torch.float) # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs1, 3) 

        phi1 = img1
        phi2 = img2
        if self.feature_encoding != 'none':
            phi1 = self.encoder(img1.view(T * B, *img_shape))
            phi2 = self.encoder(img2.view(T * B, *img_shape))
            phi1 = phi1.view(T, B, -1)
            phi2 = phi2.view(T, B, -1)

        predicted_action = self.inverse_model(torch.cat([phi1, phi2], 2))
        predicted_phi2 = self.forward_model(phi1.detach(), action.view(T, B, -1))

        return phi1, phi2, predicted_phi2, predicted_action

    def compute_bonus(self, observations, actions):
        obs1 = observations.clone()[:-1]
        obs2 = observations.clone()[1:]
        phi1, phi2, predicted_phi2, predicted_action = self.forward(obs1, obs2, actions)
        reward = 0.5 * (nn.functional.mse_loss(predicted_phi2, phi2, reduction='none').sum(-1)/self.feature_size)
        return self.prediction_beta * reward

    def compute_loss(self, observations, actions):
        obs1 = observations.clone()[:-1]
        obs2 = observations.clone()[1:]
        # dimension add for when you have only one environment
        if actions.dim() == 2: actions = actions.unsqueeze(1)

        phi1, phi2, predicted_phi2, predicted_action = self.forward(obs1, obs2, actions)
        actions = torch.max(actions.view(-1, *actions.shape[2:]), 1)[1] # convert action to (T * B, action_size)
        inverse_loss = nn.functional.cross_entropy(predicted_action.view(-1, *predicted_action.shape[2:]), actions.detach())
        forward_loss = 0.5 * nn.functional.mse_loss(predicted_phi2, phi2.detach())
        return inverse_loss, forward_loss





