
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
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

class Disagreement(nn.Module):
    """Curiosity model for intrinsically motivated agents: similar to ICM
    except there is an ensemble of forward models that each make predictions.
    The intrinsic reward is defined as the variance between these predictions.
    """

    def __init__(
            self, 
            image_shape, 
            action_size,
            ensemble_size=5,
            feature_encoding='idf', 
            batch_norm=False,
            prediction_beta=1.0,
            obs_stats=None,
            std_rew_scaling=1.0,
            device="cpu",
            forward_loss_wt=0.2,
            ):
        super(Disagreement, self).__init__()

        self.ensemble_size = ensemble_size
        self.prediction_beta = prediction_beta
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
        self.std_rew_scaling = std_rew_scaling
        self.device = torch.device("cuda:0" if device == "gpu" else "cpu")

        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        if forward_loss_wt == -1.0:
            self.forward_loss_wt = 1.0
            self.inverse_loss_wt = 1.0
        else:
            self.forward_loss_wt = forward_loss_wt
            self.inverse_loss_wt = 1-forward_loss_wt

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

        self.forward_model_1 = ResForward(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_2 = ResForward(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_3 = ResForward(feature_size=self.feature_size, action_size=action_size).to(self.device)
        self.forward_model_4 = ResForward(feature_size=self.feature_size, action_size=action_size).to(self.device)

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
            phi1 = phi1.view(T, B, -1) # make sure you're not mixing data up here
            phi2 = phi2.view(T, B, -1)

        predicted_action = self.inverse_model(torch.cat([phi1, phi2], 2))

        predicted_phi2 = []

        predicted_phi2.append(self.forward_model_1(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_2(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_3(phi1.detach(), action.view(T, B, -1).detach()))
        predicted_phi2.append(self.forward_model_4(phi1.detach(), action.view(T, B, -1).detach()))

        predicted_phi2_stacked = torch.stack(predicted_phi2)

        return phi1, phi2, predicted_phi2, predicted_phi2_stacked, predicted_action

    def compute_bonus(self, observations, next_observations, actions):
        phi1, phi2, predicted_phi2, predicted_phi2_stacked, predicted_action = self.forward(observations, next_observations, actions)
        feature_var = torch.var(predicted_phi2_stacked, dim=0) # feature variance across forward models
        reward = torch.mean(feature_var, axis=-1) # mean over feature
        return self.prediction_beta * reward * self.std_rew_scaling

    def compute_loss(self, observations, next_observations, actions, valid):
        #------------------------------------------------------------#
        # hacky dimension add for when you have only one environment (debugging)
        if actions.dim() == 2: 
            actions = actions.unsqueeze(1)
        #------------------------------------------------------------#
        phi1, phi2, predicted_phi2, predicted_phi2_stacked, predicted_action = self.forward(observations, next_observations, actions)
        actions = torch.max(actions.view(-1, *actions.shape[2:]), 1)[1] # conver action to (T * B, action_size), then get target indexes
        inverse_loss = nn.functional.cross_entropy(predicted_action.view(-1, *predicted_action.shape[2:]), actions.detach(), reduction='none').view(phi1.shape[0], phi2.shape[1])
        inverse_loss = valid_mean(inverse_loss, valid)
        
        forward_loss = torch.tensor(0.0, device=self.device)

        forward_loss_1 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[0], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_1, valid)

        forward_loss_2 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[1], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_2, valid)

        forward_loss_3 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[2], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_3, valid)

        forward_loss_4 = nn.functional.dropout(nn.functional.mse_loss(predicted_phi2[3], phi2.detach(), reduction='none'), p=0.2).sum(-1)/self.feature_size
        forward_loss += valid_mean(forward_loss_4, valid)

        return self.inverse_loss_wt*inverse_loss, self.forward_loss_wt*forward_loss






