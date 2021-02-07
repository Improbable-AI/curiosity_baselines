import os
from PIL import Image
import numpy as np
import torch
from torch import nn
torch.set_printoptions(edgeitems=3)

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.models.utils import Flatten
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead
import cv2


class RND(nn.Module):
    """Curiosity model for intrinsically motivated agents: 
    """

    def __init__(
            self, 
            image_shape, 
            prediction_beta=1.0,
            drop_probability=1.0,
            gamma=0.99,
            device='cpu'
            ):
        super(RND, self).__init__()

        self.prediction_beta = prediction_beta
        self.drop_probability = drop_probability
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        c, h, w = 1, image_shape[1], image_shape[2] # assuming grayscale inputs
        self.obs_rms = RunningMeanStd(shape=(1, c, h, w)) # (T, B, c, h, w)
        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma)
        self.feature_size = 512
        self.conv_feature_size = 7*7*64

        # Learned predictor model
        # self.forward_model = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
        #                                    nn.LeakyReLU(),
        #                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #                                    nn.LeakyReLU(),
        #                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #                                    nn.LeakyReLU(),
        #                                    Flatten(),
        #                                    nn.Linear(self.conv_feature_size, self.feature_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(self.feature_size, self.feature_size),
        #                                    nn.ReLU(),
        #                                    nn.Linear(self.feature_size, self.feature_size))

        self.forward_model = nn.Sequential(
                                            nn.Conv2d(
                                                in_channels=1,
                                                out_channels=32,
                                                kernel_size=8,
                                                stride=4),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=32,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=64,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1),
                                            nn.LeakyReLU(),
                                            Flatten(),
                                            nn.Linear(self.conv_feature_size, self.feature_size),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_size, self.feature_size),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_size, self.feature_size)
                                            )

        for param in self.forward_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()

        # Fixed weight target model
        # self.target_model = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
        #                                   nn.LeakyReLU(),
        #                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #                                   nn.LeakyReLU(),
        #                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #                                   nn.LeakyReLU(),
        #                                   Flatten(),
        #                                   nn.Linear(self.conv_feature_size, self.feature_size))

        self.target_model = nn.Sequential(
                                            nn.Conv2d(
                                                in_channels=1,
                                                out_channels=32,
                                                kernel_size=8,
                                                stride=4),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=32,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=64,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1),
                                            nn.LeakyReLU(),
                                            Flatten(),
                                            nn.Linear(self.conv_feature_size, self.feature_size)
                                        )

        for param in self.target_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()
        for param in self.target_model.parameters():
            param.requires_grad = False


    def forward(self, obs, done=None):

        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)

        # img = np.squeeze(obs.data.numpy()[0][0])
        # mean = np.squeeze(self.obs_rms.mean)
        # var = np.squeeze(self.obs_rms.var)
        # std = np.squeeze(np.sqrt(self.obs_rms.var))
        # cv2.imwrite('images/original.png', img)
        # cv2.imwrite('images/mean.png', mean)
        # cv2.imwrite('images/var.png', var)
        # cv2.imwrite('images/std.png', std)
        # cv2.imwrite('images/whitened.png', img-mean)
        # cv2.imwrite('images/final.png', (img-mean)/std)
        # cv2.imwrite('images/scaled_final.png', ((img-mean)/std)*111)

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        
        # normalize observations and clip (see paper for details)
        if done is not None:
            obs_cpu = obs.clone().cpu().data.numpy()
            done = done.cpu().data.numpy()
            done = np.sum(np.abs(done-1), axis=0)
            obs_cpu = np.swapaxes(obs_cpu, 0, 1)
            sliced_obs = obs_cpu[0][:int(done[0].item())]
            for i in range(1, B):
                c = obs_cpu[i]
                data_chunk = obs_cpu[i][:int(done[i].item())]
                sliced_obs = np.concatenate((sliced_obs, data_chunk))
            self.obs_rms.update(sliced_obs)
        
        if self.device == torch.device('cuda:0'):
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().cuda()
            obs_var = torch.from_numpy(self.obs_rms.var).float().cuda()
        else:
            obs_mean = torch.from_numpy(self.obs_rms.mean).float()
            obs_var = torch.from_numpy(self.obs_rms.var).float()

        obs = ((obs - obs_mean) / torch.sqrt(obs_var))
        obs = torch.clamp(obs, -5, 5)
        obs = obs.type(torch.float) # expect torch.uint8 inputs

        # prediction target
        phi = self.target_model(obs.clone().detach().view(T * B, *img_shape)).view(T, B, -1)

        # make prediction
        predicted_phi = self.forward_model(obs.detach().view(T * B, *img_shape)).view(T, B, -1)

        return phi, predicted_phi, T, B

    def compute_bonus(self, next_observation, done):
        phi, predicted_phi, T, _ = self.forward(next_observation, done=done)
        # rewards = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size
        rewards = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/2
        rewards_cpu = rewards.clone().cpu().data.numpy()
        done = torch.abs(done-1).cpu().data.numpy()
        total_rew_per_env = list()
        for i in range(T):
            update = self.rew_rff.update(rewards_cpu[i], done=done[i])
            total_rew_per_env.append(update)
        total_rew_per_env = np.array(total_rew_per_env)
        mean_length = np.mean(np.sum(np.swapaxes(done, 0, 1), axis=1))

        self.rew_rms.update_from_moments(np.mean(total_rew_per_env), np.var(total_rew_per_env), mean_length)
        if self.device == torch.device('cuda:0'):
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float().cuda()
            done = torch.from_numpy(np.array(done)).float().cuda()
        else:
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float()
            done = torch.from_numpy(np.array(done)).float()
        rewards /= torch.sqrt(rew_var)

        rewards *= done
        return self.prediction_beta * rewards

    def compute_loss(self, observations, valid):
        phi, predicted_phi, T, B = self.forward(observations, done=None)
        forward_loss = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size
        mask = torch.rand(forward_loss.shape)
        mask = (mask > self.drop_probability).type(torch.FloatTensor).to(self.device)
        forward_loss = forward_loss * mask.detach()
        forward_loss = valid_mean(forward_loss, valid.detach())
        return forward_loss


