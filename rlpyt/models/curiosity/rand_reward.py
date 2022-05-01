import cv2
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead
from rlpyt.models.utils import Flatten
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
torch.set_printoptions(edgeitems=3)


class RandReward(nn.Module):

    def __init__(self,
                 image_shape,
                 device='cpu'
                 ):

        super(RandReward, self).__init__()

        # assuming grayscale inputs
        c, h, w = 1, image_shape[1], image_shape[2]
        self.feature_size = 512
        self.conv_feature_size = 7*7*64
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        # Initialize network with random weigths, just to figure out the dimensions of args
        with torch.no_grad():
            self.network = nn.Sequential(
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


    def forward(self, obs, done=None):
        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)

        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        with torch.no_grad():
            obs = obs.type(torch.float) # expect torch.uint8 inputs
            predicted_phi = self.network(obs.detach().view(T * B, *img_shape)).view(T, B, -1)

        return predicted_phi, T, B

    def compute_bonus(self, next_observation, done):
        with torch.no_grad():
            predicted_phi, T, B = self.forward(next_observation, done)
            rewards = predicted_phi.detach().sum(-1)/self.feature_size
        return rewards

    def compute_loss(self, observations, valid):
        with torch.no_grad():
            predicted_phi, T, B = self.forward(observations, done=None)
            R = torch.rand_like(predicted_phi)
            forward_loss = (nn.functional.mse_loss(predicted_phi.detach(), R, reduction='none').sum(-1)/self.feature_size).mean(dim=()).detach()

        return forward_loss


