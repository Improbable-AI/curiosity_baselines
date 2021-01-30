import os
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.utils.averages import RunningMeanStd
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
            device='cpu'
            ):
        super(RND, self).__init__()

        self.prediction_beta = prediction_beta
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        c, h, w = 1, image_shape[1], image_shape[2] # assuming grayscale inputs
        self.obs_rms = RunningMeanStd(shape=(1, c, h, w)) # (T, B, c, h, w)
        self.feature_size = 512
        self.conv_feature_size = 3136

        # Fixed weight target model
        self.target_model = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                                          nn.LeakyReLU(),
                                          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                          nn.LeakyReLU(),
                                          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                          nn.LeakyReLU(),
                                          Flatten(),
                                          nn.Linear(self.conv_feature_size, self.feature_size))
        for param in self.target_model.parameters():
            param.requires_grad = False
        for param in self.target_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()

        # Learned predictor model
        self.forward_model = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                           nn.LeakyReLU(),
                                           Flatten(),
                                           nn.Linear(self.conv_feature_size, self.feature_size),
                                           nn.ReLU(),
                                           nn.Linear(self.feature_size, self.feature_size),
                                           nn.ReLU(),
                                           nn.Linear(self.feature_size, self.feature_size))
        for param in self.forward_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()


    def forward(self, obs, computing_loss=False):

        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)

        # img = np.squeeze(obs.data.numpy()[0][0])
        # mean = np.squeeze(self.obs_rms.mean)
        # var = np.squeeze(self.obs_rms.var)
        # std = np.squeeze(np.sqrt(self.obs_rms.var))
        # print(img)
        # cv2.imwrite('images/test.png', img)
        # cv2.imwrite('images/mean.png', mean)
        # cv2.imwrite('images/var.png', var)
        # cv2.imwrite('images/std.png', std)
        # cv2.imwrite('images/whitened.png', img-mean)
        # print((img-mean)/std)
        # print('-'*100)
        # cv2.imwrite('images/final.png', (img-mean)/std)
        # cv2.imwrite('images/scaled_final.png', ((img-mean)/std)*111)

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        
        # normalize observations and clip (see paper for details)
        obs_cpu = obs.clone().cpu().data.numpy()
        if computing_loss is False:
            self.obs_rms.update(obs_cpu.reshape(T*B, *img_shape))
        
        if self.device == 'cuda:0':
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().cuda()
            obs_var = torch.from_numpy(self.obs_rms.var).float().cuda()
        else:
            obs_mean = torch.from_numpy(self.obs_rms.mean).float()
            obs_var = torch.from_numpy(self.obs_rms.var).float()
            
        obs = (obs - obs_mean) / torch.sqrt(obs_var) 
        obs = torch.clamp(obs, -5, 5)
        obs = obs.type(torch.float) # expect torch.uint8 inputs

        # prediction target
        phi = self.target_model(obs.clone().detach().view(T * B, *img_shape)).view(T, B, -1)

        # make prediction
        predicted_phi = self.forward_model(obs.clone().detach().view(T * B, *img_shape)).view(T, B, -1)

        return phi, predicted_phi

    def compute_bonus(self, next_observation):
        phi, predicted_phi = self.forward(next_observation, computing_loss=False)
        rewards = nn.functional.mse_loss(predicted_phi, phi, reduction='none').sum(-1)/self.feature_size
        return self.prediction_beta * rewards

    def compute_loss(self, observations, valid):
        phi, predicted_phi = self.forward(observations, computing_loss=True)
        forward_loss = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size
        forward_loss = valid_mean(forward_loss, valid)
        return forward_loss





