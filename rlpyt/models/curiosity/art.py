from collections import defaultdict
from sofm.art import OnlineFuzzyART
from rlpyt.models.curiosity.fuzzy_art import FuzzyART
from typing import Tuple, Callable, Generator
import collections
import warnings
import scipy.stats
import scipy as scp
import cv2
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead, ARTHead
from rlpyt.models.utils import Flatten
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
torch.set_printoptions(edgeitems=3)

# Kohonen imports


class ART(nn.Module):

    def __init__(self,
                 image_shape,
                 rho=0.2,
                 alpha=0.1,
                 beta=0.01,
                 art_input_dim=16,
                 gamma=0.99,
                 std_rew_scaling=1.0,
                 headless=False,
                 frame_stacking=True,
                 device='cpu'
                 ):

        super(ART, self).__init__()

        # assuming grayscale inputs
        self.feature_size = 512
        self.conv_feature_size = 7*7*64
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma)
        self.std_rew_scaling = std_rew_scaling

        # TODO(marius): Make into parameters defined externally
        # TODO(odin): Fix to whatever is actual
        self.encoded_input_dim = art_input_dim
        self.encoding_batch_norm = True
        self.frame_stacking = frame_stacking

        if self.frame_stacking:
            # Set image shape to be without the frames stacked
            image_shape = (1,) + image_shape[1:]

        self.headless = headless

        with torch.no_grad():
            if headless:
                self.feature_encoder = nn.Flatten()
                art_input_dim = np.prod(image_shape)
            else:
                self.feature_encoder = ARTHead(
                    image_shape=image_shape,
                    output_size=self.encoded_input_dim
                )

        self.encoded_input_dim = art_input_dim
        self.fuzzy_art = OnlineFuzzyART(
            rho=rho, alpha=alpha, beta=beta, num_features=self.encoded_input_dim)

        self.seen_classes = defaultdict(lambda: 0)

    def forward(self, obs, done=None):
        if self.frame_stacking:
            # in case of frame stacking
            obs = obs[:, :, -1, :, :]
            obs = obs.unsqueeze(2)

        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        obs = obs.type(torch.float)
        with torch.no_grad():
            obs_feature_mapped = self.feature_encoder.forward(
                obs.view(T * B, *img_shape))

        obs_map = obs_feature_mapped.detach().cpu().numpy()
        predictions = self.fuzzy_art.run_online(obs_map, max_epochs=20)
        self.update_seen_classes(predictions)
        rewards = self.compute_rewards(predictions)

        ret = rewards.reshape(T, B)

        return ret, T, B

    def update_seen_classes(self, predictions):
        for prediction in predictions:
            self.seen_classes[prediction] += 1

    def compute_rewards(self, predictions):
        num_seen = np.array([self.seen_classes[prediction]
                            for prediction in predictions], dtype=float)
        return 1.0 / np.sqrt(num_seen)

    def compute_bonus(self, next_observation, done):
        rewards_cpu, T, B = self.forward(next_observation, done)
        done = torch.abs(done-1).cpu().data.numpy()
        total_rew_per_env = list()
        for i in range(T):
            update = self.rew_rff.update(rewards_cpu[i], done=done[i])
            total_rew_per_env.append(update)
        total_rew_per_env = np.array(total_rew_per_env)
        mean_length = np.mean(np.sum(np.swapaxes(done, 0, 1), axis=1))

        self.rew_rms.update_from_moments(
            np.mean(total_rew_per_env), np.var(total_rew_per_env), mean_length)
        if self.device == torch.device('cuda:0'):
            rew_var = torch.from_numpy(
                np.array(self.rew_rms.var)).float().cuda()
            done = torch.from_numpy(np.array(done)).float().cuda()
            rewards = torch.from_numpy(rewards_cpu).float().cuda()
        else:
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float()
            done = torch.from_numpy(np.array(done)).float()
            rewards = torch.from_numpy(rewards_cpu).float()

        rewards /= torch.sqrt(rew_var)

        rewards *= done

        return rewards * self.std_rew_scaling

    def compute_loss(self, observations, valid):
        return torch.zeros(tuple()), len(self.seen_classes)
