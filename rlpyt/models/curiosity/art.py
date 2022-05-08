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

# Kohonen imports
import scipy as scp
import scipy.stats
import warnings

import collections
from typing import Tuple, Callable, Generator

from rlpyt.models.curiosity.fuzzy_art import FuzzyART
from sofm.art import OnlineFuzzyART

from collections import defaultdict


class ScalingSigmoid(nn.Sigmoid):
    def __init__(self, scaling=1.0):
        super().__init__()
        self.scaling = scaling

    def forward(self, feature):
        return super().forward(feature*self.scaling)


class ART(nn.Module):

    def __init__(self,
                 image_shape,
                 rho=0.2,
                 alpha=0.1,
                 beta=0.01,
                 art_input_dim=16,
                gamma=0.99,
                std_rew_scaling=1.0,
                 device='cpu'
                 ):

        super(ART, self).__init__()

        # assuming grayscale inputs
        c, h, w = 1, image_shape[1], image_shape[2]
        self.feature_size = 512
        self.conv_feature_size = 7*7*64
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma)
        self.std_rew_scaling = std_rew_scaling

        # TODO(marius): Make into parameters defined externally
        self.encoded_input_dim = art_input_dim  # TODO(odin): Fix to whatever is actual
        self.encoding_batch_norm = True

        self.feature_encoder = nn.Sequential(
            MazeHead((1, h, w), output_size=self.encoded_input_dim, batch_norm=self.encoding_batch_norm),
            ScalingSigmoid(scaling=0.5)
        )

        # self.fuzzy_art = FuzzyART(rho=rho, alpha=alpha, beta=beta)
        self.fuzzy_art = OnlineFuzzyART(rho=rho, alpha=alpha, beta=beta, num_features=art_input_dim)

        self.seen_classes = defaultdict(lambda : 0)


    def forward(self, obs, done=None):
        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)

        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        obs = obs.type(torch.float)
        obs_feature_mapped = self.feature_encoder.forward(obs.view(T * B, *img_shape))
        # rewards = torch.zeros(T*B)

        # assert reduced_dim_sample.shape == (self.encoded_input_dim,)

        # with torch.no_grad():
        #     obs = obs.type(torch.float) # expect torch.uint8 inputs
        #     predicted_phi = self.network(obs.detach().view(T * B, *img_shape)).view(T, B, -1)
        # TODO(marius): Handle being done

        obs_map = obs_feature_mapped.detach().cpu().numpy()
        # self.fuzzy_art.fit(obs_map)
        # predictions = torch.LongTensor(self.fuzzy_art.predict(obs_map))
        predictions = self.fuzzy_art.run_online(obs_map, max_epochs=20)
        self.update_seen_classes(predictions)
        rewards = self.compute_rewards(predictions)

        # for idx, obs_map in enumerate(obs_feature_mapped.detach().cpu().numpy()):
        #     predictions = int(torch.Tensor(self.fuzzy_art.predict(obs_map[None])).item())
        #     self.update_seen_classes(predictions)
        #     rewards[idx] = self.compute_reward(predictions)

        ret = rewards.reshape(T, B)

        return ret, T, B
        # return predicted_phi, T, B

    def update_seen_classes(self, predictions):
        for prediction in predictions:
            self.seen_classes[prediction] += 1

    def compute_rewards(self, predictions):
        num_seen = np.array([self.seen_classes[prediction] for prediction in predictions], dtype=float)
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

        self.rew_rms.update_from_moments(np.mean(total_rew_per_env), np.var(total_rew_per_env), mean_length)
        if self.device == torch.device('cuda:0'):
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float().cuda()
            done = torch.from_numpy(np.array(done)).float().cuda()
        else:
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float()
            done = torch.from_numpy(np.array(done)).float()

        rewards = torch.from_numpy(rewards_cpu)
        rewards /= (torch.sqrt(rew_var) * self.std_rew_scaling)

        rewards *= done

        return rewards

    def compute_loss(self, observations, valid):
        # # TODO(marius): Verify observations shape
        # observations = observations[:,:,-1,:,:]
        # observations = observations.unsqueeze(2)

        # lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        # observations = observations.type(torch.float)
        # obs_feature_mapped = self.feature_encoder.forward(observations.view(T * B, *img_shape))
        # # Scale input and mapt to [0, 1]
        # obs_feature_mapped = nn.functional.sigmoid(obs_feature_mapped*0.1)

        # def get_sample_gen():
        #     while True:
        #         rnd_idx = np.random.randint(0, len(observations))
        #         obs = obs_feature_mapped[rnd_idx]
        #         # TODO(odin): Do feature mammping on obs
        #         yield obs

        # sample_generator = get_sample_gen()

        return torch.zeros(tuple()), len(self.seen_classes)
