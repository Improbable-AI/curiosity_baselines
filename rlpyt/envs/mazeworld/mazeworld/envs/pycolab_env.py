"""The pycolab environment interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import abc
import time
import numbers
import gym
from gym import spaces
from gym import logger
from gym.utils import seeding

import numpy as np
from collections import namedtuple

from rlpyt.samplers.collections import TrajInfo
import matplotlib.pyplot as plt

EnvInfo = namedtuple("EnvInfo", ["visitation_frequency", "first_visit_time", "traj_done"])

class PycolabTrajInfo(TrajInfo):
    """TrajInfo class for use with Pycolab Env, to store visitation
    frequencies and any other custom metrics. Has room to store up to 5 sprites
    currently, but can be expanded. Can store fewer automatically as well. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visit_freq_a = 0
        self.visit_freq_b = 0
        self.visit_freq_c = 0
        self.visit_freq_d = 0
        self.visit_freq_e = 0
        self.visit_freq_f = 0
        self.visit_freq_g = 0
        self.visit_freq_h = 0
        self.visit_freq_i = 0
        self.first_visit_a = 500
        self.first_visit_b = 500
        self.first_visit_c = 500
        self.first_visit_d = 500
        self.first_visit_e = 500
        self.first_visit_f = 500
        self.first_visit_g = 500
        self.first_visit_h = 500
        self.first_visit_i = 500

    def step(self, observation, action, reward_ext, done, agent_info, env_info):
        visitation_frequency = getattr(env_info, 'visitation_frequency', None)
        first_visit_time = getattr(env_info, 'first_time_visit', None)

        if visitation_frequency is not None and first_visit_time is not None:
            if len(visitation_frequency) >= 1:
                if first_visit_time[0] == 500 and visitation_frequency[0] == 1:
                    self.first_visit_a = self.Length
                self.visit_freq_a = visitation_frequency[0]
            if len(visitation_frequency) >= 2:
                if first_visit_time[1] == 500 and visitation_frequency[1] == 1:
                    self.first_visit_b = self.Length
                self.visit_freq_b = visitation_frequency[1]
            if len(visitation_frequency) >= 3:
                if first_visit_time[2] == 500 and visitation_frequency[2] == 1:
                    self.first_visit_c = self.Length
                self.visit_freq_c = visitation_frequency[2]
            if len(visitation_frequency) >= 4:
                if first_visit_time[3] == 500 and visitation_frequency[3] == 1:
                    self.first_visit_d = self.Length
                self.visit_freq_d = visitation_frequency[3]
            if len(visitation_frequency) >= 5:
                if first_visit_time[4] == 500 and visitation_frequency[4] == 1:
                    self.first_visit_e = self.Length
                self.visit_freq_e = visitation_frequency[4]
            if len(visitation_frequency) >= 6:
                if first_visit_time[5] == 500 and visitation_frequency[5] == 1:
                    self.first_visit_f = self.Length
                self.visit_freq_f = visitation_frequency[5]
            if len(visitation_frequency) >= 7:
                if first_visit_time[6] == 500 and visitation_frequency[6] == 1:
                    self.first_visit_g = self.Length
                self.visit_freq_g = visitation_frequency[6]
            if len(visitation_frequency) >= 8:
                if first_visit_time[7] == 500 and visitation_frequency[7] == 1:
                    self.first_visit_h = self.Length
                self.visit_freq_h = visitation_frequency[7]
            if len(visitation_frequency) >= 9:
                if first_visit_time[8] == 500 and visitation_frequency[8] == 1:
                    self.first_visit_i = self.Length
                self.visit_freq_i = visitation_frequency[8]

        super().step(observation, action, reward_ext, done, agent_info, env_info)

def _repeat_axes(x, factor, axis=[0, 1]):
    """Repeat np.array tiling it by `factor` on all axes.

    Args:
        x: input array.
        factor: number of repeats per axis.
        axis: axes to repeat x by factor.

    Returns:
        repeated array with shape `[x.shape[ax] * factor for ax in axis]`
    """
    x_ = x
    for ax in axis:
        x_ = np.repeat(
            x_, factor, axis=ax)
    return x_


class PyColabEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 max_iterations,
                 obs_type,
                 default_reward,
                 action_space,
                 act_null_value=4,
                 delay=30,
                 resize_scale=8,
                 crop_window=[5, 5]):
        """Create an `PyColabEnv` adapter to a `pycolab` game as a `gym.Env`.

        You can access the `pycolab.Engine` instance with `env.current_game`.

        Args:
            max_iterations: maximum number of steps.
            obs_type: type of observation to return.
            default_reward: default reward if reward is None returned by the
                `pycolab` game.
            action_space: the action `Space` of the environment.
            delay: renderer delay.
            resize_scale: number of pixels per observation pixel.
                Used only by the renderer.
            crop_window: dimensions of observation cropping.
        """
        assert max_iterations > 0
        assert isinstance(default_reward, numbers.Number)

        self._max_iterations = max_iterations
        self._default_reward = default_reward

        # At this point, the game would only want to access the random
        # property, although it is set to None initially.
        self.np_random = None

        self._colors = self.make_colors()
        test_game = self.make_game()
        test_game.the_plot.info = {}
        observations, _, _ = test_game.its_showtime()
        layers = list(observations.layers.keys())
        not_ordered = list(set(layers) - set(test_game.z_order))
        self._render_order = list(reversed(not_ordered + test_game.z_order))

        # Create the observation space.
        self.obs_type = obs_type

        if self.obs_type == 'mask':
            self.observation_space = spaces.Box(0., 1., [len(self.state_layer_chars)] + crop_window) # don't count empty space layer
        elif self.obs_type == 'rgb':
            self.observation_space = spaces.Box(0., 255., [crop_window[0]*resize_scale, crop_window[1]*resize_scale] + [3])
        self.action_space = action_space
        self.act_null_value = act_null_value

        self.current_game = None
        self._croppers = []
        self._state = None

        self._last_uncropped_observations = None
        self._empty_uncropped_board = None
        self._last_cropped_observations = None
        self._empty_cropped_board = None

        self._last_reward = None
        self._game_over = False

        self.viewer = None
        self.resize_scale = resize_scale
        self.delay = delay

        # Metrics
        self.visitation_frequency = {char:0 for char in self.objects}
        self.first_visit_time = {char:500 for char in self.objects}

        # Heatmaps
        self.episodes = 0 # number of episodes run (to determine when to save heatmaps)
        self.heatmap_save_freq = 3 # save heatmaps every 3 episodes
        self.heatmap = np.ones((5, 5)) # stores counts each episode (5x5 is a placeholder)

    def pycolab_init(self, logdir, log_heatmaps):
        self.log_heatmaps = log_heatmaps
        root_path = os.path.abspath(__file__).split('/')[1:]
        root_path = root_path[:root_path.index('curiosity_baselines')+1]
        self.heatmap_path = '/' + '/'.join(root_path) + '/' + '/'.join(logdir.split('/')[1:]) + '/heatmaps'
        if os.path.isdir(self.heatmap_path) == False and log_heatmaps == True:
            os.makedirs(self.heatmap_path)

    @abc.abstractmethod
    def make_game(self):
        """Function that creates a new pycolab game.

        Returns:
            pycolab.Engine.
        """
        pass

    def make_colors(self):
        """Functions that returns colors.

        Returns:
            Dictionary mapping key name to `tuple(R, G, B)`.
        """

        return {'P' : (255., 255., 255.),
                'a' : (175., 255., 15.),
                'b' : (21., 0., 255.),
                'c' : (250., 0., 129.),
                'd' : (0., 250., 71.),
                'e' : (255., 0., 0.),
                'f' : (252., 28., 3.),
                'g' : (136., 3., 252.),
                'h' : (20., 145., 60.),
                '#' : (61., 61., 61.),
                '@' : (255., 255., 0.),
                ' ' : (0., 0., 0.)}


    def _paint_board(self, layers, cropped=False):
        """Method to privately paint layers to RGB.

        Args:
            layers: a dictionary mapping a character to the respective curtain.
            cropped: whether or not this is being called to paint cropped or
                     uncropped images.

        Returns:
            3D np.array (np.uint32) representing the RGB of the observation
                layers.
        """
        if not cropped:
            board_shape = self._last_uncropped_observations.board.shape
        else:
            board_shape = self._last_cropped_observations.board.shape

        board = np.zeros(list(board_shape) + [3], np.uint32)
        board_mask = np.zeros(list(board_shape) + [3], np.bool)

        for key in self._render_order:
            color = self._colors.get(key, (0, 0, 0))
            color = np.reshape(color, [1, 1, -1]).astype(np.uint32)

            # Broadcast the layer to [H, W, C].
            board_layer_mask = np.array(layers[key])[..., None]
            board_layer_mask = np.repeat(board_layer_mask, 3, axis=-1)

            # Update the board with the new layer.
            board = np.where(
                np.logical_not(board_mask),
                board_layer_mask * color,
                board)

            # Update the mask.
            board_mask = np.logical_or(board_layer_mask, board_mask)
        return board

    def _update_for_game_step(self, observations, reward):
        """Update internal state with data from an environment interaction."""
        # disentangled one hot state

        if self.obs_type == 'mask':
            self._state = []
            for char in self.state_layer_chars:
                if char != ' ':
                    mask = observations.layers[char].astype(float)
                    if char in self.objects and 1. in mask:
                        self.visitation_frequency[char] += 1
                    self._state.append(mask)
            self._state = np.array(self._state)

        elif self.obs_type == 'rgb':
            rgb_img = self._paint_board(observations.layers, cropped=True).astype(float)
            self._state = self.resize(rgb_img)
            for char in self.state_layer_chars:
                if char != ' ':
                    mask = observations.layers[char].astype(float)
                    if char in self.objects and 1. in mask:
                        self.visitation_frequency[char] += 1

        # update heatmap metric
        if self.log_heatmaps == True:
            pr, pc = self.current_game.things['P'].position
            self.heatmap[pr, pc] += 1

        self._last_reward = reward if reward is not None else \
            self._default_reward

        self._game_over = self.current_game.game_over

        if self.current_game.the_plot.frame >= self._max_iterations:
            self._game_over = True

    def reset(self):
        """Start a new episode."""
        self.current_game = self.make_game()
        for cropper in self._croppers:
            cropper.set_engine(self.current_game)
        self._colors = self.make_colors()
        self.current_game.the_plot.info = {}
        self._game_over = None
        self._last_observations = None
        self._last_reward = None

        observations, reward, _ = self.current_game.its_showtime()
        self._last_uncropped_observations = observations
        self._empty_uncropped_board = np.zeros_like(self._last_uncropped_observations.board)
        if len(self._croppers) > 0:
            observations = [cropper.crop(observations) for cropper in self._croppers][0]
            self._last_cropped_observations = observations
            self._empty_cropped_board = np.zeros_like(self._last_cropped_observations)

        # save and reset metrics
        self.visitation_frequency = {char:0 for char in self.objects}
        if self.log_heatmaps == True and self.episodes % self.heatmap_save_freq == 0:
            np.save('{}/{}.npy'.format(self.heatmap_path, self.episodes), self.heatmap)
            heatmap_normed = self.heatmap / np.linalg.norm(self.heatmap)
            plt.imsave('{}/{}.png'.format(self.heatmap_path, self.episodes), heatmap_normed, cmap='afmhot', vmin=0.0, vmax=1.0)
        self.episodes += 1
        self.heatmap = np.zeros(self._last_uncropped_observations.board.shape)
        
        # run update
        self._update_for_game_step(observations, reward)
        return self._state

    def step(self, action):
        """Apply action, step the world forward, and return observations.

        Args:
            action: the desired action to apply to the environment.

        Returns:
            state, reward, done, info.
        """
        if self.current_game is None:
            logger.warn("Episode has already ended, call `reset` instead..")
            self._state = None
            reward = self._last_reward
            done = self._game_over
            return self._state, reward, done, {}

        # Execute the action in pycolab.
        self.current_game.the_plot.info = {}
        observations, reward, _ = self.current_game.play(action)
        self._last_uncropped_observations = observations
        self._empty_uncropped_board = np.zeros_like(self._last_uncropped_observations.board)

        # Crop and update
        if len(self._croppers) > 0:
            observations = [cropper.crop(observations) for cropper in self._croppers][0]
            self._last_cropped_observations = observations
            self._empty_cropped_board = np.zeros_like(self._last_cropped_observations.board)

        self._update_for_game_step(observations, reward)
        info = self.current_game.the_plot.info

        # Add custom metrics
        info['visitation_frequency'] = self.visitation_frequency
        info['first_time_visit'] = self.first_visit_time

        # Check the current status of the game.
        reward = self._last_reward
        done = self._game_over

        if self._game_over:
            self.current_game = None

        return self._state, reward, done, info

    def render(self, mode='rgb_array', close=False):
        """Render the board to an image viewer or an np.array.

        Args:
            mode: One of the following modes:
                - 'human': render to an image viewer.
                - 'rgb_array': render to an RGB np.array (np.uint8)

        Returns:
            3D np.array (np.uint8) or a `viewer.isopen`.
        """
        img = self._empty_uncropped_board
        if self._last_uncropped_observations:
            img = self._last_uncropped_observations.board
            layers = self._last_uncropped_observations.layers
            if self._colors:
                img = self._paint_board(layers, cropped=False)
            else:
                assert img is not None, '`board` must not be `None`.'

        img = self.resize(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control.rendering import (
                    SimpleImageViewer)
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            time.sleep(self.delay / 1e3)
            return self.viewer.isopen

    def resize(self, img):
        img = _repeat_axes(img, self.resize_scale, axis=[0, 1])
        if len(img.shape) != 3:
            img = np.repeat(img[..., None], 3, axis=-1)
        return img.astype(np.uint8)

    def seed(self, seed=None):
        """Seeds the environment.

        Args:
            seed: seed of the random engine.

        Returns:
            [seed].
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Tears down the renderer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None

