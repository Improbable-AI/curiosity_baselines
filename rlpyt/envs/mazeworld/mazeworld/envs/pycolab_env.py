"""The pycolab environment interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import time
import numbers
import gym
from gym import spaces
from gym import logger
from gym.utils import seeding

import numpy as np

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
                 default_reward,
                 action_space,
                 delay=30,
                 resize_scale=8,
                 crop_window=[5, 5]):
        """Create an `PyColabEnv` adapter to a `pycolab` game as a `gym.Env`.

        You can access the `pycolab.Engine` instance with `env.current_game`.

        Args:
            max_iterations: maximum number of steps.
            default_reward: default reward if reward is None returned by the
                `pycolab` game.
            action_space: the action `Space` of the environment.
            delay: renderer delay.
            resize_scale: number of pixels per observation pixel.
                Used only by the renderer.
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
        observation_layers = list(set(layers))
        self._observation_order = sorted(observation_layers)
        self.observation_space = spaces.Box(0., 1., [len(self.state_layer_chars)] + crop_window) # don't count empty space layer
        self.action_space = action_space

        self.current_game = None
        self._croppers = []
        self._state = None
        self._last_observations = None
        self._empty_board = None
        self._last_painted = None
        self._last_reward = None
        self._game_over = False

        self.viewer = None
        self.resize_scale = resize_scale
        self.delay = delay

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
                'c' : (0., 250., 71.),
                'd' : (250., 0., 129.),
                'e' : (255., 0., 0.),
                '#' : (61., 61., 61.),
                '@' : (255., 255., 0.),
                ' ' : (0., 0., 0.)}

    def _paint_board(self, layers):
        """Method to privately paint layers to RGB.

        Args:
            layers: a dictionary mapping a character to the respective curtain.

        Returns:
            3D np.array (np.uint32) representing the RGB of the observation
                layers.
        """
        board_shape = self._last_observations.board.shape
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
        self._state = []
        for char in self.state_layer_chars:
            if char != ' ':
                mask = observations.layers[char].astype(float)
                self._state.append(mask)
        self._state = np.array(self._state)

        # rendering purposes (RGB)
        self._last_observations = observations
        self._empty_board = np.zeros_like(self._last_observations.board)
        self._last_painted = self._paint_board(observations.layers).astype(np.float32)

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
        if len(self._croppers) > 0:
            observations = [cropper.crop(observations) for cropper in self._croppers][0]
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
            state = self._last_painted
            reward = self._last_reward
            done = self._game_over
            return state, reward, done, {}

        # Execute the action in pycolab.
        self.current_game.the_plot.info = {}
        observations, reward, _ = self.current_game.play(action)
        if len(self._croppers) > 0:
            observations = [cropper.crop(observations) for cropper in self._croppers][0]
        self._update_for_game_step(observations, reward)
        info = self.current_game.the_plot.info

        # Check the current status of the game.
        state = self._last_painted # for rendering
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
        img = self._empty_board
        if self._last_observations:
            img = self._last_observations.board
            layers = self._last_observations.layers
            if self._colors:
                img = self._paint_board(layers)
            else:
                assert img is not None, '`board` must not be `None`.'

        img = _repeat_axes(img, self.resize_scale, axis=[0, 1])
        if len(img.shape) != 3:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = img.astype(np.uint8)

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
