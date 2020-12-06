import gym
from gym import spaces
import numpy as np
from PIL import Image

from collections import namedtuple

from numpy.core.defchararray import lower
from rlpyt.samplers.collections import TrajInfo

class FetchTrajInfo(TrajInfo):
    """TrajInfo class for use with Pycolab Env, to store visitation
    frequencies and any other custom metrics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_block_visits = 0

    def step(self, observation, action, reward_ext, reward_int, done, agent_info, env_info):
        block_visit = getattr(env_info, 'block_visit', None)

        if block_visit is not None and block_visit == True:
            self.total_block_visits += 1

        super().step(observation, action, reward_ext, reward_int, done, agent_info, env_info)

class GridActions(gym.Wrapper):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.03, 0.03)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.action_space = spaces.Discrete(9)
        self.distance = distance

        self.action_mapping = { # maps from action index to (dx, dy)
            0 : (1, 0),
            1 : (-1, 0),
            2 : (0, 1),
            3 : (0, -1),
            4 : (1, 1),
            5 : (-1, -1),
            6 : (1, -1),
            7 : (-1, 1),
            8 : (0, 0)
        }

    def step(self, action):
        delta_x, delta_y = self.action_mapping[action]
        n_steps = int(2 * (self.distance // 0.03)) # 2 steps with force 1 gets you 0.03 distance
        # Keep track of visits throughout all steps instead of just last one
        info = {'block_visit': False}
        for _ in range(n_steps):
            obs, reward, done, info_tmp = self.env.step([delta_x, delta_y, 0.0055, 0.0])
            info_tmp['block_visit'] = info['block_visit'] or info_tmp['block_visit']
            info = info_tmp
        return obs, reward, done, info

class MicroGridActions(GridActions):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.003, 0.003)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.distance = distance

        self.action_mapping = self._init_action_mapping()
        self.action_space = spaces.Discrete(len(self.action_mapping))

    def _init_action_mapping(self, step=0.5):
        lower_bound = -1
        upper_bound = 1 + step

        val_range = np.arange(lower_bound, upper_bound, step)

        action_mapping = dict()

        counter = 0

        for dx in val_range:
            for dy in val_range:
                action_mapping[counter] = (dx, dy)
                counter += 1

        return action_mapping.copy()

class NanoGridActions(MicroGridActions):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.0003, 0.0003)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.distance = distance

        self.action_mapping = self._init_action_mapping(step=0.25)
        self.action_space = spaces.Discrete(len(self.action_mapping))

class PicoGridActions(MicroGridActions):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.0003, 0.0003)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.distance = distance

        self.action_mapping = self._init_action_mapping(step=0.1)
        self.action_space = spaces.Discrete(len(self.action_mapping))

class GridActions3D(gym.Wrapper):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.03, 0.03, 0.03)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.action_space = spaces.Discrete(9)
        self.distance = distance

        self.action_mapping = self._init_action_mapping()
        self.action_space = spaces.Discrete(len(self.action_mapping))

    def _init_action_mapping(self, step=1):
        lower_bound = -1
        upper_bound = 1 + step

        val_range = np.arange(lower_bound, upper_bound, step)

        action_mapping = dict()

        counter = 0

        for dx in val_range:
            for dy in val_range:
                for dz in val_range:
                    action_mapping[counter] = (dx, dy, dz)
                    counter += 1

        return action_mapping.copy()

    def step(self, action):
        delta_x, delta_y, delta_z = self.action_mapping[action]
        n_steps = int(2 * (self.distance // 0.03)) # 2 steps with force 1 gets you 0.03 distance
        # Keep track of visits throughout all steps instead of just last one
        info = {'block_visit': False}
        for _ in range(n_steps):
            obs, reward, done, info_tmp = self.env.step([delta_x, delta_y, delta_z, 0.0])
            info_tmp['block_visit'] = info['block_visit'] or info_tmp['block_visit']
            info = info_tmp
        return obs, reward, done, info

class MicroGridActions3D(GridActions3D):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.003, 0.003, 0.003)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.distance = distance

        self.action_mapping = self._init_action_mapping(step=0.5)
        self.action_space = spaces.Discrete(len(self.action_mapping))

class NanoGridActions3D(MicroGridActions3D):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.0003, 0.0003, 0.0003)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.distance = distance

        self.action_mapping = self._init_action_mapping(step=0.25)
        self.action_space = spaces.Discrete(len(self.action_mapping))

class PicoGridActions3D(MicroGridActions3D):
    '''
        A Wrapper that maps the usual continuous action
        space of the RoboticsEnv to a grid-structured
        environment, where our arm is constrained roughly
        to tiles of size (0.0003, 0.0003, 0.0003)

        Going with Wrapper instead of ActionWrapper, as
        `step()` requires some modification to make this
        work best.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `distance` :: float : multiple of 0.03. Corresponds to unit distance in each direction travelled at each step.
    '''
    def __init__(self, env, distance=0.03):
        super().__init__(env)
        self.distance = distance

        self.action_mapping = self._init_action_mapping(step=0.1)
        self.action_space = spaces.Discrete(len(self.action_mapping))

class ActionNoise(gym.ActionWrapper):
    '''
        A ActionWrapper that applies noise to the actual actions in Fetch environment. This
        MUST BE APPLIED BEFORE the GridActions() wrapper, due to the multiple substeps that
        GridActions uses.

        The RobotEnv environment clips the actions to it's own high low limits, so the noise
        may not be visually noticeable since the actions are already random.

        Parameters
        ----------
        `env` :: OpenAI Gym Env : the env to wrap
        `std`:: float :: the std. deviation of mean-zero guassian normal noise to apply
        `actions_with_noise` :: List of ints or "all" : the indices of the actions
            actually sent to step() that the noise should be applied to.
    '''

    def __init__(self, env, std=0.1, actions_with_noise='all'):
        super(ActionNoise, self).__init__(env)

        # There are 4 actions [pos_control(x,y,z), gripper_control] sent to the
        # _set_action method.  Choose which ones should have noise based on their
        # index.
        if actions_with_noise == 'all':
            self.noise_indices = np.arange(4)
        else:
            self.noise_indices = np.array(actions_with_noise)
                
        self.mu = 0 # Keep zero mean
        self.std = std
        # Random noise generator
        self.rng = np.random.default_rng()

    def step(self, action):
        # Ensure action is numpy array format
        action = np.array(action)
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action[self.noise_indices] += self.rng.normal(loc=self.mu, scale=self.std, size=self.noise_indices.shape)
        return self.env.step(action)

class ResizeImage(gym.ObservationWrapper):
    '''
    A Wrapper takes the image obervation from step and
    resizes it so that it can be smaller, and more sims
    can be trained at once.

    Parameters
    ----------
    `env` :: OpenAI Gym Env : the env to wrap
    `height` :: int : Pixel height to scale image to.
    `width` :: int : Pixel width to scale image to.
    '''

    def __init__(self, env, height=84, width=84):
        super(ResizeImage, self).__init__(env)
        assert self.obs_type == 'img', self.__class__.__name__ + "only works for `img` obs_type"
        self.height = height
        self.width = width

        # Reformat observation space to new image size
        shape = self.observation_space.shape
        assert (len(shape) == 2) or (len(shape) == 3), 'Make sure observation is 2D or 3D img'
        if len(shape) == 2:
            shape = (self.height, self.width)
        else:
            shape = (self.height, self.width, shape[2])
        high = self.observation_space.high
        low = self.observation_space.low
        obs_space_type = self.observation_space.dtype
        low = np.array(Image.fromarray(low).resize((self.height, self.width), resample=Image.BILINEAR), dtype=obs_space_type)
        high = np.array(Image.fromarray(high).resize((self.height, self.width), resample=Image.BILINEAR), dtype=obs_space_type)
        self.observation_space = spaces.Box(low, high, shape, dtype=obs_space_type)
    
    def observation(self, obs):
        """
        Resizes image
        """
        return np.array(Image.fromarray(obs).resize((self.height, self.width), resample=Image.BILINEAR), dtype=self.observation_space.dtype)

class GrayscaleImage(gym.ObservationWrapper):
    '''
    A Wrapper that converts the 3 channel RGB image observation
    into a 1 channel grayscale image

    Parameters
    ----------
    `env` :: OpenAI Gym Env : the env to wrap
    '''

    def __init__(self, env):
        super(GrayscaleImage, self).__init__(env)
        assert self.obs_type == 'img', self.__class__.__name__ + "only works for `img` obs_type"

        # Reformat observation space from 3 channel to 1 channel grayscale
        assert (self.observation_space.shape[2]==3), 'Make sure observation is a 3 channel RGB img'
        shape = (self.height, self.width, 1)
        self.observation_space = spaces.Box(0., 255., shape, dtype=self.observation_space.dtype)

    def observation(self, obs):
        # Grayscale conversion ITU-R601-2 luma transform (http://effbot.org/imagingbook/image.htm)
        # Using 0:1 for channel so that output is still (h, w, 1)
        return (obs[:, :, 0:1] * 0.299 + obs[:, :, 1:2] * 0.587 + obs[:, :, 2:3] * 0.114).astype(self.observation_space.dtype)
