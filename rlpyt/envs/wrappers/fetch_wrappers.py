import gym
from gym import spaces
import numpy as np
from PIL import Image

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
        for _ in range(n_steps):
            obs, reward, done, info = self.env.step([delta_x, delta_y, 0.0055, 0.0])
        return obs, reward, done, info

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
