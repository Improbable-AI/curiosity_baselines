import gym
from gym import spaces
import numpy as np

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
        self.action_space = spaces.Discrete(8)
        self.distance = distance

        self.action_mapping = { # maps from action index to (dx, dy)
            0 : (1, 0),
            1 : (-1, 0),
            2 : (0, 1),
            3 : (0, -1),
            4 : (1, 1),
            5 : (-1, -1),
            6 : (1, -1),
            7 : (-1, 1)
        }

    def step(self, action):
        delta_x, delta_y = self.action_mapping[action]
        n_steps = int(2 * (self.distance // 0.03)) # 2 steps with force 1 gets you 0.03 distance
        for _ in range(n_steps):
            obs, reward, done, info = self.env.step([delta_x, delta_y, 0.0055, 0.0])
        return obs, reward, done, info


