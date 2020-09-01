"""An example implementation of pycolab games as environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import better_scrolly_maze, deepmind_maze
from pycolab import cropping
from mazeworld.envs import pycolab_env

class MazeWorld(pycolab_env.PyColabEnv):
    """Custom maze world game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.state_layer_chars = ['P', '#', 'a', 'b', 'c', 'd', 'e', '@']
        super(MazeWorld, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        self._croppers = self.make_croppers()
        return better_scrolly_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 default_reward=0.):
        self.level = level
        self.state_layer_chars = ['#', 'a', 'b', 'c', 'd', 'e', '@'] # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=8)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]










        