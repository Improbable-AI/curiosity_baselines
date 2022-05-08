from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.sharedctypes import Value

import numpy as np

from gym import spaces

from pycolab.examples import (better_scrolly_maze,
                              deepmind_maze,
                              deepmind_5room,
                              deepmind_5room_randomfixed,
                              deepmind_5room_bouncing,
                              deepmind_5room_brownian,
                              deepmind_8room,
                              deepmind_8room_v1,
                              deepmind_5room_moveable,
                              deepmind_5room_moveable_v1,
                              ordeal
                              )
from pycolab import cropping
from . import pycolab_env


class MazeWorld(pycolab_env.PyColabEnv):
    """Custom maze world game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', '@']
        self.state_layer_chars = ['P', '#'] + self.objects
        super(MazeWorld, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return better_scrolly_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_5room, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_randomfixed(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 2.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_5room_randomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_randomfixed.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_bouncing(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 3.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_5room_bouncing, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_bouncing.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_brownian(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 4.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_5room_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_brownian.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_maze(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 5.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_maze, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_8room(pycolab_env.PyColabEnv):
    """An eight room environment with many fixed objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_8room, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_8room_v1(pycolab_env.PyColabEnv):
    """An eight room environment with one fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['a']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_8room_v1, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room_v1.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_moveable(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['e', 'b']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_5room_moveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_moveable_v1(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object that has stochastic movement.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.):
        self.level = level
        self.objects = ['e', 'b']
        # each char will produce a layer in the disentangled state
        self.state_layer_chars = ['#'] + self.objects
        super(DeepmindMazeWorld_5room_moveable_v1, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            resize_scale=17)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable_v1.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class OrdealEnv(pycolab_env.PyColabEnv):
    def __init__(self,
                 obs_type='mask',
                 default_reward=0.0,
                 max_iterations=500):

        self.objects = ['S', 'D']
        self.state_layer_chars = ['#', '%', '~', '@', 'w', 'P'] + self.objects

        super(OrdealEnv, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            # left, right, up, down, no action
            action_space=spaces.Discrete(4 + 1)
        )

    def make_game(self):
        return ordeal.make_game()

    def make_colors(self):
        return ordeal.COLOURS



# class OrdealEnv(pycolab_env.PyColabEnv):
#     def __init__(self,
#                  obs_type='mask',
#                  default_reward=0.0,
#                  max_iterations=500):
#         self.chapters = ['kansas', 'cavern', 'castle']

#         super(OrdealEnv, self).__init__(
#             max_iterations=max_iterations,
#             obs_type=obs_type,
#             default_reward=default_reward,
#             # left, right, up, down, no action
#             action_space=spaces.Discrete(4 + 1)
#         )

#     @property
#     def objects(self):
#         chapter = self.game._current_game.the_plot.this_chapter
#         if chapter == 'kansas':
#             return []
#         elif chapter == 'cavern':
#             return ['S']
#         elif chapter == 'castle':
#             return ['D']
#         else:
#             raise ValueError(f"current plot is {chapter}, must be one of {self.chapters}")

#     @property
#     def state_layer_chars(self):
#         return ['#', '%', '~', '@', 'w', 'P'] + self.objects

#     def make_game(self):
#         self.game = ordeal.make_game()
#         return self.game

#     def make_colors(self):
#         return ordeal.COLOURS
