# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of the environments from 'World Discovery Models'[https://arxiv.org/pdf/1902.07685.pdf]. 
Learn to explore!

This environment uses a simple scrolling mechanism: cropping! As far as the pycolab engine is concerned, 
the game world doesn't scroll at all: it just renders observations that are the size
of the entire map. Only later do "cropper" objects crop out a part of the
observation to give the impression of a moving world/partial observability.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import sys
import numpy as np

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the object sprites
    # 'a', 'b', 'c', 'd' and 'e'. I guess if you really don't want them in your maze
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': bouncing object A.
    #     'P': player starting location.    'b': fixed object B.
    #     ' ': boring old maze floor.       'c': fixed object C.
    #                                       'd': fixed object D.
    #                                       'e': fixed object E.
    #

    # Maze #0: (paper: maze environment)
    ['#####################',
     '#     e      d      #',
     '# @      ##         #',
     '#################   #',
     '#################   #',
     '#          ###   c  #',
     '#    #    ###       #',
     '#   #P  ###     #####',
     '# ############      #',
     '#    #       ###    #',
     '#   ## a  #b  ##    #',
     '#         #         #',
     '#####################'],

    # Maze #1: (paper: 5 rooms environment)
    ['###################',
     '##               ##',
     '# # d           # #',
     '#  #           #  #',
     '#   #         #   #',
     '#    #### ####    #',
     '#    #### ####    #',
     '# a  ##     ##    #',
     '#    ##     ##    #',
     '#        P        #',
     '#    ##     ##    #',
     '#    ##     ##c   #',
     '#    #### ####    #',
     '#    #### ####    #',
     '#   #         #   #',
     '#  #           #  #',
     '# # b           # #',
     '##               ##',
     '###################']
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '@': (999, 862, 110),  # Shimmering golden coins
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780),    # Patroller A
             'b': (145, 987, 341),  # Patroller B
             'c': (987, 623, 145),  # Patroller C
             'd': (987, 623, 145),  # Patroller D
             'e': (987, 623, 145)}  # Patroller E

COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.

ENEMIES = {'a', 'b', 'c', 'd', 'e'} # Globally accessible set of sprites

def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  return ascii_art.ascii_art_to_game(
      MAZES_ART[level], what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'a': BouncingObject,
          'b': FixedObject,
          'c': FixedObject,
          'd': FixedObject,
          'e': FixedObject},
      drapes={
          '@': CashDrape},
      update_schedule=['P', 'a', 'b', 'c', 'd', 'e', '@'],
      z_order='abcde@P')


def make_croppers(level):
  """Builds and returns `ObservationCropper`s for the selected level.

  We make one cropper for each level: centred on the player. Room
  to add more if needed.

  Args:
    level: level to make `ObservationCropper`s for.

  Returns:
    a list of all the `ObservationCropper`s needed.
  """
  return [
      # The player view.
      cropping.ScrollingCropper(rows=5, cols=5, to_track=['P']),
  ]


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer."""

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # stay put? (Not strictly necessary.)
      self._stay(board, the_plot)
    if actions == 5:    # just quit?
      the_plot.terminate_episode()

    # for t in things:
    #   if t in ENEMIES and self.position == things[t].position:
    #     the_plot.terminate_episode()


class BouncingObject(prefab_sprites.MazeWalker):
  """Wanders back and forth horizontally."""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(BouncingObject, self).__init__(
        corner, position, character, impassable='#')
    # Choose our initial direction based on our character value.
    self._moving_east = bool(ord(character) % 2)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)
      if self.position == things['P'].position:
        the_plot.terminate_episode()
      return

    # If there is a wall next to us, we ought to switch direction.
    row, col = self.position
    if layers['#'][row, col-1]: self._moving_east = True
    if layers['#'][row, col+1]: self._moving_east = False

    # Make our move. 
    (self._east if self._moving_east else self._west)(board, the_plot)
    if self.position == things['P'].position:
      the_plot.terminate_episode()

class BrownianObject(prefab_sprites.MazeWalker):
  """Randomly sample direction from left/right/up/down"""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(BrownianObject, self).__init__(corner, position, character, impassable='#')
    # Choose our initial direction.
    self._direction = np.random.choice(4) # 0 = east, 1 = west, 2 = north, 3 = south

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)
      if self.position == things['P'].position:
        the_plot.terminate_episode()
      return

    # Sample a move
    self._direction = np.random.choice(4) # 0 = east, 1 = west, 2 = north, 3 = south

    # Make a move
    if self._direction == 0: self._east(board, the_plot)
    elif self._direction == 1: self._west(board, the_plot)
    elif self._direction == 2: self._north(board, the_plot)
    elif self._direction == 3: self._south(board, the_plot)
    if self.position == things['P'].position:
      the_plot.terminate_episode()

class WhiteNoiseObject(prefab_sprites.MazeWalker):
  """Randomly sample direction from left/right/up/down"""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(WhiteNoiseObject, self).__init__(corner, position, character, impassable='#')
    # Initialize empty space in surrounding radius.
    self._empty_coords = np.array([
                                    [3, 17], [3, 18], [3, 19], 
                                    [4, 17], [4, 18], [4, 19],
                                    [5, 15], [5, 16], [5, 17], [5, 18], [5, 19],
                                    [6, 14], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19],
                                    [7, 12], [7, 13], [7, 14], [7, 15], [7, 16]
                                ])

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

    # We only move once every two game iterations.
    if the_plot.frame % 2:
      self._stay(board, the_plot)
      if self.position == things['P'].position:
        the_plot.terminate_episode()
      return

    # Sample and make a move
    self._teleport(self._empty_coords[np.random.choice(len(self._empty_coords))])
    if self.position == things['P'].position:
      the_plot.terminate_episode()

class FixedObject(plab_things.Sprite):
  """Static object. Doesn't move."""

  def __init__(self, corner, position, character):
    super(FixedObject, self).__init__(
        corner, position, character)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.
    if self.position == things['P'].position:
      the_plot.terminate_episode()

class CashDrape(plab_things.Drape):
  """A `Drape` handling all of the coins.

  This Drape detects when a player traverses a coin, removing the coin and
  crediting the player for the collection. Terminates if all coins are gone.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has reached a coin, credit one reward and remove the coin
    # from the scrolling pattern. If the player has obtained all coins, quit!
    player_pattern_position = things['P'].position

    if self.curtain[player_pattern_position]:
      the_plot.log('Coin collected at {}!'.format(player_pattern_position))
      the_plot.add_reward(1.0)
      self.curtain[player_pattern_position] = False
      if not self.curtain.any(): the_plot.terminate_episode()


def main(argv=()):
  level = int(argv[1]) if len(argv) > 1 else 0

  # Build the game.
  game = make_game(level)
  # Build the croppers we'll use to scroll around in it, etc.
  croppers = make_croppers(level)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
      croppers=croppers)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
