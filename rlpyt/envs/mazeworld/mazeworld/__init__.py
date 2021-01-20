from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from .envs import (MazeWorld, 
                    DeepmindMazeWorld_maze, 
                    DeepmindMazeWorld_5room, 
                    DeepmindMazeWorld_5room_randomfixed, 
                    DeepmindMazeWorld_5room_bouncing,
                    DeepmindMazeWorld_5room_brownian,
                    DeepmindMazeWorld_8room,
                    DeepmindMazeWorld_8room_v1,
                    DeepmindMazeWorld_5room_moveable,
                    DeepmindMazeWorld_5room_moveable_v1)

def register(id, entry_point, max_episode_steps, kwargs):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        del env_specs[id]
    gym.register(id=id, 
                 entry_point=entry_point, 
                 max_episode_steps=max_episode_steps, 
                 kwargs=kwargs)

register(
    id='Maze-v0',
    entry_point='mazeworld.envs:MazeWorld',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5Room-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5RoomRandomFixed-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_randomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5RoomBouncing-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_bouncing',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5RoomBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='DeepmindMaze-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_maze',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind8Room-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind8Room-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_v1',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5RoomMoveable-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})

register(
    id='Deepmind5RoomMoveable-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable_v1',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500})


