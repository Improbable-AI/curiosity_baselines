import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')
BOTTOM_LEFT = [1.5, 0.45, 0.42469975]
BOTTOM_RIGHT = [1.5, 1.05, 0.42469975] 
BOTTOM_CENTER = [1.5, 0.75, 0.42469975] 
TOP_LEFT = [1.1, 0.45, 0.42469975] 
TOP_RIGHT = [1.1, 1.05, 0.42469975] 
TOP_CENTER = [1.1, 0.75, 0.42469975]
CENTER_LEFT = [1.3, 0.45, 0.42469975] 
CENTER_RIGHT = [1.3, 1.05, 0.42469975] 
CENTER = [1.3, 0.75, 0.42469975]

class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='state', 
            camera_name='external_camera_0', fixed_start=None, fixed_goal=None,
            fixed_obj=None, time_limit=None)
        utils.EzPickle.__init__(self)

# 881 experiment 1 - start is close to goal
OBJ_1 = [1.4, 0.75, 0.42469975]
TARGET_1 = BOTTOM_CENTER
class FetchPushExp1(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': OBJ_1+[1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='img', 
            camera_name='external_camera_0', fixed_start=None, fixed_goal=np.array(TARGET_1),
            fixed_obj=np.array(OBJ_1), time_limit=50*2)
        utils.EzPickle.__init__(self)

# 881 experiment 1 - start is close to goal
ARM_START_2 = TOP_CENTER
OBJ_2     = CENTER
TARGET_2  = BOTTOM_CENTER
class FetchPushExp2(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': OBJ_1+[1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='img', 
            camera_name='external_camera_0', fixed_start=np.array(ARM_START_2), fixed_goal=np.array(TARGET_2),
            fixed_obj=np.array(OBJ_1), time_limit=50*2)
        utils.EzPickle.__init__(self)











