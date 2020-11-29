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
            fixed_obj=None, time_limit=None, visitation_thresh=0.055)
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
            fixed_obj=np.array(OBJ_1), time_limit=50*2, visitation_thresh=0.055)
        utils.EzPickle.__init__(self)

# 881 experiment 2 - arm at the back of table, start at center
ARM_START_2 = TOP_CENTER
OBJ_2     = CENTER
TARGET_2  = BOTTOM_CENTER
class FetchPushExp2(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': OBJ_2+[1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='img', 
            camera_name='external_camera_0', fixed_start=np.array(ARM_START_2), fixed_goal=np.array(TARGET_2),
            fixed_obj=np.array(OBJ_2), time_limit=50*2, visitation_thresh=0.055)
        utils.EzPickle.__init__(self)

# 881 experiment 3 - arm at the back of table, start is not close to the arm
ARM_START_3 = TOP_CENTER
OBJ_3     = CENTER_RIGHT
TARGET_3  = BOTTOM_CENTER
class FetchPushExp3(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': OBJ_3+[1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='img', 
            camera_name='external_camera_0', fixed_start=np.array(ARM_START_3), fixed_goal=np.array(TARGET_3),
            fixed_obj=np.array(OBJ_3), time_limit=50*2, visitation_thresh=0.055)
        utils.EzPickle.__init__(self)

# 881 experiment 4 - arm at the back of table, start is not close to the arm
ARM_START_4 = TOP_CENTER
OBJ_4     = TOP_LEFT
TARGET_4  = BOTTOM_CENTER
class FetchPushExp4(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': OBJ_4+[1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='img', 
            camera_name='external_camera_0', fixed_start=np.array(ARM_START_4), fixed_goal=np.array(TARGET_4),
            fixed_obj=np.array(OBJ_4), time_limit=50*2, visitation_thresh=0.055)
        utils.EzPickle.__init__(self)

# 881 experiment 5 - arm at back corner, object center, goal far right (diagonal push)
ARM_START_5 = TOP_LEFT
OBJ_5     = CENTER
TARGET_5  = BOTTOM_RIGHT
class FetchPushExp5(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': OBJ_5+[1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, obs_type='img', 
            camera_name='external_camera_0', fixed_start=np.array(ARM_START_5), fixed_goal=np.array(TARGET_5),
            fixed_obj=np.array(OBJ_5), time_limit=50*2, visitation_thresh=0.055)
        utils.EzPickle.__init__(self)











