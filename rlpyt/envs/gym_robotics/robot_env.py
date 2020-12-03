import os
import copy
import numpy as np
from datetime import datetime

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class RobotEnv(gym.Env):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, obs_type):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        self.action_space = spaces.Box(-2., 2., shape=(n_actions,), dtype='float32')

        self.obs_type = obs_type
        if self.obs_type == 'state':
            state_obs = self._get_state_obs()
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=state_obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=state_obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=state_obs['observation'].shape, dtype='float32'),
            ))
        elif self.obs_type == 'img':
            # img_obs = self._get_img_obs()
            self.observation_space = spaces.Box(0., 255., (500,500,3), dtype='uint8')

        self.time_elapsed = 0

        self.table_center = [1.3, 0.75, 0.42469975] # Hardcoded center of tabletop.  TODO: figure out how to import properly
        self.heatmap_spacing = 0.03
        self.heatmap_edge = 2 # The length of heatmap in m.  If gripper ends up outside of this, the heatmap
        # is just not updated
        # Store discrete grid of where gripper is after each step.
        # TODO: Make this scale with action discretation
        # TODO: Make 3D also
        self.visitation_heatmap = np.zeros((int(self.heatmap_edge//self.heatmap_spacing),)*2, dtype=np.int64)

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        state_obs = self._get_state_obs()
        img_obs = self._get_img_obs()

        # Update that we've visited the heatmap
        self._update_heatmap(state_obs['observation'][:3])

        done = False
        self.time_elapsed += 1
        if self.time_elapsed == self.time_limit:
            done = True
            # Save visitation heatmap to file
            timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S') 
            np.save(f"heatmap_data_{timestamp}.npy", self.visitation_heatmap)
        info = {
            'is_success': self._is_success(state_obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(state_obs['achieved_goal'], self.goal, info)

        info = self._metric_info(info)

        if self.obs_type == 'state':
            return state_obs, reward, done, info
        elif self.obs_type == 'img':
            return img_obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        # super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        state_obs = self._get_state_obs()
        img_obs = self._get_img_obs()

        self.time_elapsed = 0
        if self.obs_type == 'state':
            return state_obs
        elif self.obs_type == 'img':
            return img_obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------
    def _update_heatmap(self, gripper_pos):
        # Only deal with 2D for now
        gripper_pos = gripper_pos[:2]
        center = self.table_center[:2]
        vec = gripper_pos - center
        if np.any(np.abs(vec) > self.heatmap_edge/2):
            # Don't update heatmap if gripper is outside heatmap grid
            return
        # Convert distance to grid indices
        indices = (vec / self.heatmap_spacing).astype(np.int16) 
        # Global coordinates are +x left, +y forward, so adjust indices accordingly
        x = int(self.visitation_heatmap.shape[1]/2) + indices[1]
        y = int(self.visitation_heatmap.shape[1]/2) + indices[0]
        self.visitation_heatmap[y, x] += 1
        return


    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_state_obs(self):
        """Returns the state space observation.
        """
        raise NotImplementedError()

    def _get_img_obs(self):
        """Returns the image space observations.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _metric_info(self, info):
        """A custom method that adds metrics to the info dictionary after each step in
        the simulation.
        """
        return info