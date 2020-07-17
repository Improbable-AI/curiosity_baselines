from collections import deque
import gym
from gym import spaces
import numpy as np

class FrameSkip(gym.Wrapper):
    """
    Repeat a single action through four n frames.
    """
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info

class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to np.ndarray before being passed to the model.
    :param frames: ([int] or [float]) environment frames
    """
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class FrameStack(gym.Wrapper):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class PytorchImage(gym.ObservationWrapper):
    """
    Switch image observation from (h, w, c) to (c, h, w) 
    which is required for the PyTorch framework.
    """
    def __init__(self, env):
        super(PytorchImage, self).__init__(env)
        current_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(current_shape[-1], current_shape[0], current_shape[1]))

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class NoExtrinsicReward(gym.RewardWrapper):
    """
    Remove external reward for experiments where you want
    to use intrinsic reward only.
    """
    def __init__(self, env):
        super(NoExtrinsicReward, self).__init__(env)

    def reward(self, reward):
        return 0.0

class NoNegativeReward(gym.RewardWrapper):
    """
    Remove negative rewards and zero them out. This can apply
    to living penalties for example.
    """
    def __init__(self, env):
        super(NoNegativeReward, self).__init__(env)

    def reward(self, reward):
        if reward < 0:
            return 0
        else:
            return reward


