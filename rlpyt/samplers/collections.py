
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict


Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
    ["action", "reward_int", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
    ["action", "reward_int", "prev_action", "agent_info", "bootstrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
    ["prev_observation", "reward", "prev_reward", "observation", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """
    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.ExtrinsicReward = 0
        self.NonzeroExtrinsicRewards = 0
        self.DiscountedExtrinsicReward = 0
        self.IntrinsicReward = 0
        self.NonzeroIntrinsicRewards = 0
        self._cur_discount = 1

    def step(self, observation, action, reward_ext, reward_int, done, agent_info, env_info):
        self.Length += 1
        self.ExtrinsicReward += reward_ext
        self.NonzeroExtrinsicRewards += reward_ext != 0
        self.DiscountedExtrinsicReward += self._cur_discount * reward_ext
        self.IntrinsicReward += reward_int
        self.NonzeroIntrinsicRewards += reward_int != 0
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self
