import numpy as np
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict


Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
    ["action", "reward_int", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
    ["action", "reward_int", "prev_action", "agent_info", "bootstrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
    ["prev_observation", "reward", "prev_reward", "observation", "next_observation", "done", "env_info"])


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
        self.EpExtrinsicReward = 0
        self.EpNonzeroExtrinsicRewards = 0
        self.EpDiscountedExtrinsicReward = 0
        self.EpIntrinsicReward = 0
        self.EpNonzeroIntrinsicRewards = 0
        self._cur_discount = 1

        self.EpAveExtrinsicReward = []
        self.EpAveIntrinsicReward = []

        # Deepmind world discovery models metrics
        self.visit_freq_a = 0
        self.visit_freq_b = 0
        self.first_visit_a = 400
        self.first_visit_b = 400

    def step(self, observation, action, reward_ext, reward_int, done, agent_info, env_info, visitation_frequency, first_visit_time):
        self.Length += 1
        self.EpExtrinsicReward += reward_ext
        self.EpNonzeroExtrinsicRewards += reward_ext != 0
        self.EpDiscountedExtrinsicReward += self._cur_discount * reward_ext
        self.EpIntrinsicReward += reward_int
        self.EpNonzeroIntrinsicRewards += reward_int != 0
        self._cur_discount *= self._discount

        self.EpAveExtrinsicReward.append(reward_ext)
        self.EpAveIntrinsicReward.append(reward_int)

        # Deepmind world discovery models metrics
        if self.visit_freq_a == 0 and visitation_frequency['a'] == 1: # first visit
            self.first_visit_a = self.Length
        if self.visit_freq_b == 0 and visitation_frequency['b'] == 1:
            self.first_visit_b = self.Length
        self.visit_freq_a = visitation_frequency['a']
        self.visit_freq_b = visitation_frequency['b']

    def terminate(self, observation):
        self.EpAveExtrinsicReward = np.mean(self.EpAveExtrinsicReward)
        self.EpAveIntrinsicReward = np.mean(self.EpAveIntrinsicReward)
        return self
