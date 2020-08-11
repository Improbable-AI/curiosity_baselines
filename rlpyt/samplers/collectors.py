from copy import deepcopy
import numpy as np

from rlpyt.agents.base import AgentInputs, AgentCuriosityInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args


class BaseCollector:
    """Class that steps environments, possibly in worker process."""

    def __init__(
            self,
            rank,
            envs,
            env_stats,
            samples_np,
            batch_T,
            TrajInfoCls,
            agent=None,  # Present or not, depending on collector class.
            sync=None,
            step_buffer_np=None,
            global_B=1,
            env_ranks=None,
            ):
        save__init__args(locals())

    def start_envs(self):
        """e.g. calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        """In CPU-collectors, call ``agent.collector_initialize()`` e.g. to set up
        vector epsilon-greedy, and reset the agent.
        """
        if getattr(self, "agent", None) is not None:  # Not in GPU collectors.
            self.agent.collector_initialize(
                global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
                env_ranks=self.env_ranks,
            )
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, agent_inputs, traj_infos):
        """Main data collection loop."""
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class BaseEvalCollector:
    """Collectors for offline agent evalution; not to record intermediate samples."""

    def __init__(
            self,
            rank,
            envs,
            TrajInfoCls,
            traj_infos_queue,
            max_T,
            agent=None,
            sync=None,
            step_buffer_np=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self):
        """Run agent evaluation in environment and return completed trajectory
        infos."""
        raise NotImplementedError


class DecorrelatingStartCollector(BaseCollector):
    """Collector which can step all environments through a random number of random
    actions during startup, to decorrelate the states in training batches.
    """

    def start_envs(self, max_decorrelation_steps=0):
        """Calls ``reset()`` on every environment instance, then steps each
        one through a random number of random actions, and returns the
        resulting agent_inputs buffer (`observation`, `prev_action`,
        `prev_reward`)."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]

        prev_action = np.stack([env.action_space.null_value() for env in self.envs]) # noop
        prev_reward = np.zeros(len(self.envs), dtype="float32") # total reward (extrinsic + intrinsic)
        prev_observations = list()
        observations = list()
        for env in self.envs:
            o = env.reset()
            prev_observations.append(o) # observation doesn't change
            observations.append(deepcopy(o)) # emulates stepping with noop
        prev_observation = buffer_from_example(prev_observations[0], len(self.envs))
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, obs in enumerate(observations):
            prev_observation[b] = prev_observations[b] # numpy array or namedarraytuple
            observation[b] = obs

        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}")
        if max_decorrelation_steps != 0:
            for b, env in enumerate(self.envs):
                n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
                for _ in range(n_steps):
                    a = env.action_space.sample()
                    if a.shape == (): # 'a' gets stored, but if form is array(3) you need to pass int(3) for env
                        action = int(a)
                    else:
                        action = a
                    o, r_ext, d, info = env.step(action)
                    r_int = 0

                    traj_infos[b].step(o, a, r_ext, r_int, d, None, info)
                    if getattr(info, "traj_done", d):
                        o = env.reset()
                        traj_infos[b] = self.TrajInfoCls()
                    if d:
                        a = env.action_space.null_value()
                        r_ext = 0
                        r_int = 0
                prev_observation[b] = deepcopy(observation[b])
                observation[b] = o
                prev_action[b] = a
                prev_reward[b] = r_ext + r_int
        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.prev_observation[:] = prev_observation
            self.step_buffer_np.prev_action[:] = prev_action
            self.step_buffer_np.prev_reward[:] = prev_reward
            self.step_buffer_np.observation[:] = observation

        # AgentInputs -> ['observation', 'prev_action', 'prev_reward']
        # AgentCuriosityInputs -> ['observation', 'action', 'next_observation']
        return AgentInputs(observation, prev_action, prev_reward), AgentCuriosityInputs(prev_observation, prev_action, observation), traj_infos

