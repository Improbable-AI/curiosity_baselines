import torch
torch.set_printoptions(precision=10)
from copy import deepcopy
import numpy as np

from rlpyt.samplers.collectors import (DecorrelatingStartCollector,
    BaseEvalCollector)
from rlpyt.agents.base import AgentInputs, AgentCuriosityInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
    buffer_method)


class CpuResetCollector(DecorrelatingStartCollector):
    """Collector which executes ``agent.step()`` in the sampling loop (i.e.
    use in CPU or serial samplers.)

    It immediately resets any environment which finishes an episode.  This is
    typically indicated by the environment returning ``done=True``.  But this
    collector defers to the ``done`` signal only after looking for
    ``env_info["traj_done"]``, so that RL episodes can end without a call to
    ``env_reset()`` (e.g. used for episodic lives in the Atari env).  The 
    agent gets reset based solely on ``done``.
    """

    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                    env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos


class CpuWaitResetCollector(DecorrelatingStartCollector):
    """Collector which executes ``agent.step()`` in the sampling loop.

    It waits to reset any environments with completed episodes until after
    the end of collecting the batch, i.e. the ``done`` environment is bypassed
    in remaining timesteps, and zeros are recorded into the batch buffer.

    Waiting to reset can be beneficial for two reasons.  One is for training
    recurrent agents; PyTorch's built-in LSTM cannot reset in the middle of a
    training sequence, so any samples in a batch after a reset would be
    ignored and the beginning of new episodes would be missed in training.
    The other reason is if the environment's reset function is very slow
    compared to its step function; it can be faster overall to leave invalid
    samples after a reset, and perform the environment resets in the workers
    while the master process is training the agent (this was true for massively
    parallelized Atari).
    """

    mid_batch_reset = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_reset = np.zeros(len(self.envs), dtype=np.bool)
        self.done = np.zeros(len(self.envs), dtype=np.bool)
        self.temp_prev_observation = buffer_method(self.samples_np.env.prev_observation[0, :len(self.envs)], "copy")
        self.temp_observation = buffer_method(self.samples_np.env.observation[0, :len(self.envs)], "copy")

    def collect_batch(self, agent_inputs, agent_curiosity_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        
        prev_observation, _, _ = agent_curiosity_inputs # shares action/observation object with agent_inputs
        observation, action, reward_tot = agent_inputs
        
        b = np.where(self.done)[0]
        # prev_observation[b] = self.temp_prev_observation[b]
        # observation[b] = self.temp_observation[b]
        self.done[:] = False  # Did resets between batches.
        
        # torchifying syncs components of agent_inputs (observation, action, reward_tot)
        # with obs_pyt, act_pyt, rew_tot_pyt. Pytorch tensors point to the original numpy 
        # array so updating observation will update obs_pyt etc.
        obs_pyt, act_pyt, rew_tot_pyt = torchify_buffer(agent_inputs)
        
        agent_buf.prev_action[0] = action # Leading prev_action
        env_buf.prev_reward[0] = reward_tot # Leading previous total reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):

            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_tot_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                if self.done[b]:
                    action[b] = 0  # Record blank.
                    reward_tot[b] = 0
                    if agent_info:
                        agent_info[b] = 0
                    # Leave self.done[b] = True, record that.
                    continue
                # Environment inputs and outputs are numpy arrays.
                o, r_ext, d, env_info = env.step(action[b])

                r_ext_log = r_ext # to ensure r_ext gets recorded regardless
                if self.no_extrinsic:
                    r_ext = 0.0

                #------------------------------------------------------------------------#
                # DEBUGGING: records observations to curiosity_baselines/images/___.jpg
                # Stops and sleeps long enough to quit out at the end of an episode. Make sure
                # frame stacking is turned off.
                # from PIL import Image
                # # img = Image.fromarray(np.squeeze(o), 'L')
                # img = Image.fromarray(env._last_painted, 'RGB')
                # img.save('images/{}_act_{}.jpg'.format(t, action[b]))
                # # o = np.expand_dims(o, 0)
                # if d:
                #     import time
                #     print("DONE! {}".format(t))
                #     time.sleep(100)

                # FRAME STACKING:
                # from PIL import Image
                # import os
                # os.mkdir('images/{}'.format(t))
                # for i in range(4):
                #     img = Image.fromarray(np.squeeze(o[i]), 'L')
                #     img.save('images/{}/{}.jpg'.format(t, i))
                #     o[i] = np.expand_dims(o[i], 0)
                # if d:
                #     import time
                #     print("DONE!")
                #     time.sleep(100)
                #------------------------------------------------------------------------#

                r_int = torch.tensor(0.0)
                if self.curiosity_alg != 'none':
                    r_int = self.agent.curiosity_step(obs_pyt[b].unsqueeze(0), act_pyt[b], torch.tensor(o).unsqueeze(0)) # torch.Tensor doesn't link memory 

                traj_infos[b].step(observation[b], action[b], r_ext_log, r_int.item(), d, agent_info[b], env_info, env.visitation_frequency, env.first_visit_time)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    self.need_reset[b] = True
                if d:
                    # self.temp_prev_observation[b] = observation[b]
                    # self.temp_observation[b] = o
                    o = 0  # Record blank.
                self.done[b] = d

                prev_observation[b] = observation[b]
                observation[b] = o
                reward_tot[b] = r_ext + r_int

                if env_info: 
                    env_buf.env_info[t, b] = env_info

            agent_buf.action[t] = action
            # agent_buf.reward_int[t] = r_int
            env_buf.reward[t] = reward_tot
            env_buf.done[t] = self.done
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_tot_pyt)

        # AgentInputs = ['observation', 'prev_action', 'prev_reward']
        # AgentCuriosityInputs = ['observation', 'action', 'next_observation']
        return AgentInputs(observation, action, reward_tot), AgentCuriosityInputs(prev_observation, action, observation), traj_infos, completed_infos

    def reset_if_needed(self, agent_inputs, agent_curiosity_inputs):
        for b in np.where(self.need_reset)[0]:
            # wipe all fields
            agent_inputs[b] = 0
            agent_curiosity_inputs[b] = 0

            # fill in first observation
            o_reset = self.envs[b].reset()
            agent_curiosity_inputs.observation[b] = o_reset
            agent_inputs.observation[b] = o_reset
            agent_curiosity_inputs.next_observation[b] = o_reset
            self.agent.reset_one(idx=b)
        self.need_reset[:] = False


class CpuEvalCollector(BaseEvalCollector):
    """Offline agent evaluation collector which calls ``agent.step()`` in 
    sampling loop.  Immediately resets any environment which finishes a
    trajectory.  Stops when the max time-steps have been reached, or when
    signaled by the master process (i.e. if enough trajectories have
    completed).
    """

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    self.traj_infos_queue.put(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Next prev_action.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if self.sync.stop_eval.value:
                break
        self.traj_infos_queue.put(None)  # End sentinel.
