
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.algos.utils import discount_return, generalized_advantage_estimation, valid_from_done

# Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase
OptInfo = namedtuple("OptInfo", ["return_",
                                 "intrinsic_rewards",
                                 "valpred",
                                 "advantage",
                                 "loss", 
                                 "pi_loss",
                                 "value_loss",
                                 "entropy_loss",
                                 "inv_loss", 
                                 "forward_loss",
                                 "reward_total_std", 
                                 "curiosity_loss", 
                                 "gradNorm", 
                                 "entropy", 
                                 "perplexity",
                                 # Rand: dummy logging parameter
                                 "rand_dummy_logging",
                                 # ART: number of classes
                                 "art_num_classes",
                                 # Kohonen: Change of weights
                                 "kohonen_dw"
                                 ])
AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])


class PolicyGradientAlgo(RlAlgorithm):
    """
    Base policy gradient / actor-critic algorithm, which includes
    initialization procedure and processing of data samples to compute
    advantages.
    """

    bootstrap_value = True  # Tells the sampler it needs Value(State')
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.
        """
        reward, done, value, bv = (samples.env.reward, samples.env.done, samples.agent.agent_info.value, samples.agent.bootstrap_value)
        done = done.type(reward.dtype)

        if self.curiosity_type == 'icm' or self.curiosity_type == 'disagreement':
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.env.next_observation.clone(), samples.agent.action.clone())
            reward += intrinsic_rewards
            self.intrinsic_rewards = intrinsic_rewards.clone().data.numpy()
        elif self.curiosity_type == 'ndigo':
            intrinsic_rewards, _ = self.agent.curiosity_step(self.curiosity_type, samples.env.observation.clone(), samples.agent.prev_action.clone(), samples.agent.action.clone()) # no grad
            reward += intrinsic_rewards
            self.intrinsic_rewards = intrinsic_rewards.clone().data.numpy()
        elif self.curiosity_type == 'rnd':
            intrinsic_rewards, _ = self.agent.curiosity_step (self.curiosity_type, samples.env.next_observation.clone(), done.clone())
            reward += intrinsic_rewards
            self.intrinsic_rewards = intrinsic_rewards.clone().data.numpy()
        elif self.curiosity_type == 'rand':
            intrinsic_rewards, _ = self.agent.curiosity_step (self.curiosity_type, samples.env.next_observation.clone(), done.clone())
            reward += intrinsic_rewards
            self.intrinsic_rewards = intrinsic_rewards.clone().data.numpy()

        # TODO MARIUS: Compute intrinsic rewards for Kohonen. Note that this passes the arguments to the agent, who passes it on to the curiosity model internally  
        elif self.curiosity_type == 'kohonen':
            intrinsic_rewards, kohonen_info = self.agent.curiosity_step (self.curiosity_type, samples.env.next_observation.clone(), done.clone())
            reward += intrinsic_rewards
            self.intrinsic_rewards = intrinsic_rewards.clone().data.numpy()

        # TODO MARIUS: Compute intrinsic rewards for ART. Note that this passes the arguments to the agent, who passes it on to the curiosity model internally
        elif self.curiosity_type == 'art':
            intrinsic_rewards, art_info = self.agent.curiosity_step (self.curiosity_type, samples.env.next_observation.clone(), done.clone())
            reward += intrinsic_rewards
            self.intrinsic_rewards = intrinsic_rewards.clone().data.numpy()

        if self.normalize_reward:
            rews = np.array([])
            for rew in reward.clone().detach().data.numpy():
                rews = np.concatenate((rews, self.reward_ff.update(rew)))
            self.reward_rms.update_from_moments(np.mean(rews), np.var(rews), len(rews))
            reward = reward / np.sqrt(self.reward_rms.var)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(reward, value, done, bv, self.discount, self.gae_lambda)
        
        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        if self.kernel_params is not None: # apply advantage kernel
            advantage[:] = torch.tensor(np.piecewise(advantage.data.numpy(), [abs(advantage.data.numpy()) < self.mu, abs(advantage.data.numpy()) >= self.mu], [self.kernel_line, self.kernel_gauss]))

        return return_, advantage, valid
