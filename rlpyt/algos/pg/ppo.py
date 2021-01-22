
import numpy as np
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, IcmAgentCuriosityInputs, NdigoAgentCuriosityInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.grad_utils import plot_grad_flow

LossInputs = namedarraytuple("LossInputs", ["agent_inputs", "agent_curiosity_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])

class PPO(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_reward=False,
            kernel_params=None,
            curiosity_type='none'
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        if self.normalize_reward:
            self.reward_ff = RewardForwardFilter(discount)
            self.reward_rms = RunningMeanStd()
        self.intrinsic_rewards = None
        
        if kernel_params is not None:
            self.mu, self.sigma = self.kernel_params
            self.kernel_line = lambda x: x
            self.kernel_gauss = lambda x: np.sign(x)*self.mu*np.exp(-(abs(x)-self.mu)**2/(2*self.sigma**2))

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        if self.curiosity_type == 'icm' or self.curiosity_type == 'disagreement':
            agent_curiosity_inputs = IcmAgentCuriosityInputs(
                observation=samples.env.observation,
                next_observation=samples.env.next_observation,
                action=samples.agent.action,
            )
        elif self.curiosity_type == 'ndigo':
            agent_curiosity_inputs = NdigoAgentCuriosityInputs(
                observation=samples.env.observation,
                prev_actions=samples.agent.prev_action,
                actions=samples.agent.action
            )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            agent_curiosity_inputs=agent_curiosity_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        T, B = samples.env.reward.shape[:2]

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None

                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, curiosity_losses = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)

                loss.backward()
                count = 0
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                
                # Tensorboard summaries
                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.item())
                opt_info.value_loss.append(value_loss.item())
                opt_info.entropy_loss.append(entropy_loss.item())

                if self.curiosity_type == 'icm' or self.curiosity_type == 'disagreement':
                    inv_loss, forward_loss = curiosity_losses
                    opt_info.inv_loss.append(inv_loss.item())
                    opt_info.forward_loss.append(forward_loss.item())
                    opt_info.intrinsic_rewards.append(np.mean(self.intrinsic_rewards))
                elif self.curiosity_type == 'ndigo':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    opt_info.intrinsic_rewards.append(np.mean(self.intrinsic_rewards))

                if self.normalize_reward:
                    opt_info.reward_total_std.append(self.reward_rms.var**0.5)

                opt_info.gradNorm.append(grad_norm.item())
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1

        opt_info.return_.append(torch.mean(return_.detach()).detach().clone().item())
        opt_info.advantage.append(torch.mean(advantage.detach()).detach().clone().item())
        opt_info.valpred.append(torch.mean(samples.agent.agent_info.value.detach()).detach().clone().item())

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        layer_info = dict() # empty dict to store model layer weights for tensorboard visualizations
        
        return opt_info, layer_info

    def loss(self, agent_inputs, agent_curiosity_inputs, action, return_, advantage, valid, old_dist_info,
            init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)
        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        perplexity = dist.mean_perplexity(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        if self.curiosity_type == 'icm' or self.curiosity_type == 'disagreement':
            inv_loss, forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += inv_loss
            loss += forward_loss
            curiosity_losses = (inv_loss, forward_loss)
        elif self.curiosity_type == 'ndigo':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        else:
            curiosity_losses = None

        return loss, pi_loss, value_loss, entropy_loss, entropy, perplexity, curiosity_losses
