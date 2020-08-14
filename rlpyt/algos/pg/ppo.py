
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, AgentCuriosityInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.averages import RunningMeanStd

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
            curiosity_kwargs={'curiosity_alg':'none'}
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        if self.normalize_reward:
            self.reward_avg = RunningMeanStd()

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
        agent_curiosity_inputs = AgentCuriosityInputs(
            observation=samples.env.prev_observation,
            action=samples.agent.action,
            next_observation=samples.env.observation
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

        layer_info = {'forward/lin1.w':None,
                      'forward/lin1.b':None,
                      'forward/res1/lin1.w':None,
                      'forward/res1/lin1.b':None,
                      'forward/res1/lin2.w':None,
                      'forward/res1/lin2.b':None,
                      'forward/res2/lin1.w':None,
                      'forward/res2/lin1.b':None,
                      'forward/res2/lin2.w':None,
                      'forward/res2/lin2.b':None,
                      'forward/res3/lin1.w':None,
                      'forward/res3/lin1.b':None,
                      'forward/res3/lin2.w':None,
                      'forward/res3/lin2.b':None,
                      'forward/res4/lin1.w':None,
                      'forward/res4/lin1.b':None,
                      'forward/res4/lin2.w':None,
                      'forward/res4/lin2.b':None,
                      'forward/lin_last.w':None,
                      'forward/lin_last.b':None,

                      'inverse/lin1.w':None, 
                      'inverse/lin1.b':None,
                      'inverse/lin2.w':None,
                      'inverse/lin2.b':None,

                      'encoder/conv1.w':None,
                      'encoder/conv1.b':None,
                      'encoder/conv2.w':None,
                      'encoder/conv2.b':None,
                      'encoder/conv3.w':None,
                      'encoder/conv3.b':None,
                      'encoder/lin_out.w':None,
                      'encoder/lin_out.b':None
                      }

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
                loss, pi_loss, value_loss, entropy_loss, inv_loss, forward_loss, curiosity_loss, entropy, perplexity = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                # Tensorboard summaries
                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.item())
                opt_info.value_loss.append(value_loss.item())
                opt_info.entropy_loss.append(entropy_loss.item())
                opt_info.inv_loss.append(inv_loss.item())
                opt_info.forward_loss.append(forward_loss.item())
                opt_info.curiosity_loss.append(curiosity_loss.item())

                if self.normalize_reward:
                    opt_info.reward_total_std.append(self.reward_avg.var**0.5)

                opt_info.gradNorm.append(torch.tensor(grad_norm).clone().detach().item())
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1

        opt_info.return_.append(torch.mean(return_).detach().clone().item())
        opt_info.advantage.append(torch.mean(advantage).detach().clone().item())
        opt_info.valpred.append(torch.mean(samples.agent.agent_info.value).detach().clone().item())

        layer_info['forward/lin1.w'] = self.agent.model.curiosity_model.forward_model.lin_1.weight
        layer_info['forward/lin1.b'] = self.agent.model.curiosity_model.forward_model.lin_1.bias
        layer_info['forward/res1/lin1.w'] = self.agent.model.curiosity_model.forward_model.res_block_1.lin_1.weight
        layer_info['forward/res1/lin1.b'] = self.agent.model.curiosity_model.forward_model.res_block_1.lin_1.bias
        layer_info['forward/res1/lin2.w'] = self.agent.model.curiosity_model.forward_model.res_block_1.lin_2.weight
        layer_info['forward/res1/lin2.b'] = self.agent.model.curiosity_model.forward_model.res_block_1.lin_2.bias
        layer_info['forward/res2/lin1.w'] = self.agent.model.curiosity_model.forward_model.res_block_2.lin_1.weight
        layer_info['forward/res2/lin1.b'] = self.agent.model.curiosity_model.forward_model.res_block_2.lin_1.bias
        layer_info['forward/res2/lin2.w'] = self.agent.model.curiosity_model.forward_model.res_block_2.lin_2.weight
        layer_info['forward/res2/lin2.b'] = self.agent.model.curiosity_model.forward_model.res_block_2.lin_2.bias
        layer_info['forward/res3/lin1.w'] = self.agent.model.curiosity_model.forward_model.res_block_3.lin_1.weight
        layer_info['forward/res3/lin1.b'] = self.agent.model.curiosity_model.forward_model.res_block_3.lin_1.bias
        layer_info['forward/res3/lin2.w'] = self.agent.model.curiosity_model.forward_model.res_block_3.lin_2.weight
        layer_info['forward/res3/lin2.b'] = self.agent.model.curiosity_model.forward_model.res_block_3.lin_2.bias
        layer_info['forward/res4/lin1.w'] = self.agent.model.curiosity_model.forward_model.res_block_4.lin_1.weight
        layer_info['forward/res4/lin1.b'] = self.agent.model.curiosity_model.forward_model.res_block_4.lin_1.bias
        layer_info['forward/res4/lin2.w'] = self.agent.model.curiosity_model.forward_model.res_block_4.lin_2.weight
        layer_info['forward/res4/lin2.b'] = self.agent.model.curiosity_model.forward_model.res_block_4.lin_2.bias
        layer_info['forward/lin_last.w'] = self.agent.model.curiosity_model.forward_model.lin_last.weight
        layer_info['forward/lin_last.b'] = self.agent.model.curiosity_model.forward_model.lin_last.bias

        layer_info['inverse/lin1.w'] = self.agent.model.curiosity_model.inverse_model[0].weight
        layer_info['inverse/lin1.b'] = self.agent.model.curiosity_model.inverse_model[0].bias
        layer_info['inverse/lin2.w'] = self.agent.model.curiosity_model.inverse_model[2].weight
        layer_info['inverse/lin2.b'] = self.agent.model.curiosity_model.inverse_model[2].bias

        layer_info['encoder/conv1.w'] = self.agent.model.curiosity_model.encoder.model[0].weight
        layer_info['encoder/conv1.b'] = self.agent.model.curiosity_model.encoder.model[0].bias
        layer_info['encoder/conv2.w'] = self.agent.model.curiosity_model.encoder.model[3].weight
        layer_info['encoder/conv2.b'] = self.agent.model.curiosity_model.encoder.model[3].bias
        layer_info['encoder/conv3.w'] = self.agent.model.curiosity_model.encoder.model[6].weight
        layer_info['encoder/conv3.b'] = self.agent.model.curiosity_model.encoder.model[6].bias
        layer_info['encoder/lin_out.w'] = self.agent.model.curiosity_model.encoder.model[10].weight
        layer_info['encoder/lin_out.b'] = self.agent.model.curiosity_model.encoder.model[10].bias

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

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
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss
        
        if self.curiosity_kwargs['curiosity_alg'] == 'icm':
            forward_loss_wt = self.curiosity_kwargs['forward_loss_wt']
            inv_loss, forward_loss = self.agent.curiosity_loss(*agent_curiosity_inputs)
            curiosity_loss = (1-forward_loss_wt)*inv_loss + (forward_loss_wt)*forward_loss
        else:
            inv_loss = torch.tensor(0.0)
            forward_loss = torch.tensor(0.0)
            curiosity_loss = torch.tensor(0.0)

        # loss += curiosity_loss
        loss += inv_loss
        loss += forward_loss # burda seems to use no scaling of forward vs inv losses

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, pi_loss, value_loss, entropy_loss, inv_loss, forward_loss, curiosity_loss, entropy, perplexity
