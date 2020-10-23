
from PIL import Image
import matplotlib.pyplot as plt
import os

import numpy as np
import torch
from torch import nn
torch.set_printoptions(threshold=500*2*100)

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.graph_utils import save_dot
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead

GruState = namedarraytuple("GruState", ["c"])  # For downstream namedarraytuples to work

class NdigoForward(nn.Module):
    """Frame predictor MLP for NDIGO curiosity algorithm"""
    def __init__(self,
                 feature_size,
                 action_size, # usually multi-action sequence
                 output_size # observation size
                 ):
        super(NdigoForward, self).__init__()

        self.model = nn.Sequential(nn.Linear(feature_size + action_size, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, output_size))

    def forward(self, belief_states, action_seqs):
        predicted_states = self.model(torch.cat([belief_states, action_seqs], 2))
        return predicted_states

class NDIGO(torch.nn.Module):
    """Curiosity model for intrinsically motivated agents: a convolutional network 
    into an FC layer into an LSTM into an MLP which outputs forward predictions on
    future states, and computes an intrinsic reward using the error in these predictions.
    """

    def __init__(
            self,
            image_shape,
            action_size,
            horizon,
            feature_encoding='idf_maze',
            gru_size=128,
            batch_norm=False,
            obs_stats=None,
            num_predictors=10
            ):
        """Instantiate neural net module according to inputs."""
        super(NDIGO, self).__init__()

        assert num_predictors >= horizon

        self.action_size = action_size
        self.horizon = horizon
        self.feature_encoding = feature_encoding
        self.obs_stats = obs_stats
        self.num_predictors = num_predictors
        if self.obs_stats is not None:
            self.obs_mean, self.obs_std = self.obs_stats

        if self.feature_encoding != 'none':
            if self.feature_encoding == 'idf':
                self.feature_size = 288
                self.encoder = UniverseHead(image_shape=image_shape, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_burda':
                self.feature_size = 512
                self.encoder = BurdaHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_maze':
                self.feature_size = 256
                self.encoder = MazeHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm)

        self.gru_size = gru_size
        self.gru = torch.nn.GRU(self.feature_size + action_size, self.gru_size)
        self.gru_states = None # state output of last batch - (1, B, gru_size) or None

        self.forward_model_1 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*1, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_2 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*2, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_3 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*3, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_4 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*4, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_5 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*5, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_6 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*6, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_7 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*7, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_8 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*8, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_9 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*9, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])
        self.forward_model_10 = NdigoForward(feature_size=self.gru_size, 
                                            action_size=action_size*10, 
                                            output_size=image_shape[0]*image_shape[1]*image_shape[2])


        # DEBUGGING
        self.ep_counter = 0
        self.vis = False

        runs = os.listdir('/curiosity_baselines/results/ppo_Deepmind5Room-v0')
        try:
            runs.remove('tmp')
        except ValueError:
            pass
        try:
            runs.remove('.DS_Store')
        except ValueError:
            pass
        sorted_runs = sorted(runs, key=lambda run: int(run.split('_')[-1]))
        self.path = '/curiosity_baselines/results/ppo_Deepmind5Room-v0/{}/images'.format(sorted_runs[-1])
        os.mkdir(self.path)


    def forward(self, observations, prev_actions, actions):

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)

        # encode batch
        images = observations.type(torch.float)
        encoded_states = self.encoder(images.view(T * B, *img_shape)).view(T, B, -1)

        # pass encoded batch through GRU
        # gru_state = None if gru_state is None else gru_state.c
        gru_inputs = torch.cat([encoded_states, prev_actions], dim=2)
        belief_states, gru_output_states = self.gru(gru_inputs, self.gru_states)

        return belief_states, gru_output_states
        

    def compute_bonus(self, observations, prev_actions, actions):
        #------------------------------------------------------------#
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        # hacky dimension add for when you have only one environment
        if prev_actions.dim() == 1: 
            prev_actions = prev_actions.view(1, 1, -1)
        if actions.dim() == 1:
            actions = actions.view(1, 1, -1)
        #------------------------------------------------------------#

        # generate belief states
        belief_states, gru_output_states = self.forward(observations, prev_actions, actions)
        self.gru_states = None # only bc we're processing exactly 1 episode per batch

        # slice beliefs and actions
        belief_states = belief_states[:T-self.horizon] # slice off last timesteps
        
        action_seqs = torch.zeros((T-self.horizon, B, self.horizon*self.action_size)) # placeholder
        for i in range(len(actions)-self.horizon):
            action_seq = actions[i:i+self.horizon]
            action_seq = torch.transpose(action_seq, 0, 1)
            action_seq = torch.reshape(action_seq, (action_seq.shape[0], -1))
            action_seqs[i] = action_seq
        
        # make forward model predictions
        if self.horizon == 1:
            predicted_states = self.forward_model_1(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-1, B, 75)
        elif self.horizon == 2:
            predicted_states = self.forward_model_2(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-2, B, 75)
        elif self.horizon == 3:
            predicted_states = self.forward_model_3(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-3, B, 75)
        elif self.horizon == 4:
            predicted_states = self.forward_model_4(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-4, B, 75)
        elif self.horizon == 5:
            predicted_states = self.forward_model_5(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-5, B, 75)
        elif self.horizon == 6:
            predicted_states = self.forward_model_6(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-6, B, 75)
        elif self.horizon == 7:
            predicted_states = self.forward_model_7(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-7, B, 75)
        elif self.horizon == 8:
            predicted_states = self.forward_model_8(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-8, B, 75)
        elif self.horizon == 9:
            predicted_states = self.forward_model_9(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-9, B, 75)
        elif self.horizon == 10:
            predicted_states = self.forward_model_10(belief_states, action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-10, B, 75)

        true_obs = observations[self.horizon:].view(-1, *predicted_states.shape[1:])

        # generate losses
        losses = nn.functional.binary_cross_entropy_with_logits(predicted_states, true_obs.detach(), reduction='none')
        losses = torch.sum(losses, dim=-1)/losses.shape[-1] # average of each feature for each environment at each timestep (T, B, ave_loss_over_feature)
        
        # subtract losses to get rewards
        r_int = torch.zeros((T, B))
        for i in range(1, len(losses)):
            r_int[i+self.horizon-1] = losses[i-1] - losses[i]

        return r_int


    def compute_loss(self, observations, prev_actions, actions):
        #------------------------------------------------------------#
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        # hacky dimension add for when you have only one environment
        if prev_actions.dim() == 2: 
            prev_actions = prev_actions.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        #------------------------------------------------------------#

        # generate belief states
        belief_states, gru_output_states = self.forward(observations, prev_actions, actions)
        self.gru_states = None # only bc we're processing exactly 1 episode per batch

        # generate loss for each forward predictor
        # loss = torch.tensor(0.0)

        # DEBUGGING
        if self.ep_counter % 200 == 0:
            os.mkdir(self.path + '/ep_{}'.format(self.ep_counter))
            self.vis = True
            rand_time = np.random.randint(480)
            b = 0
            start = observations[rand_time, b].detach().clone().data.numpy() # (T, B, 3, 5, 5)
            for i in range(3):
                plt.imsave(self.path + '/ep_{}/o_t{}_{}.jpg'.format(self.ep_counter, rand_time, i), start[i])
        else:
            self.vis = False

        for k in range(1, self.num_predictors+1):
            action_seqs = torch.zeros((T-k, B, k*self.action_size)) # placeholder
            for i in range(len(actions)-k):
                action_seq = actions[i:i+k]
                action_seq = torch.transpose(action_seq, 0, 1)
                action_seq = torch.reshape(action_seq, (action_seq.shape[0], -1))
                action_seqs[i] = action_seq

            # make forward model predictions for this predictor
            if k == 1:
                predicted_states = self.forward_model_1(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-1, B, 75)
            elif k == 2:
                predicted_states = self.forward_model_2(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-2, B, 75)
            elif k == 3:
                predicted_states = self.forward_model_3(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-3, B, 75)
            elif k == 4:
                predicted_states = self.forward_model_4(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-4, B, 75)
            elif k == 5:
                predicted_states = self.forward_model_5(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-5, B, 75)
            elif k == 6:
                predicted_states = self.forward_model_6(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-6, B, 75)
            elif k == 7:
                predicted_states = self.forward_model_7(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-7, B, 75)
            elif k == 8:
                predicted_states = self.forward_model_8(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-8, B, 75)
            elif k == 9:
                predicted_states = self.forward_model_9(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-9, B, 75)
            elif k == 10:
                predicted_states = self.forward_model_10(belief_states[:T-k], action_seqs.detach()).view(-1, B, img_shape[0]*img_shape[1]*img_shape[2]) # (T-10, B, 75)

            # generate losses for this predictor
            true_obs = observations[k:].view(-1, *predicted_states.shape[1:]).detach()

            if k == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(predicted_states, true_obs.detach(), reduction='mean')
            else:
                loss += nn.functional.binary_cross_entropy_with_logits(predicted_states, true_obs.detach(), reduction='mean')

            # DEBUGGING
            if self.vis:
                with torch.no_grad():

                    os.mkdir(self.path + '/ep_{}/pred_{}'.format(self.ep_counter, k))

                    np.savetxt(self.path + '/ep_{}/pred_{}/actions.txt'.format(self.ep_counter, k, i),
                               action_seqs[rand_time, b].detach().clone().data.numpy())
                    predicted_states = predicted_states.detach().clone().data.numpy() # (T, B, 75)
                    true = true_obs.detach().clone().data.numpy() # (T, B, 75)
                    predicted_states = np.reshape(predicted_states[rand_time, b], (3, 5, 5))
                    true = np.reshape(true[rand_time, b], (3, 5, 5))

                    for i in range(3):
                        plt.imsave(self.path + '/ep_{}/pred_{}/pred_{}.jpg'.format(self.ep_counter, k, i), predicted_states[i])
                        plt.imsave(self.path + '/ep_{}/pred_{}/true_{}.jpg'.format(self.ep_counter, k, i), true[i])


        # print("SAVING")
        # save_dot(loss,
        #          {loss: 'forward_loss',
        #           self.encoder.model[0].weight: 'encoder.conv1.w',
        #           self.encoder.model[0].bias: 'encoder.conv1.b',
        #           self.encoder.model[2].weight: 'encoder.conv2.w',
        #           self.encoder.model[2].bias: 'encoder.conv2.b',
        #           self.encoder.model[5].weight: 'encoder.lin_out.w',
        #           self.encoder.model[5].bias: 'encoder.lin_out.b',
        #           self.gru.weight_ih_l0: 'gru.input_h.w',
        #           self.gru.bias_ih_l0: 'gru.input_h.b',
        #           self.gru.weight_hh_l0: 'gru.hidden_h.w',
        #           self.gru.bias_hh_l0: 'gru.hidden_h.b',
        #           # self.forward_model_1.model[0].weight: 'forward_1.lin_1.w',
        #           # self.forward_model_1.model[0].bias: 'forward_1.lin_1.b',
        #           # self.forward_model_1.model[2].weight: 'forward_1.lin_2.w',
        #           # self.forward_model_1.model[2].bias: 'forward_1.lin_2.b',
        #           self.forward_model_2.model[0].weight: 'forward_2.lin_1.w',
        #           self.forward_model_2.model[0].bias: 'forward_2.lin_1.b',
        #           self.forward_model_2.model[2].weight: 'forward_2.lin_2.w',
        #           self.forward_model_2.model[2].bias: 'forward_2.lin_2.b',}, 
        #          open('./ndigo.dot', 'w'))
        # print('DONE SAVING')
        self.ep_counter += 1
        return loss



