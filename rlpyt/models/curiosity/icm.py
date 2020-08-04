
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UniverseHead(nn.Module):
    '''
    Universe agent example: https://github.com/openai/universe-starter-agent
    '''
    def __init__(
            self, 
            image_shape,
            batch_norm=False
            ):
        super(UniverseHead, self).__init__()
        c, h, w = image_shape
        sequence = list()
        for l in range(5):
            if l == 0:
                conv = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            else:
                conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            block = [conv, nn.ELU()]
            if batch_norm:
                block.append(nn.BatchNorm2d(32))
            sequence.extend(block)
        self.model = nn.Sequential(*sequence)


    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state.view(encoded_state.shape[0], -1)

class BurdaHead(nn.Module):
    '''
    Large scale curiosity paper
    '''
    def __init__(
            self, 
            image_shape,
            output_size=512,
            conv_output_size=3136,
            batch_norm=False,
            hook=False
            ):
        super(BurdaHead, self).__init__()
        c, h, w = image_shape
        self.output_size = output_size
        self.conv_output_size = conv_output_size
        self.model = nn.Sequential(
                                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
                                nn.LeakyReLU(),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
                                nn.LeakyReLU(),
                                nn.BatchNorm2d(64),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                                nn.LeakyReLU(),
                                nn.BatchNorm2d(64),
                                Flatten(),
                                nn.Linear(in_features=self.conv_output_size, out_features=self.output_size),
                                # nn.BatchNorm1d(self.output_size)
                                )

        # if hook:
        #     self.model[-1].weight.register_hook(lambda grad: print(grad))

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state

class ResBlock(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size):
        super(ResBlock, self).__init__()

        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.lin_2 = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, x, action):
        res = nn.functional.leaky_relu(self.lin_1(torch.cat([x, action], 2)))
        res = self.lin_2(torch.cat([res, action], 2))
        return res + x

class ResForward(nn.Module):
    def __init__(self,
                 feature_size,
                 action_size):
        super(ResForward, self).__init__()

        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.res_block_1 = ResBlock(feature_size, action_size)
        self.res_block_2 = ResBlock(feature_size, action_size)
        self.res_block_3 = ResBlock(feature_size, action_size)
        self.res_block_4 = ResBlock(feature_size, action_size)
        self.lin_last = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, phi1, action):
        x = nn.functional.leaky_relu(self.lin_1(torch.cat([phi1, action], 2)))
        x = self.res_block_1(x, action)
        x = self.res_block_2(x, action)
        x = self.res_block_3(x, action)
        x = self.res_block_4(x, action)
        x = self.lin_last(torch.cat([x, action], 2))
        return x

class ICM(nn.Module):

    def __init__(
            self, 
            image_shape, 
            action_size, 
            feature_encoding='idf', 
            batch_norm=False,
            prediction_beta=0.01
            ):
        super(ICM, self).__init__()

        self.prediction_beta = prediction_beta
        self.feature_encoding = feature_encoding

        if self.feature_encoding != 'none':
            if self.feature_encoding == 'idf':
                self.feature_size = 288
                self.encoder = UniverseHead(image_shape=image_shape, batch_norm=batch_norm)
            elif self.feature_encoding == 'idf_burda':
                self.feature_size = 512
                self.encoder = BurdaHead(image_shape=image_shape, output_size=self.feature_size, batch_norm=batch_norm, hook=True)

        # self.forward_model = nn.Sequential(
        #     nn.Linear(self.feature_size + action_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.feature_size)
        #     )
        self.forward_model = ResForward(feature_size=self.feature_size,
                                        action_size=action_size)

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_size * 2, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, action_size)
            )


    def forward(self, obs1, obs2, action):
        img1 = obs1.type(torch.float)
        img2 = obs2.type(torch.float) # Expect torch.uint8 inputs
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs1, 3) 

        phi1 = img1
        phi2 = img2
        if self.feature_encoding != 'none':
            phi1 = self.encoder(img1.view(T * B, *img_shape))
            phi2 = self.encoder(img2.view(T * B, *img_shape))
            phi1 = phi1.view(T, B, -1) # make sure you're not mixing data up here
            phi2 = phi2.view(T, B, -1)

        # predicted_action = nn.functional.softmax(self.inverse_model(torch.cat([phi1.view(T, B, -1), phi2.view(T, B, -1)], 2)), dim=-1) # SOFTMAX IS PERFORMED INSIDE CROSS ENT
        predicted_action = self.inverse_model(torch.cat([phi1, phi2], 2))

        # predicted_phi2 = self.forward_model(torch.cat([phi1.detach(), action.view(T, B, -1)], 2)) # USED WITH OLD FORWARD MODEL
        predicted_phi2 = self.forward_model(phi1.detach(), action.view(T, B, -1))

        return phi1, phi2, predicted_phi2, predicted_action

    def compute_bonus(self, obs, action, next_obs):
        phi1, phi2, predicted_phi2, predicted_action =  self.forward(obs, next_obs, action)
        forward_loss = 0.5 * nn.functional.mse_loss(predicted_phi2, phi2).sum(-1).unsqueeze(-1)
        return self.prediction_beta * forward_loss.squeeze()

    def compute_loss(self, obs, action, next_obs):
        #------------------------------------------------------------#
        # hacky dimension add for when you have only one environment
        if action.dim() == 2: 
            action = action.unsqueeze(1)
        #------------------------------------------------------------#
        phi1, phi2, predicted_phi2, predicted_action = self.forward(obs, next_obs, action)
        action = torch.max(action.view(-1, *action.shape[2:]), 1)[1] # conver action to (T * B, action_size), then get target indexes
        inverse_loss = nn.functional.cross_entropy(predicted_action.view(-1, *predicted_action.shape[2:]), action.detach())
        forward_loss = 0.5 * nn.functional.mse_loss(predicted_phi2, phi2.detach())
        print('INVLOSS: ', inverse_loss.squeeze())
        print('FORLOSS: ', forward_loss.squeeze())
        return inverse_loss.squeeze(), forward_loss.squeeze()





