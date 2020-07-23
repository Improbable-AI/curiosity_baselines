
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

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
        for l in range(4):
            if l == 0:
                conv = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            else:
                conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            block = [conv, nn.elu()]
            if batch_norm:
                block.append(nn.BatchNorm2d(32))
            sequence.extend(block)
        self.model = nn.Sequential(*sequence)

    def forward(self, state):
        """Compute the feature encoding convolution + head on the input;
        assumes correct input shape: [B,C,H,W]."""
        encoded_state = self.model(state)
        return encoded_state.view(-1, 288)

class ICM(nn.Module):

    def __init__(
            self, 
            image_shape, 
            action_size, 
            feature_encoding='idf', 
            batch_norm=False
            ):
        super(ICM, self).__init__()

        self.feature_encoding = feature_encoding
        if self.feature_encoding != 'none':
            if self.feature_encoding == 'idf':
                self.encoder = UniverseHead(image_shape=image_shape, batch_norm=batch_norm) # universe head from original paper (ICM 2017)

        self.forward_model = nn.Sequential(
            nn.Linear(288 + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 288)
            )
        self.inverse_model = nn.Sequential(
            nn.Linear(288 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.ReLU()
            )

    def forward(self, obs1, obs2, action):
        img1 = obs1.type(torch.float)
        img2 = obs2.type(torch.float) # Expect torch.uint8 inputs
        img1 = img1.mul_(1. / 255)
        img2 = img2.mul_(1. / 255) # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(obs1, 3) 

        phi1 = img1
        phi2 = img2
        if self.feature_encoding != 'none':
            phi1 = self.encoder(img1.view(T * B, *img_shape))
            phi2 = self.encoder(img2.view(T * B, *img_shape))
        predicted_action = nn.softmax(self.inverse_model(torch.cat([phi1, phi2], 1)), dim=-1)
        predicted_phi2 = self.forward_model(torch.cat([phi1, action], 1))

        return phi1, phi2, predicted_phi2, predicted_action

    def bonus(self, eta, states, next_states, actions, action_probs):
        action_pred, phi2_pred, phi1, phi2 =  self.icm(states, next_states, action_probs)
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2, reduce=False).sum(-1).unsqueeze(-1)
        return eta * forward_loss

    def loss(self, states, next_states, actions, action_probs):
        action_pred, phi2_pred, phi1, phi2 =  self.icm(states, next_states, action_probs)
        inverse_loss = F.cross_entropy(action_pred, actions.view(-1))
        forward_loss = 0.5 * F.mse_loss(phi2_pred, phi2.detach(), reduce=False).sum(-1).mean()
        return inverse_loss, forward_loss





