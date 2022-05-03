import cv2
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead
from rlpyt.models.utils import Flatten
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
torch.set_printoptions(edgeitems=3)

# Kohonen imports
from tqdm import tqdm

import scipy as scp
import scipy.stats
import warnings

import collections
from typing import Tuple, Callable, Generator


#### Kohonen implementaion:

class KohonenSOM:
    def __init__(self,
                 input_dim: int = 2, node_shape: Tuple = (10, 10), init_scaling: float = 1E-3,
                 ):
        """A Kohonen Self-Orgranizing Map

        :param input_dim: Dimensions of the input to the map. (Default: 2)
        :param node_shape: Tuple of dimension (o,) containing ints specifying the output node shape. (Default: (10, 10))
        :param init_scaling: Initial scaling of weights.
        """
        self.input_dim = input_dim  # <- Dimension of input data
        self.shape = node_shape  # <- Shape of output map of nodes
        self.ndim = len(node_shape)

        self.W = np.random.random((*self.shape, self.input_dim)) * init_scaling  # <- Weights of the map
        self.W += np.array((1/2,)*input_dim)  # Centering, assuming a square on [0, 1]^input_dim

        self.node_grid = np.stack(np.meshgrid(*(np.arange(dim) for dim in self.shape)), axis=-1)  # <- A grid of coordinates for the nodes

        #  data_dim = (1, 1, ..., 1, input_dim) to reshape each sample to, ensuring good broadcasting
        #  node_dim = (1, 1, ..., 1, self.ndim) to reshape each sample to, ensuring good broadcasting
        self.data_shape = (1,)*(self.W.ndim-1) + (self.input_dim,)  # <- Shape of datapoint to broadcast over self.W
        self.node_shape = (1,)*(self.node_grid.ndim-1) + (self.ndim,)  # <- Shape of datapoint to broadcast over self.node_grid

        # Logging:
        self.total_its = 0  # <- Total number of sampling iterations
        self.W_log = collections.OrderedDict()  # <- Log of W-matrix after 'key' its. Preserves insertion ordering
        self.W_log[self.total_its] = self.W

    def train(self,
              lr: float, max_its: int,
              sample_generator: Generator[np.ndarray, None, None],
              neighborhood_fcn: Callable[[np.ndarray, np.ndarray], float],
              log_interval: int = np.inf,
              ) -> np.ndarray:
        """'Train' the SOM, letting it self-organize.

        :param lr: Learning rate
        :param max_its: Samples to train for. Exits early if sample_generator exits early.
        :param sample_generator: Callable giving samples each time it is called. Samples should be np.ndarrays of shape (input_dim,)
        :param neighborhood_fcn: Takes (node_to_update_weight_for, best_matching_unit). Should be vectorized.
        :param log_interval: Interval between logging. np.inf for no logging. (Default: 100)
        :return: self.W, final weights
        """
        self._last_generator = sample_generator

        # Wrap the generator to get no more than max_its samples.
        def sample_enumerator():
            for idx in tqdm(range(max_its)):
                yield idx, next(sample_generator)

        # Iterate over samples
        for idx, sample in sample_enumerator():
            sample = sample.reshape(self.data_shape)  # <- Sample reshaped for broadcasting

            # Find index of best matching weight ((self.W.shape-1) dimensional tuple corresponding to an output unit)
            # Canonically $i(x) = argmin_j ||x-w_j||
            best_match_idx = np.unravel_index(
                np.argmin(
                    np.linalg.norm(self.W - sample, axis=-1)
                ), shape=self.shape
            )

            # Get the neighborhood function scaling value for each of the output units. Canonically $h(j, i(x))$
            neighborhood_scaling = neighborhood_fcn(
                self.node_grid,
                np.array(best_match_idx).reshape(self.node_shape)
            ).T

            # Get the diff, canonically $w_j += \eta h(j, i(x)) (x - w_j)$
            dW = lr * np.expand_dims(neighborhood_scaling, -1) * (sample - self.W)
            self.W = self.W + lr*self.get_dW(sample, neighborhood_fcn) # dW

            # Logging:
            self.total_its += 1
            if self.total_its % log_interval == 0:
                self.W_log[self.total_its] = self.W

        return self.W

    def get_dW(self, sample: np.ndarray, neighborhood_fcn: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
        best_match_idx = np.unravel_index(
            np.argmin(
                np.linalg.norm(self.W - sample, axis=-1)
            ), shape=self.shape
        )

        # Get the neighborhood function scaling value for each of the output units. Canonically $h(j, i(x))$
        neighborhood_scaling = neighborhood_fcn(
            self.node_grid,
            np.array(best_match_idx).reshape(self.node_shape)
        ).T

        # Get the diff, canonically $w_j += \eta h(j, i(x)) (x - w_j)$
        dW = np.expand_dims(neighborhood_scaling, -1) * (sample - self.W)

        return dW


class Kohonen(nn.Module):
    kohonen_map: KohonenSOM

    def __init__(self,
                 image_shape,
                 device='cpu'
                 ):

        super(Kohonen, self).__init__()

        # assuming grayscale inputs
        c, h, w = 1, image_shape[1], image_shape[2]
        self.feature_size = 512
        self.conv_feature_size = 7*7*64
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')


        # TODO(marius): Make into parameters defined externally
        self.encoded_input_dim = 3  # TODO(odin): Fix to whatever is actual
        self.encoding_batch_norm = True
        kohonen_nodes_shape = (10, 10)
        self.lr = 1
        self.train_its_on_batch = 10

        self.feature_encoder = BurdaHead(image_shape, output_size=self.encoded_input_dim, batch_norm=self.encoding_batch_norm)

        self.kohonen_map = KohonenSOM(self.encoded_input_dim, node_shape=kohonen_nodes_shape)
        self.neighborhood_fcn = NeighborhoodFcns.gaussian(self.encoded_input_dim, cov=1)


    def forward(self, obs, done=None):
        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)

        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        obs_feature_mapped = self.feature_encoder.forward(obs.view(T * B, *img_shape))
        rewards = torch.zeros(T*B)

        # assert reduced_dim_sample.shape == (self.encoded_input_dim,)

        # with torch.no_grad():
        #     obs = obs.type(torch.float) # expect torch.uint8 inputs
        #     predicted_phi = self.network(obs.detach().view(T * B, *img_shape)).view(T, B, -1)
        # TODO(marius): Handle being done
        for idx, obs_map in enumerate(obs_feature_mapped):
            weight_update = torch.Tensor(self.kohonen_map.get_dW(obs_map, self.neighborhood_fcn))
            rewards[idx] = torch.linalg.norm(weight_update)
        ret = rewards.view(T, B)

        return ret, T, B
        # return predicted_phi, T, B

    def compute_bonus(self, next_observation, done):

        # with torch.no_grad():
        #     predicted_phi, T, B = self.forward(next_observation, done)
        #     rewards = predicted_phi.detach().sum(-1)/self.feature_size
        return self.forward(next_observation, done)

    def compute_loss(self, observations, valid):
        # TODO(marius): Verify observations shape
        lead_dim, T, B, img_shape = infer_leading_dims(observations, 3)
        obs_feature_mapped = self.feature_encoder.forward(observations.view(T * B, *img_shape))

        def get_sample_gen():
            while True:
                rnd_idx = np.random.randint(0, len(observations))
                obs = observations[rnd_idx]
                # TODO(odin): Do feature mammping on obs
                yield obs

        sample_generator = get_sample_gen()

        self.kohonen_map.train(
            lr=self.lr,
            max_its=len(observations)*self.train_its_on_batch,
            sample_generator=sample_generator,
            neighborhood_fcn=self.neighborhood_fcn
        )

        return torch.zeros(1)


class NeighborhoodFcns:
    @staticmethod
    def gaussian(ndim, mean=None, cov=None) -> Callable[[np.ndarray, np.ndarray], float]:

        # Make sure mean is specified with the right dimension.
        # This allows the pdf to know what dimension the input points are, broadcasting/evaluating correctly.
        mean = np.zeros(ndim) if mean is None else mean

        if not np.allclose(mean, np.zeros_like(mean)):
            warnings.warn(f"Mean is not 0 in SOM_utils.NeighborhoodFcns.gaussian, but is {mean}.")

        def gaussian_pdf(x, y):
            return scp.stats.multivariate_normal.pdf(x-y, mean=mean, cov=cov)

        return gaussian_pdf


def _test_som():
    from SOM_utils import NeighborhoodFcns, SampleGenerators

    input_ndim = 2  # <- Dimensionality of input
    node_shape = (25, 25)  # <- Shape of node-array

    som = KohonenSOM(input_ndim, node_shape)  # <- The map object
    gen, plotting_params = SampleGenerators.v1_density(a=0.2, b=1.5)  # <- Data generator (must give arrays with dimension input_dim)

    # nbh = NeighborhoodFcns.gaussian(len(node_shape))  # <- Neighborhood function, taking the node ndim for vectorization
    nbh = lambda cov: NeighborhoodFcns.gaussian(len(node_shape), cov=cov)  # <- Neighborhood function, taking the node ndim for vectorization

    # som.train(0.5, 5_001, gen, nbh, log_interval=500)
    # som.train(0.2, 10_001, gen, nbh, log_interval=500)
    # som.train(10, 100_001, gen, nbh, log_interval=500)

    # som.W = som.node_grid / 10
    som.train(3, 401, gen, nbh(3), log_interval=20)
    som.train(3, 20_001, gen, nbh(3), log_interval=400)
    som.train(1, 20_001, gen, nbh(1), log_interval=400)
    som.train(0.3, 20_001, gen, nbh(0.8), log_interval=400)
    # som.train(0.3, 2_001, gen, nbh(1), log_interval=50)
    # som.train(0.3, 20_001, gen, nbh(0.1), log_interval=500)
    # som.train(0.03, 10_001, gen, nbh01, log_interval=500)

    print(np.round(10*som.W.T, 0))

    return som, plotting_params