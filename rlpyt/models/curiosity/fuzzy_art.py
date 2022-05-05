"""
   Copyright 2019 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
# References:
# [1] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast
#     stable learning and categorization of analog patterns by an adaptive
#     resonance system," Neural networks, vol. 4, no. 6, pp. 759�771, 1991.

import random

import numpy as np
import numpy.linalg
from sklearn.base import BaseEstimator, ClusterMixin

__author__ = 'Islam Elnabarawy'

def max_norm(x):
    # noinspection PyTypeChecker
    return np.linalg.norm(x, ord=1)


def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)


class FuzzyART(BaseEstimator, ClusterMixin):
    class MaxClustersReached(BaseException):
        def __init__(self, num_clusters, max_clusters) -> None:
            message = "Current number of clusters {} exceeds the maximum value of {}".format(num_clusters, max_clusters)
            super().__init__(message)

    def __init__(self, rho, alpha, beta, max_epochs=np.inf, shuffle=True, random_seed=None, w_init=None,
                 distance_fn=None, match_fn=None, vigilance_fn=None, update_fn=None, max_clusters=None):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.w_init = w_init

        self.distance_fn = distance_fn
        self.match_fn = match_fn
        self.vigilance_fn = vigilance_fn
        self.update_fn = update_fn

        self.max_clusters = max_clusters

        self.w = None
        self.num_clusters = None
        self.num_features = None
        self.labels = None
        self.iterations = 0

        self.initialized = False

    def set_params(self, **params):
        keys = ['rho', 'alpha', 'beta', 'w_init', 'max_epochs', 'shuffle', 'random_seed']

        for p in params.keys():
            if p not in keys:
                raise KeyError('Invalid parameter specified')
            self.__setattr__(p, params[p])

        return self

    def get_params(self, deep=True):
        return {
            'rho': self.rho, 'alpha': self.alpha, 'beta': self.beta,
            'w_init': self.w_init, 'max_epochs': self.max_epochs,
            'shuffle': self.shuffle, 'random_seed': self.random_seed
        }

    def initialize(self, inputs):
        self.num_features = inputs.shape[1]
        self.w = self.w_init if self.w_init is not None else np.ones((0, self.num_features * 2))

        self.initialized = True


    def fit(self, inputs, labels=None):

        self.num_features = inputs.shape[1]

        if self.w_init is not None:
            assert self.w_init.shape[1] == (self.num_features * 2)

        if not self.initialized:
            self.initialize(inputs)
        
        self.num_clusters = self.w.shape[0]

        # complement-code the data
        dataset = np.concatenate((inputs, 1 - inputs), axis=1)

        # initialize variables
        self.labels = np.zeros(dataset.shape[0])
        self.iterations = 0
        w_old = None

        if self.shuffle and self.random_seed is not None:
            random.seed(self.random_seed)

        # repeat the learning until either convergence or max_epochs
        while not np.array_equal(self.w, w_old) and self.iterations < self.max_epochs:
            w_old = self.w
            self.labels = np.zeros(dataset.shape[0])

            indices = list(range(dataset.shape[0]))

            if self.shuffle:
                random.shuffle(indices)

            # present the input patters to the Fuzzy ART module
            for ix in indices:
                self.labels[ix] = self.train_pattern(dataset[ix, :])

            self.iterations += 1

        # return results
        return self.labels

    def predict(self, inputs):
        if not self.initialized:
            self.initialize(inputs)
        dataset = np.concatenate((inputs, 1 - inputs), axis=1)
        labels = np.array(list(map(self.eval_pattern, dataset)), dtype=np.int32)
        return labels

    def train_pattern(self, pattern):
        # evaluate the pattern to get the winning category
        winner = self.eval_pattern(pattern)

        # check if the uncommitted node was the winner
        if (winner + 1) > self.num_clusters:
            self.num_clusters += 1
            self.w = np.concatenate((self.w, np.ones((1, self.w.shape[1]))))

        # check if the number of clusters has exceeded max_clusters
        if self.max_clusters is not None and self.num_clusters > self.max_clusters:
            raise FuzzyART.MaxClustersReached(self.num_clusters, self.max_clusters)

        # update the weight of the winning neuron
        self.w[winner, :] = self.weight_update(pattern, self.w[winner, :], self.beta)

        return winner

    def eval_pattern(self, pattern):
        # initialize variables
        matches = np.zeros(self.w.shape[0])
        # calculate the category match values
        for jx in range(self.w.shape[0]):
            matches[jx] = self.category_choice(pattern, self.w[jx, :], self.alpha)
        # pick the winning category
        match_attempts = 0
        while match_attempts < len(matches):
            # winner-take-all selection
            winner = np.argmax(matches)
            # vigilance test
            if self.vigilance_check(pattern, self.w[winner, :], self.rho):
                # the winning category passed the vigilance test
                return winner
            else:
                # shut off this category from further testing
                matches[winner] = 0
                match_attempts += 1
        return len(matches)

    def pattern_compare(self, pattern, category_w):
        if self.distance_fn is not None:
            return self.distance_fn(pattern, category_w)
        return fuzzy_and(pattern, category_w)

    def category_choice(self, pattern, category_w, alpha):
        if self.match_fn is not None:
            return self.match_fn(pattern, category_w, alpha)
        return max_norm(self.pattern_compare(pattern, category_w)) / (alpha + max_norm(category_w))

    def vigilance_check(self, pattern, category_w, rho):
        if self.vigilance_fn is not None:
            return self.vigilance_fn(pattern, category_w, rho)
        return max_norm(self.pattern_compare(pattern, category_w)) >= rho * max_norm(pattern)

    def weight_update(self, pattern, category_w, beta):
        if self.update_fn is not None:
            return self.update_fn(pattern, category_w, beta)
        return beta * self.pattern_compare(pattern, category_w) + (1 - beta) * category_w

    @property
    def cluster_prototypes(self):
        clusters = np.array([
            self.w[:self.num_clusters, :self.num_features],
            1 - self.w[:self.num_clusters, self.num_features:]
        ])

        return clusters.round(2)


def fa_cluster(args):
    from sklearn.metrics import adjusted_rand_score
    rho, inputs, targets = args
    fa = FuzzyART(rho, 0.001, 1.0, shuffle=False)
    labels = fa.fit(inputs)
    return adjusted_rand_score(targets, labels)


def eval_fuzzy_art(inputs, targets, rho_values):
    import multiprocessing as mp
    with mp.Pool() as pool:
        ari_values = pool.map(fa_cluster, [(rho, inputs, targets) for rho in rho_values])

    return ari_values
