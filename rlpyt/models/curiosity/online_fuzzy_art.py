import numpy as np
from numpy import linalg as la
import random

__author__ = 'Islam Elnabarawy'


def max_norm(x):
    # noinspection PyTypeChecker
    return la.norm(x, ord=1)


def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)


def default_category_choice(pattern, category_w, alpha):
    return max_norm(fuzzy_and(pattern, category_w)) / (alpha + max_norm(category_w))



def scale_range(x, x_range, y_range=(0.0, 1.0)):
    """
    scale the number x from the range specified by x_range to the range specified by y_range

    :param x: the number to scale
    :type x: float
    :param x_range: the number range that x belongs to
    :type x_range: tuple
    :param y_range: the number range to convert x to, defaults to (0.0, 1.0)
    :type y_range: tuple
    :return: the scaled value
    :rtype: float
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min



class OnlineFuzzyART(object):
    def __init__(self, rho, alpha, beta, num_features, choice_fn=default_category_choice, w=None):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.num_features = num_features
        self.w = w if w is not None else np.ones((1, num_features * 2))
        self.num_clusters = self.w.shape[0] - 1
        self.clusters = np.zeros(0)
        self.choice_fn = choice_fn

    def run_batch(self, dataset, max_epochs=np.inf, seed=None):
        # complement-code the data
        dataset = np.concatenate((dataset, 1 - dataset), axis=1)

        # initialize variables
        cluster_choices = np.zeros(dataset.shape[0])
        iterations = 1
        w_old = None

        if seed is not None:
            random.seed(seed)

        # repeat the learning until either convergence or max_epochs
        while not np.array_equal(self.w, w_old) and iterations < max_epochs:
            w_old = self.w
            cluster_choices = np.zeros(dataset.shape[0])

            indices = list(range(dataset.shape[0]))
            random.shuffle(indices)

            # present the input patters to the Fuzzy ART module
            for ix in indices:
                cluster_choices[ix] = self.train_pattern(dataset[ix, :])

            iterations += 1

        # return results
        return iterations, cluster_choices

    def run_online(self, data_reader, data_ranges, max_epochs=np.inf, seed=None):
        # initialize variables
        cluster_choices = np.zeros(len(data_reader))
        iterations = 1
        w_old = None

        if seed is not None:
            random.seed(seed)

        def normalize(p):
            for i in range(len(p)):
                p[i] = scale_range(p[i], data_ranges[i])

        # repeat the learning until either convergence or max_epochs
        while not np.array_equal(self.w, w_old) and iterations < max_epochs:
            w_old = self.w
            indices = list(range(len(data_reader)))
            random.shuffle(indices)
            for ix in indices:
                pattern = np.array(data_reader[ix], dtype=float)
                normalize(pattern)
                pattern = np.concatenate((pattern, 1.0 - pattern))
                choice = self.train_pattern(pattern)
                cluster_choices[ix] = choice
            iterations += 1

        # return results
        return iterations, np.array(cluster_choices)

    def train_pattern(self, pattern):
        # evaluate the pattern to get the winning category
        winner = self.eval_pattern(pattern)

        # update the weight of the winning neuron
        self.w[winner, :] = self.beta * fuzzy_and(pattern, self.w[winner, :]) + (1 - self.beta) * self.w[winner, :]

        # check if the uncommitted node was the winner
        if (winner + 1) > self.num_clusters:
            self.num_clusters += 1
            self.w = np.concatenate((self.w, np.ones((1, self.w.shape[1]))))

        return winner

    def eval_pattern(self, pattern):
        # initialize variables
        matches = np.zeros(self.w.shape[0])
        # calculate the category match values
        for jx in range(self.w.shape[0]):
            matches[jx] = self.choice_fn(pattern, self.w[jx, :], self.alpha)
        # pick the winning category
        vigilance_test = self.rho * max_norm(pattern)
        match_attempts = 0
        while match_attempts < len(matches):
            # winner-take-all selection
            winner = np.argmax(matches)
            # vigilance test
            if max_norm(fuzzy_and(pattern, self.w[winner, :])) >= vigilance_test:
                # the winning category passed the vigilance test
                return winner
            else:
                # shut off this category from further testing
                matches[winner] = 0
                match_attempts += 1
        return len(matches) - 1


# def main():
#     from data import XCSVFileReader
#     with XCSVFileReader('data/users.csv') as reader:
#         fa = OnlineFuzzyART(0.95, 0.001, 0.9, reader.num_fields)
#         data_ranges = [(0, 1), (0, 6)] + [(0, 1)] * 21
#         iterations, clusters = fa.run_online(reader, data_ranges, 100)
#     print(iterations, fa.num_clusters)
#     np.set_printoptions(suppress=True)
#     print(fa.w)


# if __name__ == '__main__':
#     main()