#include "OnlineFuzzyArt.h"
#include <optional>
#include <numeric>
#include <algorithm>
#include <random>
#include <iostream>
#include "utils.h"

namespace art
{

    OnlineFuzzyART::OnlineFuzzyART(double rho, double alpha, double beta, int num_features)
        : rho_(rho),
          alpha_(alpha),
          beta_(beta),
          num_features_(num_features),
          num_clusters_(0),
          w_(Matrix::Ones(1, 2 * num_features))
    {
    }

    std::vector<int> OnlineFuzzyART::run_online(const MatrixConstRef features, int max_epochs)
    {
        std::vector<int> cluster_choices;
        size_t num_datapoints = features.rows();
        size_t num_features = features.cols();

        for (int i = 0; i < num_datapoints; i++)
        {
            cluster_choices.push_back(0);
        }
        int iterations = 0;

        Matrix w_old = Matrix::Constant(w_.rows(), w_.cols(), NAN);

        std::vector<int> indices(features.rows());
        std::iota(indices.begin(), indices.end(), 0);
        auto rng = std::mt19937{std::random_device{}()};

        while (!w_.isApprox(w_old) && iterations < max_epochs)
        {
            std::cout << "Entered while loop C++!\n";
            w_old = w_;

            // std::shuffle(indices.begin(), indices.end(), rng);

            for (const auto &ix : indices)
            {
                Vector pattern(2 * num_features);
                pattern.head(num_features) = features.row(ix);
                pattern.tail(num_features) = 1.0 - features.row(ix).array();
                cluster_choices[ix] = train_pattern(pattern);
            }
            iterations++;
        }

        std::cout << w_ << "\n" << std::endl;

        return cluster_choices;
    }

    int OnlineFuzzyART::train_pattern(const VectorConstRef pattern)
    {
        // evaluate the pattern to get the winning category
        int winner = eval_pattern(pattern);

        // update the weight of the winning neuron
        w_.row(winner) = beta_ * utils::fuzzy_and(pattern, w_.row(winner)) + (1.0 - beta_) * w_.row(winner);

        // check if the uncommitted node was the winner
        if ((winner + 1) > num_clusters_)
        {
            num_clusters_++;
            w_.conservativeResize(w_.rows() + 1, Eigen::NoChange);
            w_.bottomRows<1>().array() = 1;
        }

        return winner;
    }

    int OnlineFuzzyART::eval_pattern(const VectorConstRef pattern)
    {
        size_t num_categories = w_.rows();

        Vector matches(Vector::NullaryExpr(num_categories, [&](Eigen::Index jx) {
            return utils::category_choice(pattern, w_.row(jx), alpha_);
        }));

        double vigilance_test = rho_ * utils::max_norm(pattern);
        int match_attempts = 0;

        Eigen::Index winner;
        while (match_attempts < num_categories) {
            // This way of using maxCoeff writes argmax to winner
            matches.maxCoeff(&winner);

            if (utils::max_norm(utils::fuzzy_and(pattern, w_.row(winner))) >= vigilance_test) {
                return winner;
            } else {
                matches(winner) = 0;
                match_attempts++;
            }
        }

        return num_categories - 1;
    }

} // namespace art
