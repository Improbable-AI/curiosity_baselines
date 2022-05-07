#pragma once

#include <vector>
#include <Eigen/Core>
#include <limits>
#include "sofm/types.h"

namespace sofm
{
    namespace art
    {
        class OnlineFuzzyART
        {
        private:
            double rho_;
            double alpha_;
            double beta_;
            int num_features_;
            int num_clusters_;
            int iterations_ = 0;
            Matrix w_;

            int eval_pattern(const VectorConstRef pattern);
            int train_pattern(const VectorConstRef pattern);

        public:
            OnlineFuzzyART(double rho, double alpha, double beta, int num_features);
            std::vector<int> run_online(const MatrixConstRef features, int max_epochs = std::numeric_limits<int>::max());
        };
    } // namespace art

} // namespace sofm
