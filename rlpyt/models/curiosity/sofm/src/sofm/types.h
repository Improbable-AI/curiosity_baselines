#pragma once

#include <Eigen/Core>

namespace sofm
{
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixRef = Eigen::Ref<Matrix>;
    using MatrixConstRef = Eigen::Ref<const Matrix>;
    using Vector = Eigen::VectorXd;
    using VectorRef = Eigen::Ref<Vector>;
    using VectorConstRef = Eigen::Ref<const Vector>;
} // namespace sofm
