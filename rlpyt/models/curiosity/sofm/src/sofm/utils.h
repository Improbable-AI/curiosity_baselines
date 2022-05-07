#pragma once

#include <Eigen/Core>
#include "sofm/types.h"
#include <iostream>

namespace sofm
{
    namespace utils {

    inline double max_norm(const VectorConstRef x)
    {
        return x.lpNorm<1>();
    }

    inline Vector fuzzy_and(const VectorConstRef x, const VectorConstRef y)
    {
        return x.cwiseMin(y);
    }

    inline double category_choice(const VectorConstRef pattern, const VectorConstRef category_w, double alpha)
    {
        return max_norm(fuzzy_and(pattern, category_w) / (alpha + max_norm(category_w)));
    }

    } // namespace utils
} // namespace sofm