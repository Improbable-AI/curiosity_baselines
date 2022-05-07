#include <Eigen/Core>
#include "types.h"


namespace art
{
    namespace utils {

    inline double max_norm(const VectorConstRef x)
    {
        return x.lpNorm<1>();
    }

    inline art::Vector fuzzy_and(const VectorConstRef x, const VectorConstRef y)
    {
        return x.cwiseMin(y);
    }

    inline double category_choice(const VectorConstRef pattern, const VectorConstRef category_w, double alpha)
    {
        return max_norm(fuzzy_and(pattern, category_w) / (alpha + max_norm(category_w)));
    }

    } // namespace utils
} // namespace art