/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cfloat>
#include <type_traits>

namespace cudnn {
namespace impl {
namespace kernel {

template <typename T>
__inline__ __host__ __device__ T Epsilon(T v) {
    static_assert(!std::is_same<T, float>::value && !std::is_same<T, float>::value,
                  "Type of double & double2 are not supported");
    return v;
}

template <>
__inline__ __host__ __device__ float Epsilon<float>(float) {
    return FLT_EPSILON;
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
