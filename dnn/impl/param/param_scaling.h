/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>

namespace cudnn {
namespace impl {
namespace param {

/** @class
 * @brief scaling parameters
 */
template <cudnnDataType_t>
struct CuParamScaling {
    typedef float Type;

    Type alpha;
    Type beta;

    __host__ __device__ __forceinline__ CuParamScaling(const void* alpha_ptr,
                                                       const void* beta_ptr) {
        alpha = *(reinterpret_cast<const Type*>(alpha_ptr));
        beta  = *(reinterpret_cast<const Type*>(beta_ptr));
    }
};

template <>
struct CuParamScaling<DNN_DATA_DOUBLE> {
    typedef double Type;

    Type alpha;
    Type beta;

    __host__ __device__ __forceinline__ CuParamScaling(const void* alpha_ptr,
                                                       const void* beta_ptr) {
        alpha = *(reinterpret_cast<const Type*>(alpha_ptr));
        beta  = *(reinterpret_cast<const Type*>(beta_ptr));
    }
};

}  // namespace param
}  // namespace impl
}  // namespace cudnn
