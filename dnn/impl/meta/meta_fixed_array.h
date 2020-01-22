/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn.h>

namespace cudnn {
namespace impl {
namespace meta {

/** @class
 * @brief raw C-style array holder with capacity set to DNN_DIM_MAX, means to hold dimension
 * related list, so one-based index is used in all APIs.
 */
template <class T>
class CuFixedArray {
 public:
    /**
     * @brief constructor
     *
     * @param[in]   nb_dims number of data to set
     * @param[in]   val     raw pointer to values to be set
     */
    __host__ __device__ CuFixedArray(int nb_dims, const T val[]) {
        int i = 0;
        for (; i < nb_dims; ++i) {
            data_[i] = val[i];
        }
        for (; i < DNN_DIM_MAX; ++i) {
            data_[i] = 0;
        }
    }

    __host__ __device__ CuFixedArray(const CuFixedArray<T>& other) {
        for (int i = 0; i < DNN_DIM_MAX; ++i) {
            data_[i] = other[i];
        }
    }

    /**
     * @brief get value of specified dimension
     *
     * @param[in]   dim     1-based dimension
     * @return  value of specified dimension
     */
    __host__ __device__ __forceinline__ T Get(int dim) const { return data_[dim - 1]; }

    /**
     * @brief get value of array index
     *
     * @param[in]   idx     0-based index
     * @return  value of specified index
     */
    __host__ __device__ __forceinline__ T operator[](int idx) const { return data_[idx]; }

 private:
    T data_[DNN_DIM_MAX];
};

}  // namespace meta
}  // namespace impl
}  // namespace cudnn
