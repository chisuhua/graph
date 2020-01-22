/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/////////////////////////////////////////
// DEPRECATED Prefer CuParamTensor::At //
/////////////////////////////////////////

#pragma once
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

namespace cudnn {
namespace impl {
namespace kernel {
/**
 * @brief obtain a reference to an 4d tensor's element
 *
 * @data[in]   ptr     device pointer to tensor's raw data
 * @data[in]   meta    tensor descriptor
 * @data[in]   n       index of batch
 * @data[in]   c       index of channel
 * @data[in]   h       index of height
 * @data[in]   w       index of width
 * @return  a areference to specific element
 */
template <class T>
__device__ __forceinline__ T&
At(T* ptr, const cudnn::impl::meta::CuMetaTensor& meta, int n, int c, int h, int w) {
    return ptr[n * meta.GetStride(1) + c * meta.GetStride(2) + h * meta.GetStride(3) +
               w * meta.GetStride(4)];
}

/**
 * @brief obtain a value of an 4d tensor's element
 *
 * @data[in]   ptr     device pointer to tensor's raw data
 * @data[in]   meta    tensor descriptor
 * @data[in]   n       index of batch
 * @data[in]   c       index of channel
 * @data[in]   h       index of height
 * @data[in]   w       index of width
 * @return  a specific element's value
 */
template <class T>
__device__ __forceinline__ T
At(const T* ptr, const cudnn::impl::meta::CuMetaTensor& meta, int n, int c, int h, int w) {
    return ptr[n * meta.GetStride(1) + c * meta.GetStride(2) + h * meta.GetStride(3) +
               w * meta.GetStride(4)];
}

/**
 * @brief obtain a reference to an 5d tensor's element
 *
 * @data[in]   ptr     device pointer to tensor's raw data
 * @data[in]   meta    tensor descriptor
 * @data[in]   n       index of batch
 * @data[in]   c       index of channel
 * @data[in]   d       index of depth
 * @data[in]   h       index of height
 * @data[in]   w       index of width
 * @return  a areference to specific element
 */
template <class T>
__device__ __forceinline__ T&
At(T* ptr, const cudnn::impl::meta::CuMetaTensor& meta, int n, int c, int d, int h, int w) {
    return ptr[n * meta.GetStride(1) + c * meta.GetStride(2) + d * meta.GetStride(3) +
               h * meta.GetStride(4) + w * meta.GetStride(5)];
}

/**
 * @brief obtain a value of an 5d tensor's element
 *
 * @data[in]   ptr     device pointer to tensor's raw data
 * @data[in]   meta    tensor descriptor
 * @data[in]   n       index of batch
 * @data[in]   c       index of channel
 * @data[in]   d       index of depth
 * @data[in]   h       index of height
 * @data[in]   w       index of width
 * @return  a specific element's value
 */
template <class T>
__device__ __forceinline__ T
At(const T* ptr, const cudnn::impl::meta::CuMetaTensor& meta, int n, int d, int c, int h, int w) {
    return ptr[n * meta.GetStride(1) + c * meta.GetStride(2) + d * meta.GetStride(3) +
               h * meta.GetStride(4) + w * meta.GetStride(5)];
}
}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
