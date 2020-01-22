/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cuda_fp16.h>  // __half
#include <cudnn.h>
#include <stdint.h>  // int8_t, uint8_t, int32_t

namespace cudnn {
namespace impl {

template <cudnnDataType_t DataType>
struct CuTensorType {};

template <>
struct CuTensorType<DNN_DATA_FLOAT> {
    using Type        = float;
    using ScalingType = float;
};

template <>
struct CuTensorType<DNN_DATA_DOUBLE> {
    using Type        = double;
    using ScalingType = double;
};

template <>
struct CuTensorType<DNN_DATA_HALF> {
    using Type        = __half;
    using ScalingType = float;
};

template <>
struct CuTensorType<DNN_DATA_INT8> {
    using Type        = int8_t;
    using ScalingType = float;
};

template <>
struct CuTensorType<DNN_DATA_UINT8> {
    using Type        = uint8_t;
    using ScalingType = float;
};

template <>
struct CuTensorType<DNN_DATA_INT32> {
    using Type        = int32_t;
    using ScalingType = float;
};

template <>
struct CuTensorType<DNN_DATA_INT8x4> {
    using Type        = int8_t[4];
    using ScalingType = float;
};

template <>
struct CuTensorType<DNN_DATA_UINT8x4> {
    using Type        = uint8_t[4];
    using ScalingType = float;
};

}  // namespace impl
}  // namespace cudnn
