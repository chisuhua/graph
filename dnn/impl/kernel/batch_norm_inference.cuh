/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>

#include <stdio.h>

namespace cudnn {
namespace impl {
namespace kernel {

using param::CuParamConstTensor;
using param::CuParamScaling;
using param::CuParamTensor;

constexpr int kBatchNormInferenceTileSize = 1;

template <class T, class ScalingT>
__global__ void BatchNormInferencePerActivation(ScalingT alpha,
                                                ScalingT beta,
                                                int dim_n,
                                                int dim_c,
                                                int dim_d,
                                                int dim_h,
                                                int dim_w,
                                                int x_stride_n,
                                                int x_stride_c,
                                                int x_stride_d,
                                                int x_stride_h,
                                                int x_stride_w,
                                                const T* x,
                                                int y_stride_n,
                                                int y_stride_c,
                                                int y_stride_d,
                                                int y_stride_h,
                                                int y_stride_w,
                                                T* y,
                                                int stride_c,
                                                int stride_d,
                                                int stride_h,
                                                int stride_w,
                                                const T* scale,
                                                const T* bias,
                                                const T* mean,
                                                const T* variance,
                                                float epsilon) {
    const auto dim_hw              = dim_h * dim_w;
    const auto nb_spatial_elements = dim_d * dim_h * dim_w;

    const auto n       = blockIdx.z;
    const auto c       = blockIdx.y;
    const auto base_id = (blockDim.x * blockIdx.x + threadIdx.x) * kBatchNormInferenceTileSize;

#pragma unroll kBatchNormInferenceTileSize
    for (int i = 0; i < kBatchNormInferenceTileSize; ++i) {
        auto gid = base_id + i;
        if (gid > nb_spatial_elements - 1) {
            return;
        }
        const auto d = gid / dim_hw;
        const auto h = gid % dim_hw / dim_w;
        const auto w = gid % dim_w;

        int idx = c * stride_c + d * stride_d + h * stride_h + w * stride_w;
        T tmp_y = bias[idx] + (scale[idx] * (x[n * x_stride_n + c * x_stride_c + d * x_stride_d +
                                               h * x_stride_h + w * x_stride_w] -
                                             mean[idx])) /
                                  sqrt(epsilon + variance[idx]);

        tmp_y = beta * y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h +
                         w * y_stride_w] +
                alpha * tmp_y;

        y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + w * y_stride_w] =
            tmp_y;
    }
}

template <class T, class ScalingT>
__global__ void BatchNormInferenceSpatial(ScalingT alpha,
                                          ScalingT beta,
                                          int dim_n,
                                          int dim_c,
                                          int dim_d,
                                          int dim_h,
                                          int dim_w,
                                          int x_stride_n,
                                          int x_stride_c,
                                          int x_stride_d,
                                          int x_stride_h,
                                          int x_stride_w,
                                          const T* x,
                                          int y_stride_n,
                                          int y_stride_c,
                                          int y_stride_d,
                                          int y_stride_h,
                                          int y_stride_w,
                                          T* y,
                                          int stride_c,
                                          const T* scale,
                                          const T* bias,
                                          const T* mean,
                                          const T* variance,
                                          float epsilon) {
    const auto dim_hw              = dim_h * dim_w;
    const auto nb_spatial_elements = dim_d * dim_h * dim_w;

    const auto n       = blockIdx.z;
    const auto c       = blockIdx.y;
    const auto base_id = (blockDim.x * blockIdx.x + threadIdx.x) * kBatchNormInferenceTileSize;

#pragma unroll kBatchNormInferenceTileSize
    for (int i = 0; i < kBatchNormInferenceTileSize; ++i) {
        auto gid = base_id + i;
        if (gid > nb_spatial_elements - 1) {
            return;
        }
        const auto d = gid / dim_hw;
        const auto h = gid % dim_hw / dim_w;
        const auto w = gid % dim_w;

        int idx = c * stride_c;
        T tmp_y = bias[idx] + (scale[idx] * (x[n * x_stride_n + c * x_stride_c + d * x_stride_d +
                                               h * x_stride_h + w * x_stride_w] -
                                             mean[idx])) /
                                  sqrt(epsilon + variance[idx]);
        tmp_y = beta * y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h +
                         w * y_stride_w] +
                alpha * tmp_y;
        y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + w * y_stride_w] =
            tmp_y;
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
