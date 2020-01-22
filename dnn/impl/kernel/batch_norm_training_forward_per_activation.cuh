/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>

namespace cudnn {
namespace impl {
namespace kernel {

template <class T, class ScalingT>
__global__ void BatchNormTrainingForwardPerActivation(ScalingT alpha,
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
                                                      const T* tensor_bn_scale,
                                                      const T* tensor_bn_bias,
                                                      float factor,
                                                      T* tensor_running_mean,
                                                      T* tensor_running_variance,
                                                      float epsilon,
                                                      T* tensor_save_mean,
                                                      T* tensor_save_inv_variance) {
    // In original paper <Batch Normalization: Accelerating Deep Network Training by Reducing
    // Internal Convariate Shift> by Sergey Ioffe, Christian Szegedy. "inv_bs" represents 1/m in
    // [Algorithm 1: Batch Normalizing Transform, applied to activation x over a mini-batch]
    // "bs_over_bs1" represents m/m-1 in [Algorithm 2: Training a Batch-Normalized Network]
    const auto inv_bs      = 1.f / dim_n;                // use float 1.0 to avoid rounding to 0
    const auto bs_over_bs1 = 1.f * dim_n / (dim_n - 1);  // use float 1.0 to avoid rounding to 0

    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= dim_d * dim_h * dim_w) {
        return;
    }

    const auto c   = blockIdx.y;
    const auto d   = gid / (dim_h * dim_w);
    const auto h   = gid % (dim_h * dim_w) / dim_w;
    const auto w   = gid % dim_w;
    const auto idx = c * stride_c + d * stride_d + h * stride_h + w * stride_w;

    // compute mean
    T sum = static_cast<T>(0);
    for (int n = 0; n < dim_n; ++n) {
        sum +=
            x[x_stride_n * n + x_stride_c * c + x_stride_d * d + x_stride_h * h + x_stride_w * w];
    }
    auto mean = sum * inv_bs;  // 'auto' makes sure not lose precision

    // compute variance
    T variance = static_cast<T>(0);
    for (int n = 0; n < dim_n; ++n) {
        auto tmp =
            x[x_stride_n * n + x_stride_c * c + x_stride_d * d + x_stride_h * h + x_stride_w * w] -
            mean;
        variance += static_cast<T>(tmp * tmp * inv_bs);
    }

    // save running mena and running variance if appliable
    if (tensor_running_mean != nullptr) {
        tensor_running_mean[idx] *= (1 - factor);
        tensor_running_mean[idx] += static_cast<T>(factor * mean);

        tensor_running_variance[idx] *= (1 - factor);
        tensor_running_variance[idx] += (factor * variance * bs_over_bs1);
    }

    T inv_variance = 1.f / sqrt(variance + epsilon);

    // save mean and invert variance
    if (tensor_save_mean != nullptr) {
        tensor_save_mean[idx]         = mean;
        tensor_save_inv_variance[idx] = inv_variance;
    }

    // normalize, scale and shift
    const auto scale = tensor_bn_scale[idx];
    const auto bias  = tensor_bn_bias[idx];
    for (int n = 0; n < dim_n; ++n) {
        const auto x_hat =
            (x[x_stride_n * n + x_stride_c * c + x_stride_d * d + x_stride_h * h + x_stride_w * w] -
             mean) *
            inv_variance;
        auto tmp = x_hat * scale + bias;
        if (beta != 0) {
            tmp = y[y_stride_n * n + y_stride_c * c + y_stride_d * d + y_stride_h * h +
                    y_stride_w * w] *
                      alpha +
                  beta;
        }
        y[y_stride_n * n + y_stride_c * c + y_stride_d * d + y_stride_h * h + y_stride_w * w] = tmp;
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
