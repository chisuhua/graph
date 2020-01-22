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
__global__ void BatchNormTrainingBackwardPerActivation(ScalingT alpha_data_scaling,
                                                       ScalingT beta_data_scaling,
                                                       ScalingT alpha_param_scaling,
                                                       ScalingT beta_param_scaling,
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
                                                       int dy_stride_n,
                                                       int dy_stride_c,
                                                       int dy_stride_d,
                                                       int dy_stride_h,
                                                       int dy_stride_w,
                                                       const T* dy,
                                                       int dx_stride_n,
                                                       int dx_stride_c,
                                                       int dx_stride_d,
                                                       int dx_stride_h,
                                                       int dx_stride_w,
                                                       T* dx,
                                                       int stride_c,
                                                       int stride_d,
                                                       int stride_h,
                                                       int stride_w,
                                                       const T* bn_scale,
                                                       T* dbn_scale_result,
                                                       T* dbn_bias_result,
                                                       float epsilon,
                                                       const T* saved_mean,
                                                       const T* saved_inv_variance) {
    // In paper <Batch Normalization: Accelerating Deep Network Training by Reducing
    // Internal Convariate Shift> by Sergey Ioffe, Christian Szegedy. "inv_bs" represents 1/m in
    // [Algorithm 1: Batch Normalizing Transform, applied to activation x over a mini-batch]
    const auto inv_bs = 1.f / dim_n;  // use float 1.0 to avoid rounding to 0

    const auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid > dim_d * dim_w * dim_h - 1) {
        return;
    }

    const auto c = blockIdx.y;
    const auto d = gid / (dim_h * dim_w);
    const auto h = gid % (dim_h * dim_w) / dim_w;
    const auto w = gid % dim_w;

    const int idx = stride_c * c + stride_d * d + stride_h * h + stride_w * w;

    T mean         = static_cast<T>(0);
    T inv_variance = static_cast<T>(0);
    if (saved_mean == nullptr) {
        // no saved mean and inv variance is available, should compute all by myself
        // compute mean
        T sum = static_cast<T>(0);
        for (int n = 0; n < dim_n; ++n) {
            sum += x[x_stride_n * n + x_stride_c * c + x_stride_d * d + x_stride_h * h +
                     x_stride_w * w];
        }
        mean = sum * inv_bs;

        // compute variance
        T variance = static_cast<T>(0);
        for (int n = 0; n < dim_n; ++n) {
            auto tmp = x[x_stride_n * n + x_stride_c * c + x_stride_d * d + x_stride_h * h +
                         x_stride_w * w] -
                       mean;
            variance += static_cast<T>(tmp * tmp * inv_bs);
        }

        inv_variance = 1 / sqrt(variance + epsilon);
    } else {
        // get saved mean and inv variance out from parameters user passed in
        mean         = saved_mean[idx];
        inv_variance = saved_inv_variance[idx];
    }

    T dvariance = static_cast<T>(0);
    T dscale    = static_cast<T>(0);
    T dbias     = static_cast<T>(0);
    T dmean     = static_cast<T>(0);
    // compute variance diff,  scale diff and bias diff and partial mean diff
    for (int n = 0; n < dim_n; ++n) {
        const auto dy_val = dy[n * dy_stride_n + c * dy_stride_c + d * dy_stride_d +
                               h * dy_stride_h + w * dy_stride_w];
        const auto dx_hat = dy_val * bn_scale[idx];

        const auto x_val =
            x[n * x_stride_n + c * x_stride_c + d * x_stride_d + h * x_stride_h + w * x_stride_w];
        dvariance = dvariance + dx_hat * (x_val - mean);

        const auto x_hat = (x_val - mean) * inv_variance;
        dscale           = dscale + dy_val * x_hat;
        dbias            = dbias + dy_val;

        dmean -= dx_hat;
    }
    dmean *= inv_variance;
    dvariance = dvariance * -0.5 * (inv_variance * inv_variance * inv_variance);

    // second part always equals to 0, is eliminated here

    // save back scale diff and bias diff
    if (beta_param_scaling != 0) {
        dscale = alpha_param_scaling * dscale + beta_param_scaling * dbn_scale_result[idx];
        dbias  = alpha_param_scaling * dbias + beta_param_scaling * dbn_bias_result[idx];
    }
    dbn_scale_result[idx] = dscale;
    dbn_bias_result[idx]  = dbias;

    // compute dx and save back
    for (int n = 0; n < dim_n; ++n) {
        const auto x_val =
            x[n * x_stride_n + c * x_stride_c + d * x_stride_d + h * x_stride_h + w * x_stride_w];
        const auto dy_val = dy[n * dy_stride_n + c * dy_stride_c + d * dy_stride_d +
                               h * dy_stride_h + w * dy_stride_w];
        const auto dx_hat = dy_val * bn_scale[idx];
        auto dx_val =
            dx_hat * inv_variance + dvariance * 2 * (x_val - mean) / dim_n + dmean / dim_n;
        if (beta_data_scaling != 0) {
            dx_val = alpha_data_scaling * dx_val +
                     beta_data_scaling * dx[n * dx_stride_n + c * dx_stride_c + d * dx_stride_d +
                                            h * dx_stride_h + w * dx_stride_w];
        }

        dx[n * dx_stride_n + c * dx_stride_c + d * dx_stride_d + h * dx_stride_h +
           w * dx_stride_w] = dx_val;
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
