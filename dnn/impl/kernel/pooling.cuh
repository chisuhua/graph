/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn/impl/kernel/cudnn_math_functions.cuh>
#include <cudnn/impl/kernel/cudnn_utility.cuh>
#include <cudnn/impl/meta/cudnn_meta_pooling.h>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>

#include <algorithm>

namespace cudnn {
namespace impl {
namespace kernel {

constexpr int kPoolingBackwardScalingTileSize = 8;

template <class T, class ScalingT>
__global__ void PoolingForward4D(cudnnPoolingMode_t mode,
                                 cudnnNanPropagation_t nan_opt,
                                 int dim_h,
                                 int dim_w,
                                 int padding_h,
                                 int padding_w,
                                 int stride_h,
                                 int stride_w,
                                 ScalingT alpha,
                                 ScalingT beta,
                                 int x_dim_h,
                                 int x_dim_w,
                                 int x_stride_n,
                                 int x_stride_c,
                                 int x_stride_h,
                                 int x_stride_w,
                                 const T* x,
                                 int y_dim_h,
                                 int y_dim_w,
                                 int y_stride_n,
                                 int y_stride_c,
                                 int y_stride_h,
                                 int y_stride_w,
                                 T* y) {
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) / y_dim_w;
    const int w = (blockDim.x * blockIdx.x + threadIdx.x) % y_dim_w;

    if (w >= y_dim_w || h >= y_dim_h) {
        return;
    }

    const int xbegin_h = stride_h * h - padding_h;
    const int xbegin_w = stride_w * w - padding_w;
    const int h_start  = xbegin_h < 0 ? 0 : xbegin_h;
    const int h_end    = min(xbegin_h + dim_h, x_dim_h);
    const int w_start  = xbegin_w < 0 ? 0 : xbegin_w;
    const int w_end    = min(xbegin_w + dim_w, x_dim_w);

    // skip the situation such as: padding >= window size
    if (h_end < h_start || w_end < w_start) {
        return;
    }

    T sum     = static_cast<T>(0);
    T max_val = x[n * x_stride_n + c * x_stride_c + h_start * x_stride_h + w_start * x_stride_w];

    int pooling_size;
    if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
        pooling_size = dim_h * dim_w;
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
        pooling_size = (w_end - w_start) * (h_end - h_start);
    }
    for (int idx_h = h_start; idx_h < h_end; ++idx_h) {
        for (int idx_w = w_start; idx_w < w_end; ++idx_w) {
            T current =
                x[n * x_stride_n + c * x_stride_c + idx_h * x_stride_h + idx_w * x_stride_w];
            max_val = fmax(max_val, current);  // FIXME(Peter Han): proper max
            sum += current / pooling_size;
        }
    }

    T new_val;
    if (mode == DNN_POOLING_MAX) {
        new_val = max_val;
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
        new_val = sum;
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
        new_val = sum;
    }

    auto tmp = new_val;
    tmp = alpha * tmp + beta * y[n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w];
    if (nan_opt == DNN_NOT_PROPAGATE_NAN) {
        if (isnan(tmp)) {
            tmp = static_cast<T>(0);
        }
    }
    y[n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w] = tmp;
}

template <class T, class ScalingT>
__global__ void PoolingForward5D(cudnnPoolingMode_t mode,
                                 cudnnNanPropagation_t nan_opt,
                                 int dim_d,
                                 int dim_h,
                                 int dim_w,
                                 int padding_d,
                                 int padding_h,
                                 int padding_w,
                                 int stride_d,
                                 int stride_h,
                                 int stride_w,
                                 ScalingT alpha,
                                 ScalingT beta,
                                 int x_dim_d,
                                 int x_dim_h,
                                 int x_dim_w,
                                 int x_stride_n,
                                 int x_stride_c,
                                 int x_stride_d,
                                 int x_stride_h,
                                 int x_stride_w,
                                 const T* x,
                                 int y_dim_d,
                                 int y_dim_h,
                                 int y_dim_w,
                                 int y_stride_n,
                                 int y_stride_c,
                                 int y_stride_d,
                                 int y_stride_h,
                                 int y_stride_w,
                                 T* y) {
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int d = (blockDim.x * blockIdx.x + threadIdx.x) / (y_dim_h * y_dim_w);
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) % (y_dim_h * y_dim_w) / y_dim_w;
    const int w = (blockDim.x * blockIdx.x + threadIdx.x) % y_dim_w;

    if (w >= y_dim_w || h >= y_dim_h || d >= y_dim_d) {
        return;
    }

    const int xbegin_d = stride_d * d - padding_d;
    const int xbegin_h = stride_h * h - padding_h;
    const int xbegin_w = stride_w * w - padding_w;
    const int d_start  = xbegin_d < 0 ? 0 : xbegin_d;
    const int d_end    = min(xbegin_d + dim_d, x_dim_d);
    const int h_start  = xbegin_h < 0 ? 0 : xbegin_h;
    const int h_end    = min(xbegin_h + dim_h, x_dim_h);
    const int w_start  = xbegin_w < 0 ? 0 : xbegin_w;
    const int w_end    = min(xbegin_w + dim_w, x_dim_w);

    // skip the situation such as: padding >= window size
    if (h_end < h_start || w_end < w_start || d_end < d_start) {
        return;
    }

    T sum     = static_cast<T>(0);
    T max_val = x[n * x_stride_n + c * x_stride_c + d_start * x_stride_d + h_start * x_stride_h +
                  w_start * x_stride_w];
    int pooling_size;
    if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
        pooling_size = dim_d * dim_h * dim_w;
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
        pooling_size = (w_end - w_start) * (h_end - h_start) * (d_end - d_start);
    }
    for (int idx_d = d_start; idx_d < d_end; ++idx_d) {
        for (int idx_h = h_start; idx_h < h_end; ++idx_h) {
            for (int idx_w = w_start; idx_w < w_end; ++idx_w) {
                T current = x[n * x_stride_n + c * x_stride_c + idx_d * x_stride_d +
                              idx_h * x_stride_h + idx_w * x_stride_w];
                max_val   = fmax(max_val, current);  // FIXME(Peter Han): proper max
                sum += current / pooling_size;
            }
        }
    }

    T new_val;
    if (mode == DNN_POOLING_MAX) {
        new_val = max_val;
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
        new_val = sum;
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
        new_val = sum;
    }

    auto tmp = new_val;
    tmp      = alpha * tmp +
          beta *
              y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + w * y_stride_w];
    if (nan_opt == DNN_NOT_PROPAGATE_NAN) {
        if (isnan(tmp)) {
            tmp = static_cast<T>(0);
        }
    }
    y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + w * y_stride_w] = tmp;
}

template <class T, class ScalingT>
__global__ void PoolingBackward4D(cudnnPoolingMode_t mode,
                                  cudnnNanPropagation_t nan_opt,
                                  int dim_h,
                                  int dim_w,
                                  int padding_h,
                                  int padding_w,
                                  int stride_h,
                                  int stride_w,
                                  ScalingT alpha,
                                  int y_dim_h,
                                  int y_dim_w,
                                  int y_stride_n,
                                  int y_stride_c,
                                  int y_stride_h,
                                  int y_stride_w,
                                  const T* y,
                                  const T* dy,
                                  int x_dim_h,
                                  int x_dim_w,
                                  int x_stride_n,
                                  int x_stride_c,
                                  int x_stride_h,
                                  int x_stride_w,
                                  const T* x,
                                  T* dx) {
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) / y_dim_w;
    const int w = (blockDim.x * blockIdx.x + threadIdx.x) % y_dim_w;

    if (w >= y_dim_w || h >= y_dim_h) {
        return;
    }
    const int xbegin_h = stride_h * h - padding_h;
    const int xbegin_w = stride_w * w - padding_w;
    const int h_start  = xbegin_h < 0 ? 0 : xbegin_h;
    const int h_end    = min(xbegin_h + dim_h, x_dim_h);
    const int w_start  = xbegin_w < 0 ? 0 : xbegin_w;
    const int w_end    = min(xbegin_w + dim_h, x_dim_w);

    // skip the situation such as: padding >= window size
    if (h_end < h_start || w_end < w_start) {
        return;
    }

    if (mode == DNN_POOLING_MAX) {
        int max_h = h_start;
        int max_w = w_start;
        T max_val = x[n * x_stride_n + c * x_stride_c + max_h * x_stride_h + max_w * x_stride_w];
        for (int idx_h = h_start; idx_h < h_end; ++idx_h) {
            for (int idx_w = w_start; idx_w < w_end; ++idx_w) {
                T current =
                    x[n * x_stride_n + c * x_stride_c + idx_h * x_stride_h + idx_w * x_stride_w];
                if (current > max_val) {
                    max_val = current;
                    max_h   = idx_h;
                    max_w   = idx_w;
                }
            }
        }

        auto tmp = dy[n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w] * alpha;
        if (nan_opt == DNN_NOT_PROPAGATE_NAN) {
            if (isnan(tmp)) {
                tmp = static_cast<T>(0);
            }
        }
        atomicAdd(&dx[n * x_stride_n + c * x_stride_c + max_h * x_stride_h + max_w * x_stride_w],
                  tmp);
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ||
               mode == DNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
        int pooling_size;
        if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
            pooling_size = dim_h * dim_w;
        } else {
            pooling_size = (w_end - w_start) * (h_end - h_start);
        }
        T tmp = dy[n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w] /
                pooling_size * alpha;
        if (nan_opt == DNN_NOT_PROPAGATE_NAN) {
            if (isnan(tmp)) {
                tmp = static_cast<T>(0);
            }
        }
        for (int idx_h = h_start; idx_h < h_end; ++idx_h) {
            for (int idx_w = w_start; idx_w < w_end; ++idx_w) {
                atomicAdd(
                    &dx[n * x_stride_n + c * x_stride_c + idx_h * x_stride_h + idx_w * x_stride_w],
                    tmp);
            }
        }
    }
}

template <class T, class ScalingT>
__global__ void PoolingBackward5D(cudnnPoolingMode_t mode,
                                  cudnnNanPropagation_t nan_opt,
                                  int dim_d,
                                  int dim_h,
                                  int dim_w,
                                  int padding_d,
                                  int padding_h,
                                  int padding_w,
                                  int stride_d,
                                  int stride_h,
                                  int stride_w,
                                  ScalingT alpha,
                                  int y_dim_d,
                                  int y_dim_h,
                                  int y_dim_w,
                                  int y_stride_n,
                                  int y_stride_c,
                                  int y_stride_d,
                                  int y_stride_h,
                                  int y_stride_w,
                                  const T* y,
                                  const T* dy,
                                  int x_dim_d,
                                  int x_dim_h,
                                  int x_dim_w,
                                  int x_stride_n,
                                  int x_stride_c,
                                  int x_stride_d,
                                  int x_stride_h,
                                  int x_stride_w,
                                  const T* x,
                                  T* dx) {
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int d = (blockDim.x * blockIdx.x + threadIdx.x) / (y_dim_h * y_dim_w);
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) % (y_dim_h * y_dim_w) / y_dim_w;
    const int w = (blockDim.x * blockIdx.x + threadIdx.x) % y_dim_w;

    if (w >= y_dim_w || h >= y_dim_h || d >= y_dim_d) {
        return;
    }

    const int xbegin_d = stride_d * d - padding_d;
    const int xbegin_h = stride_h * h - padding_h;
    const int xbegin_w = stride_w * w - padding_w;
    const int d_start  = xbegin_d < 0 ? 0 : xbegin_d;
    const int d_end    = min(xbegin_d + dim_d, x_dim_d);
    const int h_start  = xbegin_h < 0 ? 0 : xbegin_h;
    const int h_end    = min(xbegin_h + dim_h, x_dim_h);
    const int w_start  = xbegin_w < 0 ? 0 : xbegin_w;
    const int w_end    = min(xbegin_w + dim_w, x_dim_w);

    // skip the situation such as: padding >= window size
    if (h_end < h_start || w_end < w_start || d_end < d_start) {
        return;
    }

    if (mode == DNN_POOLING_MAX) {
        int max_d = d_start;
        int max_h = h_start;
        int max_w = w_start;
        T max_val = x[n * x_stride_n + c * x_stride_c + max_d * x_stride_d + max_h * x_stride_h +
                      max_w * x_stride_w];
        for (int idx_d = d_start; idx_d < d_end; ++idx_d) {
            for (int idx_h = h_start; idx_h < h_end; ++idx_h) {
                for (int idx_w = w_start; idx_w < w_end; ++idx_w) {
                    T current = x[n * x_stride_n + c * x_stride_c + idx_d * x_stride_d +
                                  idx_h * x_stride_h + idx_w * x_stride_w];
                    if (current > max_val) {
                        max_val = current;
                        max_d   = idx_d;
                        max_h   = idx_h;
                        max_w   = idx_w;
                    }
                }
            }
        }
        auto tmp =
            dy[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + w * y_stride_w] *
            alpha;
        if (nan_opt == DNN_NOT_PROPAGATE_NAN) {
            if (isnan(tmp)) {
                tmp = static_cast<T>(0);
            }
        }
        atomicAdd(&dx[n * x_stride_n + c * x_stride_c + max_d * x_stride_d + max_h * x_stride_h +
                      max_w * x_stride_w],
                  tmp);
    } else if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ||
               mode == DNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
        int pooling_size;
        if (mode == DNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
            pooling_size = dim_d * dim_h * dim_w;
        } else {
            pooling_size = (w_end - w_start) * (h_end - h_start) * (d_end - d_start);
        }

        T tmp =
            dy[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + w * y_stride_w] /
            pooling_size * alpha;
        if (nan_opt == DNN_NOT_PROPAGATE_NAN) {
            if (isnan(tmp)) {
                tmp = static_cast<T>(0);
            }
        }
        for (int idx_d = d_start; idx_d < d_end; ++idx_d) {
            for (int idx_h = h_start; idx_h < h_end; ++idx_h) {
                for (int idx_w = w_start; idx_w < w_end; ++idx_w) {
                    atomicAdd(&dx[n * x_stride_n + c * x_stride_c + idx_d * x_stride_d +
                                  idx_h * x_stride_h + idx_w * x_stride_w],
                              tmp);
                }
            }
        }
    }
}

template <class T, class ScalingT>
__global__ void PoolingBackwardScaling(ScalingT beta,
                                       int dim_n,
                                       int dim_c,
                                       int dim_d, /* 1 for 4d tensor */
                                       int dim_h,
                                       int dim_w,
                                       int stride_n,
                                       int stride_c,
                                       int stride_d, /* stride_h * stride_w for 4d tensor */
                                       int stride_h,
                                       int stride_w,
                                       T* dx) {
    auto global_tid_base =
        (blockDim.x * blockIdx.x + threadIdx.x) * kPoolingBackwardScalingTileSize;

#pragma unroll kPoolingBackwardScalingTileSize
    for (int i = 0; i < kPoolingBackwardScalingTileSize; ++i) {
        auto global_tid = global_tid_base + i;
        if (global_tid >= dim_n * dim_c * dim_d * dim_h * dim_w) {
            return;
        }

        const auto n = global_tid / (dim_c * dim_d * dim_h * dim_w);
        const auto c = global_tid % (dim_c * dim_d * dim_h * dim_w) / (dim_d * dim_h * dim_w);
        const auto d = global_tid % (dim_d * dim_h * dim_w) / (dim_h * dim_w);
        const auto h = global_tid % (dim_h * dim_w) / dim_w;
        const auto w = global_tid % dim_w;

        const auto idx = n * stride_n + c * stride_c + d * stride_d + h * stride_h + w * stride_w;
        dx[idx]        = dx[idx] * beta;
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
