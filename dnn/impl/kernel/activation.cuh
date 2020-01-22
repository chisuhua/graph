/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn/impl/kernel/cudnn_intrinsic.h>
#include <cudnn/impl/kernel/cudnn_math_functions.cuh>
#include <cudnn/impl/meta/cudnn_meta_activation.h>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>

namespace {

using cudnn::impl::kernel::CuExp;
using cudnn::impl::kernel::CuFmax;
using cudnn::impl::kernel::CuFmin;
using cudnn::impl::meta::CuMetaActivation;
using cudnn::impl::meta::CuMetaTensor;

template <class T>
__forceinline__ __device__ void forward_op_0(T x, float coef, T& y) {
    T one = static_cast<T>(1);
    y     = one / (one + CuExp(-x)); /** NO need to worry about overflow */
}

template <class T>
__forceinline__ __device__ void forward_op_1(T x, float coef, T& y) {
    y = x * (x > static_cast<T>(0));
}

template <class T>
__forceinline__ __device__ void forward_op_2(T x, float coef, T& y) {
    // TODO(Peter Han): use CuTanh to overload for different types
    y = tanhf(x);
}

template <class T>
__forceinline__ __device__ void forward_op_3(T x, float coef, T& y) {
    T zero = static_cast<T>(0);
    y      = CuFmin(static_cast<T>(coef), CuFmax(x, zero));
}

template <class T>
__forceinline__ __device__ void forward_op_4(T x, float alpha, T& y) {
    T zero = static_cast<T>(0);
    T one  = static_cast<T>(1);
    y      = x > zero ? x : (static_cast<T>(alpha) * (CuExp(x) - one));
}

template <class T>
__forceinline__ __device__ void forward_op_5(T x, float coef, T& y) {
    y = x;
}

template <class T>
__forceinline__ __device__ void
backward_op_0(const T* /*x*/, const T* y, const T* dy, float /*coef*/, T& dx) {
    T y_val = *y;
    dx      = y_val * (static_cast<T>(1) - y_val) * (*dy);
}

template <class T>
__forceinline__ __device__ void
backward_op_1(const T* x, const T* /*y*/, const T* dy, float /*coef*/, T& dx) {
    dx = (*dy) * ((*x) > static_cast<T>(0));
}

template <class T>
__forceinline__ __device__ void
backward_op_2(const T* /*x*/, const T* y, const T* dy, float /*coef*/, T& dx) {
    T y_val = *y;
    dx      = (static_cast<T>(1) - y_val * y_val) * (*dy);
}

template <class T>
__forceinline__ __device__ void
backward_op_3(const T* x, const T* /*y*/, const T* dy, float coef, T& dx) {
    T zero  = static_cast<T>(0);
    T x_val = *x;
    dx      = (x_val > zero && x_val <= coef) ? (*dy) : zero;
}

template <class T>
__forceinline__ __device__ void
backward_op_4(const T* x, const T* y, const T* dy, float alpha, T& dx) {
    T zero = static_cast<T>(0);
    T one  = static_cast<T>(1);
    dx     = (*dy) * (((*x) > zero) ? one : ((*y) + static_cast<T>(alpha)));
}

template <class T>
__forceinline__ __device__ void
backward_op_5(const T* /*x*/, const T* /*y*/, const T* dy, float /*alpha*/, T& dx) {
    dx = *dy;
}

}  // namespace

namespace cudnn {
namespace impl {
namespace kernel {

constexpr int kActivationTileSize = 8;

#define ACTIVATION(mode)                                                                      \
    template <class T, class ScalingT>                                                        \
    __global__ void Activation##mode(cudnnNanPropagation_t nan_opt,                           \
                                     float coef,                                              \
                                     ScalingT alpha,                                          \
                                     ScalingT beta,                                           \
                                     int dim_n,                                               \
                                     int dim_c,                                               \
                                     int dim_d,                                               \
                                     int dim_h,                                               \
                                     int dim_w,                                               \
                                     int x_stride_n,                                          \
                                     int x_stride_c,                                          \
                                     int x_stride_d,                                          \
                                     int x_stride_h,                                          \
                                     int x_stride_w,                                          \
                                     const T* x,                                              \
                                     int y_stride_n,                                          \
                                     int y_stride_c,                                          \
                                     int y_stride_d,                                          \
                                     int y_stride_h,                                          \
                                     int y_stride_w,                                          \
                                     T* y) {                                                  \
        const auto base_idx = (blockDim.x * blockIdx.x + threadIdx.x) * kActivationTileSize;  \
                                                                                              \
        for (int i = 0; i < kActivationTileSize; ++i) {                                       \
            auto idx = base_idx + i;                                                          \
            if (idx > dim_n * dim_c * dim_d * dim_h * dim_w - 1) {                            \
                return;                                                                       \
            }                                                                                 \
            const auto n = idx / (dim_c * dim_d * dim_h * dim_w);                             \
            const auto c = idx % (dim_c * dim_d * dim_h * dim_w) / (dim_d * dim_h * dim_w);   \
            const auto d = idx % (dim_d * dim_h * dim_w) / (dim_h * dim_w);                   \
            const auto h = idx % (dim_h * dim_w) / dim_w;                                     \
            const auto w = idx % dim_w;                                                       \
                                                                                              \
            T tmp_y;                                                                          \
            forward_op_##mode<T>(x[n * x_stride_n + c * x_stride_c + d * x_stride_d +         \
                                   h * x_stride_h + w * x_stride_w],                          \
                                 coef,                                                        \
                                 tmp_y);                                                      \
                                                                                              \
            T y_prior = y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h + \
                          w * y_stride_w];                                                    \
            tmp_y     = alpha * tmp_y + beta * y_prior;                                       \
            if (nan_opt == DNN_NOT_PROPAGATE_NAN) {                                         \
                if (isnan(tmp_y)) {                                                           \
                    tmp_y = static_cast<T>(0);                                                \
                }                                                                             \
            }                                                                                 \
            y[n * y_stride_n + c * y_stride_c + d * y_stride_d + h * y_stride_h +             \
              w * y_stride_w] = tmp_y;                                                        \
        }                                                                                     \
    }

#define ACTIVATION_DIFF(mode)                                                                   \
    template <class T, class ScalingT>                                                          \
    __global__ void ActivationDiff##mode(cudnnNanPropagation_t nan_opt,                         \
                                         float coef,                                            \
                                         ScalingT alpha,                                        \
                                         ScalingT beta,                                         \
                                         int dim_n,                                             \
                                         int dim_c,                                             \
                                         int dim_d,                                             \
                                         int dim_h,                                             \
                                         int dim_w,                                             \
                                         int y_stride_n,                                        \
                                         int y_stride_c,                                        \
                                         int y_stride_d,                                        \
                                         int y_stride_h,                                        \
                                         int y_stride_w,                                        \
                                         const T* y,                                            \
                                         const T* dy,                                           \
                                         int x_stride_n,                                        \
                                         int x_stride_c,                                        \
                                         int x_stride_d,                                        \
                                         int x_stride_h,                                        \
                                         int x_stride_w,                                        \
                                         const T* x,                                            \
                                         T* dx) {                                               \
        const auto base_idx = (blockDim.x * blockIdx.x + threadIdx.x) * kActivationTileSize;    \
                                                                                                \
        for (int i = 0; i < kActivationTileSize; ++i) {                                         \
            auto idx = base_idx + i;                                                            \
            if (idx > dim_n * dim_c * dim_d * dim_h * dim_w - 1) {                              \
                return;                                                                         \
            }                                                                                   \
            const auto n = idx / (dim_c * dim_d * dim_h * dim_w);                               \
            const auto c = idx % (dim_c * dim_d * dim_h * dim_w) / (dim_d * dim_h * dim_w);     \
            const auto d = idx % (dim_d * dim_h * dim_w) / (dim_h * dim_w);                     \
            const auto h = idx % (dim_h * dim_w) / dim_w;                                       \
            const auto w = idx % dim_w;                                                         \
                                                                                                \
            T tmp_dx;                                                                           \
            backward_op_##mode<T>(&x[n * x_stride_n + c * x_stride_c + d * x_stride_d +         \
                                     h * x_stride_h + w * x_stride_w],                          \
                                  &y[n * y_stride_n + c * y_stride_c + d * y_stride_d +         \
                                     h * y_stride_h + w * y_stride_w],                          \
                                  &dy[n * y_stride_n + c * y_stride_c + d * y_stride_d +        \
                                      h * y_stride_h + w * y_stride_w],                         \
                                  coef,                                                         \
                                  tmp_dx);                                                      \
                                                                                                \
            T dx_prior = dx[n * x_stride_n + c * x_stride_c + d * x_stride_d + h * x_stride_h + \
                            w * x_stride_w];                                                    \
            tmp_dx     = alpha * tmp_dx + beta * dx_prior;                                      \
            if (nan_opt == DNN_NOT_PROPAGATE_NAN) {                                           \
                if (isnan(tmp_dx)) {                                                            \
                    tmp_dx = static_cast<T>(0);                                                 \
                }                                                                               \
            }                                                                                   \
            dx[n * x_stride_n + c * x_stride_c + d * x_stride_d + h * x_stride_h +              \
               w * x_stride_w] = tmp_dx;                                                        \
        }                                                                                       \
    }

ACTIVATION(0);
ACTIVATION(1);
ACTIVATION(2);
ACTIVATION(3);
ACTIVATION(4);
ACTIVATION(5);

ACTIVATION_DIFF(0);
ACTIVATION_DIFF(1);
ACTIVATION_DIFF(2);
ACTIVATION_DIFF(3);
ACTIVATION_DIFF(4);
ACTIVATION_DIFF(5);

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
