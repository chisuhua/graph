/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

namespace cudnn {
namespace impl {
namespace kernel {

template <typename T1, typename T2>
__global__ void
CudnnAddTensorEQDimKernel(const T1* A, const T2 alpha, const T2 beta, T1* C, const int data_size) {
    int data_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (data_id < data_size) {
        C[data_id] = alpha * A[data_id] + beta * C[data_id];
    }
}

template <typename T1, typename T2>
__global__ void CudnnAddTensorNotEQDimKernel(const T1* a,
                                             const int a_chls_stride,
                                             const int a_feature_size,
                                             const T2 alpha,
                                             const T2 beta,
                                             T1* c,
                                             const int c_chls,
                                             const int c_chls_stride,
                                             const int c_image_size) {
    int data_id    = threadIdx.x + blockIdx.x * blockDim.x;
    int c_batch_id = blockIdx.y;

    if (data_id < blockDim.x*gridDim.x) {
        if (a_feature_size == 1) {
            for (int chls_id = 0; chls_id < c_chls; chls_id++) {
                c[c_batch_id * c_image_size + chls_id * c_chls_stride + data_id] =
                alpha*a[chls_id * a_chls_stride]+beta*c[c_batch_id * c_image_size + chls_id * c_chls_stride + data_id];
            }
        } else {
            for (int chls_id = 0; chls_id < c_chls; chls_id++) {
                c[c_batch_id * c_image_size + chls_id * c_chls_stride + data_id] =
                alpha*a[chls_id * a_chls_stride + data_id]+beta*c[c_batch_id * c_image_size + chls_id * c_chls_stride + data_id];
            }
        }
    }
}

template <class T>
__global__ void CudnnTransformTensorKernel(const T* x,
                                           const float alpha,
                                           const float beta,
                                           T* y,
                                           const int y_Stride_n,
                                           const int y_Stride_c,
                                           const int y_Stride_h,
                                           const int y_Stride_w,
                                           const int x_Stride_n,
                                           const int x_Stride_c,
                                           const int x_Stride_h,
                                           const int x_Stride_w,
                                           const int x_w,
                                           const int x_h) {
    int xy_threads = threadIdx.x + blockIdx.z * blockDim.x;
    int data_id_n  = blockIdx.x;
    int data_id_c  = blockIdx.y;
    int data_id_h  = xy_threads / x_w;
    int data_id_w  = xy_threads - data_id_h * x_w;

    int xpos = data_id_n * x_Stride_n + data_id_c * x_Stride_c + data_id_h * x_Stride_h +
               data_id_w * x_Stride_w;
    int ypos = data_id_n * y_Stride_n + data_id_c * y_Stride_c + data_id_h * y_Stride_h +
               data_id_w * y_Stride_w;

    if (data_id_h < x_h && data_id_w < x_w) {
        y[ypos] = alpha * x[xpos] + beta * y[ypos];
    }
}

template <class T>
__global__ void CudnnTransformScaToVecKernel(const T* x,
                                             const float alpha,
                                             const float beta,
                                             T* y,
                                             const int stride_n,
                                             const int stride_c,
                                             const int stride_w,
                                             const int quad_hw) {
    int n_id      = blockIdx.x;
    int n_bias    = n_id * stride_n;
    int thread_id = threadIdx.x + blockIdx.y * blockDim.x;

    if (thread_id < stride_n) {
        int vec_c_id  = thread_id / quad_hw;
        int rem_vec_c = thread_id - vec_c_id * quad_hw;
        int sca_w_id  = rem_vec_c >> 2;
        int sca_c_id  = rem_vec_c - (sca_w_id << 2);

        int y_pos = n_bias + thread_id;
        int x_pos = n_bias + ((vec_c_id << 2) + sca_c_id) * stride_c + sca_w_id * stride_w;

        y[y_pos] = alpha * x[x_pos] + beta * y[y_pos];
    }
}

template <class T>
__global__ void CudnnTransformVecToScaKernel(const T* x,
                                             const float alpha,
                                             const float beta,
                                             T* y,
                                             const int stride_n,
                                             const int stride_c,
                                             const int stride_w,
                                             const int quad_hw) {
    int n_id      = blockIdx.x;
    int n_bias    = n_id * stride_n;
    int thread_id = threadIdx.x + blockIdx.y * blockDim.x;

    if (thread_id < stride_n) {
        int vec_c_id  = thread_id / quad_hw;
        int rem_vec_c = thread_id - vec_c_id * quad_hw;
        int sca_w_id  = rem_vec_c >> 2;
        int sca_c_id  = rem_vec_c - (sca_w_id << 2);

        int x_pos = n_bias + thread_id;
        int y_pos = n_bias + ((vec_c_id << 2) + sca_c_id) * stride_c + sca_w_id * stride_w;

        y[y_pos] = alpha * x[x_pos] + beta * y[y_pos];
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
