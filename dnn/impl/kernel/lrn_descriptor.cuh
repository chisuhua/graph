/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <algorithm>

namespace cudnn {
namespace impl {
namespace kernel {

template <class T>
__global__ void CudnnLRNCrossChannelFwdKernel(const T* x,
                                              const float alpha,
                                              const float beta,
                                              T* y,
                                              const int lrn_n,
                                              const T lrn_alpha,
                                              const T lrn_beta,
                                              const T lrn_k,
                                              const int stride_h,
                                              const int stride_c,
                                              const int stride_n,
                                              const int in_h) {
    int c_id = blockIdx.y;
    int in_c = gridDim.y;
    int idx = 0;

    int max_c = min((in_c - 1), (c_id + (lrn_n >> 1)));
    int min_c = max(0, (c_id - ((lrn_n - 1) >> 1)));

    int thread_id = threadIdx.x + blockIdx.z * blockDim.x;

    if (thread_id < stride_h * in_h) {
        T sum_x = 0;

        int cur_pos = blockIdx.x * stride_n + c_id * stride_c + thread_id;

        for (idx = min_c; idx <= max_c; ++idx) {
            sum_x += x[cur_pos + (idx - c_id) * stride_c] * x[cur_pos + (idx - c_id) * stride_c];
        }

        // yj = lrn_k + lrn_alpha / n * sum(xj * xj)
        y[cur_pos] = x[cur_pos] * alpha * exp2f(-lrn_beta * log2f(lrn_k + lrn_alpha * sum_x /((T)lrn_n))) + beta * y[cur_pos];
    }
}

template <class T>
__global__ void CudnnLRNCrossChannelBwdKernel(const T* x,
                                              const float alpha,
                                              const float beta,
                                              const T* y,
                                              const T* dy,
                                              T* dx,
                                              const int lrn_n,
                                              T lrn_alpha,
                                              T lrn_beta,
                                              T lrn_k,
                                              const int stride_n,
                                              const int stride_c,
                                              const int stride_h,
                                              const int in_h,
                                              const int in_c) {
    extern __shared__ T shared_mem[];
    // total (4 * lrn_n - 1) * threads  store x value 2 * lrn_n - 1,  store dy value lrn_n, store scale_i value lrn_n,
    // store scale_i value lrn_n,  scale_i = lrn_k + lrn_alpha / n * sum(xj * xj)

    int ahead = (lrn_n - 1) >> 1;
    int tail  = lrn_n >> 1;
    int x_len = (lrn_n << 1) - 1;

    int c_id = blockIdx.y;

    int idx = 0;
    T ave_alpha = lrn_alpha / lrn_n;

    int thread_id = threadIdx.x + blockIdx.z * blockDim.x;
    if (thread_id < stride_h * in_h) {
        int chls_bias = blockIdx.x * stride_n + thread_id;
        int shared_mem_bias = ((lrn_n << 2) - 1) * threadIdx.x;

        T sum_s_s = 0;
        T dx_tmp;

        int j = 0;

        // store x value
        for (idx = c_id - (ahead << 1); idx < c_id + (tail << 1) + 1; ++idx, ++j) {
            if (idx < 0 || idx >= in_c) {
                shared_mem[shared_mem_bias + j] = 0;
            } else {
                shared_mem[shared_mem_bias + j] = x[chls_bias + idx * stride_c];
            }
        }

        j = 0;
        // store dy value and compute scale_i value
        // compute sum_s_s = dy * y * scale_i
        for (idx = c_id - ahead; idx < c_id + tail + 1; ++idx, ++j) {
            if (idx < 0 || idx >= in_c) {
                shared_mem[shared_mem_bias + x_len + j] = 0;
                shared_mem[shared_mem_bias + x_len + lrn_n + j] = 0;
            } else {
                T sum_s = 0;
                shared_mem[shared_mem_bias + x_len + j] = dy[chls_bias + idx * stride_c];
                for (int i = 0; i < lrn_n; ++i) {
                    sum_s += shared_mem[shared_mem_bias + j + i] * shared_mem[shared_mem_bias + j + i];
                }
                shared_mem[shared_mem_bias + x_len + lrn_n + j] = lrn_k + ave_alpha * sum_s;
                sum_s_s += shared_mem[shared_mem_bias + ahead + j] * exp2f(-(1 + lrn_beta) * log2f(shared_mem[shared_mem_bias + x_len + lrn_n + j])) * shared_mem[shared_mem_bias + x_len + j];
            }
        }
        // dx = (scale_i ^ -lrn_beta) * dy - 2 lrn_alpha * lrn_beta * xi * sum_s_s / lrn_n
        dx_tmp = shared_mem[shared_mem_bias + x_len + ahead] * exp2f(-lrn_beta * log2f(shared_mem[shared_mem_bias + x_len + lrn_n + ahead])) - 2 * lrn_beta * ave_alpha * shared_mem[(shared_mem_bias + (ahead << 1))] * sum_s_s;

        dx[chls_bias + c_id * stride_c] = alpha * dx_tmp + beta * dx[chls_bias + c_id * stride_c];
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
