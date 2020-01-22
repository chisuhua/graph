/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <algorithm>
#include <cfloat>
namespace cudnn {
namespace impl {
namespace kernel {

// support nchw ,nhwc, ncdhw format
template <class T>
__global__ void CudnnSoftMaxModeChlFwdKernel(const T* x,
                                             const T alpha,
                                             const T beta,
                                             T* y,
                                             const int chls,
                                             const int featuremap_size,
                                             const int n_stride,
                                             const int c_stride,
                                             const int w_stride) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int n_id       = thread_id / featuremap_size;
    int feature_id = thread_id % featuremap_size;

    int data_id = 0;

    T max_in_chl = -FLT_MAX;
    T sum_in_chl = 0;

    for (int c_id = 0; c_id < chls; c_id++) {
        if (c_stride >= featuremap_size) {
            // nchw or ncdhw format
            data_id = n_id * n_stride + c_id * c_stride + feature_id;
        } else {
            // nhwc format
            data_id = n_id * n_stride + feature_id * w_stride + c_id;
        }
        max_in_chl = max(max_in_chl, x[data_id]);
    }

    for (int c_id = 0; c_id < chls; c_id++) {
        if (c_stride >= featuremap_size) {
            // nchw or ncdhw format
            data_id = n_id * n_stride + c_id * c_stride + feature_id;
        } else {
            // nhwc format
            data_id = n_id * n_stride + feature_id * w_stride + c_id;
        }
        y[data_id] = exp(x[data_id] - max_in_chl);
        sum_in_chl += y[data_id];
    }

    for (int c_id = 0; c_id < chls; c_id++) {
        if (c_stride >= featuremap_size) {
            // nchw or ncdhw format
            data_id = n_id * n_stride + c_id * c_stride + feature_id;
        } else {
            // nhwc format
            data_id = n_id * n_stride + feature_id * w_stride + c_id;
        }
        y[data_id] = y[data_id] / sum_in_chl;
    }
}

inline __device__ float float_sum(float a, float b) {
    float s       = a + b;
    float a_p     = s - b;
    float b_p     = s - a_p;
    float delta_a = a - a_p;
    float delta_b = b - b_p;
    float r       = delta_a + delta_b;
    float result  = s + r;
    return result;
}

// per channel
// SOFTMAX_ACCURATE and SOFTMAX_FAST will use this one
template <typename T1, typename T2>
__global__ void CudnnSoftmaxModeChlBwdKernel(const T1* dy,
                                             const T1* y,
                                             T1* dx,
                                             const int grid_size /*nhw*/,
                                             const int spatial_dim /*hw*/,
                                             const int channel,
                                             const T2 alpha,
                                             const T2 beta,
                                             int softmax_algo) {
    __shared__ float mid_result[256];

    int tid = threadIdx.x;
    // in one loop each threadblock will work on one channel
    for (int gid = blockIdx.x; gid < grid_size; gid += gridDim.x) {
        int n = gid / spatial_dim;
        int s = gid % spatial_dim;

        float channel_dot = 0.f;

        for (int i = tid; i < channel; i += blockDim.x) {
            float value;
            if (softmax_algo == 1) {
                value = dy[(n * channel + i) * spatial_dim + s] *
                        y[(n * channel + i) * spatial_dim + s];
            } else if (softmax_algo == 2) {
                value = dy[(n * channel + i) * spatial_dim + s] *
                        expf(y[(n * channel + i) * spatial_dim + s]);
            }

            channel_dot = float_sum(channel_dot, value);
        }

        mid_result[tid] = channel_dot;

        __syncthreads();
        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (tid < i) {
                mid_result[tid] += mid_result[tid + i];
            }
            __syncthreads();
        }
        channel_dot = mid_result[0];

        for (int i = tid; i < channel; i += blockDim.x) {
            if (softmax_algo == 1) {
                dx[(n * channel + i) * spatial_dim + s] =
                    y[(n * channel + i) * spatial_dim + s] *
                    (dy[(n * channel + i) * spatial_dim + s] - channel_dot);
            } else if (softmax_algo == 2) {
                dx[(n * channel + i) * spatial_dim + s] =
                    dy[(n * channel + i) * spatial_dim + s] - channel_dot;
            }
        }
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
