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
__global__ void implCrossCorrelation2DKernel(const T* x,
                                             const T* w,
                                             const int in_w,
                                             const int in_h,
                                             const int in_c,
                                             const int kernel_h,
                                             const int kernel_w,
                                             const int pad_h,
                                             const int pad_w,
                                             const int stride_h,
                                             const int stride_w,
                                             const int dilation_h,
                                             const int dilation_w,
                                             const int out_w,
                                             const int out_h,
                                             const int out_c,
                                             const float alpha,
                                             const float beta,
                                             T* y) {
    int batch_id   =  blockIdx.x;
    int out_c_id   = (blockIdx.y << 4) + threadIdx.y;
    int out_h_id   = ((blockIdx.z << 4) + threadIdx.x) / out_w;
    int out_w_id   = ((blockIdx.z << 4) + threadIdx.x) - out_h_id * out_w;

    int kernel_size = kernel_w * kernel_h;

    const int out_stride_c = out_h * out_w;
    const int out_stride_n = out_stride_c * out_c;
    const int in_stride_c = in_h * in_w;
    const int in_stride_n = in_stride_c * in_c;

    __shared__ T knl_block[16][16];
    __shared__ T inp_block[16][16];

    T sum = 0;

    for (int idx = 0; idx < ((in_c * kernel_size + 15) >> 4); ++idx) {
        int in_h_init = out_h_id * stride_h - pad_h;
        int in_w_init = out_w_id * stride_w - pad_w;
        int in_c_id = ((idx << 4) + threadIdx.y) / kernel_size;
        int in_rem  = ((idx << 4) + threadIdx.y) - in_c_id * kernel_size;
        int knl_h_id = in_rem / kernel_w;
        int knl_w_id = in_rem - knl_h_id * kernel_w;
        int in_h_id  = in_h_init + knl_h_id * dilation_h;
        int in_w_id  = in_w_init + knl_w_id * dilation_w;

        if ((out_c_id < out_c) && (((idx << 4) + threadIdx.x) < in_c * kernel_size)) {
            int knl_idx = out_c_id * in_c * kernel_size + ((idx << 4) + threadIdx.x);
            knl_block[threadIdx.y][threadIdx.x] = w[knl_idx];
        } else {
            knl_block[threadIdx.y][threadIdx.x] = 0;
        }

        if (in_c_id < in_c && in_h_id < in_h && in_w_id < in_w && in_c_id >= 0 && in_h_id >= 0 && in_w_id >= 0) {
            int inp_idx = batch_id * in_stride_n + in_c_id * in_stride_c + in_h_id * in_w + in_w_id;
            inp_block[threadIdx.y][threadIdx.x] = x[inp_idx];
        } else {
            inp_block[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = 0; i < 16; ++i) {
            sum += knl_block[threadIdx.y][i] * inp_block[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (out_c_id < out_c && out_h_id < out_h && out_w_id < out_w) {
        int out_id = batch_id * out_stride_n + out_c_id * out_stride_c + out_h_id * out_w + out_w_id;
        y[out_id] = alpha * sum + beta * y[out_id];
    }
}

template <class T>
__global__ void implConvolution2DKernel(const T* x,
                                        const T* w,
                                        const int in_w,
                                        const int in_h,
                                        const int in_c,
                                        const int kernel_h,
                                        const int kernel_w,
                                        const int pad_h,
                                        const int pad_w,
                                        const int stride_h,
                                        const int stride_w,
                                        const int dilation_h,
                                        const int dilation_w,
                                        const int out_w,
                                        const int out_h,
                                        const int out_c,
                                        const float alpha,
                                        const float beta,
                                        T* y) {
    int batch_id   =  blockIdx.x;
    int out_c_id   = (blockIdx.y << 4) + threadIdx.y;
    int out_h_id   = ((blockIdx.z << 4) + threadIdx.x) / out_w;
    int out_w_id   = ((blockIdx.z << 4) + threadIdx.x) - out_h_id * out_w;

    int kernel_size = kernel_w * kernel_h;

    const int out_stride_c = out_h * out_w;
    const int out_stride_n = out_stride_c * out_c;
    const int in_stride_c = in_h * in_w;
    const int in_stride_n = in_stride_c * in_c;

    __shared__ T knl_block[16][16];
    __shared__ T inp_block[16][16];

    T sum = 0;

    for (int idx = 0; idx < ((in_c * kernel_size + 15) >> 4); ++idx) {
        int in_h_init = out_h_id * stride_h - pad_h;
        int in_w_init = out_w_id * stride_w - pad_w;
        int in_c_id = ((idx << 4) + threadIdx.y) / kernel_size;
        int in_rem  = ((idx << 4) + threadIdx.y) - in_c_id * kernel_size;
        int knl_h_id = in_rem / kernel_w;
        int knl_w_id = in_rem - knl_h_id * kernel_w;
        int in_h_id  = in_h_init + knl_h_id * dilation_h;
        int in_w_id  = in_w_init + knl_w_id * dilation_w;

        if ((out_c_id < out_c) && (((idx << 4) + threadIdx.x) < in_c * kernel_size)) {
            int n = ((idx << 4) + threadIdx.x) / kernel_size;
            int knl_idx = out_c_id * in_c * kernel_size + (2 * n + 1) * kernel_size - ((idx << 4) + threadIdx.x) - 1;
            knl_block[threadIdx.y][threadIdx.x] = w[knl_idx];
        } else {
            knl_block[threadIdx.y][threadIdx.x] = 0;
        }

        if (in_c_id < in_c && in_h_id < in_h && in_w_id < in_w && in_c_id >= 0 && in_h_id >= 0 && in_w_id >= 0) {
            int inp_idx = batch_id * in_stride_n + in_c_id * in_stride_c + in_h_id * in_w + in_w_id;
            inp_block[threadIdx.y][threadIdx.x] = x[inp_idx];
        } else {
            inp_block[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = 0; i < 16; ++i) {
            sum += knl_block[threadIdx.y][i] * inp_block[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (out_c_id < out_c && out_h_id < out_h && out_w_id < out_w) {
        int out_id = batch_id * out_stride_n + out_c_id * out_stride_c + out_h_id * out_w + out_w_id;
        y[out_id] = alpha * sum + beta * y[out_id];
    }
}

template <class T>
__global__ void implCrossCorrelation3DKernel(const T* x,
                                             const T* w,
                                             const int in_w,
                                             const int in_h,
                                             const int in_d,
                                             const int in_c,
                                             const int kernel_d,
                                             const int kernel_h,
                                             const int kernel_w,
                                             const int pad_d,
                                             const int pad_h,
                                             const int pad_w,
                                             const int stride_d,
                                             const int stride_h,
                                             const int stride_w,
                                             const int dilation_d,
                                             const int dilation_h,
                                             const int dilation_w,
                                             const int out_w,
                                             const int out_h,
                                             const int out_d,
                                             const int out_c,
                                             const float alpha,
                                             const float beta,
                                             T* y) {
    int batch_id   =  blockIdx.x;

    int temValue   = ((blockIdx.z << 4) + threadIdx.x);
    int out_c_id   = (blockIdx.y << 4) + threadIdx.y;
    int out_d_id   = temValue / (out_w * out_h);
    temValue      -= out_d_id * out_w * out_h;
    int out_h_id   = temValue / out_w;
    int out_w_id   = temValue - out_h_id * out_w;

    int kernel_size = kernel_w * kernel_h * kernel_d;

    const int out_stride_c = out_h * out_w * out_d;
    const int out_stride_n = out_stride_c * out_c;
    const int in_stride_c = in_h * in_w * in_d;
    const int in_stride_n = in_stride_c * in_c;

    __shared__ T knl_block[16][16];
    __shared__ T inp_block[16][16];

    T sum = 0;

    for (int idx = 0; idx < ((in_c * kernel_size + 15) >> 4); ++idx) {
        int in_h_init = out_h_id * stride_h - pad_h;
        int in_w_init = out_w_id * stride_w - pad_w;
        int in_d_init = out_d_id * stride_d - pad_d;
        int in_c_id = ((idx << 4) + threadIdx.y) / kernel_size;
        temValue  = ((idx << 4) + threadIdx.y) - in_c_id * kernel_size;
        int knl_d_id = temValue / (kernel_w * kernel_h);
        temValue -= knl_d_id * kernel_w * kernel_h;
        int knl_h_id = temValue / kernel_w;
        int knl_w_id = temValue - knl_h_id * kernel_w;
        int in_d_id  = in_d_init + knl_d_id * dilation_d;
        int in_h_id  = in_h_init + knl_h_id * dilation_h;
        int in_w_id  = in_w_init + knl_w_id * dilation_w;

        if ((out_c_id < out_c) && (((idx << 4) + threadIdx.x) < in_c * kernel_size)) {
            int knl_idx = out_c_id * in_c * kernel_size + ((idx << 4) + threadIdx.x);
            knl_block[threadIdx.y][threadIdx.x] = w[knl_idx];
        } else {
            knl_block[threadIdx.y][threadIdx.x] = 0;
        }

        if (in_d_id < in_d && in_c_id < in_c && in_h_id < in_h && in_w_id < in_w && in_c_id >= 0 && in_h_id >= 0 && in_w_id >= 0 && in_d_id >= 0) {
            int inp_idx = batch_id * in_stride_n + in_c_id * in_stride_c + in_d_id * in_w * in_h + in_h_id * in_w + in_w_id;
            inp_block[threadIdx.y][threadIdx.x] = x[inp_idx];
        } else {
            inp_block[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = 0; i < 16; ++i) {
            sum += knl_block[threadIdx.y][i] * inp_block[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (out_c_id < out_c && out_h_id < out_h && out_w_id < out_w && out_d_id < out_d) {
        int out_id = batch_id * out_stride_n + out_c_id * out_stride_c + out_d_id * out_w * out_h + out_h_id * out_w + out_w_id;
        y[out_id] = alpha * sum + beta * y[out_id];
    }
}

template <class T>
__global__ void implConvolution3DKernel(const T* x,
                                        const T* w,
                                        const int in_w,
                                        const int in_h,
                                        const int in_d,
                                        const int in_c,
                                        const int kernel_d,
                                        const int kernel_h,
                                        const int kernel_w,
                                        const int pad_d,
                                        const int pad_h,
                                        const int pad_w,
                                        const int stride_d,
                                        const int stride_h,
                                        const int stride_w,
                                        const int dilation_d,
                                        const int dilation_h,
                                        const int dilation_w,
                                        const int out_w,
                                        const int out_h,
                                        const int out_d,
                                        const int out_c,
                                        const float alpha,
                                        const float beta,
                                        T* y) {
    int batch_id   =  blockIdx.x;
    int temValue   = ((blockIdx.z << 4) + threadIdx.x);
    int out_c_id   = (blockIdx.y << 4) + threadIdx.y;
    int out_d_id   = temValue / (out_w * out_h);
    temValue      -= out_d_id * out_w * out_h;
    int out_h_id   = temValue / out_w;
    int out_w_id   = temValue - out_h_id * out_w;

    int kernel_size = kernel_w * kernel_h * kernel_d;

    const int out_stride_c = out_h * out_w * out_d;
    const int out_stride_n = out_stride_c * out_c;
    const int in_stride_c = in_h * in_w * in_d;
    const int in_stride_n = in_stride_c * in_c;

    __shared__ T knl_block[16][16];
    __shared__ T inp_block[16][16];

    T sum = 0;

    for (int idx = 0; idx < ((in_c * kernel_size + 15) >> 4); ++idx) {
        int in_h_init = out_h_id * stride_h - pad_h;
        int in_w_init = out_w_id * stride_w - pad_w;
        int in_d_init = out_d_id * stride_d - pad_d;
        int in_c_id = ((idx << 4) + threadIdx.y) / kernel_size;
        temValue  = ((idx << 4) + threadIdx.y) - in_c_id * kernel_size;
        int knl_d_id = temValue / (kernel_w * kernel_h);
        temValue -= knl_d_id * kernel_w * kernel_h;
        int knl_h_id = temValue / kernel_w;
        int knl_w_id = temValue - knl_h_id * kernel_w;

        int in_d_id  = in_d_init + knl_d_id * dilation_d;
        int in_h_id  = in_h_init + knl_h_id * dilation_h;
        int in_w_id  = in_w_init + knl_w_id * dilation_w;

        if ((out_c_id < out_c) && (((idx << 4) + threadIdx.x) < in_c * kernel_size)) {
            int n = ((idx << 4) + threadIdx.x) / kernel_size;
            int knl_idx = out_c_id * in_c * kernel_size + (2 * n + 1) * kernel_size - ((idx << 4) + threadIdx.x) - 1;
            knl_block[threadIdx.y][threadIdx.x] = w[knl_idx];
        } else {
            knl_block[threadIdx.y][threadIdx.x] = 0;
        }

        if (in_d_id < in_d && in_c_id < in_c && in_h_id < in_h && in_w_id < in_w && in_c_id >= 0 && in_h_id >= 0 && in_w_id >= 0 && in_d_id >= 0) {
            int inp_idx = batch_id * in_stride_n + in_c_id * in_stride_c + in_d_id * in_w * in_h + in_h_id * in_w + in_w_id;
            inp_block[threadIdx.y][threadIdx.x] = x[inp_idx];
        } else {
            inp_block[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int i = 0; i < 16; ++i) {
            sum += knl_block[threadIdx.y][i] * inp_block[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (out_c_id < out_c && out_h_id < out_h && out_w_id < out_w && out_d_id < out_d) {
        int out_id = batch_id * out_stride_n + out_c_id * out_stride_c + out_d_id * out_w * out_h + out_h_id * out_w + out_w_id;
        y[out_id] = alpha * sum + beta * y[out_id];
    }
}
}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
