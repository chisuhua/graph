/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <algorithm>

namespace cudnn {
namespace impl {
namespace kernel {

/**
 * This kernel only support zero padding
 * Img2col ,the format of col matrix is (height_col*width_col)*(channel*kernel_h*kernel_w)
 */
template <class T>
__global__ void Img2colKernel(const int n,
                              const T* data_im,
                              const int height,
                              const int width,
                              const int kernel_h,
                              const int kernel_w,
                              const int pad_h,
                              const int pad_w,
                              const int stride_h,
                              const int stride_w,
                              const int dilation_h,
                              const int dilation_w,
                              const int height_col,
                              const int width_col,
                              const int input_len,
                              T* data_col) {
    const int batchid               = blockIdx.y;
    const int batch_size            = gridDim.y;
    const int batch_col_matrix_size = height_col * width_col * batch_size;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
         index += blockDim.x * gridDim.x) {
        const int tmp_index = index / width_col;
        const int h_col     = tmp_index % height_col;
        const int w_col     = index % width_col;
        const int c_im      = tmp_index / height_col;
        const int c_col     = c_im * kernel_h * kernel_w;
        const int h_offset  = h_col * stride_h - pad_h;
        const int w_offset  = w_col * stride_w - pad_w;
        T* data_col_ptr     = data_col + batchid * height_col * width_col;

        data_col_ptr += c_col * batch_col_matrix_size + h_col * width_col + w_col;
        const T* data_im_ptr = data_im + batchid * input_len;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im      = h_offset + i * dilation_h;
                int w_im      = w_offset + j * dilation_w;
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                    ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                                    : 0;
                data_col_ptr += batch_col_matrix_size;
            }
        }
    }
}

// col matrix dimension is  (chls*kernel_h*kernel_w)*(height_out*weight_out)
// height_out= (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
// weight_out= (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
template <typename T>
__global__ void Col2imgKernel(const int n,
                              const T* data_col,
                              const int height,
                              const int width,
                              const int channels,
                              const int kernel_h,
                              const int kernel_w,
                              const int pad_h,
                              const int pad_w,
                              const int stride_h,
                              const int stride_w,
                              const int dilation_h,
                              const int dilation_w,
                              const int height_col,
                              const int width_col,
                              const int input_len,
                              const int output_len,
                              T* data_im) {
    const int batchid = blockIdx.y;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
         index += blockDim.x * gridDim.x) {
        float val           = 0;
        const int w_im      = index % width + pad_w;
        const int h_im      = (index / width) % height + pad_h;
        const int c_im      = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);

        // TODO(fbh): use LCM of stride and dilation to avoid unnecessary loops

        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index =
                        (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                            width_col +
                        w_col;
                    val += data_col[data_col_index + batchid * input_len];
                }
            }
        }
        data_im[index + batchid * output_len] = val;
    }
}

// now follow warp is 32 threads, if the hardware support wavefront 64 threads, need to change this
// code
template <class T>
__device__ void warpReduce(volatile T* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// dstValue = alpha[0]*result + beta[0]*priorDstValue
template <class T>
__global__ void CudnnConvBwdBiasKernel(const T alpha,
                                       const T* dy,
                                       const T beta,
                                       T* db,
                                       int batch_size,
                                       int chl,
                                       int stride_c,
                                       int stride_higest_dim,
                                       int perfeature_size,
                                       int tile) {
    T sum = 0.0f;
    extern __shared__ T local_sum[];

    int perfeature_offset = blockIdx.x * stride_c;
    int data_id_x         = threadIdx.x;
    int glb_offset        = 0;
    // }
    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        glb_offset = perfeature_offset + b_idx * chl * perfeature_size;

        for (int idx = data_id_x; idx < perfeature_size; idx += tile) {
            sum += dy[glb_offset + idx * stride_higest_dim];
        }
    }
    local_sum[data_id_x] = sum;

    __syncthreads();

    // the below need to reduce local_sum
    for (int local_size = (tile >> 1); (data_id_x < local_size) && (local_size > 32);
         local_size     = (local_size >> 1)) {
        local_sum[data_id_x] += local_sum[data_id_x + local_size];
        __syncthreads();
    }

    // now follow warp is 32 threads, if the hardware support wavefront 64 threads, need to
    // change this code
    if (data_id_x < 32) {
        warpReduce(local_sum, data_id_x);
    }

    if (data_id_x == 0) {
        db[blockIdx.x] = alpha * local_sum[0] + beta * db[blockIdx.x];
    }
}

/*
// this version don't do any optimization
// a matrix dimension is m*n, b matrix dimension is n*k, c matrix dimension is m*k.
template <typename T1, typename T2>
__global__ void CudnnConvMatrixMulKernel(
    const T2 alpha, const T1* a, const T1* b, const T2 beta, T1* c, int m, int n, int k) {
    __shared__ T1 ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ T1 ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  // w direction block
    int by = blockIdx.y;  // h direction block
    int bz = blockIdx.z;

    int tx = threadIdx.x;  // w direction thread, k direction
    int ty = threadIdx.y;  // h direction thread, m direction

    // the result matrix row and column
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    T1 c_value = 0;
    int c_idx  = bz * m * k + row * k + col;
    // loop loading data to tile a,b  and computing result matrix c
    int tile_len_x    = tx;
    int tile_len_y    = ty;
    int num_tile_in_n = (n - 1) / TILE_WIDTH + 1;

    for (int t = 0; t < num_tile_in_n; ++t) {

        if (row < m && tile_len_x < n) {
            // load tile ds_a by coalesced access way
            ds_a[ty][tx] = a[row * n + tile_len_x];

        } else {
            // printf("ds_a[ty][tx] = 0.0,m=%d,n=%d\n,alpha=%f,beta=%f", m, n, alpha, beta);
            ds_a[ty][tx] = 0.0;
        }

        if (col < k && (tile_len_y < n)) {
            ds_b[ty][tx] = b[tile_len_y * gridDim.z * k + bz * k + col];
        } else {
            ds_b[ty][tx] = 0.0;
        }

        // make sure the element of tile to be loaded
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            c_value += ds_a[ty][i] * ds_b[i][tx];
        }
        __syncthreads();

        tile_len_x += TILE_WIDTH;
        tile_len_y += TILE_WIDTH;
    }

    if (alpha == 1 && beta == 0) {
        if (row < m && col < k) {
            c[c_idx] = c_value;
        }
    } else {
        if (row < m && col < k) {
            c[c_idx] = alpha * c_value + beta * c[c_idx];
        }
    }
}*/

/*
// this version test 32 thread coalesced access global memory
template <typename T1, typename T2>
__global__ void CudnnConvMatrixMulKernel(
    const T2 alpha, const T1* a, const T1* b, const T2 beta, T1* c, int m, int n, int k) {
    __shared__ T1 ds_a[TILE_WIDTH][TILE_WIDTH * 2];
    __shared__ T1 ds_b[TILE_WIDTH * 2][TILE_WIDTH];

    int bx = blockIdx.x;  // w direction block
    int by = blockIdx.y;  // h direction block
    int bz = blockIdx.z;

    int tx = threadIdx.x;  // w direction thread, k direction
    int ty = threadIdx.y;  // h direction thread, m direction

    // the result matrix row and column
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    T1 c_value = 0;
    int c_idx  = bz * m * k + row * k + col;
    // loop loading data to tile a,b  and computing result matrix c

    int num_tile_in_n = (n - 1) / (TILE_WIDTH * 2) + 1;

    int a_row = by * TILE_WIDTH + (ty >> 1);
    int a_col = (ty & 1) * TILE_WIDTH;

    int tile_width_mul_2 = TILE_WIDTH << 1;
    int tile_width_div_2 = TILE_WIDTH >> 1;
    int ty_div_2         = ty >> 1;

    int tile_len_x = tx;
    int tile_len_y = ty;

    for (int t = 0; t < num_tile_in_n; ++t) {

        // int tile_len_x = t * tile_width_mul_2 + tx;
        // int tile_len_y = t * tile_width_mul_2 + ty;

        if ((tile_len_x + a_col) < n) {
            if (a_row < m) {
                // load tile ds_a by coalesced access way
                // ds_a[ty][tx] = a[row * n + tile_len_x];

                ds_a[ty_div_2][a_col + tx] = a[a_row * n + tile_len_x + a_col];
            } else {
                // printf("ds_a[ty][tx] = 0.0,m=%d,n=%d\n,alpha=%f,beta=%f", m, n, alpha, beta);
                ds_a[ty_div_2][a_col + tx] = 0.0;
            }

            if (a_row + tile_width_div_2 < m) {
                // load tile ds_a by coalesced access way
                ds_a[tile_width_div_2 + ty_div_2][a_col + tx] =
                    a[(a_row + tile_width_div_2) * n + tile_len_x + a_col];
            } else {
                // printf("ds_a[ty][tx] = 0.0,m=%d,n=%d\n,alpha=%f,beta=%f", m, n, alpha, beta);
                ds_a[tile_width_div_2 + ty_div_2][a_col + tx] = 0.0;
            }
        }

        if (col < k) {
            if (tile_len_y < n) {
                ds_b[ty][tx] = b[tile_len_y * gridDim.z * k + bz * k + col];
            } else {
                ds_b[ty][tx] = 0.0;
            }

            if (tile_len_y + TILE_WIDTH < n) {

                ds_b[TILE_WIDTH + ty][tx] =
                    b[(tile_len_y + TILE_WIDTH) * gridDim.z * k + bz * k + col];

            } else {
                ds_b[TILE_WIDTH + ty][tx] = 0.0;
            }
        }

        // make sure the element of tile to be loaded
        __syncthreads();

        for (int i = 0; i < tile_width_mul_2; ++i) {
            c_value += ds_a[ty][i] * ds_b[i][tx];
        }
        __syncthreads();

        tile_len_x += tile_width_mul_2;
        tile_len_y += tile_width_mul_2;
    }

    if (alpha == 1 && beta == 0) {
        if (row < m && col < k) {
            c[c_idx] = c_value;
        }
    } else {
        if (row < m && col < k) {
            c[c_idx] = alpha * c_value + beta * c[c_idx];
        }
    }
}*/

// this version reuses ds_a data
template <typename T1, typename T2>
__global__ void CudnnConvMatrixMulKernel(
    const T2 alpha, const T1* a, const T1* b, const T2 beta, T1* c, int m, int n, int k) {
    __shared__ T1 ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ T1 ds_b[TILE_WIDTH][(TILE_WIDTH << 2)];

    int bx = blockIdx.x;  // w direction block
    int by = blockIdx.y;  // h direction block
    int bz = blockIdx.z;

    int tx = threadIdx.x;  // w direction thread, k direction
    int ty = threadIdx.y;  // h direction thread, m direction

    int tile_width2 = TILE_WIDTH << 1;
    int tile_width3 = TILE_WIDTH * 3;
    int tile_width4 = (TILE_WIDTH << 2);

    // the result matrix row and column
    int row = by * TILE_WIDTH + ty;
    int col = bx * tile_width4 + tx;

    T1 c_value1 = 0;
    T1 c_value2 = 0;
    T1 c_value3 = 0;
    T1 c_value4 = 0;

    int c_idx1 = bz * m * k + row * k + col;
    int c_idx2 = c_idx1 + TILE_WIDTH;
    int c_idx3 = c_idx2 + TILE_WIDTH;
    int c_idx4 = c_idx3 + TILE_WIDTH;

    // loop loading data to tile a,b  and computing result matrix c
    int tile_len_x    = tx;
    int tile_len_y    = ty;
    int num_tile_in_n = (n - 1) / TILE_WIDTH + 1;

    for (int t = 0; t < num_tile_in_n; ++t) {
        // int tile_len_x = t * TILE_WIDTH + tx;
        // int tile_len_y = t * TILE_WIDTH + ty;

        if (row < m && tile_len_x < n) {
            // load tile ds_a by coalesced access way
            ds_a[ty][tx] = a[row * n + tile_len_x];

        } else {
            // printf("ds_a[ty][tx] = 0.0,m=%d,n=%d\n,alpha=%f,beta=%f", m, n, alpha, beta);
            ds_a[ty][tx] = 0.0;
        }

        int b_idx = tile_len_y * gridDim.z * k + bz * k + col;
        if (col < k && (tile_len_y < n)) {
            ds_b[ty][tx] = b[b_idx];
        } else {
            ds_b[ty][tx] = 0.0;
        }

        if (tile_len_y < n) {
            if ((col + TILE_WIDTH) < k) {
                ds_b[ty][tx + TILE_WIDTH] = b[b_idx + TILE_WIDTH];

            } else {
                ds_b[ty][tx + TILE_WIDTH] = 0.0;
            }

            if ((col + tile_width2) < k) {
                ds_b[ty][tx + tile_width2] = b[b_idx + tile_width2];

            } else {
                ds_b[ty][tx + tile_width2] = 0.0;
            }

            if ((col + tile_width3) < k) {
                ds_b[ty][tx + tile_width3] = b[b_idx + tile_width3];

            } else {
                ds_b[ty][tx + tile_width3] = 0.0;
            }
        }
        // make sure the element of tile to be loaded
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            c_value1 += ds_a[ty][i] * ds_b[i][tx];
            c_value2 += ds_a[ty][i] * ds_b[i][tx + TILE_WIDTH];
            c_value3 += ds_a[ty][i] * ds_b[i][tx + tile_width2];
            c_value4 += ds_a[ty][i] * ds_b[i][tx + tile_width3];
        }
        __syncthreads();

        tile_len_x += TILE_WIDTH;
        tile_len_y += TILE_WIDTH;
    }

    if (alpha == 1 && beta == 0) {
        if (row < m && col + tile_width3 < k) {
            c[c_idx1] = c_value1;
            c[c_idx2] = c_value2;
            c[c_idx3] = c_value3;
            c[c_idx4] = c_value4;
        } else if (row < m && col < k && (col + TILE_WIDTH) >= k) {
            c[c_idx1] = c_value1;
        } else if (row < m && (col + TILE_WIDTH) < k && (col + tile_width2) >= k) {
            c[c_idx1] = c_value1;
            c[c_idx2] = c_value2;
        } else if (row < m && (col + tile_width2) < k && (col + tile_width3) >= k) {
            c[c_idx1] = c_value1;
            c[c_idx2] = c_value2;
            c[c_idx3] = c_value3;
        }
    } else {
        if (row < m && col < k) {
            c[c_idx1] = alpha * c_value1 + beta * c[c_idx1];
        }

        if (row < m && col + tile_width3 < k) {
            c[c_idx1] = alpha * c_value1 + beta * c[c_idx1];
            c[c_idx2] = alpha * c_value2 + beta * c[c_idx2];
            c[c_idx3] = alpha * c_value3 + beta * c[c_idx3];
            c[c_idx4] = alpha * c_value4 + beta * c[c_idx4];
        } else if (row < m && col < k && (col + TILE_WIDTH) >= k) {
            c[c_idx1] = alpha * c_value1 + beta * c[c_idx1];
        } else if (row < m && (col + TILE_WIDTH) < k && (col + tile_width2) >= k) {
            c[c_idx1] = alpha * c_value1 + beta * c[c_idx1];
            c[c_idx2] = alpha * c_value2 + beta * c[c_idx2];
        } else if (row < m && (col + tile_width2) < k && (col + tile_width3) >= k) {
            c[c_idx1] = alpha * c_value1 + beta * c[c_idx1];
            c[c_idx2] = alpha * c_value2 + beta * c[c_idx2];
            c[c_idx3] = alpha * c_value3 + beta * c[c_idx3];
        }
    }
}

template <class T>
__global__ void
CudnnConvFilterTransDy2AKernel(const T* dy, T* matrix_a, int n, int c, int h, int w) {
    int featuremap_size   = h * w;
    int threads_x         = blockIdx.x * blockDim.x + threadIdx.x;
    int idx               = blockIdx.y * c * featuremap_size + threads_x;
    int c_id              = threads_x / featuremap_size;
    int idx_in_featuremap = threads_x % featuremap_size;

    if (threads_x < c * h * w) {
        matrix_a[c_id * n * featuremap_size + blockIdx.y * featuremap_size + idx_in_featuremap] =
            dy[idx];
    }
}

/**
 * This kernel only support zero padding
 * Img2col ,the format of col matrix is (height_col*width_col)*(channel*kernel_h*kernel_w)
 * input_len=channels_img * height_img *width_img;
 */
template <class T>
__global__ void CudnnConvFilterTransDy2BKernel(const int n,
                                               const T* data_im,
                                               const int chls,
                                               const int height,
                                               const int width,
                                               const int kernel_h,
                                               const int kernel_w,
                                               const int pad_h,
                                               const int pad_w,
                                               const int stride_h,
                                               const int stride_w,
                                               const int dilation_h,
                                               const int dilation_w,
                                               const int height_col,
                                               const int width_col,
                                               const int input_len,
                                               T* data_col) {
    const int batchid = blockIdx.y;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
         index += blockDim.x * gridDim.x) {
        const int tmp_index = index / width_col;
        const int h_col     = tmp_index % height_col;
        const int w_col     = index % width_col;
        const int c_im      = tmp_index / height_col;
        const int h_offset  = h_col * stride_h - pad_h;
        const int w_offset  = w_col * stride_w - pad_w;

        T* data_col_ptr = data_col + batchid * height_col * width_col * chls * kernel_h * kernel_w;
        data_col_ptr += c_im * height_col * width_col + h_col * width_col + w_col;

        const T* data_im_ptr = data_im + batchid * input_len;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;

        int col_matrix_width = height_col * width_col * chls;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;

                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                    ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                                    : 0;
                data_col_ptr += col_matrix_width;
            }
        }
    }
}

// This function is general matrix multiplication, a matrix dimension is m*n, b matrix dimension
// is n*k, c matrix dimension is m*k.
template <typename T1, typename T2>
__global__ void CudnnMatrixMulKernel(
    const T2 alpha, const T1* a, const T1* b, const T2 beta, T1* c, int m, int n, int k) {
    __shared__ T1 ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ T1 ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  // w direction block
    int by = blockIdx.y;  // h direction block

    int tx = threadIdx.x;  // w direction thread
    int ty = threadIdx.y;  // h direction thread

    // the result matrix row and column
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    T1 c_value = 0;
    int c_idx  = row * k + col;
    // loop loading data to tile a,b  and computing result matrix c

    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
        int tile_len_x = t * TILE_WIDTH + tx;
        int tile_len_y = t * TILE_WIDTH + ty;

        if (row < m && tile_len_x < n) {
            // load tile ds_a by coalesced access way
            ds_a[ty][tx] = a[row * n + tile_len_x];
        } else {
            ds_a[ty][tx] = 0.0;
        }

        if ((col < (k * gridDim.z)) && (tile_len_y < n)) {
            ds_b[ty][tx] = b[tile_len_y * gridDim.z * k + col];
        } else {
            ds_b[ty][tx] = 0.0;
        }

        // make sure the element of tile to be loaded
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            c_value += ds_a[ty][i] * ds_b[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        c[c_idx] = alpha * c_value + beta * c[c_idx];
    }
}

// This function is conv_bwd_data matrix multiplication, a matrix dimension is m*n, b matrix
// dimension is m*k, c matrix dimension is n*k. A matrix will do transposition before matrixMul.
template <typename T1, typename T2>
__global__ void CudnnBwdDataMatrixMulKernel(
    const T2 alpha, const T1* a, const T1* b, const T2 beta, T1* c, int m, int n, int k) {
    __shared__ T1 ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ T1 ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  // w direction block
    int by = blockIdx.y;  // h direction block
    int bz = blockIdx.z;  // batch_id

    int tx = threadIdx.x;  // w direction thread
    int ty = threadIdx.y;  // h direction thread

    // the result matrix row and column
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    T1 c_value = 0;
    int c_idx  = bz * m * k + row * k + col;
    // loop loading data to tile a,b  and computing result matrix c
    int b_offset = bz * n * k;
    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
        int tile_len_x = t * TILE_WIDTH + tx;
        int tile_len_y = t * TILE_WIDTH + ty;

        if (row < m && tile_len_x < n) {
            // load tile ds_a by coalesced access way
            // ds_a[ty][tx] = a[row * n + tile_len_x];
            ds_a[ty][tx] = a[tile_len_x * m + row];
        } else {
            ds_a[ty][tx] = 0.0;
        }
        if ((col < k) && (tile_len_y < n)) {
            ds_b[ty][tx] = b[b_offset + tile_len_y * k + col];
        } else {
            ds_b[ty][tx] = 0.0;
        }

        // make sure the element of tile to be loaded
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            c_value += ds_a[ty][i] * ds_b[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        c[c_idx] = alpha * c_value + beta * c[c_idx];
    }
}

/**
 * brief@ this kernel make matrix A used for convolution backward data
 */
template <class T>
__global__ void CudnnConvBwdDataTransW2AKernel(
    const T* w, T* matrix_a, int out_c, int inp_c, int filter_h, int filter_w) {
    int idx_in_block     = blockIdx.x * blockDim.x + threadIdx.x;
    int threads_in_block = filter_h * filter_w;
    int chl_idx          = idx_in_block / threads_in_block;
    int threadid_in_chl  = idx_in_block % threads_in_block;
    int by               = blockIdx.y;

    matrix_a[chl_idx * out_c * threads_in_block + by * threads_in_block +
             ((threads_in_block - 1) - threadid_in_chl)] =
        w[(by * inp_c * threads_in_block) + (chl_idx * threads_in_block) + threadid_in_chl];
}

template <class T>
__global__ void CudnnConvBwdDataTransDy2BKernel(const int n,
                                                const T* data_im,
                                                const int height,
                                                const int width,
                                                const int kernel_h,
                                                const int kernel_w,
                                                const int pad_h,
                                                const int pad_w,
                                                const int stride_h,
                                                const int stride_w,
                                                const int dilation_h,
                                                const int dilation_w,
                                                const int height_col,
                                                const int width_col,
                                                const int input_len,
                                                T* data_col) {
    const int batchid    = blockIdx.y;
    const int batch_size = gridDim.y;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
         index += blockDim.x * gridDim.x) {
        const int tmp_index = index / width_col;
        const int h_col     = tmp_index % height_col;
        const int w_col     = index % width_col;
        const int c_im      = tmp_index / height_col;
        const int c_col     = c_im * kernel_h * kernel_w;
        const int h_offset  = h_col * stride_h - pad_h;
        const int w_offset  = w_col * stride_w - pad_w;
        T* data_col_ptr     = data_col + batchid * height_col * width_col;

        data_col_ptr += c_col * height_col * width_col * batch_size + h_col * width_col + w_col;
        const T* data_im_ptr = data_im + batchid * input_len;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im      = h_offset + i * dilation_h;
                int w_im      = w_offset + j * dilation_w;
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                    ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                                    : 0;
                data_col_ptr += height_col * width_col * batch_size;
            }
        }
    }
}

template <typename T1, typename T2>
__global__ void CudnnConvActBiasMatrixMulKernel(
    const T2 alpha1, const T1* a, const T1* b, T1* c, int m, int n, int k, const T2 alpha2, const T1* z, const T1* bias) {
    __shared__ T1 ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ T1 ds_b[TILE_WIDTH][(TILE_WIDTH << 2)];

    int bx = blockIdx.x;  // w direction block
    int by = blockIdx.y;  // h direction block
    int bz = blockIdx.z;

    int tx = threadIdx.x;  // w direction thread, k direction
    int ty = threadIdx.y;  // h direction thread, m direction

    int tile_width2 = TILE_WIDTH << 1;
    int tile_width3 = TILE_WIDTH * 3;
    int tile_width4 = (TILE_WIDTH << 2);

    // the result matrix row and column
    int row = by * TILE_WIDTH + ty;
    int col = bx * tile_width4 + tx;

    T1 c_value1 = 0;
    T1 c_value2 = 0;
    T1 c_value3 = 0;
    T1 c_value4 = 0;

    int c_idx1 = bz * m * k + row * k + col;
    int c_idx2 = c_idx1 + TILE_WIDTH;
    int c_idx3 = c_idx2 + TILE_WIDTH;
    int c_idx4 = c_idx3 + TILE_WIDTH;

    // loop loading data to tile a,b  and computing result matrix c
    int tile_len_x    = tx;
    int tile_len_y    = ty;
    int num_tile_in_n = (n - 1) / TILE_WIDTH + 1;

    for (int t = 0; t < num_tile_in_n; ++t) {
        // int tile_len_x = t * TILE_WIDTH + tx;
        // int tile_len_y = t * TILE_WIDTH + ty;

        if (row < m && tile_len_x < n) {
            // load tile ds_a by coalesced access way
            ds_a[ty][tx] = a[row * n + tile_len_x];

        } else {
            // printf("ds_a[ty][tx] = 0.0,m=%d,n=%d\n,alpha=%f,beta=%f", m, n, alpha, beta);
            ds_a[ty][tx] = 0.0;
        }

        int b_idx = tile_len_y * gridDim.z * k + bz * k + col;
        if (col < k && (tile_len_y < n)) {
            ds_b[ty][tx] = b[b_idx];
        } else {
            ds_b[ty][tx] = 0.0;
        }

        if (tile_len_y < n) {
            if ((col + TILE_WIDTH) < k) {
                ds_b[ty][tx + TILE_WIDTH] = b[b_idx + TILE_WIDTH];

            } else {
                ds_b[ty][tx + TILE_WIDTH] = 0.0;
            }

            if ((col + tile_width2) < k) {
                ds_b[ty][tx + tile_width2] = b[b_idx + tile_width2];

            } else {
                ds_b[ty][tx + tile_width2] = 0.0;
            }

            if ((col + tile_width3) < k) {
                ds_b[ty][tx + tile_width3] = b[b_idx + tile_width3];

            } else {
                ds_b[ty][tx + tile_width3] = 0.0;
            }
        }
        // make sure the element of tile to be loaded
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            c_value1 += ds_a[ty][i] * ds_b[i][tx];
            c_value2 += ds_a[ty][i] * ds_b[i][tx + TILE_WIDTH];
            c_value3 += ds_a[ty][i] * ds_b[i][tx + tile_width2];
            c_value4 += ds_a[ty][i] * ds_b[i][tx + tile_width3];
        }
        __syncthreads();

        tile_len_x += TILE_WIDTH;
        tile_len_y += TILE_WIDTH;
    }

    T1 bias_value;

    if (row < m && col + tile_width3 < k) {
        bias_value = bias[row];
        c_value1 = alpha1 * c_value1 + alpha2 * z[c_idx1] + bias_value;
        c[c_idx1] = c_value1 * (c_value1 > 0);
        c_value2 = alpha1 * c_value2 + alpha2 * z[c_idx2] + bias_value;
        c[c_idx2] = c_value2 * (c_value2 > 0);
        c_value3 = alpha1 * c_value3 + alpha2 * z[c_idx3] + bias_value;
        c[c_idx3] = c_value3 * (c_value3 > 0);
        c_value4 = alpha1 * c_value4 + alpha2 * z[c_idx4] + bias_value;
        c[c_idx4] = c_value4 * (c_value4 > 0);
    } else if (row < m && col < k && (col + TILE_WIDTH) >= k) {
        bias_value = bias[row];
        c_value1 = alpha1 * c_value1 + alpha2 * z[c_idx1] + bias_value;
        c[c_idx1] = c_value1 * (c_value1 > 0);
    } else if (row < m && (col + TILE_WIDTH) < k && (col + tile_width2) >= k) {
        bias_value = bias[row];
        c_value1 = alpha1 * c_value1 + alpha2 * z[c_idx1] + bias_value;
        c[c_idx1] = c_value1 * (c_value1 > 0);
        c_value2 = alpha1 * c_value2 + alpha2 * z[c_idx2] + bias_value;
        c[c_idx2] = c_value2 * (c_value2 > 0);
    } else if (row < m && (col + tile_width2) < k && (col + tile_width3) >= k) {
        bias_value = bias[row];
        c_value1 = alpha1 * c_value1 + alpha2 * z[c_idx1] + bias_value;
        c[c_idx1] = c_value1 * (c_value1 > 0);
        c_value2 = alpha1 * c_value2 + alpha2 * z[c_idx2] + bias_value;
        c[c_idx2] = c_value2 * (c_value2 > 0);
        c_value3 = alpha1 * c_value3 + alpha2 * z[c_idx3] + bias_value;
        c[c_idx3] = c_value3 * (c_value3 > 0);
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
