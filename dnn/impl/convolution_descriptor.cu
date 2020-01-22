
/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <cudnn.h>
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_convolution_descriptor.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_filter_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/kernel/cudnn_convolution_descriptor.cuh>
#include <cudnn/impl/kernel/cudnn_implicit_conv_fwd.cuh>
#include <cudnn/impl/kernel/cudnn_tensor_descriptor.cuh>

#include <algorithm>

#include <iostream>
#include <memory>
#include <vector>
// #include <random>
// #include <tuple>

using std::cout;
using std::endl;
using std::make_shared;
// using std::make_tuple;
using std::shared_ptr;
using std::vector;

namespace cudnn {
namespace impl {

// Im2col function
void CuIm2Col(const CuHandle& handle,
              const CuTensorDescriptor& src_desc,
              const void* src_data,
              const CuFilterDescriptor& filter_desc,
              const CuConvolutionDescriptor& conv_desc,
              void* col_buffer) {
    cudnnDataType_t in_data_type;
    in_data_type = src_desc.GetDataType();
    if (DNN_DATA_FLOAT == in_data_type) {
        const float* in_src_data = reinterpret_cast<const float*>(src_data);
        float* out_col_buffer    = reinterpret_cast<float*>(col_buffer);

        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            LaunchIm2ColKernel(src_desc, in_src_data, filter_desc, conv_desc, out_col_buffer);
        } else {
            // LaunchIm2ColKernel(src_desc, in_src_data, filter_desc, conv_desc, out_col_buffer);
        }
    } else if (DNN_DATA_HALF == in_data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == in_data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}

template <class T>
void LaunchIm2ColKernel(const CuTensorDescriptor& src_desc,
                        const T* src_data,
                        const CuFilterDescriptor& filter_desc,
                        const CuConvolutionDescriptor& conv_desc,
                        T* col_buffer) {
    int channels_img, height_img, width_img, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, batch_num;

    channels_img = src_desc.GetC();
    height_img   = src_desc.GetH();
    width_img    = src_desc.GetW();
    kernel_h     = filter_desc.GetFilterH();
    kernel_w     = filter_desc.GetFilterW();
    pad_h        = conv_desc.GetPadH();
    pad_w        = conv_desc.GetPadW();
    stride_h     = conv_desc.GetFilterStrideH();
    stride_w     = conv_desc.GetFilterStrideW();
    dilation_h   = conv_desc.GetDilationH();
    dilation_w   = conv_desc.GetDilationW();
    batch_num    = src_desc.GetN();

    // weight slides in height direction,the col matrix height
    int height_col = (height_img + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;

    // weight slides in weight direction,the col matrix width
    int width_col   = (width_img + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels_img * height_col * width_col;
    int input_size  = channels_img * height_img * width_img;

    // Setting the number of threads is an integer multiple of kWarpSize
    unsigned int block_size =
        ((std::min(num_kernels, kMaxThreadNbPerBlock / 2) + (kWarpSize - 1)) / kWarpSize) *
        kWarpSize;
    dim3 grid  = {(unsigned int)((num_kernels + block_size - 1) / block_size),
                 (unsigned int)(batch_num)};
    dim3 block = {block_size};

    kernel::Img2colKernel<<<grid, block>>>(num_kernels,
                                           src_data,
                                           height_img,
                                           width_img,
                                           kernel_h,
                                           kernel_w,
                                           pad_h,
                                           pad_w,
                                           stride_h,
                                           stride_w,
                                           dilation_h,
                                           dilation_w,
                                           height_col,
                                           width_col,
                                           input_size,
                                           col_buffer);
}

template <class T>
void LaunchCol2ImgKernel(const CuTensorDescriptor& src_desc,
                         const T* data_col,
                         const CuFilterDescriptor& filter_desc,
                         const CuConvolutionDescriptor& conv_desc,
                         T* data_im) {
    int channels_img, height_img, width_img, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, batch_num;

    channels_img = src_desc.GetC();
    height_img   = src_desc.GetH();
    width_img    = src_desc.GetW();
    kernel_h     = filter_desc.GetFilterH();
    kernel_w     = filter_desc.GetFilterW();
    pad_h        = conv_desc.GetPadH();
    pad_w        = conv_desc.GetPadW();
    stride_h     = conv_desc.GetFilterStrideH();
    stride_w     = conv_desc.GetFilterStrideW();
    dilation_h   = conv_desc.GetDilationH();
    dilation_w   = conv_desc.GetDilationW();
    batch_num    = src_desc.GetN();

    // weight slides in height direction,the col matrix height
    int height_col = (height_img + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    // weight slides in weight direction,the col matrix width
    int width_col   = (width_img + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels_img * height_img * width_img;
    int input_size  = height_col * width_col * kernel_h * kernel_w * channels_img;
    int output_size = channels_img * height_img * width_img;

    // Setting the number of threads is an integer multiple of kWarpSize
    unsigned int block_size =
        ((std::min(num_kernels, kMaxThreadNbPerBlock / 2) + (kWarpSize - 1)) / kWarpSize) *
        kWarpSize;
    dim3 grid  = {(unsigned int)((num_kernels + block_size - 1) / block_size),
                 (unsigned int)(batch_num)};
    dim3 block = {block_size};
    kernel::Col2imgKernel<<<grid, block>>>(num_kernels,
                                           data_col,
                                           height_img,
                                           width_img,
                                           channels_img,
                                           kernel_h,
                                           kernel_w,
                                           pad_h,
                                           pad_w,
                                           stride_h,
                                           stride_w,
                                           dilation_h,
                                           dilation_w,
                                           height_col,
                                           width_col,
                                           input_size,
                                           output_size,
                                           data_im);
}

/**
 *@brief this function executes convolutions over x using filters specified with w,
 * returning results in y. Scaling factors alpha and beta can be used to scale the input tensor and
 * the output tensor respectively.
 */
void CuConvolutionForward(const CuHandle& handle,
                          const void* alpha,
                          const CuTensorDescriptor& x_desc,
                          const void* x,
                          const CuFilterDescriptor& w_desc,
                          const void* w,
                          const CuConvolutionDescriptor& conv_desc,
                          cudnnConvolutionFwdAlgo_t algo,
                          void* workSpace,
                          size_t workspace_size_in_bytes,
                          const void* beta,
                          const CuTensorDescriptor& y_desc,
                          void* y) {
    cudnnDataType_t data_type;
    data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_alpha     = reinterpret_cast<const float*>(alpha);
        const float* ptr_beta      = reinterpret_cast<const float*>(beta);
        const float* ptr_w         = reinterpret_cast<const float*>(w);
        const float* ptr_x         = reinterpret_cast<const float*>(x);
        const float* ptr_workSpace = reinterpret_cast<const float*>(workSpace);
        float* ptr_y               = reinterpret_cast<float*>(y);

        if (algo == DNN_CONVOLUTION_FWD_ALGO_GEMM) {
            if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
                // for CNN conv
                CuIm2Col(handle, x_desc, x, w_desc, conv_desc, workSpace);
                LaunchConvMatrixMulKernel(ptr_alpha,
                                            w_desc,
                                            ptr_w,
                                            conv_desc,
                                            algo,
                                            ptr_workSpace,
                                            workspace_size_in_bytes,
                                            ptr_beta,
                                            y_desc,
                                            ptr_y);
            } else {
                 /* NOT SUPPORT */
            }
        } else if (algo == DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) {
                LaunchImplConvFwdKernel(ptr_alpha,
                                        x_desc,
                                        ptr_x,
                                        w_desc,
                                        ptr_w,
                                        conv_desc,
                                        ptr_beta,
                                        y_desc,
                                        ptr_y);
        } else {
            /* NOT SUPPORT */
        }
    } else if (DNN_DATA_HALF == data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}

template <typename T1, typename T2>
void LaunchImplConvFwdKernel(const T2* alpha,
                             const CuTensorDescriptor& x_desc,
                             const T1* x,
                             const CuFilterDescriptor& w_desc,
                             const T1* w,
                             const CuConvolutionDescriptor& conv_desc,
                             const T2* beta,
                             const CuTensorDescriptor& y_desc,
                             T1* y) {
    int nDims = w_desc.GetNbDim();
    if (4 == nDims) {
        int in_c         = x_desc.GetC();
        int in_h         = x_desc.GetH();
        int in_w         = x_desc.GetW();
        int in_n         = x_desc.GetN();
        int kernel_h     = w_desc.GetFilterH();
        int kernel_w     = w_desc.GetFilterW();
        int pad_h        = conv_desc.GetPadH();
        int pad_w        = conv_desc.GetPadW();
        int stride_h     = conv_desc.GetFilterStrideH();
        int stride_w     = conv_desc.GetFilterStrideW();
        int dilation_h   = conv_desc.GetDilationH();
        int dilation_w   = conv_desc.GetDilationW();
        int out_c        = y_desc.GetC();

        int out_h  = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        int out_w  = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        dim3 grid       = {(unsigned int)(in_n), (unsigned int)((out_c + 15) >> 4), (unsigned int)((out_h * out_w + 15) >> 4)};
        dim3 block      = {16, 16, 1};

        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            kernel::implCrossCorrelation2DKernel<<<grid, block>>>(x,
                                                                  w,
                                                                  in_w,
                                                                  in_h,
                                                                  in_c,
                                                                  kernel_h,
                                                                  kernel_w,
                                                                  pad_h,
                                                                  pad_w,
                                                                  stride_h,
                                                                  stride_w,
                                                                  dilation_h,
                                                                  dilation_w,
                                                                  out_w,
                                                                  out_h,
                                                                  out_c,
                                                                  *alpha,
                                                                  *beta,
                                                                  y);
        } else {
            kernel::implConvolution2DKernel<<<grid, block>>>(x,
                                                             w,
                                                             in_w,
                                                             in_h,
                                                             in_c,
                                                             kernel_h,
                                                             kernel_w,
                                                             pad_h,
                                                             pad_w,
                                                             stride_h,
                                                             stride_w,
                                                             dilation_h,
                                                             dilation_w,
                                                             out_w,
                                                             out_h,
                                                             out_c,
                                                             *alpha,
                                                             *beta,
                                                             y);
        }
    } else if (5 == nDims) {
        int dimA_in[DNN_DIM_MAX];
        int stridesA_in[DNN_DIM_MAX];
        int n_dim;
        cudnnDataType_t in_data_type;
        x_desc.Get(
            DNN_DIM_MAX, &in_data_type, &n_dim, dimA_in, stridesA_in);

        int dimA_out[DNN_DIM_MAX];
        int stridesA_out[DNN_DIM_MAX];
        cudnnDataType_t out_data_type;
        y_desc.Get(
            DNN_DIM_MAX, &out_data_type, &n_dim, dimA_out, stridesA_out);

        int padA[3];
        int strideA[3];
        int dilationA[3];
        int array_length;
        cudnnConvolutionMode_t convolutionMode;
        conv_desc.Get(
            DNN_DIM_MAX, &array_length, padA, strideA, dilationA, &convolutionMode, &in_data_type);

        int filterDimA[5];
        int nbDims;
        cudnnTensorFormat_t tensorFormat;
        w_desc.Get(DNN_DIM_MAX, &in_data_type, &tensorFormat, &nbDims, filterDimA);

        int n_thread = stridesA_out[1];
        dim3 grid       = {(unsigned int)(dimA_out[0]), (unsigned int)((dimA_out[1] + 15) >> 4), (unsigned int)((stridesA_out[1] + 15) >> 4)};
        dim3 block      = {16, 16, 1};

        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            kernel::implCrossCorrelation3DKernel<<<grid, block>>>(x,
                                                                  w,
                                                                  dimA_in[4],
                                                                  dimA_in[3],
                                                                  dimA_in[2],
                                                                  dimA_in[1],
                                                                  filterDimA[2],
                                                                  filterDimA[3],
                                                                  filterDimA[4],
                                                                  padA[0],
                                                                  padA[1],
                                                                  padA[2],
                                                                  strideA[0],
                                                                  strideA[1],
                                                                  strideA[2],
                                                                  dilationA[0],
                                                                  dilationA[1],
                                                                  dilationA[2],
                                                                  dimA_out[4],
                                                                  dimA_out[3],
                                                                  dimA_out[2],
                                                                  dimA_out[1],
                                                                  *alpha,
                                                                  *beta,
                                                                  y);
        } else {
            kernel::implConvolution3DKernel<<<grid, block>>>(x,
                                                             w,
                                                             dimA_in[4],
                                                             dimA_in[3],
                                                             dimA_in[2],
                                                             dimA_in[1],
                                                             filterDimA[2],
                                                             filterDimA[3],
                                                             filterDimA[4],
                                                             padA[0],
                                                             padA[1],
                                                             padA[2],
                                                             strideA[0],
                                                             strideA[1],
                                                             strideA[2],
                                                             dilationA[0],
                                                             dilationA[1],
                                                             dilationA[2],
                                                             dimA_out[4],
                                                             dimA_out[3],
                                                             dimA_out[2],
                                                             dimA_out[1],
                                                             *alpha,
                                                             *beta,
                                                             y);
        }
    }
}

template <typename T1, typename T2>
void LaunchConvMatrixMulKernel(const T2* alpha,
                               const CuFilterDescriptor& w_desc,
                               const T1* w,
                               const CuConvolutionDescriptor& conv_desc,
                               cudnnConvolutionFwdAlgo_t algo,
                               const T1* workSpace,
                               size_t workspace_size_in_bytes,
                               const T2* beta,
                               const CuTensorDescriptor& y_desc,
                               T1* y) {
    // the last matrix C rows
    int m = w_desc.GetOutC();

    // the first matrix A colums

    int n = w_desc.GetFilterH() * w_desc.GetFilterW() * w_desc.GetInC();

    // the last matrix C columns
    int k = y_desc.GetH() * y_desc.GetW();

    int batch_size = y_desc.GetN();

    // set thread numbers
    // dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, batch_size);
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    dim3 dimGrid((k - 1) / (TILE_WIDTH * 4) + 1, (m - 1) / TILE_WIDTH + 1, batch_size);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // call kernel function
    kernel::CudnnConvMatrixMulKernel<<<dimGrid, dimBlock>>>(
        *alpha, w, workSpace, *beta, y, m, n, k);
}

void CuConvolutionBackwardBias(const CuHandle& handle,
                               const void* alpha,
                               const CuTensorDescriptor& dy_desc,
                               const void* dy,
                               const void* beta,
                               const CuTensorDescriptor& db_desc,
                               void* db) {
    cudnnDataType_t data_type;
    data_type = dy_desc.GetDataType();
    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_alpha = reinterpret_cast<const float*>(alpha);
        const float* ptr_beta  = reinterpret_cast<const float*>(beta);
        const float* ptr_dy    = reinterpret_cast<const float*>(dy);
        float* ptr_db          = reinterpret_cast<float*>(db);

        LaunchConvBwdBiasKernel(ptr_alpha, dy_desc, ptr_dy, ptr_beta, db_desc, ptr_db);
    } else if (DNN_DATA_HALF == data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}

template <class T>
void LaunchConvBwdBiasKernel(const T* p_alpha,
                             const CuTensorDescriptor& dy_desc,
                             const T* dy,
                             const T* p_beta,
                             const CuTensorDescriptor& db_desc,
                             T* db) {
    cudnnDataType_t dy_data_type;
    int nb_dims;
    int nums_perfeatureMap = 1;
    int dims_a[DNN_DIM_MAX];
    int stride_a[DNN_DIM_MAX];
    int stride_highest_dim = 1;

    dy_desc.Get(&dy_data_type, &nb_dims, dims_a, stride_a);
    stride_highest_dim = stride_a[nb_dims - 1];

    for (int idx = 2; idx < nb_dims; ++idx) {
        nums_perfeatureMap *= dims_a[idx];
    }

    int threads_perblk_x = std::min(nums_perfeatureMap, kMaxThreadNbPerBlock);

    // tile is maximum 2 N-th power less than nums_perfeatureMap
    int tile = 32;
    for (tile = 32; tile <= (threads_perblk_x >> 1); tile = tile * 2) {
        /*do nothing */
    }
    threads_perblk_x = tile;

    dim3 gridSize(dims_a[1], 1, 1);  // batch_size is block number
    dim3 blockSize(threads_perblk_x, 1, 1);

    kernel::CudnnConvBwdBiasKernel<<<gridSize, blockSize, tile * sizeof(T)>>>((*p_alpha),
                                                                              dy,
                                                                              (*p_beta),
                                                                              db,
                                                                              dims_a[0],
                                                                              dims_a[1],
                                                                              stride_a[1],
                                                                              stride_highest_dim,
                                                                              nums_perfeatureMap,
                                                                              tile);
}

/**
 *@brief this function computes the convolution weight (filter) gradient of the tensor dy,
 * where y is the output of the forward convolution in cudnnConvolutionForward().
 * It uses the specified algo, and returns the results in the output tensor dw.
 */

void CuConvolutionBackwardFilter(const CuHandle& handle,
                                 const void* alpha,
                                 const CuTensorDescriptor& x_desc,
                                 const void* x,
                                 const CuTensorDescriptor& dy_desc,
                                 const void* dy,
                                 const CuConvolutionDescriptor& conv_desc,
                                 cudnnConvolutionBwdFilterAlgo_t algo,
                                 void* workspace,
                                 size_t work_space_size_in_bytes,
                                 const void* beta,
                                 const CuFilterDescriptor& dw_desc,
                                 void* dw) {
    cudnnDataType_t data_type;
    data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_alpha = reinterpret_cast<const float*>(alpha);
        const float* ptr_beta  = reinterpret_cast<const float*>(beta);
        const float* ptr_x     = reinterpret_cast<const float*>(x);
        const float* ptr_dy    = reinterpret_cast<const float*>(dy);
        float* ptr_workSpace   = reinterpret_cast<float*>(workspace);
        float* ptr_dw          = reinterpret_cast<float*>(dw);
        int matrix_a_workspace_size =
            dy_desc.GetW() * dy_desc.GetH() * dy_desc.GetN() * dy_desc.GetC();

        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            // for CNN conv

            LanunchConvFilterTransDy2AKernel(
                dy_desc, ptr_dy, ptr_workSpace, matrix_a_workspace_size);
            LanunchConvBwdFilterTransDy2BKernel(x_desc,
                                                ptr_x,
                                                dy_desc,
                                                dw_desc,
                                                conv_desc,
                                                ptr_workSpace + matrix_a_workspace_size);
            LaunchMatrixMulKernel(ptr_alpha,
                                  dy_desc,
                                  ptr_workSpace,  // matrix a
                                  x_desc,
                                  ptr_workSpace + matrix_a_workspace_size,  // matrix b
                                  conv_desc,
                                  ptr_beta,
                                  dw_desc,
                                  ptr_dw);

        } else if (DNN_CONVOLUTION == conv_desc.GetMode()) {
            // TODO(fbh): to do  corresponding mathematically to a convolution
        }
    } else if (DNN_DATA_HALF == data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}

template <class T>
void LanunchConvFilterTransDy2AKernel(const CuTensorDescriptor& dy_desc,
                                      const T* dy,
                                      T* matrix_a,
                                      int matrix_a_work_space_size) {
    int n = dy_desc.GetN();
    int c = dy_desc.GetC();
    int h = dy_desc.GetH();
    int w = dy_desc.GetW();

    int threads_perblk_x = std::min(h * w * c, kMaxThreadNbPerBlock);
    dim3 gridSize((h * w * c + threads_perblk_x - 1) / threads_perblk_x, n);
    dim3 blockSize(threads_perblk_x, 1);

    kernel::CudnnConvFilterTransDy2AKernel<<<gridSize, blockSize>>>(dy, matrix_a, n, c, h, w);
}

template <class T>
void LanunchConvBwdFilterTransDy2BKernel(const CuTensorDescriptor& src_desc,
                                         const T* src_data,
                                         const CuTensorDescriptor& dy_desc,
                                         const CuFilterDescriptor& dw_desc,
                                         const CuConvolutionDescriptor& conv_desc,
                                         T* col_buffer) {
    int channels_img, height_img, width_img, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, batch_num;

    channels_img = src_desc.GetC();
    height_img   = src_desc.GetH();
    width_img    = src_desc.GetW();
    kernel_h     = dy_desc.GetH();
    kernel_w     = dy_desc.GetW();
    pad_h        = conv_desc.GetPadH();
    pad_w        = conv_desc.GetPadW();
    dilation_h   = conv_desc.GetFilterStrideH();  // forward stride_h is backward dilation_h
    dilation_w   = conv_desc.GetFilterStrideW();  // forward stride_w is backward dilation_w
    stride_h     = conv_desc.GetDilationH();      // forward dilation_h is backward stride_h
    stride_w     = conv_desc.GetDilationW();      // forward dilation_w is backward stride_w
    batch_num    = src_desc.GetN();

    // weight slides in height direction,the col matrix height
    int height_col = dw_desc.GetFilterH();

    // weight slides in weight direction,the col matrix width
    int width_col   = dw_desc.GetFilterW();
    int num_kernels = channels_img * height_col * width_col;
    int input_size  = channels_img * height_img * width_img;

    // Setting the number of threads is an integer multiple of kWarpSize
    unsigned int block_size =
        ((std::min(num_kernels, kMaxThreadNbPerBlock / 2) + (kWarpSize - 1)) / kWarpSize) *
        kWarpSize;

    dim3 grid  = {(unsigned int)((num_kernels + block_size - 1) / block_size),
                 (unsigned int)(batch_num)};
    dim3 block = {block_size};

    kernel::CudnnConvFilterTransDy2BKernel<<<grid, block>>>(num_kernels,
                                                            src_data,
                                                            channels_img,
                                                            height_img,
                                                            width_img,
                                                            kernel_h,
                                                            kernel_w,
                                                            pad_h,
                                                            pad_w,
                                                            stride_h,
                                                            stride_w,
                                                            dilation_h,
                                                            dilation_w,
                                                            height_col,
                                                            width_col,
                                                            input_size,
                                                            col_buffer);
}

template <class T>
void LaunchMatrixMulKernel(const T* alpha,
                           const CuTensorDescriptor& dy_desc,
                           const T* dy,  // matrix a
                           const CuTensorDescriptor& x_desc,
                           const T* x,  // matrix b
                           const CuConvolutionDescriptor& conv_desc,
                           const T* beta,
                           const CuFilterDescriptor& dw_desc,
                           T* dw) {
    // the last matrix C rows
    int m = dw_desc.GetOutC();

    // the first matrix A colums
    int n = dy_desc.GetH() * dy_desc.GetW() * dy_desc.GetN();

    // the last matrix C columns
    int k = dw_desc.GetFilterH() * dw_desc.GetFilterW() * x_desc.GetC();

    // set thread numbers
    dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // call kernel function
    kernel::CudnnMatrixMulKernel<<<dimGrid, dimBlock>>>(*alpha, dy, x, *beta, dw, m, n, k);
}

/*
void CuConvolutionBackwardData(const CuHandle& handle,
                               const void* alpha,
                               const CuFilterDescriptor& w_desc,
                               const void* w,
                               const CuTensorDescriptor& dy_desc,
                               const void* dy,
                               const CuConvolutionDescriptor& conv_desc,
                               cudnnConvolutionBwdDataAlgo_t algo,
                               void* workspace,
                               size_t workSpaceSizeInBytes,
                               const void* beta,
                               const CuTensorDescriptor& dx_desc,
                               void* dx) {
    cudnnDataType_t data_type;
    data_type = dx_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_alpha = reinterpret_cast<const float*>(alpha);
        const float* ptr_beta  = reinterpret_cast<const float*>(beta);
        const float* ptr_dy    = reinterpret_cast<const float*>(dy);
        float* ptr_workSpace   = reinterpret_cast<float*>(workspace);
        const float* ptr_w     = reinterpret_cast<const float*>(w);
        float* ptr_dx          = reinterpret_cast<float*>(dx);

        int matrix_a_workspace_size = w_desc.GetFilterH() * w_desc.GetFilterW() * w_desc.GetInC() *
                                      w_desc.GetOutC() * w_desc.GetDataTypeSizeInBytes();
        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            LanunchConvBwdDataTransW2AKernel(w_desc, ptr_w, ptr_workSpace, matrix_a_workspace_size);

            LanunchConvBwdDataTransDy2BKernel(dx_desc,
                                              dy_desc,
                                              ptr_dy,
                                              w_desc,
                                              conv_desc,
                                              ptr_workSpace + matrix_a_workspace_size);

            LaunchConvBwdDataMatrixMulKernel(ptr_alpha,
                                             w_desc,
                                             ptr_workSpace,  // matrix a
                                             dy_desc,
                                             ptr_workSpace + matrix_a_workspace_size,  // matrix b
                                             conv_desc,
                                             ptr_beta,
                                             dx_desc,
                                             ptr_dx);

        } else if (DNN_CONVOLUTION == conv_desc.GetMode()) {
            // TODO(fbh): to do  corresponding mathematically to a convolution
        }
    } else if (DNN_DATA_HALF == data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}
*/
template <class T>
void LaunchBwdDataMatrixMulKernel(const T* alpha,
                                  const CuFilterDescriptor& w_desc,
                                  const T* w,  // matrix a
                                  const CuTensorDescriptor& dy_desc,
                                  const T* dy,  // matrix b
                                  const CuConvolutionDescriptor& conv_desc,
                                  T* workspace,
                                  size_t workSpaceSizeInBytes,
                                  const T* beta,
                                  const CuTensorDescriptor& dx_desc) {
    // the A matrix dimension is (m*n),B matirx dimension is (m*k),C matrix dimension is (n*k),
    // the A matrix should do transposition

    // the last matrix A colums
    int m = w_desc.GetFilterH() * w_desc.GetFilterW() * w_desc.GetInC();

    // the first matrix A rows or B matrix rows
    int n = w_desc.GetOutC();

    // the last matrix C columns
    int k = dy_desc.GetH() * dy_desc.GetW();

    // set thread numbers
    dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, dx_desc.GetN());
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // call kernel function
    kernel::CudnnBwdDataMatrixMulKernel<<<dimGrid, dimBlock>>>(
        *alpha, w, dy, *beta, workspace, m, n, k);
}

void CuConvolutionBackwardData(const CuHandle& handle,
                               const void* alpha,
                               const CuFilterDescriptor& w_desc,
                               const void* w,
                               const CuTensorDescriptor& dy_desc,
                               const void* dy,
                               const CuConvolutionDescriptor& conv_desc,
                               cudnnConvolutionBwdDataAlgo_t algo,
                               void* workspace,
                               size_t workSpaceSizeInBytes,
                               const void* beta,
                               const CuTensorDescriptor& dx_desc,
                               void* dx) {
    cudnnDataType_t data_type;
    data_type = dx_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_alpha = reinterpret_cast<const float*>(alpha);
        const float* ptr_beta  = reinterpret_cast<const float*>(beta);
        const float* ptr_dy    = reinterpret_cast<const float*>(dy);
        float* ptr_workSpace   = reinterpret_cast<float*>(workspace);
        const float* ptr_w     = reinterpret_cast<const float*>(w);
        float* ptr_dx          = reinterpret_cast<float*>(dx);

        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            // shared_ptr<vector<float>> cu_h_x = make_shared<vector<float>>(100);
            // shared_ptr<vector<float>> cu_h_b = make_shared<vector<float>>(100);

            LaunchBwdDataMatrixMulKernel(ptr_alpha,
                                         w_desc,
                                         ptr_w,  // matrix a
                                         dy_desc,
                                         ptr_dy,  // matrix b
                                         conv_desc,
                                         ptr_workSpace,
                                         workSpaceSizeInBytes,
                                         ptr_beta,
                                         dx_desc);

            // cudaDeviceSynchronize();
            // cudaMemcpy(cu_h_x->data(), ptr_workSpace, workSpaceSizeInBytes,
            // cudaMemcpyDeviceToHost);

            // std::cout<<"matrix_res="<<std::endl;
            // for (int i = 0; i < workSpaceSizeInBytes/4; i++) {
            //     std::cout << (*cu_h_x)[i] << " ";
            // }

            LaunchCol2ImgKernel(dx_desc, ptr_workSpace, w_desc, conv_desc, ptr_dx);

        } else if (DNN_CONVOLUTION == conv_desc.GetMode()) {
            // TODO(fbh): to do  corresponding mathematically to a convolution
        }
    } else if (DNN_DATA_HALF == data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}

template <class T>
void LanunchConvBwdDataTransW2AKernel(const CuFilterDescriptor& w_desc,
                                      const T* w,
                                      T* matrix_a,
                                      int matrix_a_work_space_size) {
    int out_c    = w_desc.GetOutC();
    int inp_c    = w_desc.GetInC();
    int filter_h = w_desc.GetFilterH();
    int filter_w = w_desc.GetFilterW();

    int threads_perblk_x = std::min((filter_h * filter_w) * inp_c, kMaxThreadNbPerBlock);
    dim3 gridSize(((filter_h * filter_w) * inp_c + threads_perblk_x - 1) / threads_perblk_x, out_c);
    dim3 blockSize(threads_perblk_x, 1);

    kernel::CudnnConvBwdDataTransW2AKernel<<<gridSize, blockSize>>>(
        w, matrix_a, out_c, inp_c, filter_h, filter_w);
}

// img2col function for convolution backward filter dy data
template <class T>
void LanunchConvBwdDataTransDy2BKernel(const CuTensorDescriptor& dx_desc,
                                       const CuTensorDescriptor& dy_desc,
                                       const T* ptr_dy,
                                       const CuFilterDescriptor& w_desc,
                                       const CuConvolutionDescriptor& conv_desc,
                                       T* col_buffer) {
    int channels_img, height_img, width_img, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, batch_num;

    channels_img = dy_desc.GetC();
    height_img   = dy_desc.GetH();
    width_img    = dy_desc.GetW();
    kernel_h     = w_desc.GetFilterH();
    kernel_w     = w_desc.GetFilterW();
    pad_h        = kernel_h - 1;  // pad 0 at the edges of dy
    pad_w        = kernel_w - 1;  // pad 0 at the edges of dy
    stride_h     = conv_desc.GetFilterStrideH();
    stride_w     = conv_desc.GetFilterStrideW();
    dilation_h   = conv_desc.GetDilationH();
    dilation_w   = conv_desc.GetDilationW();
    batch_num    = dx_desc.GetN();

    // weight slides in height direction,the col matrix height
    int height_col = (height_img + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;

    // weight slides in weight direction,the col matrix width
    int width_col   = (width_img + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels_img * height_col * width_col;
    int input_size  = channels_img * height_img * width_img;

    // Setting the number of threads is an integer multiple of kWarpSize
    unsigned int block_size =
        ((std::min(num_kernels, kMaxThreadNbPerBlock / 2) + (kWarpSize - 1)) / kWarpSize) *
        kWarpSize;
    dim3 grid  = {(unsigned int)((num_kernels + block_size - 1) / block_size),
                 (unsigned int)(batch_num)};
    dim3 block = {block_size};

    kernel::CudnnConvBwdDataTransDy2BKernel<<<grid, block>>>(num_kernels,
                                                             ptr_dy,
                                                             height_img,
                                                             width_img,
                                                             kernel_h,
                                                             kernel_w,
                                                             pad_h,
                                                             pad_w,
                                                             stride_h,
                                                             stride_w,
                                                             dilation_h,
                                                             dilation_w,
                                                             height_col,
                                                             width_col,
                                                             input_size,
                                                             col_buffer);
}

template <typename T1, typename T2>
void LaunchConvBwdDataMatrixMulKernel(const T2* alpha,
                                      const CuFilterDescriptor& w_desc,
                                      const T1* w,  // matrix a
                                      const CuTensorDescriptor& dy_desc,
                                      const T1* dy,  // matrix b
                                      const CuConvolutionDescriptor& conv_desc,
                                      const T2* beta,
                                      const CuTensorDescriptor& dx_desc,
                                      T1* dx) {
    // the last matrix C rows
    int m = w_desc.GetInC();

    // the first matrix A colums
    int n = w_desc.GetFilterH() * w_desc.GetFilterW() * w_desc.GetOutC();

    // the last matrix C columns
    int k = dx_desc.GetH() * dx_desc.GetW();

    int batch_size = dx_desc.GetN();
    // set thread numbers
    dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, batch_size);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // call kernel function
    kernel::CudnnConvMatrixMulKernel<<<dimGrid, dimBlock>>>(*alpha, w, dy, *beta, dx, m, n, k);
}

void CuConvolutionBiasActivationForward(const CuHandle& handle,
                                        const void* alpha1,
                                        const CuTensorDescriptor& x_desc,
                                        const void* x,
                                        const CuFilterDescriptor& w_desc,
                                        const void* w,
                                        const CuConvolutionDescriptor& conv_desc,
                                        cudnnConvolutionFwdAlgo_t algo,
                                        void* workSpace,
                                        size_t workspace_size_in_bytes,
                                        const void* alpha2,
                                        const CuTensorDescriptor& z_desc,
                                        const void* z,
                                        const CuTensorDescriptor& bias_desc,
                                        const void* bias,
                                        const CuTensorDescriptor& y_desc,
                                        void* y) {
    cudnnDataType_t data_type;
    data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_alpha1    = reinterpret_cast<const float*>(alpha1);
        const float* ptr_alpha2    = reinterpret_cast<const float*>(alpha2);
        const float* ptr_w         = reinterpret_cast<const float*>(w);
        const float* ptr_workSpace = reinterpret_cast<const float*>(workSpace);
        const float* ptr_z         = reinterpret_cast<const float*>(z);
        const float* ptr_bias      = reinterpret_cast<const float*>(bias);
        float* ptr_y               = reinterpret_cast<float*>(y);

        if (DNN_CROSS_CORRELATION == conv_desc.GetMode()) {
            // for CNN conv
            CuIm2Col(handle, x_desc, x, w_desc, conv_desc, workSpace);
            LaunchConvActBiasMatMulKernel(ptr_alpha1,
                                          w_desc,
                                          ptr_w,
                                          conv_desc,
                                          algo,
                                          ptr_workSpace,
                                          workspace_size_in_bytes,
                                          ptr_alpha2,
                                          ptr_z,
                                          ptr_bias,
                                          y_desc,
                                          ptr_y);
        } else if (DNN_CONVOLUTION == conv_desc.GetMode()) {
            // TODO(fbh): to do  corresponding mathematically to a convolution
        }
    } else if (DNN_DATA_HALF == data_type) {
        // for DNN_DATA_HALF
    } else if (DNN_DATA_INT8 == data_type) {
        // for DNN_DATA_INT8
    } else {
    }
}

template <typename T1, typename T2>
void LaunchConvActBiasMatMulKernel(const T2* alpha1,
                                   const CuFilterDescriptor& w_desc,
                                   const T1* w,
                                   const CuConvolutionDescriptor& conv_desc,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   const T1* workSpace,
                                   size_t workspace_size_in_bytes,
                                   const T2* alpha2,
                                   const T1* z,
                                   const T1* bias,
                                   const CuTensorDescriptor& y_desc,
                                   T1* y) {
    // the last matrix C rows
    int m = w_desc.GetOutC();

    // the first matrix A colums

    int n = w_desc.GetFilterH() * w_desc.GetFilterW() * w_desc.GetInC();

    // the last matrix C columns
    int k = y_desc.GetH() * y_desc.GetW();

    int batch_size = y_desc.GetN();

    // set thread numbers
    // dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, batch_size);
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    dim3 dimGrid((k - 1) / (TILE_WIDTH * 4) + 1, (m - 1) / TILE_WIDTH + 1, batch_size);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // call kernel function
    kernel::CudnnConvActBiasMatrixMulKernel<<<dimGrid, dimBlock>>>(
        *alpha1, w, workSpace, y, m, n, k, *alpha2, z, bias);
}

}  // namespace impl
}  // namespace cudnn
