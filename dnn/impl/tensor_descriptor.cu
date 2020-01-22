/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/kernel/cudnn_tensor_descriptor.cuh>

#include <algorithm>

namespace cudnn {
namespace impl {

void AddTensor(CuHandle& handle,
               const void* alpha,
               const CuTensorDescriptor& a_desc,
               const void* a,
               const void* beta,
               const CuTensorDescriptor& c_desc,
               void* c) {
    cudnnDataType_t in_data_type;

    in_data_type = a_desc.GetDataType();

    if (DNN_DATA_FLOAT == in_data_type) {
        const float* in_a     = reinterpret_cast<const float*>(a);
        const float* in_alpha = reinterpret_cast<const float*>(alpha);
        const float* in_beta  = reinterpret_cast<const float*>(beta);
        float* out_c          = reinterpret_cast<float*>(c);
        LaunchAddTensorKernel(in_alpha, a_desc, in_a, in_beta, c_desc, out_c);
    } else if (DNN_DATA_HALF == in_data_type) {
        /** to do half addtensor*/
    } else if (DNN_DATA_INT8 == in_data_type) {
        /** to do int8 addtensor*/
    } else {
    }
}

void TransformTensor(CuHandle& handle,
                     const void* alpha,
                     const CuTensorDescriptor& x_desc,
                     const void* x,
                     const void* beta,
                     const CuTensorDescriptor& y_desc,
                     void* y) {
    cudnnDataType_t in_data_type;

    in_data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == in_data_type) {
        const float* in_x     = reinterpret_cast<const float*>(x);
        const float* in_alpha = reinterpret_cast<const float*>(alpha);
        const float* in_beta  = reinterpret_cast<const float*>(beta);
        float* out_y          = reinterpret_cast<float*>(y);

        LaunchTransformTensorKernel(in_alpha, x_desc, in_x, in_beta, y_desc, out_y);
    } else if (DNN_DATA_HALF == in_data_type) {
        /** to do half transformTensor */
    } else if (DNN_DATA_INT8 == in_data_type || DNN_DATA_INT8x4 == in_data_type) {
        const char* in_x      = reinterpret_cast<const char*>(x);
        const float* in_alpha = reinterpret_cast<const float*>(alpha);
        const float* in_beta  = reinterpret_cast<const float*>(beta);
        char* out_y           = reinterpret_cast<char*>(y);

        LaunchTransformTensorKernel(in_alpha, x_desc, in_x, in_beta, y_desc, out_y);
    } else {
    }
}

// AddTensor support 2d and 3d, NCHW NCDHW and NHWC format
template <typename T1, typename T2>
void LaunchAddTensorKernel(const T2* alpha,
                           const CuTensorDescriptor& a_desc,
                           const T1* a,
                           const T2* beta,
                           const CuTensorDescriptor& c_desc,
                           T1* c) {
    if (a_desc.GetN() == c_desc.GetN() && a_desc.GetH() == c_desc.GetH() && a_desc.GetW() == c_desc.GetW()) {
        int image_size       = 0;
        image_size           = a_desc.GetStride(1);  // only support batch_size is first dimension
        int n_out            = c_desc.GetN();
        int threads_total_x  = n_out * image_size;
        int threads_perblk_x = std::min(threads_total_x, kMaxThreadNbPerBlock);
        int nblocks_x        = (threads_total_x + threads_perblk_x - 1) / threads_perblk_x;

        dim3 gridSize(nblocks_x, 1, 1);
        dim3 blockSize(threads_perblk_x, 1, 1);

        kernel::CudnnAddTensorEQDimKernel<<<gridSize, blockSize>>>(
            a, *alpha, *beta, c, threads_total_x);
    } else {
        int c_batch        = c_desc.GetN();
        int a_chls_stride  = a_desc.GetStride(2);
        int c_chls_stride  = c_desc.GetStride(2);
        int c_chls         = c_desc.GetDim(2);
        int c_image_size   = c_desc.GetStride(1);  // c*h*w
        int a_feature_size = 1;
        int c_feature_size = 1;
        for (int idx = 3; idx <= c_desc.GetNbDims(); idx++) {
            a_feature_size = a_feature_size * a_desc.GetDim(idx);
        }
        for (int idx = 3; idx <= c_desc.GetNbDims(); idx++) {
            c_feature_size = c_feature_size * c_desc.GetDim(idx);
        }

        int threads_total_x  = c_feature_size;
        int threads_perblk_x = std::min(threads_total_x, kMaxThreadNbPerBlock);
        int nblocks_x        = (threads_total_x + threads_perblk_x - 1) / threads_perblk_x;

        dim3 gridSize(nblocks_x, c_batch, 1);
        dim3 blockSize(threads_perblk_x, 1, 1);
        kernel::CudnnAddTensorNotEQDimKernel<<<gridSize, blockSize>>>(
            a, a_chls_stride, a_feature_size, *alpha, *beta, c, c_chls, c_chls_stride, c_image_size);
    }
}

template <class T>
void LaunchTransformTensorKernel(const float* alpha,
                                 const CuTensorDescriptor& x_desc,
                                 const T* x,
                                 const float* beta,
                                 const CuTensorDescriptor& y_desc,
                                 T* y) {
    int in_n, in_c, in_w, in_h, in_stride_n, in_stride_c, in_stride_h, in_stride_w;
    int out_n, out_c, out_w, out_h, out_stride_n, out_stride_c, out_stride_h, out_stride_w;
    cudnnDataType_t in_data_type, out_data_type;

    x_desc.Get(&in_data_type,
               &in_n,
               &in_c,
               &in_h,
               &in_w,
               &in_stride_n,
               &in_stride_c,
               &in_stride_h,
               &in_stride_w);

    y_desc.Get(&out_data_type,
               &out_n,
               &out_c,
               &out_h,
               &out_w,
               &out_stride_n,
               &out_stride_c,
               &out_stride_h,
               &out_stride_w);

    cudnnTensorFormat_t x_format = x_desc.GetTensorFormat();
    cudnnTensorFormat_t y_format = y_desc.GetTensorFormat();

    if (x_format == DNN_TENSOR_NCHW_VECT_C && y_format != DNN_TENSOR_NCHW_VECT_C) {
        int feature_num = in_h * in_w * in_c;
        int quad_hw     = in_h * in_w << 2;

        int threads_x;
        int blocks_x = in_n;
        int blocks_y;

        if (feature_num <= kMaxThreadNbPerBlock) {
            threads_x = feature_num;
            blocks_y  = 1;
        } else {
            threads_x = kMaxThreadNbPerBlock;
            blocks_y  = (feature_num % kMaxThreadNbPerBlock == 0)
                           ? feature_num / kMaxThreadNbPerBlock
                           : feature_num / kMaxThreadNbPerBlock + 1;
        }

        dim3 gridSize(blocks_x, blocks_y, 1);
        dim3 blockSize(threads_x, 1, 1);

        kernel::CudnnTransformVecToScaKernel<<<gridSize, blockSize>>>(
            x, *alpha, *beta, y, out_stride_n, out_stride_c, out_stride_w, quad_hw);

    } else if (x_format != DNN_TENSOR_NCHW_VECT_C && y_format == DNN_TENSOR_NCHW_VECT_C) {
        int feature_num = in_h * in_w * in_c;
        int quad_hw     = in_h * in_w << 2;

        int threads_x;
        int blocks_x = in_n;
        int blocks_y;

        if (feature_num <= kMaxThreadNbPerBlock) {
            threads_x = feature_num;
            blocks_y  = 1;
        } else {
            threads_x = kMaxThreadNbPerBlock;
            blocks_y  = (feature_num % kMaxThreadNbPerBlock == 0)
                           ? feature_num / kMaxThreadNbPerBlock
                           : feature_num / kMaxThreadNbPerBlock + 1;
        }

        dim3 gridSize(blocks_x, blocks_y, 1);
        dim3 blockSize(threads_x, 1, 1);

        kernel::CudnnTransformScaToVecKernel<<<gridSize, blockSize>>>(
            x, *alpha, *beta, y, in_stride_n, in_stride_c, in_stride_w, quad_hw);
    } else {
        int muti_hw = in_h * in_w;

        int threads_x;
        int blocks_x = in_n;
        int blocks_y = in_c;
        int blocks_z;

        if (muti_hw <= kMaxThreadNbPerBlock) {
            threads_x = muti_hw;
            blocks_z  = 1;
        } else {
            threads_x = kMaxThreadNbPerBlock;
            blocks_z  = (muti_hw % kMaxThreadNbPerBlock == 0) ? muti_hw / kMaxThreadNbPerBlock
                                                             : muti_hw / kMaxThreadNbPerBlock + 1;
        }

        dim3 gridSize(blocks_x, blocks_y, blocks_z);
        dim3 blockSize(threads_x, 1, 1);

        kernel::CudnnTransformTensorKernel<<<gridSize, blockSize>>>(x,
                                                                    *alpha,
                                                                    *beta,
                                                                    y,
                                                                    out_stride_n,
                                                                    out_stride_c,
                                                                    out_stride_h,
                                                                    out_stride_w,
                                                                    in_stride_n,
                                                                    in_stride_c,
                                                                    in_stride_h,
                                                                    in_stride_w,
                                                                    in_w,
                                                                    in_h);
    }
}

}  // namespace impl
}  // namespace cudnn
