/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_softmax.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>

#include <cudnn/impl/kernel/cudnn_softmax.cuh>

#include <algorithm>

namespace cudnn {
namespace impl {

void CuSoftmaxForward(CuHandle& handle,
                      cudnnSoftmaxAlgorithm_t algorithm,
                      cudnnSoftmaxMode_t mode,
                      const void* alpha,
                      const CuTensorDescriptor& x_desc,
                      const void* x,
                      const void* beta,
                      const CuTensorDescriptor& y_desc,
                      void* y) {
    cudnnDataType_t data_type;

    data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* in_x     = reinterpret_cast<const float*>(x);
        const float* in_alpha = reinterpret_cast<const float*>(alpha);
        const float* in_beta  = reinterpret_cast<const float*>(beta);
        float* out_y          = reinterpret_cast<float*>(y);
        LaunchSoftMaxFwdKernel(algorithm, mode, in_alpha, x_desc, in_x, in_beta, y_desc, out_y);
    } else if (DNN_DATA_HALF == data_type) {
        /** to do half addtensor*/
    } else if (DNN_DATA_INT8 == data_type) {
        /** to do int8 addtensor*/
    } else {
    }
}

template <class T>
void LaunchSoftMaxFwdKernel(cudnnSoftmaxAlgorithm_t algorithm,
                            cudnnSoftmaxMode_t mode,
                            const T* alpha,
                            const CuTensorDescriptor& x_desc,
                            const T* x,
                            const T* beta,
                            const CuTensorDescriptor& y_desc,
                            T* y) {
    cudnnDataType_t data_type;
    int nb_dims;
    int dim_a[DNN_DIM_MAX];
    int stride_a[DNN_DIM_MAX];
    x_desc.Get(&data_type, &nb_dims, dim_a, stride_a);

    int featuremap_size  = 0;
    int threads_total_x  = 1;
    int threads_perblk_x = 1;
    int nblocks_x        = 1;

    if (DNN_SOFTMAX_MODE_CHANNEL == mode) {
        if (x_desc.GetPackMode()) {
            // total threads is equal to n*h*w or n*d*h*w
            for (int idx = 0; idx < nb_dims; idx++) {
                if (idx != 1) {
                    threads_total_x = threads_total_x * dim_a[idx];
                }
            }
            featuremap_size = threads_total_x / dim_a[0];

            threads_perblk_x = std::min(threads_total_x, kMaxThreadNbPerBlock);
            nblocks_x        = (threads_total_x + threads_perblk_x - 1) / threads_perblk_x;

            dim3 gridSize(nblocks_x, 1, 1);
            dim3 blockSize(threads_perblk_x, 1, 1);
            kernel::CudnnSoftMaxModeChlFwdKernel<<<gridSize, blockSize>>>(x,
                                                                          *alpha,
                                                                          *beta,
                                                                          y,
                                                                          dim_a[1],
                                                                          featuremap_size,
                                                                          stride_a[0],
                                                                          stride_a[1],
                                                                          stride_a[nb_dims - 1]);
        } else {
            // TODO(fbh): not support unpacked now
        }

    } else if (DNN_SOFTMAX_MODE_INSTANCE == mode) {
        /*to do softmax mode instance*/
    }
}

void CuSoftmaxBackward(CuHandle& handle,
                       cudnnSoftmaxAlgorithm_t algo,
                       cudnnSoftmaxMode_t mode,
                       const void* alpha,
                       const CuTensorDescriptor& y_desc,
                       const void* y,
                       const CuTensorDescriptor& dy_desc,
                       const void* dy,
                       const void* beta,
                       const CuTensorDescriptor& dx_desc,
                       void* dx) {
    cudnnDataType_t data_type;

    data_type = dx_desc.GetDataType();

    if (DNN_DATA_FLOAT == data_type) {
        const float* ptr_y     = reinterpret_cast<const float*>(y);
        const float* ptr_alpha = reinterpret_cast<const float*>(alpha);
        const float* ptr_beta  = reinterpret_cast<const float*>(beta);
        const float* ptr_dy    = reinterpret_cast<const float*>(dy);
        float* ptr_dx          = reinterpret_cast<float*>(dx);

        LaunchSoftmaxBackward(
            algo, mode, ptr_alpha, y_desc, ptr_y, dy_desc, ptr_dy, ptr_beta, dx_desc, ptr_dx);

    } else if (DNN_DATA_HALF == data_type) {
        /** to do half addtensor*/
    } else if (DNN_DATA_INT8 == data_type) {
        /** to do int8 addtensor*/
    } else {
    }
}

template <typename T1, typename T2>
void LaunchSoftmaxBackward(cudnnSoftmaxAlgorithm_t algo,
                           cudnnSoftmaxMode_t mode,
                           const T2* alpha,
                           const CuTensorDescriptor& y_desc,
                           const T1* y,
                           const CuTensorDescriptor& dy_desc,
                           const T1* dy,
                           const T2* beta,
                           const CuTensorDescriptor& dx_desc,
                           T1* dx) {
    cudnnDataType_t data_type;
    int nb_dims;
    int dim_a[DNN_DIM_MAX];
    int stride_a[DNN_DIM_MAX];
    dx_desc.Get(&data_type, &nb_dims, dim_a, stride_a);

    int featuremap_size  = 0;
    int threads_total_x  = 1;
    int threads_perblk_x = 1;
    int nblocks_x        = 1;

    int output_n = y_desc.GetN();
    int channel  = y_desc.GetC();

    int output_h = y_desc.GetH();
    int output_w = y_desc.GetW();

    int spatial_dim = output_h * output_w;
    int grid_size   = output_n * spatial_dim;

    if (DNN_SOFTMAX_MODE_CHANNEL == mode) {
        if (dx_desc.GetPackMode()) {
            // total threads is equal to n*h*w or n*d*h*w
            size_t grid_x =
                (mode == 0) ? output_n : std::min(output_n * output_h * output_w, 64 * 40 * 8);
            dim3 gridSize(grid_x, 1, 1);
            dim3 blockSize(256, 1, 1);

            kernel::CudnnSoftmaxModeChlBwdKernel<<<gridSize, blockSize>>>(
                dy, y, dx, grid_size, spatial_dim, channel, *alpha, *beta, algo);
        } else {
            // TODO(fbh): not support unpacked now
        }

    } else if (DNN_SOFTMAX_MODE_INSTANCE == mode) {
        /*to do softmax mode instance*/
    }
}

}  // namespace impl
}  // namespace cudnn
