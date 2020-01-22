/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_lrn_descriptor.h>
#include <cudnn/impl/kernel/cudnn_lrn_descriptor.cuh>

#include <algorithm>

namespace cudnn {
namespace impl {

void CuLRNCrossChannelForward(CuHandle handle,
                              const CuLRNDescriptor& norm_desc,
                              const void* alpha,
                              const CuTensorDescriptor& x_desc,
                              const void* in_x,
                              const void* beta,
                              const CuTensorDescriptor& y_desc,
                              void* out_y) {
    cudnnDataType_t in_data_type;
    unsigned lrn_n_;
    double   lrn_alpha_;
    double   lrn_beta_;
    double   lrn_k_;

    norm_desc.Get(&lrn_n_, &lrn_alpha_, &lrn_beta_, &lrn_k_);

    in_data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == in_data_type) {
        const float* x        = reinterpret_cast<const float*>(in_x);
        const float* alpha_   = reinterpret_cast<const float*>(alpha);
        const float* beta_    = reinterpret_cast<const float*>(beta);
        float* y              = reinterpret_cast<float*>(out_y);
        const int lrn_n       = static_cast<int>(lrn_n_);
        const float lrn_alpha = static_cast<float>(lrn_alpha_);
        const float lrn_beta  = static_cast<float>(lrn_beta_);
        const float lrn_k     = static_cast<float>(lrn_k_);
        LaunchLRNCrossChannelFwdKernel(x_desc, x, lrn_alpha, lrn_beta,
                                       lrn_k, lrn_n, alpha_, beta_, y_desc, y);
    } else if (DNN_DATA_HALF == in_data_type) {
        /** to do half lrn */
    } else {
        /*NOT SUPPORT*/
    }
}

template <class T>
void LaunchLRNCrossChannelFwdKernel(const CuTensorDescriptor& x_desc,
                                    const T* x,
                                    const T lrn_alpha,
                                    const T lrn_beta,
                                    const T lrn_k,
                                    const int lrn_n,
                                    const float* alpha,
                                    const float* beta,
                                    const CuTensorDescriptor& y_desc,
                                    T* y) {
    int dims_a[DNN_DIM_MAX];
    int strides_a[DNN_DIM_MAX];
    int n_dim;
    cudnnDataType_t in_data_type;

    x_desc.Get(
        DNN_DIM_MAX, &in_data_type, &n_dim, dims_a, strides_a);

    int blocks_x = dims_a[0];
    int blocks_y = dims_a[1];
    int blocks_z;
    int threads_x;
    int muti_hw;

    // lrn only support 4 or 5 dimension if dim = 4 w*h threads if dim = 5 w*h*d threads
    if (n_dim == 4) {
        muti_hw = dims_a[2] * dims_a[3];
    } else if (n_dim == 5) {
        muti_hw = dims_a[2] * dims_a[3] * dims_a[4];
    } else {
        /*NOT SUPPORT*/
    }

    if (muti_hw <= kMaxThreadNbPerBlock) {
        threads_x = muti_hw;
        blocks_z = 1;
    } else {
        threads_x = kMaxThreadNbPerBlock;
        blocks_z = (muti_hw % kMaxThreadNbPerBlock == 0) ? muti_hw/kMaxThreadNbPerBlock : muti_hw/kMaxThreadNbPerBlock + 1;
    }

    dim3 gridSize(blocks_x, blocks_y, blocks_z);
    dim3 blockSize(threads_x, 1, 1);

    kernel::CudnnLRNCrossChannelFwdKernel<<<gridSize, blockSize>>>(x,
                                                                   *alpha,
                                                                   *beta,
                                                                   y,
                                                                   lrn_n,
                                                                   lrn_alpha,
                                                                   lrn_beta,
                                                                   lrn_k,
                                                                   strides_a[2],
                                                                   strides_a[1],
                                                                   strides_a[0],
                                                                   dims_a[2]);
}
void CuLRNCrossChannelBackward(CuHandle handle,
                               const CuLRNDescriptor& norm_desc,
                               const void* alpha,
                               const CuTensorDescriptor& y_desc,
                               const void* in_y,
                               const CuTensorDescriptor& dy_desc,
                               const void* in_dy,
                               const CuTensorDescriptor& x_desc,
                               const void* in_x,
                               const void* beta,
                               const CuTensorDescriptor& dx_desc,
                               void* out_dx) {
    cudnnDataType_t in_data_type;
    unsigned lrn_n_;
    double   lrn_alpha_;
    double   lrn_beta_;
    double   lrn_k_;

    norm_desc.Get(&lrn_n_, &lrn_alpha_, &lrn_beta_, &lrn_k_);

    in_data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == in_data_type) {
        const float* x        = reinterpret_cast<const float*>(in_x);
        const float* alpha_   = reinterpret_cast<const float*>(alpha);
        const float* beta_    = reinterpret_cast<const float*>(beta);
        const float* y        = reinterpret_cast<const float*>(in_y);
        const float* dy       = reinterpret_cast<const float*>(in_dy);
        float *dx             = reinterpret_cast<float*>(out_dx);
        const int lrn_n       = static_cast<int>(lrn_n_);
        const float lrn_alpha = static_cast<float>(lrn_alpha_);
        const float lrn_beta  = static_cast<float>(lrn_beta_);
        const float lrn_k     = static_cast<float>(lrn_k_);
        LaunchLRNCrossChannelBwdKernel(lrn_alpha, lrn_beta, lrn_k, lrn_n, alpha_,
                                       y_desc, y, dy, beta_, x_desc, x, dx);
    } else if (DNN_DATA_HALF == in_data_type) {
        /** to do half lrn */
    } else {
        /*NOT SUPPORT*/
    }
}

template <class T>
void LaunchLRNCrossChannelBwdKernel(const T lrn_alpha,
                                    const T lrn_beta,
                                    const T lrn_k,
                                    const int lrn_n,
                                    const float* alpha,
                                    const CuTensorDescriptor& y_desc,
                                    const T* y,
                                    const T* dy,
                                    const float* beta,
                                    const CuTensorDescriptor& x_desc,
                                    const T* x,
                                    T* dx) {
    int dims_a[DNN_DIM_MAX];
    int strides_a[DNN_DIM_MAX];
    int n_dim;
    cudnnDataType_t in_data_type;

    x_desc.Get(
        DNN_DIM_MAX, &in_data_type, &n_dim, dims_a, strides_a);

    int blocks_x = dims_a[0];
    int blocks_y = dims_a[1];
    int blocks_z;
    int threads_x;
    int muti_hw;

    // lrn only support 4 or 5 dimension if dim = 4 w*h threads if dim = 5 w*h*d threads
    if (n_dim == 4) {
        muti_hw = dims_a[2] * dims_a[3];
    } else if (n_dim == 5) {
        muti_hw = dims_a[2] * dims_a[3] * dims_a[4];
    } else {
        /*NOT SUPPORT*/
    }
    if (lrn_n > 6) {
        if (muti_hw <= (kMaxThreadNbPerBlock >> 2)) {
            threads_x = muti_hw;
            blocks_z = 1;
        } else {
            threads_x = (kMaxThreadNbPerBlock >> 2);
            blocks_z = (muti_hw % threads_x == 0) ? muti_hw/threads_x : muti_hw/threads_x + 1;
        }
    } else {
        if (muti_hw <= (kMaxThreadNbPerBlock >> 1)) {
            threads_x = muti_hw;
            blocks_z = 1;
        } else {
            threads_x = (kMaxThreadNbPerBlock >> 1);
            blocks_z = (muti_hw % threads_x == 0) ? muti_hw/threads_x : muti_hw/threads_x + 1;
        }
    }

    dim3 gridSize(blocks_x, blocks_y, blocks_z);
    dim3 blockSize(threads_x, 1, 1);

    // shared memory size
    int tile = ((lrn_n << 2) - 1) * threads_x;

    kernel::CudnnLRNCrossChannelBwdKernel<<<gridSize, blockSize, tile * sizeof(T)>>>(x,
                                                                                     *alpha,
                                                                                     *beta,
                                                                                     y,
                                                                                     dy,
                                                                                     dx,
                                                                                     lrn_n,
                                                                                     lrn_alpha,
                                                                                     lrn_beta,
                                                                                     lrn_k,
                                                                                     strides_a[0],
                                                                                     strides_a[1],
                                                                                     strides_a[2],
                                                                                     dims_a[2],
                                                                                     dims_a[1]);
}

}  // namespace impl
}  // namespace cudnn
