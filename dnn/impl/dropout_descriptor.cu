/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_dropout_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/kernel/cudnn_dropout.cuh>

#include <algorithm>

namespace cudnn {
namespace impl {
void CuDropoutForward(CuHandle& handle,
                      const CuTensorDescriptor& x_desc,
                      const CuDropoutDescriptor& dropout_desc,
                      const void* in_x,
                      const CuTensorDescriptor& y_desc,
                      void* out_y,
                      void* reserveSpace,
                      size_t reserveSpaceSizeInBytes) {
    cudaStream_t stream = handle.GetStream();
    void* states;
    float dropout_ratio;
    dropout_desc.Get(&states, &dropout_ratio);
    int tensorSize = static_cast<int>(x_desc.GetSizeInBytes());
    cudnnDataType_t in_data_type;
    in_data_type = x_desc.GetDataType();

    if (DNN_DATA_FLOAT == in_data_type) {
        const float* x       = reinterpret_cast<const float*>(in_x);
        float* y             = reinterpret_cast<float*>(out_y);
        float* freserveSpace = reinterpret_cast<float*>(reserveSpace);
        LaunchDropoutFwdKernel(
            stream, x, y, freserveSpace, states, (tensorSize >> 2), dropout_ratio);
    } else if (DNN_DATA_HALF == in_data_type) {
        /** to do half lrn */
    } else {
        /*NOT SUPPORT*/
    }
}

void CuDropoutBackward(CuHandle& handle,
                       const CuDropoutDescriptor& dropout_desc,
                       const CuTensorDescriptor& dy_desc,
                       const void* in_dy,
                       const CuTensorDescriptor& dx_desc,
                       void* out_dx,
                       void* reserveSpace,
                       size_t reserveSpaceSizeInBytes) {
    cudaStream_t stream = handle.GetStream();
    void* states;
    float dropout_ratio;
    dropout_desc.Get(&states, &dropout_ratio);
    int tensorSize = static_cast<int>(dx_desc.GetSizeInBytes());
    cudnnDataType_t in_data_type;
    in_data_type = dx_desc.GetDataType();

    if (DNN_DATA_FLOAT == in_data_type) {
        float* dx            = reinterpret_cast<float*>(out_dx);
        const float* dy      = reinterpret_cast<const float*>(in_dy);
        float* freserveSpace = reinterpret_cast<float*>(reserveSpace);
        LaunchDropoutBwdKernel(stream, dy, dx, freserveSpace, (tensorSize >> 2), dropout_ratio);
    } else if (DNN_DATA_HALF == in_data_type) {
        /** to do half lrn */
    } else {
        /*NOT SUPPORT*/
    }
}

void randstate_init(void* states, unsigned long long seed) {
    // the whole thread number is 5120, match the allocated state number
    kernel::randstate_init_kernel<<<20, 256>>>(states, seed);
}

template <class T>
void LaunchDropoutFwdKernel(cudaStream_t& stream,
                            const T* x,
                            T* y,
                            float* reserveSpace,
                            void* states,
                            int tensorLen,
                            float dropout_ratio) {
    // make sure only maxmimum 5120 threads are launched
    int gridNum   = (tensorLen > 5120) ? 5120 : tensorLen;
    int grid_size = (gridNum + kMaxThreadNbPerBlock - 1) / kMaxThreadNbPerBlock;
    kernel::dropout_fwd_kernel<T><<<grid_size, kMaxThreadNbPerBlock, 0, stream>>>(
        tensorLen, x, y, reserveSpace, states, dropout_ratio);
}

template <typename T>
void LaunchDropoutBwdKernel(
    cudaStream_t& stream, const T* dy, T* dx, float* reserveSpace, int tensorLen, float dropout) {
    int grid_size = (tensorLen + kMaxThreadNbPerBlock - 1) / kMaxThreadNbPerBlock;
    kernel::dropout_bwd_kernel<T>
        <<<grid_size, kMaxThreadNbPerBlock, 0, stream>>>(tensorLen, dy, dx, reserveSpace, dropout);
}

}  // namespace impl
}  // namespace cudnn
