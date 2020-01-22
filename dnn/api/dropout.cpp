/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_dropout_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>
#include <cudnn/impl/kernel/rand_generator.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>

using cudnn::GetLogger;
using cudnn::CuException;
using cudnn::Try;
using cudnn::api::CheckNull;
using cudnn::impl::Deref;
using cudnn::impl::CuDropoutDescriptor;

extern "C" {
cudnnStatus_t DNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropout_desc) {
    return Try([&] {
        if (dropout_desc == nullptr) {
            GetLogger()->info("dropout_desc is nullptr");
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        *dropout_desc = new CuDropoutDescriptor();
    });
}

cudnnStatus_t DNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropout_desc) {
    return Try([&] { delete dropout_desc; });
}

cudnnStatus_t DNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t* sizeInBytes) {
    return Try(
        [&] { *sizeInBytes = (sizeof(ixrand_state_xorwow) * 5120); });
    return DNN_STATUS_SUCCESS;
}

cudnnStatus_t DNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc,
                                                          size_t* sizeInBytes) {
    size_t tensorSize = Deref(xdesc).GetSizeInBytes();
    cudnnDataType_t in_data_type;
    in_data_type = Deref(xdesc).GetDataType();
    if (DNN_DATA_FLOAT == in_data_type) {
        *sizeInBytes = tensorSize;
    } else {
        /* NOT SUPPORT*/
    }
    return DNN_STATUS_SUCCESS;
}

cudnnStatus_t DNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropout_desc,
                                                    cudnnHandle_t handle,
                                                    float dropout,
                                                    void* states,
                                                    size_t state_size_in_bytes,
                                                    unsigned long long seed) {
    return Try([&] {
        Deref(dropout_desc).Set(Deref(handle), dropout, states, state_size_in_bytes, seed);
    });
}

cudnnStatus_t DNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropout_desc,
                                                        cudnnHandle_t handle,
                                                        float dropout,
                                                        void* states,
                                                        size_t state_size_in_bytes,
                                                        unsigned long long seed) {
    return Try([&] {
        Deref(dropout_desc).Restore(Deref(handle), dropout, states, state_size_in_bytes, seed);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropout_desc,
                                                    cudnnHandle_t handle,
                                                    float* dropout,
                                                    void** states,
                                                    unsigned long long* seed) {
    return Try([&] {
        CheckNull(dropout, states, seed);
        size_t state_size_in_bytes;
        std::tie(*dropout, *states, state_size_in_bytes, *seed) = Deref(dropout_desc).Get();
    });
}

cudnnStatus_t DNNWINAPI cudnnDropoutForward(cudnnHandle_t handle,
                                              const cudnnDropoutDescriptor_t dropout_desc,
                                              const cudnnTensorDescriptor_t xdesc,
                                              const void* x,
                                              const cudnnTensorDescriptor_t ydesc,
                                              void* y,
                                              void* reserve_space,
                                              size_t reserve_space_size_in_bytes) {
    void *states;
    float dropout;
    Deref(dropout_desc).Get(&states, &dropout);
    return Try([&] {
        if (x == nullptr || y == nullptr || reserve_space == nullptr || states == NULL) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuDropoutForward(Deref(handle),
                         Deref(xdesc),
                         Deref(dropout_desc),
                         x,
                         Deref(ydesc),
                         y,
                         reserve_space,
                         reserve_space_size_in_bytes);
    });
    return DNN_STATUS_SUCCESS;
}

cudnnStatus_t DNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle,
                                               const cudnnDropoutDescriptor_t dropout_desc,
                                               const cudnnTensorDescriptor_t dydesc,
                                               const void* dy,
                                               const cudnnTensorDescriptor_t dxdesc,
                                               void* dx,
                                               void* reserve_space,
                                               size_t reserve_space_size_in_bytes) {
    return Try([&] {
        if (dx == nullptr || dy == nullptr || reserve_space == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuDropoutBackward(Deref(handle),
                          Deref(dropout_desc),
                          Deref(dydesc),
                          dy,
                          Deref(dxdesc),
                          dx,
                          reserve_space,
                          reserve_space_size_in_bytes);
    });
    return DNN_STATUS_SUCCESS;
}
}
