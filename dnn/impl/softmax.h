/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

namespace cudnn {
namespace impl {

/**
 * @brief cudnnSoftmaxForward implementation function
 */
void CuSoftmaxForward(CuHandle& handle,
                      cudnnSoftmaxAlgorithm_t algorithm,
                      cudnnSoftmaxMode_t mode,
                      const void* alpha,
                      const CuTensorDescriptor& x_desc,
                      const void* x,
                      const void* beta,
                      const CuTensorDescriptor& y_desc,
                      void* y);

template <class T>
void LaunchSoftMaxFwdKernel(cudnnSoftmaxAlgorithm_t algorithm,
                            cudnnSoftmaxMode_t mode,
                            const T* alpha,
                            const CuTensorDescriptor& x_desc,
                            const T* x,
                            const T* beta,
                            const CuTensorDescriptor& y_desc,
                            T* y);

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
                       void* dx);

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
                           T1* dx);

}  // namespace impl
}  // namespace cudnn
