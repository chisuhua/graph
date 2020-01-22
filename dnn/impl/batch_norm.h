/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

namespace cudnn {
namespace impl {
void CuDeriveBnTensorDescriptor(CuTensorDescriptor& bn_desc,
                                const CuTensorDescriptor& x_desc,
                                cudnnBatchNormMode_t mode);

void CuBatchNormForwardTraining(const CuHandle& handle,
                                cudnnBatchNormMode_t mode,
                                const void* alpha,
                                const void* beta,
                                const meta::CuMetaTensor& x_desc,
                                const void* x,
                                const meta::CuMetaTensor& y_desc,
                                void* y,
                                const meta::CuMetaTensor& bn_scale_bias_mean_var_desc,
                                const void* bn_scale,
                                const void* bn_bias,
                                double exponential_average_factor,
                                void* running_mean,
                                void* running_variance,
                                double epsilon,
                                void* save_mean,
                                void* save_inv_variance);

void CuBatchNormForwardInference(const CuHandle& handle,
                                 cudnnBatchNormMode_t mode,
                                 const void* alpha,
                                 const void* beta,
                                 const meta::CuMetaTensor& x_desc,
                                 const void* x,
                                 const meta::CuMetaTensor& y_desc,
                                 void* y,
                                 const meta::CuMetaTensor& bn_scale_bias_mean_var_desc,
                                 const void* bn_scale,
                                 const void* bn_bias,
                                 const void* estimated_mean,
                                 const void* estimated_variance,
                                 double epsilon);

void CuBatchNormBackward(const CuHandle& handle,
                         cudnnBatchNormMode_t mode,
                         const void* alpha_data_diff,
                         const void* beta_data_diff,
                         const void* alpha_param_diff,
                         const void* beta_param_diff,
                         const meta::CuMetaTensor& x_desc,
                         const void* x,
                         const meta::CuMetaTensor& dy_desc,
                         const void* dy,
                         const meta::CuMetaTensor& dx_desc,
                         void* dx,
                         const meta::CuMetaTensor& dbn_scale_bias_desc,
                         const void* bn_scale,
                         void* dbn_scale_result,
                         void* dbn_bias_result,
                         double epsilon,
                         const void* saved_mean,
                         const void* saved_inv_variance);
}  // namespace impl
}  // namespace cudnn
