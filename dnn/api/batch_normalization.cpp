/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_batch_norm.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

#include <iostream>
#include <vector>

using cudnn::Try;
using cudnn::api::CheckDataTypeDiffer;
using cudnn::api::CheckDimensionDiffer;
using cudnn::api::CheckIs4dOr5dTensor;
using cudnn::api::CheckNull;
using cudnn::impl::Deref;
using cudnn::impl::CuTensorDescriptor;

namespace {
void CheckBnScaleBiasMeanVarDesc(const cudnn::impl::CuTensorDescriptor& x_desc,
                                 const cudnn::impl::CuTensorDescriptor& bn_desc,
                                 cudnnBatchNormMode_t mode) {
    const auto nb_dims = x_desc.GetNbDims();
    const auto n       = x_desc.GetDim(1);
    const auto c       = x_desc.GetDim(2);
    const auto d       = x_desc.GetDim(nb_dims - 2);
    const auto h       = x_desc.GetDim(nb_dims - 1);
    const auto w       = x_desc.GetDim(nb_dims);
    const auto bn_dim  = bn_desc.GetDim();

    if (mode == DNN_BATCHNORM_PER_ACTIVATION) {
        if (nb_dims == 4) {
            if (bn_dim != std::vector<int>({1, c, h, w})) {
                throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
            }
        } else {
            if (bn_dim != std::vector<int>({1, c, d, h, w})) {
                throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
            }
        }
    } else {  // for spatial
        if (nb_dims == 4) {
            if (bn_dim != std::vector<int>({1, c, 1, 1})) {
                throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
            }
        } else {
            if (bn_dim != std::vector<int>({1, c, 1, 1, 1})) {
                throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
            }
        }
    }
}

void CheckSimultaneouslyNullOrNot(const void* first, const void* second) {
    if ((first == nullptr && second == nullptr) || (first != nullptr && second != nullptr)) {
        return;
    }
    throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
}

}  // namespace

extern "C" {
cudnnStatus_t DNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derived_bn_desc,
                                                        const cudnnTensorDescriptor_t x_desc,
                                                        cudnnBatchNormMode_t mode) {
    return Try([&]() {
        CHECK_RANGE(mode,
                    DNN_BATCHNORM_PER_ACTIVATION,
                    DNN_BATCHNORM_SPATIAL_PERSISTENT,
                    DNN_STATUS_BAD_PARAM);
        CheckIs4dOr5dTensor(Deref(x_desc));

        CuDeriveBnTensorDescriptor(Deref(derived_bn_desc), Deref(x_desc), mode);
    });
}

cudnnStatus_t DNNWINAPI
cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle,
                                       cudnnBatchNormMode_t mode,
                                       const void* alpha,
                                       const void* beta,
                                       const cudnnTensorDescriptor_t x_desc,
                                       const void* x,
                                       const cudnnTensorDescriptor_t y_desc,
                                       void* y,
                                       const cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
                                       const void* bn_scale,
                                       const void* bn_bias,
                                       double exponential_average_factor,
                                       void* result_running_mean,
                                       void* result_running_variance,
                                       double epsilon,
                                       void* result_save_mean,
                                       void* result_save_inv_variance) {
    return Try([&]() {
        CheckNull(alpha, beta, x, y, bn_scale, bn_bias);
        CHECK_RANGE(Deref(x_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(Deref(y_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CheckBnScaleBiasMeanVarDesc(Deref(x_desc), Deref(bn_scale_bias_mean_var_desc), mode);
        CheckSimultaneouslyNullOrNot(result_running_mean, result_running_variance);
        CheckSimultaneouslyNullOrNot(result_save_mean, result_save_inv_variance);
        CHECK_LOWER_BOUND(epsilon, DNN_BN_MIN_EPSILON, DNN_STATUS_BAD_PARAM);
        CheckDataTypeDiffer(Deref(x_desc), Deref(y_desc));
        CheckDimensionDiffer(Deref(x_desc), Deref(y_desc));

        CuBatchNormForwardTraining(Deref(handle),
                                   mode,
                                   alpha,
                                   beta,
                                   Deref(x_desc),
                                   x,
                                   Deref(y_desc),
                                   y,
                                   Deref(bn_scale_bias_mean_var_desc),
                                   bn_scale,
                                   bn_bias,
                                   exponential_average_factor,
                                   result_running_mean,
                                   result_running_variance,
                                   epsilon,
                                   result_save_mean,
                                   result_save_inv_variance);
    });
}

cudnnStatus_t DNNWINAPI
cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
                                        cudnnBatchNormMode_t mode,
                                        const void* alpha,
                                        const void* beta,
                                        const cudnnTensorDescriptor_t x_desc,
                                        const void* x,
                                        const cudnnTensorDescriptor_t y_desc,
                                        void* y,
                                        const cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
                                        const void* bn_scale,
                                        const void* bn_bias,
                                        const void* estimated_mean,
                                        const void* estimated_variance,
                                        double epsilon) {
    return Try([&]() {
        CheckNull(alpha, beta, x, y, bn_scale, bn_bias, estimated_mean, estimated_variance);
        CHECK_RANGE(Deref(y_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CheckBnScaleBiasMeanVarDesc(Deref(x_desc), Deref(bn_scale_bias_mean_var_desc), mode);
        CHECK_LOWER_BOUND(epsilon, DNN_BN_MIN_EPSILON, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(Deref(x_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CheckDataTypeDiffer(Deref(x_desc), Deref(y_desc));
        CheckDimensionDiffer(Deref(x_desc), Deref(y_desc));

        CuBatchNormForwardInference(Deref(handle),
                                    mode,
                                    alpha,
                                    beta,
                                    Deref(x_desc),
                                    x,
                                    Deref(y_desc),
                                    y,
                                    Deref(bn_scale_bias_mean_var_desc),
                                    bn_scale,
                                    bn_bias,
                                    estimated_mean,
                                    estimated_variance,
                                    epsilon);
    });
}

cudnnStatus_t DNNWINAPI
cudnnBatchNormalizationBackward(cudnnHandle_t handle,
                                cudnnBatchNormMode_t mode,
                                const void* alpha_data_diff,
                                const void* beta_data_diff,
                                const void* alpha_param_diff,
                                const void* beta_param_diff,
                                const cudnnTensorDescriptor_t x_desc,
                                const void* x,
                                const cudnnTensorDescriptor_t dy_desc,
                                const void* dy,
                                const cudnnTensorDescriptor_t dx_desc,
                                void* dx,
                                const cudnnTensorDescriptor_t dbn_scale_bias_desc,
                                const void* bn_scale,
                                void* dbn_scale_result,
                                void* dbn_bias_result,
                                double epsilon,
                                const void* saved_mean,
                                const void* saved_inv_variance) {
    return Try([&]() {
        CheckNull(alpha_data_diff,
                  beta_data_diff,
                  alpha_param_diff,
                  beta_param_diff,
                  x,
                  dy,
                  dx,
                  bn_scale,
                  dbn_scale_result,
                  dbn_bias_result);
        CHECK_RANGE(Deref(x_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(Deref(dy_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(Deref(dx_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
        CheckBnScaleBiasMeanVarDesc(Deref(x_desc), Deref(dbn_scale_bias_desc), mode);
        CheckSimultaneouslyNullOrNot(saved_mean, saved_inv_variance);
        CHECK_LOWER_BOUND(epsilon, DNN_BN_MIN_EPSILON, DNN_STATUS_BAD_PARAM);
        CheckDataTypeDiffer(Deref(x_desc), Deref(dy_desc));
        CheckDimensionDiffer(Deref(x_desc), Deref(dy_desc));
        CheckDataTypeDiffer(Deref(x_desc), Deref(dx_desc));
        CheckDimensionDiffer(Deref(x_desc), Deref(dx_desc));

        CuBatchNormBackward(Deref(handle),
                            mode,
                            alpha_data_diff,
                            beta_data_diff,
                            alpha_param_diff,
                            beta_param_diff,
                            Deref(x_desc),
                            x,
                            Deref(dy_desc),
                            dy,
                            Deref(dx_desc),
                            dx,
                            Deref(dbn_scale_bias_desc),
                            bn_scale,
                            dbn_scale_result,
                            dbn_bias_result,
                            epsilon,
                            saved_mean,
                            saved_inv_variance);
    });
}

// cudnnStatus_t DNNWINAPI
// cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle,
//                                         cudnnBatchNormMode_t mode,
//                                         cudnnBatchNormOps_t bn_ops,
//                                         const void* alpha,
//                                         const void* beta,
//                                         const cudnnTensorDescriptor_t x_desc,
//                                         const void* x_data,
//                                         const cudnnTensorDescriptor_t z_desc,
//                                         const void* z_data,
//                                         const cudnnTensorDescriptor_t y_desc,
//                                         void* y_data,
//                                         const cudnnTensorDescriptor_t
//                                         bn_scale_bias_mean_var_desc, const void* bn_scale_data,
//                                         const void* bn_bias_data,
//                                         double exponential_average_factor,
//                                         void* result_running_mean_data,
//                                         void* result_running_variance_data,
//                                         double epsilon,
//                                         void* save_mean,
//                                         void* save_inv_variance,
//                                         cudnnActivationDescriptor_t activation_desc,
//                                         void* workspace,
//                                         size_t workspacesize_in_bytes,
//                                         void* reserve_space,
//                                         size_t reserve_spacesize_in_bytes) {
//    return Try([&] {
//        CheckNull(alpha, beta, x_data, z_data, y_data, bn_scale_data, bn_bias_data);
//        CHECK_RANGE(Deref(x_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
//        CHECK_RANGE(Deref(y_desc).GetNbDims(), 4, 5, DNN_STATUS_BAD_PARAM);
//        CheckBnScaleBiasMeanVarDesc(Deref(x_desc), Deref(bn_scale_bias_mean_var_desc), mode);
//        CheckSimultaneouslyNullOrNot(result_running_mean_data, result_running_variance_data);
//        CheckSimultaneouslyNullOrNot(save_mean, save_inv_variance);
//        CHECK_LOWER_BOUND(epsilon, DNN_BN_MIN_EPSILON, DNN_STATUS_BAD_PARAM);
//        CheckDataTypeDiffer(Deref(x_desc), Deref(y_desc));
//        CheckDimensionDiffer(Deref(x_desc), Deref(y_desc));
//
//        // TBD(Peter Han): this api isn't supported in tf1.13
//    });
//}
//
// cudnnStatus_t DNNWINAPI
// cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,
//                                  cudnnBatchNormMode_t mode,
//                                  cudnnBatchNormOps_t bn_ops,
//                                  const void* alpha_data_diff,
//                                  const void* beta_data_diff,
//                                  const void* alpha_param_diff,
//                                  const void* beta_param_diff,
//                                  const cudnnTensorDescriptor_t x_desc,
//                                  const void* x_data,
//                                  const cudnnTensorDescriptor_t y_desc,
//                                  const void* y_data,
//                                  const cudnnTensorDescriptor_t dy_desc,
//                                  const void* dy_data,
//                                  const cudnnTensorDescriptor_t dz_desc,
//                                  void* dz_data,
//                                  const cudnnTensorDescriptor_t dx_desc,
//                                  void* dx_data,
//                                  /* Shared tensor desc for the 4 tensors below */
//                                  const cudnnTensorDescriptor_t d_bn_scale_bias_desc,
//                                  const void* bn_scale_data,
//                                  const void* bn_bias_data, /* needed if there is activation */
//                                  void* d_bn_scale_data,
//                                  void* d_bn_bias_data,
//                                  double epsilon, /* Same epsilon as forward pass */
//                                  /* Optionally cached intermediate results from
//                                     forward pass */
//                                  const void* saved_mean,
//                                  const void* saved_inv_variance,
//                                  cudnnActivationDescriptor_t activation_desc,
//                                  void* workspace,
//                                  size_t workspace_size_in_bytes,
//                                  void* reserve_space,
//                                  size_t reserve_space_size_in_bytes) {
//    return Try([&] {
//        // TBD(Peter Han): this api isn't supported in tf1.13
//    });
//}
//
// cudnnStatus_t DNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
//    cudnnHandle_t handle,
//    cudnnBatchNormMode_t mode,
//    cudnnBatchNormOps_t bn_ops,
//    const cudnnTensorDescriptor_t x_desc,
//    const cudnnTensorDescriptor_t z_desc,
//    const cudnnTensorDescriptor_t y_desc,
//    const cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc,
//    const cudnnActivationDescriptor_t activation_desc,
//    size_t* size_in_bytes) {
//
//    return Try([&] {
//        // TBD(Peter Han): this api isn't supported in tf1.13
//    });
//}
//
// cudnnStatus_t DNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
//    cudnnHandle_t handle,
//    cudnnBatchNormMode_t mode,
//    cudnnBatchNormOps_t bn_ops,
//    const cudnnTensorDescriptor_t x_desc,
//    const cudnnTensorDescriptor_t y_desc,
//    const cudnnTensorDescriptor_t dy_desc,
//    const cudnnTensorDescriptor_t dz_desc,
//    const cudnnTensorDescriptor_t dx_desc,
//    const cudnnTensorDescriptor_t d_bn_scale_bias_desc,
//    const cudnnActivationDescriptor_t activation_desc,
//    size_t* size_in_bytes) {
//    return Try([&] {
//        // TBD(Peter Han): this api isn't supported in tf1.13
//    });
//}
//
// cudnnStatus_t DNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
//    cudnnHandle_t handle,
//    cudnnBatchNormMode_t mode,
//    cudnnBatchNormOps_t bn_ops,
//    const cudnnActivationDescriptor_t activation_desc,
//    const cudnnTensorDescriptor_t x_desc,
//    size_t* size_in_bytes) {
//    return Try([&] {
//        // TBD(Peter Han): this api isn't supported in tf1.13
//    });
//}
}
