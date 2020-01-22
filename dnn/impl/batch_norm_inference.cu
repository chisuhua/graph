/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_batch_norm.h>

#include <cudnn/impl/cudnn_launch_config.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/kernel/cudnn_batch_norm_inference.cuh>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <cudnn/cudnn_logger.h>

#include <tuple>

namespace cudnn {
namespace impl {

using cudnn::impl::kernel::kBatchNormInferenceTileSize;
using meta::CuMetaTensor;

#define BN_INFERENCE_CASE_PER_ACTIVATION(data_type)                                         \
    case data_type:                                                                         \
        kernel::BatchNormInferencePerActivation<CuTensorType<data_type>::Type,              \
                                                CuTensorType<data_type>::ScalingType>       \
            <<<grid, block, 0, stream>>>(                                                   \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha),      \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),       \
                x_desc.GetDimN(),                                                           \
                x_desc.GetDimC(),                                                           \
                x_desc.GetDimD(),                                                           \
                x_desc.GetDimH(),                                                           \
                x_desc.GetDimW(),                                                           \
                x_desc.GetStrideN(),                                                        \
                x_desc.GetStrideC(),                                                        \
                x_desc.GetStrideD(),                                                        \
                x_desc.GetStrideH(),                                                        \
                x_desc.GetStrideW(),                                                        \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),                  \
                y_desc.GetStrideN(),                                                        \
                y_desc.GetStrideC(),                                                        \
                y_desc.GetStrideD(),                                                        \
                y_desc.GetStrideH(),                                                        \
                y_desc.GetStrideW(),                                                        \
                reinterpret_cast<CuTensorType<data_type>::Type*>(y),                        \
                bn_scale_bias_mean_var_desc.GetStrideC(),                                   \
                bn_scale_bias_mean_var_desc.GetStrideD(),                                   \
                bn_scale_bias_mean_var_desc.GetStrideH(),                                   \
                bn_scale_bias_mean_var_desc.GetStrideW(),                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(bn_scale),           \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(bn_bias),            \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(estimated_mean),     \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(estimated_variance), \
                epsilon);                                                                   \
        break

#define BN_INFERENCE_CASE_SPATIAL(data_type)                                                \
    case data_type:                                                                         \
        kernel::BatchNormInferenceSpatial<CuTensorType<data_type>::Type,                    \
                                          CuTensorType<data_type>::ScalingType>             \
            <<<grid, block, 0, stream>>>(                                                   \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha),      \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),       \
                x_desc.GetDimN(),                                                           \
                x_desc.GetDimC(),                                                           \
                x_desc.GetDimD(),                                                           \
                x_desc.GetDimH(),                                                           \
                x_desc.GetDimW(),                                                           \
                x_desc.GetStrideN(),                                                        \
                x_desc.GetStrideC(),                                                        \
                x_desc.GetStrideD(),                                                        \
                x_desc.GetStrideH(),                                                        \
                x_desc.GetStrideW(),                                                        \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),                  \
                y_desc.GetStrideN(),                                                        \
                y_desc.GetStrideC(),                                                        \
                y_desc.GetStrideD(),                                                        \
                y_desc.GetStrideH(),                                                        \
                y_desc.GetStrideW(),                                                        \
                reinterpret_cast<CuTensorType<data_type>::Type*>(y),                        \
                bn_scale_bias_mean_var_desc.GetStrideC(),                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(bn_scale),           \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(bn_bias),            \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(estimated_mean),     \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(estimated_variance), \
                epsilon);                                                                   \
        break

void CuBatchNormForwardInference(const CuHandle& handle,
                                 cudnnBatchNormMode_t mode,
                                 const void* alpha,
                                 const void* beta,
                                 const CuMetaTensor& x_desc,
                                 const void* x,
                                 const CuMetaTensor& y_desc,
                                 void* y,
                                 const CuMetaTensor& bn_scale_bias_mean_var_desc,
                                 const void* bn_scale,
                                 const void* bn_bias,
                                 const void* estimated_mean,
                                 const void* estimated_variance,
                                 double epsilon) {
    dim3 grid;
    dim3 block;
    std::tie(grid, block) = CalcLaunchConfig(x_desc, kBatchNormInferenceTileSize);
    auto stream           = handle.GetStream();
    cudnn::GetLogger()->info("{} grid={}, block={}", __func__, grid, block);

    if (mode == DNN_BATCHNORM_PER_ACTIVATION) {
        switch (x_desc.GetDataType()) {
            BN_INFERENCE_CASE_PER_ACTIVATION(DNN_DATA_FLOAT);
            // BN_INFERENCE_CASE_PER_ACTIVATION(DNN_DATA_HALF);
            // BN_INFERENCE_CASE_PER_ACTIVATION(DNN_DATA_INT8);
            // BN_INFERENCE_CASE_PER_ACTIVATION(DNN_DATA_UINT8);
            // BN_INFERENCE_CASE_PER_ACTIVATION(DNN_DATA_INT32);
        }
    } else {
        switch (x_desc.GetDataType()) {
            BN_INFERENCE_CASE_SPATIAL(DNN_DATA_FLOAT);
            // BN_INFERENCE_CASE_SPATIAL(DNN_DATA_HALF);
            // BN_INFERENCE_CASE_SPATIAL(DNN_DATA_INT8);
            // BN_INFERENCE_CASE_SPATIAL(DNN_DATA_UINT8);
            // BN_INFERENCE_CASE_SPATIAL(DNN_DATA_INT32);
        }
    }
}

}  // namespace impl
}  // namespace cudnn
