/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_batch_norm.h>

#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_device_memory.h>
#include <cudnn/impl/cudnn_launch_config.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/kernel/cudnn_batch_norm_training_forward_per_activation.cuh>
#include <cudnn/impl/kernel/cudnn_batch_norm_training_forward_spatial.cuh>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>
#include <cudnn/cudnn_logger.h>
#include <cudnn/cudnn_spdlog_patch.h>

#include <tuple>

namespace {
using cudnn::impl::kMaxBlockDimension;

std::tuple<int, int> CalcLaunchConfigSpatialFactor(const cudnn::impl::meta::CuMetaTensor& x_desc) {
    int grid_dim    = 1;
    int nb_channels = x_desc.GetDim(2);
    int block_dim   = nb_channels;

    if (nb_channels > kMaxBlockDimension.x) {
        block_dim = kMaxBlockDimension.x;
    }
    grid_dim = (nb_channels + block_dim - 1) / block_dim;
    return std::make_tuple(grid_dim, block_dim);
}

}  // namespace

namespace cudnn {
namespace impl {

using meta::CuMetaTensor;
using param::CuParamConstTensor;
using param::CuParamScaling;
using param::CuParamTensor;

#define BN_TRAINING_FWD_CASE_PER_ACTIVATION(data_type)                                       \
    case data_type:                                                                          \
        do {                                                                                 \
            using Type        = CuTensorType<data_type>::Type;                               \
            using ScalingType = CuTensorType<data_type>::ScalingType;                        \
            kernel::BatchNormTrainingForwardPerActivation<Type, ScalingType>                 \
                <<<grid, block, 0, stream>>>(*reinterpret_cast<const ScalingType*>(alpha),   \
                                             *reinterpret_cast<const ScalingType*>(beta),    \
                                             x_desc.GetDimN(),                               \
                                             x_desc.GetDimC(),                               \
                                             x_desc.GetDimD(),                               \
                                             x_desc.GetDimH(),                               \
                                             x_desc.GetDimW(),                               \
                                             x_desc.GetStrideN(),                            \
                                             x_desc.GetStrideC(),                            \
                                             x_desc.GetStrideD(),                            \
                                             x_desc.GetStrideH(),                            \
                                             x_desc.GetStrideW(),                            \
                                             reinterpret_cast<const Type*>(x),               \
                                             y_desc.GetStrideN(),                            \
                                             y_desc.GetStrideC(),                            \
                                             y_desc.GetStrideD(),                            \
                                             y_desc.GetStrideH(),                            \
                                             y_desc.GetStrideW(),                            \
                                             reinterpret_cast<Type*>(y),                     \
                                             bn_scale_bias_mean_var_desc.GetStrideC(),       \
                                             bn_scale_bias_mean_var_desc.GetStrideD(),       \
                                             bn_scale_bias_mean_var_desc.GetStrideH(),       \
                                             bn_scale_bias_mean_var_desc.GetStrideW(),       \
                                             reinterpret_cast<const Type*>(bn_scale),        \
                                             reinterpret_cast<const Type*>(bn_bias),         \
                                             static_cast<float>(exponential_average_factor), \
                                             reinterpret_cast<Type*>(running_mean),          \
                                             reinterpret_cast<Type*>(running_variance),      \
                                             static_cast<float>(epsilon),                    \
                                             reinterpret_cast<Type*>(save_mean),             \
                                             reinterpret_cast<Type*>(save_inv_variance));    \
        } while (false);                                                                     \
        break

#define BN_TRAINING_FWD_CASE_SPATIAL(data_type)                                                  \
    case data_type: {                                                                            \
        using TensorType                                       = CuParamTensor<data_type>::Type; \
        std::shared_ptr<CuDeviceMemory<TensorType>> smart_pool = nullptr;                        \
        if (running_mean == nullptr) {                                                           \
            const auto len   = x_desc.GetDim(2);                                                 \
            smart_pool       = std::make_shared<CuDeviceMemory<TensorType>>(len * 2, stream);    \
            TensorType* pool = &*smart_pool;                                                     \
            running_mean     = &pool[len * 0];                                                   \
            running_variance = &pool[len * 1];                                                   \
        }                                                                                        \
        kernel::BatchNormTraiingForwardSpatialFactorPreviousMeanVariance<data_type>              \
            <<<grid_dim, block_dim, 0, stream>>>(                                                \
                CuParamTensor<data_type>(bn_scale_bias_mean_var_desc, running_mean),             \
                CuParamTensor<data_type>(bn_scale_bias_mean_var_desc, running_variance),         \
                exponential_average_factor);                                                     \
        kernel::BatchNormTrainingForwardSpatialCalcMean<data_type><<<grid, block, 0, stream>>>(  \
            CuParamConstTensor<data_type>(static_cast<CuMetaTensor>(x_desc), x),                 \
            exponential_average_factor,                                                          \
            CuParamTensor<data_type>(bn_scale_bias_mean_var_desc, running_mean),                 \
            CuParamTensor<data_type>(bn_scale_bias_mean_var_desc, save_mean));                   \
        kernel::BatchNormTrainingForwardSpatialCalcVariance<data_type>                           \
            <<<grid, block, 0, stream>>>(                                                        \
                CuParamConstTensor<data_type>(static_cast<CuMetaTensor>(x_desc), x),             \
                exponential_average_factor,                                                      \
                CuParamConstTensor<data_type>(bn_scale_bias_mean_var_desc, running_mean),        \
                CuParamTensor<data_type>(bn_scale_bias_mean_var_desc, running_variance));        \
        kernel::BatchNormTrainingForwardSpatialNormalScaleShift<data_type>                       \
            <<<grid, block, 0, stream>>>(                                                        \
                CuParamScaling<data_type>(alpha, beta),                                          \
                CuParamConstTensor<data_type>(static_cast<CuMetaTensor>(x_desc), x),             \
                CuParamTensor<data_type>(static_cast<CuMetaTensor>(y_desc), y),                  \
                CuParamConstTensor<data_type>(bn_scale_bias_mean_var_desc, bn_scale),            \
                CuParamConstTensor<data_type>(bn_scale_bias_mean_var_desc, bn_bias),             \
                CuParamConstTensor<data_type>(bn_scale_bias_mean_var_desc, running_mean),        \
                CuParamConstTensor<data_type>(bn_scale_bias_mean_var_desc, running_variance),    \
                epsilon,                                                                         \
                CuParamTensor<data_type>(bn_scale_bias_mean_var_desc, save_inv_variance));       \
    } break

void CuBatchNormForwardTraining(const CuHandle& handle,
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
                                double exponential_average_factor,
                                void* running_mean,
                                void* running_variance,
                                double epsilon,
                                void* save_mean,
                                void* save_inv_variance) {
    dim3 grid;
    dim3 block;
    auto stream = handle.GetStream();
    if (mode == DNN_BATCHNORM_PER_ACTIVATION) {
        std::tie(grid, block) = CalcLaunchConfig(x_desc, 1, 256);
        grid.z                = 1;  // NOTE: batch size will be taken cared inside kernel
        cudnn::GetLogger()->info(
            "{} per activation launch config <<<{}, {}>>>", __func__, grid, block);

        switch (x_desc.GetDataType()) {
            BN_TRAINING_FWD_CASE_PER_ACTIVATION(DNN_DATA_FLOAT);
            // BN_TRAINING_FWD_CASE_PER_ACTIVATION(DNN_DATA_HALF);
            // BN_TRAINING_FWD_CASE_PER_ACTIVATION(DNN_DATA_INT8);
            // BN_TRAINING_FWD_CASE_PER_ACTIVATION(DNN_DATA_UINT8);
            // BN_TRAINING_FWD_CASE_PER_ACTIVATION(DNN_DATA_INT32);
        }
    } else {
        int grid_dim;
        int block_dim;
        std::tie(grid_dim, block_dim) = CalcLaunchConfigSpatialFactor(x_desc);
        std::tie(grid, block) =
            CalcLaunchConfig(x_desc, 1, cudnn::impl::kernel::kNbThreadsPerBlockTrainingForward);
        cudnn::GetLogger()->info("{} grid_dim={}, block_dim={}", __func__, grid_dim, block_dim);
        cudnn::GetLogger()->info("{} grid={}, block={}", __func__, grid, block);

        switch (x_desc.GetDataType()) {
            BN_TRAINING_FWD_CASE_SPATIAL(DNN_DATA_FLOAT);
            // BN_TRAINING_FWD_CASE_SPATIAL(DNN_DATA_HALF);
            // BN_TRAINING_FWD_CASE_SPATIAL(DNN_DATA_INT8);
            // BN_TRAINING_FWD_CASE_SPATIAL(DNN_DATA_UINT8);
            // BN_TRAINING_FWD_CASE_SPATIAL(DNN_DATA_INT32);
        }
    }
}

}  // namespace impl
}  // namespace cudnn
