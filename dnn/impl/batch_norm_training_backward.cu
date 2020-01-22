/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_batch_norm.h>

#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_device_memory.h>
#include <cudnn/impl/cudnn_launch_config.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/kernel/cudnn_batch_norm_training_backward_per_activation.cuh>
#include <cudnn/impl/kernel/cudnn_batch_norm_training_backward_spatial.cuh>
#include <cudnn/impl/kernel/cudnn_batch_norm_training_common.cuh>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>
#include <cudnn/cudnn_logger.h>

#include <tuple>

namespace cudnn {
namespace impl {

using cudnn::impl::kernel::BatchNormTrainingCalcMean;
using cudnn::impl::kernel::BatchNormTrainingCalcVariance;
using cudnn::impl::kernel::BatchNormTrainingInvertVariance;
using meta::CuMetaTensor;
using param::CuParamConstTensor;
using param::CuParamScaling;
using param::CuParamTensor;

#define PER_ACTIVATION_CASE(data_type)                                        \
    case data_type:                                                           \
        do {                                                                  \
            using Type        = CuTensorType<data_type>::Type;                \
            using ScalingType = CuTensorType<data_type>::ScalingType;         \
            kernel::BatchNormTrainingBackwardPerActivation<Type, ScalingType> \
                <<<grid, block, 0, stream>>>(                                 \
                    *reinterpret_cast<const ScalingType*>(alpha_data_diff),   \
                    *reinterpret_cast<const ScalingType*>(beta_data_diff),    \
                    *reinterpret_cast<const ScalingType*>(alpha_param_diff),  \
                    *reinterpret_cast<const ScalingType*>(beta_param_diff),   \
                    x_desc.GetDimN(),                                         \
                    x_desc.GetDimC(),                                         \
                    x_desc.GetDimD(),                                         \
                    x_desc.GetDimH(),                                         \
                    x_desc.GetDimW(),                                         \
                    x_desc.GetStrideN(),                                      \
                    x_desc.GetStrideC(),                                      \
                    x_desc.GetStrideD(),                                      \
                    x_desc.GetStrideH(),                                      \
                    x_desc.GetStrideW(),                                      \
                    reinterpret_cast<const Type*>(x),                         \
                    dy_desc.GetStrideN(),                                     \
                    dy_desc.GetStrideC(),                                     \
                    dy_desc.GetStrideD(),                                     \
                    dy_desc.GetStrideH(),                                     \
                    dy_desc.GetStrideW(),                                     \
                    reinterpret_cast<const Type*>(dy),                        \
                    dx_desc.GetStrideN(),                                     \
                    dx_desc.GetStrideC(),                                     \
                    dx_desc.GetStrideD(),                                     \
                    dx_desc.GetStrideH(),                                     \
                    dx_desc.GetStrideW(),                                     \
                    reinterpret_cast<Type*>(dx),                              \
                    dbn_scale_bias_desc.GetStrideC(),                         \
                    dbn_scale_bias_desc.GetStrideD(),                         \
                    dbn_scale_bias_desc.GetStrideH(),                         \
                    dbn_scale_bias_desc.GetStrideW(),                         \
                    reinterpret_cast<const Type*>(bn_scale),                  \
                    reinterpret_cast<Type*>(dbn_scale_result),                \
                    reinterpret_cast<Type*>(dbn_bias_result),                 \
                    static_cast<float>(epsilon),                              \
                    reinterpret_cast<const Type*>(saved_mean),                \
                    reinterpret_cast<const Type*>(saved_inv_variance));       \
        } while (false);                                                      \
        break

#define SPATIAL_CASE(data_type)                                                                    \
    case data_type:                                                                                \
        using TensorType       = typename CuParamTensor<data_type>::Type;                          \
        const auto unit        = sizeof(TensorType);                                               \
        const auto nb_channels = x_desc.GetDim(2);                                                 \
        TensorType* mean;                                                                          \
        TensorType* inv_variance;                                                                  \
        TensorType* dvariance;                                                                     \
        TensorType* dmean;                                                                         \
        int pool_len = nb_channels * 2;                                                            \
        if (saved_mean == nullptr) {                                                               \
            GetLogger()->info("saved_mean is null, alloc device memories");                        \
            pool_len += nb_channels * 2;                                                           \
        }                                                                                          \
        {                                                                                          \
            int sm_size;                                                                           \
            std::tie(grid, block, sm_size) = CalcLaunchConfigReduction(x_desc, unit * 4);          \
            GetLogger()->info("{} diff variance, diff mean, diff scale and diff bias launch "      \
                              "config<<<{}, {}, {}>>> ",                                           \
                              __func__,                                                            \
                              grid,                                                                \
                              block,                                                               \
                              sm_size);                                                            \
            dim3 grid2;                                                                            \
            dim3 block2;                                                                           \
            std::tie(grid2, block2) = CalcLaunchConfig(x_desc, 1, 256);                            \
            GetLogger()->info("{} diff data launch config <<<{}, {}>>>", __func__, grid, block);   \
            std::vector<TensorType> zeros(pool_len);                                               \
            std::shared_ptr<CuDeviceMemory<TensorType>> smart_pool =                               \
                std::make_shared<CuDeviceMemory<TensorType>>(pool_len, stream);                    \
            TensorType* pool = &*smart_pool;                                                       \
            smart_pool->SyncToDevice(zeros.data());                                                \
            dvariance = &pool[nb_channels * 0];                                                    \
            dmean     = &pool[nb_channels * 1];                                                    \
            if (saved_mean == nullptr) {                                                           \
                mean         = &pool[nb_channels * 2];                                             \
                inv_variance = &pool[nb_channels * 3];                                             \
                int sm_size;                                                                       \
                std::tie(grid, block, sm_size) = CalcLaunchConfigReduction(                        \
                    x_desc, sizeof(typename CuParamTensor<data_type>::Type));                      \
                kernel::BatchNormTrainingCalcMean<data_type><<<grid, block, sm_size, stream>>>(    \
                    CuParamConstTensor<data_type>(x_desc, x),                                      \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, mean));                          \
                kernel::BatchNormTrainingCalcVariance<data_type>                                   \
                    <<<grid, block, sm_size, stream>>>(                                            \
                        CuParamConstTensor<data_type>(x_desc, x),                                  \
                        CuParamConstTensor<data_type>(dbn_scale_bias_desc, mean),                  \
                        CuParamTensor<data_type>(dbn_scale_bias_desc, inv_variance));              \
                kernel::BatchNormTrainingInvertVariance<data_type><<<1, nb_channels, 0, stream>>>( \
                    static_cast<CuMetaTensor>(x_desc).GetNbSpatialElements(),                      \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, inv_variance),                   \
                    epsilon);                                                                      \
            }                                                                                      \
            kernel::BatchNormTrainingBackwardSpatialFactorScaleDiffBiasDiff<data_type>             \
                <<<1, nb_channels, 0, stream>>>(                                                   \
                    CuParamScaling<data_type>(alpha_param_diff, beta_param_diff),                  \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, dbn_scale_result),               \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, dbn_bias_result));               \
            kernel::BatchNormTrainingBackwardSpatialDiffVarianceMeanScaleBias<data_type>           \
                <<<grid, block, sm_size, stream>>>(                                                \
                    CuParamConstTensor<data_type>(x_desc, x),                                      \
                    CuParamConstTensor<data_type>(dy_desc, dy),                                    \
                    CuParamConstTensor<data_type>(dbn_scale_bias_desc, bn_scale),                  \
                    CuParamConstTensor<data_type>(dbn_scale_bias_desc,                             \
                                                  saved_mean == nullptr ? mean : saved_mean),      \
                    CuParamConstTensor<data_type>(                                                 \
                        dbn_scale_bias_desc,                                                       \
                        saved_inv_variance == nullptr ? inv_variance : saved_inv_variance),        \
                    CuParamScaling<data_type>(alpha_param_diff, beta_param_diff),                  \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, dmean),                          \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, dvariance),                      \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, dbn_scale_result),               \
                    CuParamTensor<data_type>(dbn_scale_bias_desc, dbn_bias_result));               \
            kernel::BatchNormTrainingBackwardSpatialDiffData<data_type>                            \
                <<<grid2, block2, 0, stream>>>(                                                    \
                    CuParamConstTensor<data_type>(x_desc, x),                                      \
                    CuParamConstTensor<data_type>(dy_desc, dy),                                    \
                    CuParamConstTensor<data_type>(dbn_scale_bias_desc, bn_scale),                  \
                    CuParamConstTensor<data_type>(dbn_scale_bias_desc,                             \
                                                  saved_mean == nullptr ? mean : saved_mean),      \
                    CuParamConstTensor<data_type>(                                                 \
                        dbn_scale_bias_desc,                                                       \
                        saved_inv_variance == nullptr ? inv_variance : saved_inv_variance),        \
                    CuParamConstTensor<data_type>(dbn_scale_bias_desc, dmean),                     \
                    CuParamConstTensor<data_type>(dbn_scale_bias_desc, dvariance),                 \
                    CuParamScaling<data_type>(alpha_data_diff, beta_data_diff),                    \
                    CuParamTensor<data_type>(dx_desc, dx));                                        \
        }                                                                                          \
        break

void CuBatchNormBackward(const CuHandle& handle,
                         cudnnBatchNormMode_t mode,
                         const void* alpha_data_diff,
                         const void* beta_data_diff,
                         const void* alpha_param_diff,
                         const void* beta_param_diff,
                         const CuMetaTensor& x_desc,
                         const void* x,
                         const CuMetaTensor& dy_desc,
                         const void* dy,
                         const CuMetaTensor& dx_desc,
                         void* dx,
                         const CuMetaTensor& dbn_scale_bias_desc,
                         const void* bn_scale,
                         void* dbn_scale_result,
                         void* dbn_bias_result,
                         double epsilon,
                         const void* saved_mean,
                         const void* saved_inv_variance) {
    dim3 grid;
    dim3 block;
    auto stream = handle.GetStream();

    if (mode == DNN_BATCHNORM_PER_ACTIVATION) {
        std::tie(grid, block) = CalcLaunchConfig(x_desc, 1, 256);
        grid.z                = 1;  // batch size will be taken care inside kernel
        GetLogger()->info(
            "{} per activation mode launch config grid={}, block={}", __func__, grid, block);

        switch (x_desc.GetDataType()) {
            PER_ACTIVATION_CASE(DNN_DATA_FLOAT);
            // PER_ACTIVATION_CASE(DNN_DATA_HALF);
            // PER_ACTIVATION_CASE(DNN_DATA_INT8);
            // PER_ACTIVATION_CASE(DNN_DATA_UINT8);
            // PER_ACTIVATION_CASE(DNN_DATA_INT32);
        }
    } else {
        switch (x_desc.GetDataType()) {
            SPATIAL_CASE(DNN_DATA_FLOAT);
            // SPATIAL_CASE(DNN_DATA_HALF);
            // SPATIAL_CASE(DNN_DATA_INT8);
            // SPATIAL_CASE(DNN_DATA_UINT8);
            // SPATIAL_CASE(DNN_DATA_INT32);
            // SPATIAL_CASE(DNN_DATA_INT8);
        }
    }
}

}  // namespace impl
}  // namespace cudnn
