/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_pooling_descriptor.h>

#include <cudnn/impl/cudnn_launch_config.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/kernel/cudnn_pooling.cuh>
#include <cudnn/impl/meta/cudnn_meta_pooling.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>

#include <tuple>

namespace {

void VerifyOutputDim(const cudnn::impl::CuPoolingDescriptor& pooling_desc,
                     const cudnn::impl::meta::CuMetaTensor& x_desc,
                     const cudnn::impl::meta::CuMetaTensor& y_desc) {
    const auto output_dim = pooling_desc.GetForwardOutputDim(x_desc, x_desc.GetNbDims());
    for (int i = 0; i < output_dim.size(); ++i) {
        if (output_dim[i] != y_desc.GetDim(i + 1)) {
            cudnn::GetLogger()->info("user specified Y tensor dimension isn't consistent with the "
                                     "one calculated by API");
            throw cudnn::CuException(DNN_STATUS_NOT_SUPPORTED);
        }
    }
}
}  // namespace

namespace cudnn {
namespace impl {

using meta::CuMetaTensor;
using param::CuParamConstTensor;
using param::CuParamScaling;
using param::CuParamTensor;

#define FORWARD4D_CASE(data_type)                                                      \
    case data_type:                                                                    \
        kernel::PoolingForward4D<CuTensorType<data_type>::Type,                        \
                                 CuTensorType<data_type>::ScalingType>                 \
            <<<grid, block, 0, stream>>>(                                              \
                meta_->mode,                                                           \
                meta_->nan_opt,                                                        \
                meta_->dim_h,                                                          \
                meta_->dim_w,                                                          \
                meta_->padding_h,                                                      \
                meta_->padding_w,                                                      \
                meta_->stride_h,                                                       \
                meta_->stride_w,                                                       \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha), \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),  \
                x_desc.GetDimH(),                                                      \
                x_desc.GetDimW(),                                                      \
                x_desc.GetStrideN(),                                                   \
                x_desc.GetStrideC(),                                                   \
                x_desc.GetStrideH(),                                                   \
                x_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),             \
                y_desc.GetDimH(),                                                      \
                y_desc.GetDimW(),                                                      \
                y_desc.GetStrideN(),                                                   \
                y_desc.GetStrideC(),                                                   \
                y_desc.GetStrideH(),                                                   \
                y_desc.GetStrideW(),                                                   \
                reinterpret_cast<CuTensorType<data_type>::Type*>(y));                  \
        break

#define FORWARD5D_CASE(data_type)                                                      \
    case data_type:                                                                    \
        kernel::PoolingForward5D<CuTensorType<data_type>::Type,                        \
                                 CuTensorType<data_type>::ScalingType>                 \
            <<<grid, block, 0, stream>>>(                                              \
                meta_->mode,                                                           \
                meta_->nan_opt,                                                        \
                meta_->dim_d,                                                          \
                meta_->dim_h,                                                          \
                meta_->dim_w,                                                          \
                meta_->padding_d,                                                      \
                meta_->padding_h,                                                      \
                meta_->padding_w,                                                      \
                meta_->stride_d,                                                       \
                meta_->stride_h,                                                       \
                meta_->stride_w,                                                       \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha), \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),  \
                x_desc.GetDimD(),                                                      \
                x_desc.GetDimH(),                                                      \
                x_desc.GetDimW(),                                                      \
                x_desc.GetStrideN(),                                                   \
                x_desc.GetStrideC(),                                                   \
                x_desc.GetStrideD(),                                                   \
                x_desc.GetStrideH(),                                                   \
                x_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),             \
                y_desc.GetDimD(),                                                      \
                y_desc.GetDimH(),                                                      \
                y_desc.GetDimW(),                                                      \
                y_desc.GetStrideN(),                                                   \
                y_desc.GetStrideC(),                                                   \
                y_desc.GetStrideD(),                                                   \
                y_desc.GetStrideH(),                                                   \
                y_desc.GetStrideW(),                                                   \
                reinterpret_cast<CuTensorType<data_type>::Type*>(y));                  \
        break

void CuPoolingDescriptor::PoolingForward(const CuHandle& handle,
                                         const void* alpha,
                                         const CuMetaTensor& x_desc,
                                         const void* x,
                                         const void* beta,
                                         const CuMetaTensor& y_desc,
                                         void* y) {
    VerifyOutputDim(*this, x_desc, y_desc);

    dim3 grid;
    dim3 block;
    std::tie(grid, block) = CalcLaunchConfig(x_desc, 1, 256);
    auto stream           = handle.GetStream();

    GetLogger()->info("PoolingForward launch config Grid={}, Block={}", grid, block);

    if (x_desc.GetNbDims() == 4) {
        switch (x_desc.GetDataType()) {
            FORWARD4D_CASE(DNN_DATA_FLOAT);
            // FORWARD4D_CASE(DNN_DATA_HALF);
            // FORWARD4D_CASE(DNN_DATA_INT8);
            // FORWARD4D_CASE(DNN_DATA_UINT8);
            // FORWARD4D_CASE(DNN_DATA_INT32);
        }
    } else {
        switch (x_desc.GetDataType()) {
            FORWARD5D_CASE(DNN_DATA_FLOAT);
            // FORWARD5D_CASE(DNN_DATA_HALF);
            // FORWARD5D_CASE(DNN_DATA_INT8);
            // FORWARD5D_CASE(DNN_DATA_UINT8);
            // FORWARD5D_CASE(DNN_DATA_INT32);
        }
    }
}

#define BACKWARD4D_CASE(data_type)                                                     \
    case data_type:                                                                    \
        kernel::PoolingBackwardScaling<CuTensorType<data_type>::Type,                  \
                                       CuTensorType<data_type>::ScalingType>           \
            <<<scaling_grid, scaling_block, 0, stream>>>(                              \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),  \
                dx_desc.GetDimN(),                                                     \
                dx_desc.GetDimC(),                                                     \
                dx_desc.GetDimD(),                                                     \
                dx_desc.GetDimH(),                                                     \
                dx_desc.GetDimW(),                                                     \
                dx_desc.GetStrideN(),                                                  \
                dx_desc.GetStrideC(),                                                  \
                dx_desc.GetStrideD(),                                                  \
                dx_desc.GetStrideH(),                                                  \
                dx_desc.GetStrideW(),                                                  \
                reinterpret_cast<CuTensorType<data_type>::Type*>(dx));                 \
        kernel::PoolingBackward4D<CuTensorType<data_type>::Type,                       \
                                  CuTensorType<data_type>::ScalingType>                \
            <<<grid, block, 0, stream>>>(                                              \
                meta_->mode,                                                           \
                meta_->nan_opt,                                                        \
                meta_->dim_h,                                                          \
                meta_->dim_w,                                                          \
                meta_->padding_h,                                                      \
                meta_->padding_w,                                                      \
                meta_->stride_h,                                                       \
                meta_->stride_w,                                                       \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha), \
                y_desc.GetDimH(),                                                      \
                y_desc.GetDimW(),                                                      \
                y_desc.GetStrideN(),                                                   \
                y_desc.GetStrideC(),                                                   \
                y_desc.GetStrideH(),                                                   \
                y_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(y),             \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(dy),            \
                x_desc.GetDimH(),                                                      \
                x_desc.GetDimW(),                                                      \
                x_desc.GetStrideN(),                                                   \
                x_desc.GetStrideC(),                                                   \
                x_desc.GetStrideH(),                                                   \
                x_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),             \
                reinterpret_cast<CuTensorType<data_type>::Type*>(dx));                 \
        break

#define BACKWARD5D_CASE(data_type)                                                     \
    case data_type:                                                                    \
        kernel::PoolingBackwardScaling<CuTensorType<data_type>::Type,                  \
                                       CuTensorType<data_type>::ScalingType>           \
            <<<scaling_grid, scaling_block, 0, stream>>>(                              \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),  \
                dx_desc.GetDimN(),                                                     \
                dx_desc.GetDimC(),                                                     \
                dx_desc.GetDimD(),                                                     \
                dx_desc.GetDimH(),                                                     \
                dx_desc.GetDimW(),                                                     \
                dx_desc.GetStrideN(),                                                  \
                dx_desc.GetStrideC(),                                                  \
                dx_desc.GetStrideD(),                                                  \
                dx_desc.GetStrideH(),                                                  \
                dx_desc.GetStrideW(),                                                  \
                reinterpret_cast<CuTensorType<data_type>::Type*>(dx));                 \
        kernel::PoolingBackward5D<CuTensorType<data_type>::Type,                       \
                                  CuTensorType<data_type>::ScalingType>                \
            <<<grid, block, 0, stream>>>(                                              \
                meta_->mode,                                                           \
                meta_->nan_opt,                                                        \
                meta_->dim_d,                                                          \
                meta_->dim_h,                                                          \
                meta_->dim_w,                                                          \
                meta_->padding_d,                                                      \
                meta_->padding_h,                                                      \
                meta_->padding_w,                                                      \
                meta_->stride_d,                                                       \
                meta_->stride_h,                                                       \
                meta_->stride_w,                                                       \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha), \
                y_desc.GetDimD(),                                                      \
                y_desc.GetDimH(),                                                      \
                y_desc.GetDimW(),                                                      \
                y_desc.GetStrideN(),                                                   \
                y_desc.GetStrideC(),                                                   \
                y_desc.GetStrideD(),                                                   \
                y_desc.GetStrideH(),                                                   \
                y_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(y),             \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(dy),            \
                x_desc.GetDimD(),                                                      \
                x_desc.GetDimH(),                                                      \
                x_desc.GetDimW(),                                                      \
                x_desc.GetStrideN(),                                                   \
                x_desc.GetStrideC(),                                                   \
                x_desc.GetStrideD(),                                                   \
                x_desc.GetStrideH(),                                                   \
                x_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),             \
                reinterpret_cast<CuTensorType<data_type>::Type*>(dx));                 \
        break

void CuPoolingDescriptor::PoolingBackward(const CuHandle& handle,
                                          const void* alpha,
                                          const CuMetaTensor& y_desc,
                                          const void* y,
                                          const CuMetaTensor& dy_desc,
                                          const void* dy,
                                          const CuMetaTensor& x_desc,
                                          const void* x,
                                          const void* beta,
                                          const CuMetaTensor& dx_desc,
                                          void* dx) {
    // pre scaling is requred for overlap pooling
    const int nb_dims = dx_desc.GetNbDims();
    dim3 scaling_grid;
    dim3 scaling_block;
    std::tie(scaling_grid, scaling_block) =
        CalcLaunchConfigOneDimension(x_desc, kernel::kPoolingBackwardScalingTileSize, 256);

    dim3 grid;
    dim3 block;
    std::tie(grid, block) = CalcLaunchConfig(y_desc, 1, 256);
    auto stream           = handle.GetStream();

    if (x_desc.GetNbDims() == 4) {
        switch (x_desc.GetDataType()) {
            BACKWARD4D_CASE(DNN_DATA_FLOAT);
            // BACKWARD4D_CASE(DNN_DATA_HALF);
            // BACKWARD4D_CASE(DNN_DATA_INT8);
            // BACKWARD4D_CASE(DNN_DATA_UINT8);
            // BACKWARD4D_CASE(DNN_DATA_INT32);
        }
    } else {
        switch (x_desc.GetDataType()) {
            BACKWARD5D_CASE(DNN_DATA_FLOAT);
            // BACKWARD5D_CASE(DNN_DATA_HALF);
            // BACKWARD5D_CASE(DNN_DATA_INT8);
            // BACKWARD5D_CASE(DNN_DATA_UINT8);
            // BACKWARD5D_CASE(DNN_DATA_INT32);
        }
    }
}
}  // namespace impl
}  // namespace cudnn
