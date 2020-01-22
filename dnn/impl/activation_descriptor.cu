/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_activation_descriptor.h>

#include <cudnn/impl/cudnn_launch_config.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/kernel/cudnn_activation.cuh>
#include <cudnn/impl/meta/cudnn_meta_activation.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <cudnn/cudnn_logger.h>
#include <tuple>

namespace cudnn {
namespace impl {

using cudnn::impl::kernel::kActivationTileSize;
using meta::CuMetaActivation;
using meta::CuMetaTensor;

#define FORWARD_CASE(data_type, mode)                                                  \
    case mode:                                                                         \
        kernel::Activation##mode<CuTensorType<data_type>::Type,                        \
                                 CuTensorType<data_type>::ScalingType>                 \
            <<<grid, block, 0, stream>>>(                                              \
                GetNanPropagation(),                                                   \
                GetCoef(),                                                             \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha), \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),  \
                x_desc.GetDimN(),                                                      \
                x_desc.GetDimC(),                                                      \
                x_desc.GetDimD(),                                                      \
                x_desc.GetDimH(),                                                      \
                x_desc.GetDimW(),                                                      \
                x_desc.GetStrideN(),                                                   \
                x_desc.GetStrideC(),                                                   \
                x_desc.GetStrideD(),                                                   \
                x_desc.GetStrideH(),                                                   \
                x_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),             \
                y_desc.GetStrideN(),                                                   \
                y_desc.GetStrideC(),                                                   \
                y_desc.GetStrideD(),                                                   \
                y_desc.GetStrideH(),                                                   \
                y_desc.GetStrideW(),                                                   \
                reinterpret_cast<CuTensorType<data_type>::Type*>(y));                  \
        break

void CuActivationDescriptor::Forward(const CuHandle& handle,
                                     const void* alpha,
                                     const CuMetaTensor& x_desc,
                                     const void* x,
                                     const void* beta,
                                     const CuMetaTensor& y_desc,
                                     void* y) const {
    dim3 grid;
    dim3 block;
    auto stream           = handle.GetStream();
    std::tie(grid, block) = CalcLaunchConfigOneDimension(x_desc, kActivationTileSize, 256);
    GetLogger()->info("{} launch configuration grid={}, block={}", __func__, grid, block);

    switch (x_desc.GetDataType()) {
    case DNN_DATA_FLOAT:
        switch (GetMode()) {
            FORWARD_CASE(DNN_DATA_FLOAT, 0);
            FORWARD_CASE(DNN_DATA_FLOAT, 1);
            FORWARD_CASE(DNN_DATA_FLOAT, 2);
            FORWARD_CASE(DNN_DATA_FLOAT, 3);
            FORWARD_CASE(DNN_DATA_FLOAT, 4);
            FORWARD_CASE(DNN_DATA_FLOAT, 5);
        }
        break;
    }
}

#define BACKWARD_CASE(data_type, mode)                                                 \
    case mode:                                                                         \
        kernel::ActivationDiff##mode<CuTensorType<data_type>::Type,                    \
                                     CuTensorType<data_type>::ScalingType>             \
            <<<grid, block, 0, stream>>>(                                              \
                GetNanPropagation(),                                                   \
                GetCoef(),                                                             \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(alpha), \
                *reinterpret_cast<const CuTensorType<data_type>::ScalingType*>(beta),  \
                x_desc.GetDimN(),                                                      \
                x_desc.GetDimC(),                                                      \
                x_desc.GetDimD(),                                                      \
                x_desc.GetDimH(),                                                      \
                x_desc.GetDimW(),                                                      \
                y_desc.GetStrideN(),                                                   \
                y_desc.GetStrideC(),                                                   \
                y_desc.GetStrideD(),                                                   \
                y_desc.GetStrideH(),                                                   \
                y_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(y),             \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(dy),            \
                x_desc.GetStrideN(),                                                   \
                x_desc.GetStrideC(),                                                   \
                x_desc.GetStrideD(),                                                   \
                x_desc.GetStrideH(),                                                   \
                x_desc.GetStrideW(),                                                   \
                reinterpret_cast<const CuTensorType<data_type>::Type*>(x),             \
                reinterpret_cast<CuTensorType<data_type>::Type*>(dx));                 \
        break

void CuActivationDescriptor::Backward(const CuHandle& handle,
                                      const void* alpha,
                                      const CuMetaTensor& y_desc,
                                      const void* y,
                                      const CuMetaTensor& dy_desc,
                                      const void* dy,
                                      const CuMetaTensor& x_desc,
                                      const void* x,
                                      const void* beta,
                                      const CuMetaTensor& dx_desc,
                                      void* dx) const {
    dim3 grid;
    dim3 block;
    auto stream           = handle.GetStream();
    std::tie(grid, block) = CalcLaunchConfigOneDimension(x_desc, kActivationTileSize, 256);

    GetLogger()->info("{} launch configuration grid={}, block={}", __func__, grid, block);
    switch (x_desc.GetDataType()) {
    case DNN_DATA_FLOAT:
        switch (GetMode()) {
            BACKWARD_CASE(DNN_DATA_FLOAT, 0);
            BACKWARD_CASE(DNN_DATA_FLOAT, 1);
            BACKWARD_CASE(DNN_DATA_FLOAT, 2);
            BACKWARD_CASE(DNN_DATA_FLOAT, 3);
            BACKWARD_CASE(DNN_DATA_FLOAT, 4);
            BACKWARD_CASE(DNN_DATA_FLOAT, 5);
        }
        break;
    }
}

}  // namespace impl
}  // namespace cudnn
