/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_batch_norm.h>

#include <cudnn.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

#include <vector>

namespace cudnn {
namespace impl {

using cudnn::impl::meta::CuMetaTensor;

void CuDeriveBnTensorDescriptor(CuTensorDescriptor& bn_desc,
                                const CuTensorDescriptor& x_desc,
                                cudnnBatchNormMode_t mode) {
    const auto nb_dims = x_desc.GetNbDims();
    const auto data_type =
        x_desc.GetDataType() == DNN_DATA_HALF ? DNN_DATA_FLOAT : x_desc.GetDataType();

    const auto data_x = static_cast<CuMetaTensor>(x_desc);
    std::vector<int> bn_dim;
    std::vector<int> bn_stride;
    if (mode == DNN_BATCHNORM_SPATIAL) {
        if (nb_dims == 4) {
            bn_dim    = {1, data_x.GetDim(2), 1, 1};
            bn_stride = {data_x.GetDim(2), 1, 1, 1};
        } else {
            bn_dim    = {1, data_x.GetDim(2), 1, 1, 1};
            bn_stride = {data_x.GetDim(2), 1, 1, 1, 1};
        }
    } else if (mode == DNN_BATCHNORM_PER_ACTIVATION) {
        if (nb_dims == 4) {
            bn_dim = {1, data_x.GetDim(2), data_x.GetDim(3), data_x.GetDim(4)};
        } else {
            bn_dim = {1, data_x.GetDim(2), data_x.GetDim(3), data_x.GetDim(4), data_x.GetDim(5)};
        }
        bn_stride = {data_x.GetStride(1),
                     data_x.GetStride(2),
                     data_x.GetStride(3),
                     data_x.GetStride(4),
                     data_x.GetStride(5)};
    }

    bn_desc.Set(data_type, nb_dims, bn_dim.data(), bn_stride.data());
}

}  // namespace impl
}  // namespace cudnn
