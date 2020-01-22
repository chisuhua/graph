/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_pooling_descriptor.h>

#include <utility>
#include <vector>

namespace cudnn {
namespace impl {

using cudnn::impl::meta::CuMetaPooling;
using cudnn::impl::meta::CuMetaTensor;
using std::vector;

void CuPoolingDescriptor::Set(cudnnPoolingMode_t pooling_mode,
                              cudnnNanPropagation_t nan_opt,
                              int nb_dims,
                              const int window_dim[],
                              const int padding[],
                              const int stride[]) {
    vector<int> window_dim_u;
    vector<int> stride_u;
    for (size_t i = 0; i < nb_dims; ++i) {
        window_dim_u.push_back(window_dim[i]);
        stride_u.push_back(stride[i]);
    }

    meta_.reset(new CuMetaPooling(
        nb_dims, pooling_mode, nan_opt, window_dim_u.data(), padding, stride_u.data()));
}

vector<int> CuPoolingDescriptor::GetWindowDim() const {
    if (meta_->nb_dims == 2) {
        vector<int> ret = {meta_->dim_h, meta_->dim_w};
        return std::move(ret);
    } else {
        vector<int> ret = {meta_->dim_d, meta_->dim_h, meta_->dim_w};
        return std::move(ret);
    }
}

vector<int> CuPoolingDescriptor::GetPadding() const {
    if (meta_->nb_dims == 2) {
        vector<int> ret = {meta_->padding_h, meta_->padding_w};
        return std::move(ret);
    } else {
        vector<int> ret = {meta_->padding_d, meta_->padding_h, meta_->padding_w};
        return std::move(ret);
    }
}

vector<int> CuPoolingDescriptor::GetStride() const {
    if (meta_->nb_dims == 2) {
        vector<int> ret = {meta_->stride_h, meta_->stride_w};
        return std::move(ret);
    } else {
        vector<int> ret = {meta_->stride_d, meta_->stride_h, meta_->stride_w};
        return std::move(ret);
    }
}

std::vector<int> CuPoolingDescriptor::GetForwardOutputDim(const CuMetaTensor& input_tensor,
                                                          int nb_dims) const {
    // copy N and C to output
    std::vector<int> output_dim(0);

    output_dim.push_back(input_tensor.GetDim(1));
    output_dim.push_back(input_tensor.GetDim(2));

    auto mapping = [](int in_dim, int dim, int padding, int stride) {
        return 1 + (in_dim + 2 * padding - dim) / stride;
    };
    if (nb_dims == 5) {
        output_dim.push_back(
            mapping(input_tensor.GetDimD(), meta_->dim_d, meta_->padding_d, meta_->stride_d));
    }
    output_dim.push_back(
        mapping(input_tensor.GetDimH(), meta_->dim_h, meta_->padding_h, meta_->stride_h));
    output_dim.push_back(
        mapping(input_tensor.GetDimW(), meta_->dim_w, meta_->padding_w, meta_->stride_w));

    return std::move(output_dim);
}

}  // namespace impl
}  // namespace cudnn
