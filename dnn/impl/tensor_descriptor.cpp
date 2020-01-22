/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>

#include <functional>
#include <numeric>

namespace {
static const size_t MAX_ELEMENTS = 2L * 1024L * 1024L * 1024L;
}  // namespace

namespace cudnn {
namespace impl {

using std::accumulate;
using std::make_tuple;
using std::multiplies;
using std::next;
using std::vector;

CuTensorDescriptor::CuTensorDescriptor() {}

void CuTensorDescriptor::Set(
    cudnnTensorFormat_t format, cudnnDataType_t data_type, int n, int c, int h, int w) {
    vector<int> dim = {n, c, h, w};
    vector<int> stride;
    if (format == DNN_TENSOR_NCHW) {
        stride.push_back(c * h * w);
        stride.push_back(h * w);
        stride.push_back(w);
        stride.push_back(1);
    } else if (format == DNN_TENSOR_NCHW_VECT_C) {
        if (c % 4 == 0 && data_type == DNN_DATA_INT8x4) {
            stride.push_back(c * h * w >> 2);
            stride.push_back(h * w);
            stride.push_back(w);
            stride.push_back(1);
        } else {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
    } else {
        stride.push_back(h * w * c);
        stride.push_back(1);
        stride.push_back(w * c);
        stride.push_back(c);
    }

    size_t total = accumulate(dim.begin(), next(dim.begin(), 4), 1ULL, multiplies<size_t>());
    if (total > MAX_ELEMENTS) {
        throw CuException(DNN_STATUS_NOT_SUPPORTED);
    }
    data_.reset(new meta::CuMetaTensor(4, dim.data(), stride.data(), data_type, format));
}

void CuTensorDescriptor::Set(cudnnDataType_t data_type,
                             int n,
                             int c,
                             int h,
                             int w,
                             int n_stride,
                             int c_stride,
                             int h_stride,
                             int w_stride) {
    vector<int> dim    = {n, c, h, w};
    vector<int> stride = {n_stride, c_stride, h_stride, w_stride};

    size_t total = accumulate(dim.begin(), next(dim.begin(), 4), 1ULL, multiplies<size_t>());
    if (total > MAX_ELEMENTS) {
        throw CuException(DNN_STATUS_NOT_SUPPORTED);
    }
    data_.reset(new meta::CuMetaTensor(4, dim.data(), stride.data(), data_type, DNN_TENSOR_NCHW));
}

void CuTensorDescriptor::Set(cudnnDataType_t data_type,
                             int nb_dims,
                             const int dim_a[],
                             const int stride_a[]) {
    vector<int> dim;
    vector<int> stride;
    for (size_t i = 0; i < nb_dims; ++i) {
        dim.push_back(dim_a[i]);
        stride.push_back(stride_a[i]);
    }

    data_.reset(
        new meta::CuMetaTensor(nb_dims, dim.data(), stride.data(), data_type, DNN_TENSOR_NCHW));
}

void CuTensorDescriptor::Set(cudnnTensorFormat_t format,
                             cudnnDataType_t data_type,
                             int nb_dims,
                             const int dim_a[]) {}

void CuTensorDescriptor::Get(cudnnDataType_t* data_type,
                             int* n,
                             int* c,
                             int* h,
                             int* w,
                             int* n_stride,
                             int* c_stride,
                             int* h_stride,
                             int* w_stride) const {
    *data_type = data_->GetDataType();
    *n         = data_->GetDim(1);
    *c         = data_->GetDim(2);
    *h         = data_->GetDim(3);
    *w         = data_->GetDim(4);
    *n_stride  = data_->GetStride(1);
    *c_stride  = data_->GetStride(2);
    *h_stride  = data_->GetStride(3);
    *w_stride  = data_->GetStride(4);
}

void CuTensorDescriptor::Get(int nb_dims_requested,
                             cudnnDataType_t* data_type,
                             int* nb_dims,
                             int dim_a[],
                             int stride_a[]) const {
    *nb_dims   = data_->GetNbDims();
    *data_type = data_->GetDataType();

    for (int i = 0; i < *nb_dims; ++i) {
        dim_a[i]    = data_->GetDim(i + 1);
        stride_a[i] = data_->GetStride(i + 1);
    }
}

TensorProperties CuTensorDescriptor::Get() const {
    const auto nb_dims = data_->GetNbDims();
    vector<int> dim;
    vector<int> stride;

    for (int i = 0; i < nb_dims; ++i) {
        dim[i]    = data_->GetDim(i + 1);
        stride[i] = data_->GetStride(i + 1);
    }
    return make_tuple(data_->GetDataType(), data_->GetFormat(), nb_dims, dim, stride);
}

size_t CuTensorDescriptor::GetSizeInBytes() const {
    if (data_->GetNbDims() == 0) {
        return 0;
    }
    auto num = data_->GetDim(1) * data_->GetStride(1);
    return num * Unit();
}

CuTensorDescriptor::operator meta::CuMetaTensor() const { return *data_; }

size_t CuTensorDescriptor::Unit() const { return kUnit.at(data_->GetDataType()); }

}  // namespace impl
}  // namespace cudnn
