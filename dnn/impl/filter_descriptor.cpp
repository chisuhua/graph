/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_filter_descriptor.h>
#include <cudnn/cudnn_exception.h>

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
static const size_t MAX_ELEMENTS = 2L * 1024L * 1024L * 1024L;
}  // namespace

namespace cudnn {
namespace impl {

using std::accumulate;
using std::multiplies;
using std::next;
using std::vector;

CuFilterDescriptor::CuFilterDescriptor() {}

void CuFilterDescriptor::Set(
    cudnnDataType_t data_type, cudnnTensorFormat_t format, int k, int c, int h, int w) {
    nbDims_    = 4;
    format_    = format;
    data_type_ = data_type;
    packed_    = true;

    filter_dimA_[0] = k;  //  Number of output feature maps.
    filter_dimA_[1] = c;  //  Number of input feature maps.
    filter_dimA_[2] = h;  //  Height of each filter.
    filter_dimA_[3] = w;  //  Width of each filter.

    size_t total =
        accumulate(filter_dimA_.begin(), next(filter_dimA_.begin(), 4), 1ULL, multiplies<size_t>());

    if (total > MAX_ELEMENTS) {
        throw CuException(DNN_STATUS_NOT_SUPPORTED);
    }

    data_.reset(new meta::CuMetaFilter(4, filter_dimA_.data(), data_type, format));
}

void CuFilterDescriptor::Set(cudnnDataType_t data_type,
                             cudnnTensorFormat_t format,
                             int nbDims,
                             const int filterDimA[]) {
    nbDims_    = nbDims;
    data_type_ = data_type;
    format_    = format;
    packed_    = true;

    for (auto iter = 0; iter < nbDims_; ++iter) {
        filter_dimA_[iter] = filterDimA[iter];
    }

    size_t total = accumulate(
        filter_dimA_.begin(), next(filter_dimA_.begin(), nbDims_), 1ULL, multiplies<int>());
    if (total > MAX_ELEMENTS) {
        throw CuException(DNN_STATUS_NOT_SUPPORTED);
    }

    data_.reset(new meta::CuMetaFilter(nbDims, filter_dimA_.data(), data_type, format));
}

void CuFilterDescriptor::Get(
    cudnnDataType_t* data_type, cudnnTensorFormat_t* format, int* k, int* c, int* h, int* w) const {
    *data_type = data_type_;
    *format    = format_;
    *k         = filter_dimA_[0];
    *c         = filter_dimA_[1];
    *h         = filter_dimA_[2];
    *w         = filter_dimA_[3];
}

void CuFilterDescriptor::Get(int nbDimsRequested,
                             cudnnDataType_t* data_type,
                             cudnnTensorFormat_t* format,
                             int* nbDims,
                             int filterDimA[]) const {
    *data_type = data_type_;
    *format    = format_;
    *nbDims    = nbDims_;

    for (auto iter = 0; iter < nbDims_; ++iter) {
        filterDimA[iter] = filter_dimA_[iter];
    }
}

CuFilterDescriptor::operator meta::CuMetaFilter() const { return *data_; }

}  // namespace impl
}  // namespace cudnn
