/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_filter_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>

using cudnn::Try;
using cudnn::api::CheckDataType;
using cudnn::api::CheckNull;
using cudnn::impl::Deref;
using cudnn::impl::CuFilterDescriptor;
using cudnn::impl::CuTensorDescriptor;

extern "C" {

cudnnStatus_t DNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filter_desc) {
    return Try([&] {
        CheckNull(filter_desc);

        *filter_desc = new CuFilterDescriptor();
    });
}

cudnnStatus_t DNNWINAPI cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filter_desc,
                                                     cudnnDataType_t data_type,
                                                     cudnnTensorFormat_t format,
                                                     int k,
                                                     int c,
                                                     int h,
                                                     int w) {
    return Try([&] {
        CHECK_LOWER_BOUND(k, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(c, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(h, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(w, 1, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(format, DNN_TENSOR_NCHW, DNN_TENSOR_NHWC, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type, DNN_STATUS_BAD_PARAM);

        Deref(filter_desc).Set(data_type, format, k, c, h, w);
    });
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filter_desc,
                                         cudnnDataType_t data_type,
                                         cudnnTensorFormat_t format,
                                         int nb_dims,
                                         const int filter_dimA[]) {
    return Try([&] {
        CHECK_LOWER_BOUND(nb_dims, 1, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(format, DNN_TENSOR_NCHW, DNN_TENSOR_NHWC, DNN_STATUS_BAD_PARAM);

        Deref(filter_desc).Set(data_type, format, nb_dims, filter_dimA);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filter_desc,
                                                     cudnnDataType_t* data_type,
                                                     cudnnTensorFormat_t* format,
                                                     int* k,
                                                     int* c,
                                                     int* h,
                                                     int* w) {
    return Try([&] {
        CheckNull(data_type, format, k, c, h, w);

        Deref(filter_desc).Get(data_type, format, k, c, h, w);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t w_desc,
                                                     int nb_dims_requested,
                                                     cudnnDataType_t* data_type,
                                                     cudnnTensorFormat_t* format,
                                                     int* nb_dims,
                                                     int filter_dim_a[]) {
    return Try([&] {
        CheckNull(data_type, format, nb_dims, filter_dim_a);

        Deref(w_desc).Get(nb_dims_requested, data_type, format, nb_dims, filter_dim_a);
    });
}

cudnnStatus_t DNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filter_desc) {
    return Try([&] { delete filter_desc; });
}
}
