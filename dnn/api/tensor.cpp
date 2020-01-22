/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

using cudnn::GetLogger;
using cudnn::CuException;
using cudnn::Try;
using cudnn::api::CheckDataType;
using cudnn::api::CheckNull;
using cudnn::impl::Deref;
using cudnn::impl::CuHandle;
using cudnn::impl::CuTensorDescriptor;

namespace {
std::shared_ptr<cudnn::logger> logger = cudnn::GetLogger();
}  // namespace

extern "C" {
cudnnStatus_t DNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensor_desc) {
    return Try([&] {
        if (tensor_desc == nullptr) {
            GetLogger()->info("tensor_desc is null");
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        *tensor_desc = new CuTensorDescriptor();
    });
}

cudnnStatus_t DNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensor_desc,
                                                     cudnnTensorFormat_t format,
                                                     cudnnDataType_t data_type,
                                                     int n,
                                                     int c,
                                                     int h,
                                                     int w) {
    return Try([&] {
        CHECK_LOWER_BOUND(n, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(c, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(h, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(w, 1, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(format, DNN_TENSOR_NCHW, DNN_TENSOR_NCHW_VECT_C, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type);

        Deref(tensor_desc).Set(format, data_type, n, c, h, w);
    });
}

cudnnStatus_t DNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensor_desc,
                                                       cudnnDataType_t data_type,
                                                       int n,
                                                       int c,
                                                       int h,
                                                       int w,
                                                       int n_stride,
                                                       int c_stride,
                                                       int h_stride,
                                                       int w_stride) {
    return Try([&] {
        CHECK_RANGE(data_type, DNN_DATA_FLOAT, DNN_DATA_UINT8x4, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type);

        Deref(tensor_desc).Set(data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensor_desc,
                                                     cudnnDataType_t* data_type,
                                                     int* n,
                                                     int* c,
                                                     int* h,
                                                     int* w,
                                                     int* n_stride,
                                                     int* c_stride,
                                                     int* h_stride,
                                                     int* w_stride) {
    return Try([&] {
        CheckNull(data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride);

        Deref(tensor_desc).Get(data_type, n, c, h, w, n_stride, c_stride, h_stride, w_stride);
    });
}

cudnnStatus_t DNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensor_desc,
                                                     cudnnDataType_t data_type,
                                                     int nb_dims,
                                                     const int dim_a[],
                                                     const int stride_a[]) {
    return Try([&] {
        CHECK_RANGE(nb_dims, 3, DNN_DIM_MAX, DNN_STATUS_NOT_SUPPORTED);
        CHECK_RANGE(data_type, DNN_DATA_FLOAT, DNN_DATA_UINT8x4, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type);

        Deref(tensor_desc).Set(data_type, nb_dims, dim_a, stride_a);
    });
}

cudnnStatus_t DNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensor_desc,
                                                       cudnnTensorFormat_t format,
                                                       cudnnDataType_t data_type,
                                                       int nb_dims,
                                                       const int dim_a[]) {
    return Try([&] {
        CHECK_RANGE(format, DNN_TENSOR_NCHW, DNN_TENSOR_NHWC, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(data_type, DNN_DATA_FLOAT, DNN_DATA_UINT8x4, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type);

        Deref(tensor_desc).Set(format, data_type, nb_dims, dim_a);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensor_desc,
                                                     int nb_dims_requested,
                                                     cudnnDataType_t* data_type,
                                                     int* nb_dims,
                                                     int dim_a[],
                                                     int stride_a[]) {
    return Try([&] {
        CheckNull(data_type, nb_dims, dim_a, stride_a);

        Deref(tensor_desc).Get(nb_dims_requested, data_type, nb_dims, dim_a, stride_a);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensor_desc,
                                                    size_t* size) {
    return Try([&] {
        CheckNull(size);

        *size = Deref(tensor_desc).GetSizeInBytes();
    });
}

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t DNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensor_desc) {
    return Try([&] { delete tensor_desc; });
}

/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t DNNWINAPI cudnnTransformTensor(cudnnHandle_t handle,
                                               const void* alpha,
                                               const cudnnTensorDescriptor_t x_desc,
                                               const void* x,
                                               const void* beta,
                                               const cudnnTensorDescriptor_t y_desc,
                                               void* y) {
    return Try(
        [&] { TransformTensor(Deref(handle), alpha, Deref(x_desc), x, beta, Deref(y_desc), y); });
}

/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t DNNWINAPI cudnnAddTensor(cudnnHandle_t handle,
                                         const void* alpha,
                                         const cudnnTensorDescriptor_t a_desc,
                                         const void* a,
                                         const void* beta,
                                         const cudnnTensorDescriptor_t c_desc,
                                         void* c) {
    // TODO(fbh): to care about return value
    // int inN, inC, inW, inH, inStrideN, inStrideC, inStrideH, inStrideW;
    // int outN, outC, outW, outH, outStrideN, outStrideC, outStrideH, outStrideW;
    // cudnnDataType_t inData_type, outData_type;

    return Try([&] {
        CHECK_EQ(Deref(a_desc).GetNbDims(), Deref(c_desc).GetNbDims(), DNN_STATUS_BAD_PARAM);
        CheckNull(alpha, a, beta, c);
        for (int idx = 1; idx <= Deref(a_desc).GetNbDims(); idx++) {
            if (Deref(a_desc).GetDim(idx) != Deref(c_desc).GetDim(idx) &&
                Deref(a_desc).GetDim(idx) != 1) {
                throw CuException(DNN_STATUS_BAD_PARAM);
            }
        }

        // dim_info==0,cudnn status is not supported
        if (Deref(a_desc).GetDim(2) == Deref(c_desc).GetDim(2)) {
            int dim_info = (Deref(a_desc).GetDim(3) == 1);
            for (int idx = 4; idx <= Deref(a_desc).GetNbDims(); idx++) {
                if (dim_info != (Deref(a_desc).GetDim(idx) == 1)) {
                    throw CuException(DNN_STATUS_NOT_SUPPORTED);
                }
            }
        }
        AddTensor(Deref(handle), alpha, Deref(a_desc), a, beta, Deref(c_desc), c);
    });
}
}
