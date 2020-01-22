/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_softmax.h>
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
/**
 * check if the lowest dimension stride is 1
 * @param[in]   desc    tensor descriptor to check
 * @param[in]   code    cudnnStatus_t code to be thrown with if check failed
 *                      default is DNN_STATUS_NOT_SUPPORTED
 * @throw CuException with 'code' if check failed
 */
inline void CheckLowestStride(const cudnn::impl::CuTensorDescriptor& desc,
                              cudnnStatus_t code = DNN_STATUS_NOT_SUPPORTED) {
    if (desc.GetStride(desc.GetNbDims() - 1) != 1) {
        cudnn::GetLogger()->info("lowest dimension stride is not 1");
        throw cudnn::CuException(code);
    }
}

void CheckSoftMaxConfig(const cudnn::impl::CuTensorDescriptor& input_tensor_desc,
                        const cudnn::impl::CuTensorDescriptor& output_tensor_desc) {
    if (input_tensor_desc.GetN() != output_tensor_desc.GetN()) {
        cudnn::GetLogger()->info("the dimension n of the input tensor and output tensors differ.",
                                 input_tensor_desc.GetN(),
                                 output_tensor_desc.GetN());
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }

    if (input_tensor_desc.GetC() != output_tensor_desc.GetC()) {
        cudnn::GetLogger()->info("the dimension c of the input tensor and output tensors differ.",
                                 input_tensor_desc.GetC(),
                                 output_tensor_desc.GetC());
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }

    if (input_tensor_desc.GetH() != output_tensor_desc.GetH()) {
        cudnn::GetLogger()->info("the dimension h of the input tensor and output tensors differ.",
                                 input_tensor_desc.GetH(),
                                 output_tensor_desc.GetH());
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }

    if (input_tensor_desc.GetW() != output_tensor_desc.GetW()) {
        cudnn::GetLogger()->info("the dimension w of the input tensor and output tensors differ.",
                                 input_tensor_desc.GetW(),
                                 output_tensor_desc.GetW());
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }

    if (input_tensor_desc.GetDataType() != output_tensor_desc.GetDataType()) {
        cudnn::GetLogger()->info("the datatype of the input tensor and output tensors differ.",
                                 input_tensor_desc.GetDataType(),
                                 output_tensor_desc.GetDataType());
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }
}
}  // namespace

extern "C" {

cudnnStatus_t DNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle,
                                              cudnnSoftmaxAlgorithm_t algorithm,
                                              cudnnSoftmaxMode_t mode,
                                              const void* alpha,
                                              const cudnnTensorDescriptor_t x_desc,
                                              const void* x,
                                              const void* beta,
                                              const cudnnTensorDescriptor_t y_desc,
                                              void* y) {
    return Try([&] {
        CheckNull(x_desc, y_desc);
        CHECK_RANGE(algorithm, DNN_SOFTMAX_FAST, DNN_SOFTMAX_LOG, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(
            mode, DNN_SOFTMAX_MODE_INSTANCE, DNN_SOFTMAX_MODE_CHANNEL, DNN_STATUS_BAD_PARAM);
        CheckSoftMaxConfig(Deref(x_desc), Deref(y_desc));
        CuSoftmaxForward(
            Deref(handle), algorithm, mode, alpha, Deref(x_desc), x, beta, Deref(y_desc), y);
    });
}

cudnnStatus_t DNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle,
                                               cudnnSoftmaxAlgorithm_t algo,
                                               cudnnSoftmaxMode_t mode,
                                               const void* alpha,
                                               const cudnnTensorDescriptor_t y_desc,
                                               const void* y,
                                               const cudnnTensorDescriptor_t dy_desc,
                                               const void* dy,
                                               const void* beta,
                                               const cudnnTensorDescriptor_t dx_desc,
                                               void* dx) {
    return Try([&] {
        CheckNull(dx_desc, y_desc, dy_desc);
        CheckNull(y, dy, alpha, beta, dx);
        CHECK_RANGE(algo, DNN_SOFTMAX_FAST, DNN_SOFTMAX_LOG, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(
            mode, DNN_SOFTMAX_MODE_INSTANCE, DNN_SOFTMAX_MODE_CHANNEL, DNN_STATUS_BAD_PARAM);
        CheckSoftMaxConfig(Deref(dy_desc), Deref(y_desc));

        CuSoftmaxBackward(Deref(handle),
                          algo,
                          mode,
                          alpha,
                          Deref(y_desc),
                          y,
                          Deref(dy_desc),
                          dy,
                          beta,
                          Deref(dx_desc),
                          dx);
    });
}
}
