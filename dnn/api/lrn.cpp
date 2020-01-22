/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_lrn_descriptor.h>
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
using cudnn::impl::CuLRNDescriptor;
using cudnn::impl::CuHandle;
using cudnn::impl::CuTensorDescriptor;

extern "C" {
cudnnStatus_t DNNWINAPI
cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* norm_desc) {
    return Try([&] { *norm_desc = new CuLRNDescriptor(); });
}

cudnnStatus_t DNNWINAPI
cudnnSetLRNDescriptor(cudnnLRNDescriptor_t norm_desc,
                      unsigned lrn_n,
                      double   lrn_alpha,
                      double   lrn_beta,
                      double   lrn_k) {
    return Try([&] {
        CHECK_RANGE(lrn_n, DNN_LRN_MIN_N, DNN_LRN_MAX_N, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(lrn_k, DNN_LRN_MIN_K, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(lrn_beta, DNN_LRN_MIN_BETA, DNN_STATUS_BAD_PARAM);

        Deref(norm_desc).Set(lrn_n, lrn_alpha, lrn_beta, lrn_k);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetLRNDescriptor(const cudnnLRNDescriptor_t norm_desc,
                      unsigned* lrn_n,
                      double*   lrn_alpha,
                      double*   lrn_beta,
                      double*   lrn_k) {
    return Try([&] {
        CheckNull(lrn_n, lrn_alpha, lrn_beta, lrn_k);

        Deref(norm_desc).Get(lrn_n, lrn_alpha, lrn_beta, lrn_k);
    });
}

cudnnStatus_t DNNWINAPI
cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                            cudnnLRNDescriptor_t norm_desc,
                            cudnnLRNMode_t lrn_mode,
                            const void* alpha,
                            const cudnnTensorDescriptor_t x_desc,
                            const void* x,
                            const void* beta,
                            const cudnnTensorDescriptor_t y_desc,
                            void* y) {
    return Try([&] {
        if (x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuLRNCrossChannelForward(Deref(handle),
                                 Deref(norm_desc),
                                 alpha,
                                 Deref(x_desc),
                                 x,
                                 beta,
                                 Deref(y_desc),
                                 y);
    });
}

cudnnStatus_t DNNWINAPI
cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
                             cudnnLRNDescriptor_t norm_desc,
                             cudnnLRNMode_t lrn_mode,
                             const void* alpha,
                             const cudnnTensorDescriptor_t y_desc,
                             const void* y,
                             const cudnnTensorDescriptor_t dy_desc,
                             const void* dy,
                             const cudnnTensorDescriptor_t x_desc,
                             const void* x,
                             const void* beta,
                             const cudnnTensorDescriptor_t dx_desc,
                             void* dx) {
    return Try([&] {
        if (alpha == nullptr || beta == nullptr || dx == nullptr || dy == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        CuLRNCrossChannelBackward(Deref(handle),
                                  Deref(norm_desc),
                                  alpha,
                                  Deref(y_desc),
                                  y,
                                  Deref(dy_desc),
                                  dy,
                                  Deref(x_desc),
                                  x,
                                  beta,
                                  Deref(dx_desc),
                                  dx);
    });
}

cudnnStatus_t DNNWINAPI
cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t norm_desc) {
    return Try([&] { delete norm_desc; });
}

}
