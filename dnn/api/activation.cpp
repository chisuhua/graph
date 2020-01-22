/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_activation_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>

using cudnn::Try;
using cudnn::api::CheckDataTypeDiffer;
using cudnn::api::CheckDimensionDiffer;
using cudnn::api::CheckNull;
using cudnn::api::CheckStrideDiffer;
using cudnn::impl::Deref;
using cudnn::impl::CuActivationDescriptor;

extern "C" {
cudnnStatus_t DNNWINAPI
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activation_desc) {
    return Try([&] {
        CheckNull(activation_desc);

        *activation_desc = new CuActivationDescriptor();
    });
}

cudnnStatus_t DNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activation_desc,
                                                       cudnnActivationMode_t mode,
                                                       cudnnNanPropagation_t relu_nan_opt,
                                                       double coef) {
    return Try([&] {
        CHECK_RANGE(
            mode, DNN_ACTIVATION_SIGMOID, DNN_ACTIVATION_IDENTITY, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(
            relu_nan_opt, DNN_NOT_PROPAGATE_NAN, DNN_PROPAGATE_NAN, DNN_STATUS_BAD_PARAM);

        Deref(activation_desc).Set(mode, relu_nan_opt, coef);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activation_desc,
                             cudnnActivationMode_t* mode,
                             cudnnNanPropagation_t* relu_nan_opt,
                             double* coef) {
    return Try([&] {
        CheckNull(mode, relu_nan_opt, coef);

        auto act_desc_impl = Deref(activation_desc);
        *mode              = act_desc_impl.GetMode();
        *relu_nan_opt      = act_desc_impl.GetNanPropagation();
        *coef              = act_desc_impl.GetCoef();
    });
}

cudnnStatus_t DNNWINAPI
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activation_desc) {
    return Try([&] { delete activation_desc; });
}

cudnnStatus_t DNNWINAPI cudnnActivationForward(cudnnHandle_t handle,
                                                 cudnnActivationDescriptor_t activation_desc,
                                                 const void* alpha,
                                                 const cudnnTensorDescriptor_t x_desc,
                                                 const void* x,
                                                 const void* beta,
                                                 const cudnnTensorDescriptor_t y_desc,
                                                 void* y) {
    return Try([&] {
        const auto act_desc_impl = Deref(activation_desc);
        const auto x_desc_impl   = Deref(x_desc);
        const auto y_desc_impl   = Deref(y_desc);

        CHECK_RANGE(act_desc_impl.GetMode(),
                    DNN_ACTIVATION_SIGMOID,
                    DNN_ACTIVATION_ELU,
                    DNN_STATUS_BAD_PARAM);
        CheckDataTypeDiffer(x_desc_impl, y_desc_impl);
        CheckDimensionDiffer(x_desc_impl, y_desc_impl);
        if (x == y) {
            CheckStrideDiffer(x_desc_impl, y_desc_impl);
        }

        act_desc_impl.Forward(Deref(handle), alpha, x_desc_impl, x, beta, y_desc_impl, y);
    });
}

cudnnStatus_t DNNWINAPI cudnnActivationBackward(cudnnHandle_t handle,
                                                  cudnnActivationDescriptor_t activation_desc,
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
        const auto act_desc_impl = Deref(activation_desc);
        const auto x_desc_impl   = Deref(x_desc);
        const auto dx_desc_impl  = Deref(dx_desc);
        const auto y_desc_impl   = Deref(y_desc);
        const auto dy_desc_impl  = Deref(dy_desc);

        CHECK_RANGE(act_desc_impl.GetMode(),
                    DNN_ACTIVATION_SIGMOID,
                    DNN_ACTIVATION_ELU,
                    DNN_STATUS_BAD_PARAM);
        CheckDataTypeDiffer(x_desc_impl, y_desc_impl, DNN_STATUS_NOT_SUPPORTED);
        CheckDimensionDiffer(x_desc_impl, y_desc_impl, DNN_STATUS_NOT_SUPPORTED);
        CheckStrideDiffer(x_desc_impl, dx_desc_impl);
        CheckStrideDiffer(y_desc_impl, dy_desc_impl);
        if (x == y) {
            CheckStrideDiffer(dx_desc_impl, dy_desc_impl);
        }

        act_desc_impl.Backward(Deref(handle),
                               alpha,
                               y_desc_impl,
                               y,
                               dy_desc_impl,
                               dy,
                               x_desc_impl,
                               x,
                               beta,
                               dx_desc_impl,
                               dx);
    });
}
}
