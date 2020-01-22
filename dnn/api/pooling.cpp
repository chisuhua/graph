/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_pooling_descriptor.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

#include <vector>

using cudnn::Try;
using cudnn::api::CheckDataTypeDiffer;
using cudnn::api::CheckDimensionDiffer;
using cudnn::api::CheckNull;
using cudnn::api::CheckStrideDiffer;
using cudnn::impl::Deref;
using cudnn::impl::CuPoolingDescriptor;
using cudnn::impl::CuTensorDescriptor;
using std::vector;

namespace {
inline void CheckLowestStride(const cudnn::impl::CuTensorDescriptor& desc,
                              cudnnStatus_t code = DNN_STATUS_NOT_SUPPORTED) {
    if (desc.GetStride(desc.GetNbDims()) != 1) {
        cudnn::GetLogger()->info("lowest dimension stride is not 1");
        throw cudnn::CuException(code);
    }
}

void CheckPoolingConfig(const cudnn::impl::CuTensorDescriptor& input_tensor_desc,
                        const cudnn::impl::CuPoolingDescriptor& pooling_desc,
                        int nb_dims) {
    const auto pooling_nb_dims = pooling_desc.GetNbDims();
    if (nb_dims != pooling_nb_dims + 2 && nb_dims != pooling_nb_dims + 3) {
        cudnn::GetLogger()->info("requested dimension {} is inconsistent with pooling dimension {}",
                                 nb_dims,
                                 pooling_nb_dims);
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }

    // The value of nbDims should be consistent with the dimensionality of inputDesc
    // Currently, implement as if nb_dims = 4, tensor dimension should be 4; if nb_dims = 5,
    // tensor dimension should be 5
    const auto tensor_nb_dims = input_tensor_desc.GetNbDims();
    if (nb_dims != tensor_nb_dims) {
        cudnn::GetLogger()->info("requested dimension {} is inconsistent with tensor dimension {}",
                                 nb_dims,
                                 tensor_nb_dims);
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }

    const auto tensor_format = input_tensor_desc.GetTensorFormat();
    if (tensor_format == DNN_TENSOR_NCHW_VECT_C) {
        cudnn::GetLogger()->info("VECT_C tensor type is not supported currently");
        throw cudnn::CuException(DNN_STATUS_NOT_SUPPORTED);
    }
}
}  // namespace

extern "C" {
cudnnStatus_t DNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* pooling_desc) {
    return Try([&] {
        CheckNull(pooling_desc);

        *pooling_desc = new CuPoolingDescriptor();
    });
}

cudnnStatus_t DNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t pooling_desc,
                                                      cudnnPoolingMode_t mode,
                                                      cudnnNanPropagation_t nan_opt,
                                                      int window_height,
                                                      int window_width,
                                                      int vertical_padding,
                                                      int horizontal_padding,
                                                      int vertical_stride,
                                                      int horizontal_stride) {
    return Try([&] {
        vector<int> window_dim = {window_height, window_width};
        vector<int> padding    = {vertical_padding, horizontal_padding};
        vector<int> stride     = {vertical_stride, horizontal_stride};

        CHECK_RANGE(
            mode, DNN_POOLING_MAX, DNN_POOLING_MAX_DETERMINISTIC, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(nan_opt, DNN_NOT_PROPAGATE_NAN, DNN_PROPAGATE_NAN, DNN_STATUS_BAD_PARAM);
        for (int i = 0; i < 2; ++i) {
            CHECK_LOWER_BOUND(window_dim[i], 1, DNN_STATUS_BAD_PARAM);
            CHECK_LOWER_BOUND(stride[i], 1, DNN_STATUS_BAD_PARAM);
        }

        Deref(pooling_desc).Set(mode, nan_opt, 2, window_dim.data(), padding.data(), stride.data());
    });
}

cudnnStatus_t DNNWINAPI cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t pooling_desc,
                                                      cudnnPoolingMode_t* mode,
                                                      cudnnNanPropagation_t* maxpooling_nan_opt,
                                                      int* window_height,
                                                      int* window_width,
                                                      int* vertical_padding,
                                                      int* horizontal_padding,
                                                      int* vertical_stride,
                                                      int* horizontal_stride) {
    return Try([&] {
        CheckNull(mode,
                  window_height,
                  window_width,
                  vertical_padding,
                  horizontal_padding,
                  vertical_stride,
                  horizontal_stride);

        const auto pooling_desc_impl = Deref(pooling_desc);
        *mode                        = pooling_desc_impl.GetMode();
        *maxpooling_nan_opt          = pooling_desc_impl.GetNanPropagation();
        vector<int> window_dim       = pooling_desc_impl.GetWindowDim();
        *window_height               = window_dim[0];
        *window_width                = window_dim[1];
        vector<int> padding          = pooling_desc_impl.GetPadding();
        *vertical_padding            = padding[0];
        *horizontal_padding          = padding[1];
        vector<int> stride           = pooling_desc_impl.GetStride();
        *vertical_stride             = stride[0];
        *horizontal_stride           = stride[1];
    });
}

cudnnStatus_t DNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t pooling_desc,
                                                      const cudnnPoolingMode_t mode,
                                                      const cudnnNanPropagation_t nan_opt,
                                                      int nb_dims,
                                                      const int window_dim_a[],
                                                      const int padding_a[],
                                                      const int stride_a[]) {
    return Try([&] {
        CHECK_RANGE(
            mode, DNN_POOLING_MAX, DNN_POOLING_MAX_DETERMINISTIC, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(nan_opt, DNN_NOT_PROPAGATE_NAN, DNN_PROPAGATE_NAN, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(nb_dims, 1, DNN_DIM_MAX - 2, DNN_STATUS_NOT_SUPPORTED);
        CheckNull(window_dim_a, padding_a, stride_a);

        for (int i = 0; i < nb_dims; ++i) {
            CHECK_LOWER_BOUND(window_dim_a[i], 1, DNN_STATUS_BAD_PARAM);
            CHECK_LOWER_BOUND(stride_a[i], 1, DNN_STATUS_BAD_PARAM);
        }

        Deref(pooling_desc).Set(mode, nan_opt, nb_dims, window_dim_a, padding_a, stride_a);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t pooling_desc,
                                                      int nb_dims_requested,
                                                      cudnnPoolingMode_t* mode,
                                                      cudnnNanPropagation_t* maxpooling_nan_opt,
                                                      int* nb_dims,
                                                      int window_dim_a[],
                                                      int padding_a[],
                                                      int stride_a[]) {
    return Try([&] {
        CHECK_RANGE(nb_dims_requested, 1, DNN_DIM_MAX, DNN_STATUS_NOT_SUPPORTED);
        CheckNull(nb_dims, mode, maxpooling_nan_opt, window_dim_a, padding_a, stride_a);

        const auto pooling_desc_impl = Deref(pooling_desc);
        *nb_dims                     = pooling_desc_impl.GetNbDims();
        *mode                        = pooling_desc_impl.GetMode();
        *maxpooling_nan_opt          = pooling_desc_impl.GetNanPropagation();
        vector<int> window_dim       = pooling_desc_impl.GetWindowDim();
        vector<int> padding          = pooling_desc_impl.GetPadding();
        vector<int> stride           = pooling_desc_impl.GetStride();

        for (int i = 0; i < std::min(*nb_dims, nb_dims_requested); ++i) {
            window_dim_a[i] = window_dim[i];
            padding_a[i]    = padding[i];
            stride_a[i]     = stride[i];
        }
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t pooling_desc,
                                  const cudnnTensorDescriptor_t input_tensor_desc,
                                  int nb_dims,
                                  int output_tensor_dim_a[]) {
    return Try([&] {
        const auto pooling_desc_impl = Deref(pooling_desc);
        const auto tensor_desc_impl  = Deref(input_tensor_desc);
        CheckPoolingConfig(tensor_desc_impl, pooling_desc_impl, nb_dims);

        vector<int> dim = pooling_desc_impl.GetForwardOutputDim(tensor_desc_impl, nb_dims);
        for (size_t i = 0; i < dim.size(); ++i) {
            output_tensor_dim_a[i] = dim[i];
        }
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t pooling_desc,
                                  const cudnnTensorDescriptor_t input_tensor_desc,
                                  int* n,
                                  int* c,
                                  int* h,
                                  int* w) {
    return Try([&] {
        const auto pooling_desc_impl = Deref(pooling_desc);
        const auto tensor_desc_impl  = Deref(input_tensor_desc);

        CheckNull(n, c, h, w);
        CheckPoolingConfig(tensor_desc_impl, pooling_desc_impl, 4);

        vector<int> dim = pooling_desc_impl.GetForwardOutputDim(tensor_desc_impl, 4);

        *n = dim[0];
        *c = dim[1];
        *h = dim[2];
        *w = dim[3];
    });
}

cudnnStatus_t DNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t pooling_desc) {
    return Try([&] { delete pooling_desc; });
}

cudnnStatus_t DNNWINAPI cudnnPoolingForward(cudnnHandle_t handle,
                                              const cudnnPoolingDescriptor_t pooling_desc,
                                              const void* alpha,
                                              const cudnnTensorDescriptor_t x_desc,
                                              const void* x,
                                              const void* beta,
                                              const cudnnTensorDescriptor_t y_desc,
                                              void* y) {
    return Try([&] {
        const auto x_desc_impl = Deref(x_desc);
        const auto y_desc_impl = Deref(y_desc);
        CheckDataTypeDiffer(x_desc_impl, y_desc_impl);
        CheckDimensionDiffer(x_desc_impl, y_desc_impl, {1, 2});  // check N and C dimension

        Deref(pooling_desc)
            .PoolingForward(Deref(handle), alpha, x_desc_impl, x, beta, y_desc_impl, y);
    });
}

cudnnStatus_t DNNWINAPI cudnnPoolingBackward(cudnnHandle_t handle,
                                               const cudnnPoolingDescriptor_t pooling_desc,
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
        const auto y_desc_impl  = Deref(y_desc);
        const auto dy_desc_impl = Deref(dy_desc);
        const auto x_desc_impl  = Deref(x_desc);
        const auto dx_desc_impl = Deref(dx_desc);

        CheckDimensionDiffer(y_desc_impl, dy_desc_impl);
        CheckStrideDiffer(y_desc_impl, dy_desc_impl);
        CheckDimensionDiffer(x_desc_impl, dx_desc_impl);
        CheckStrideDiffer(x_desc_impl, dx_desc_impl);
        CheckDataTypeDiffer(y_desc_impl, dy_desc_impl);
        CheckDataTypeDiffer(x_desc_impl, dx_desc_impl);
        CheckDataTypeDiffer(x_desc_impl, y_desc_impl);

        // CheckLowestStride(x_desc_impl);
        // CheckLowestStride(y_desc_impl);  // I know it's redundant

        Deref(pooling_desc)
            .PoolingBackward(Deref(handle),
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
