#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_convolution_descriptor.h>
#include <cudnn/impl/cudnn_filter_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

using cudnn::GetLogger;
using cudnn::CuException;
using cudnn::Try;
using cudnn::api::CheckDataType;
using cudnn::api::CheckDataTypeDiffer;
using cudnn::api::CheckDimensionDiffer;
using cudnn::api::CheckNull;
using cudnn::impl::Deref;
using cudnn::impl::CuConvolutionDescriptor;
using cudnn::impl::CuGetConvolutionBackwardFilterWorkspaceSize;
using cudnn::impl::CuGetConvolutionNdForwardOutputDim;
using cudnn::impl::CuHandle;
using cudnn::impl::CuTensorDescriptor;
extern "C" {

/* This function constructs the A matrix necessary to perform a forward pass of GEMM convolution
 */
cudnnStatus_t DNNWINAPI cudnnIm2Col(cudnnHandle_t handle,
                                      cudnnTensorDescriptor_t src_desc,
                                      const void* src_data,
                                      cudnnFilterDescriptor_t filter_desc,
                                      cudnnConvolutionDescriptor_t conv_desc,
                                      void* col_buffer) {
    return Try([&] {
        CheckNull(src_data, col_buffer);

        CuIm2Col(Deref(handle),
                 Deref(src_desc),
                 src_data,
                 Deref(filter_desc),
                 Deref(conv_desc),
                 col_buffer);
    });
}

cudnnStatus_t DNNWINAPI
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* conv_desc) {
    return Try([&] { *conv_desc = new CuConvolutionDescriptor(); });
}

cudnnStatus_t DNNWINAPI cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t conv_desc,
                                                          int pad_h,
                                                          int pad_w,
                                                          int u,
                                                          int v,
                                                          int dilation_h,
                                                          int dilation_w,
                                                          cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t compute_type) {
    return Try([&] {
        CHECK_LOWER_BOUND(pad_h, 0, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(pad_w, 0, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(u, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(v, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(dilation_h, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(dilation_w, 1, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(mode, DNN_CONVOLUTION, DNN_CROSS_CORRELATION, DNN_STATUS_BAD_PARAM);
        CheckDataType(compute_type, DNN_STATUS_BAD_PARAM);

        Deref(conv_desc).Set(pad_h, pad_w, u, v, dilation_h, dilation_w, mode, compute_type);
    });
}

cudnnStatus_t DNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t conv_desc,
                                                        int group_count) {
    return Try([&] {
        CHECK_LOWER_BOUND(group_count, 1, DNN_STATUS_BAD_PARAM);

        Deref(conv_desc).Set(group_count);
    });
}

cudnnStatus_t DNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t conv_desc,
                                                      cudnnMathType_t math_type) {
    return Try([&] { Deref(conv_desc).Set(math_type); });
}

cudnnStatus_t DNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t conv_desc,
                                                          int array_length,
                                                          const int padA[],
                                                          const int filter_strideA[],
                                                          const int dilationA[],
                                                          cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t data_type) {
    return Try([&] {
        CHECK_LOWER_BOUND(array_length, 1, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(mode, DNN_CONVOLUTION, DNN_CROSS_CORRELATION, DNN_STATUS_BAD_PARAM);
        CheckDataType(data_type, DNN_STATUS_BAD_PARAM);

        Deref(conv_desc).Set(array_length, padA, filter_strideA, dilationA, mode, data_type);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t conv_desc,
                                int* pad_h,
                                int* pad_w,
                                int* u,
                                int* v,
                                int* dilation_h,
                                int* dilation_w,
                                cudnnConvolutionMode_t* mode,
                                cudnnDataType_t* compute_type) {
    return Try([&] {
        CheckNull(pad_h, pad_w, u, v, dilation_h, dilation_w, mode, compute_type);

        Deref(conv_desc).Get(pad_h, pad_w, u, v, dilation_h, dilation_w, mode, compute_type);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t conv_desc,
                                      const cudnnTensorDescriptor_t input_tensor_desc,
                                      const cudnnFilterDescriptor_t filter_desc,
                                      int* n,
                                      int* c,
                                      int* h,
                                      int* w) {
    return Try([&] {
        CheckNull(n, c, h, w);
        CuGetConvolution2dForwardOutputDim(
            Deref(conv_desc), Deref(input_tensor_desc), Deref(filter_desc), n, c, h, w);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                    const cudnnTensorDescriptor_t x_desc,
                                    const cudnnFilterDescriptor_t w_desc,
                                    const cudnnConvolutionDescriptor_t conv_desc,
                                    const cudnnTensorDescriptor_t y_desc,
                                    cudnnConvolutionFwdPreference_t preference,
                                    size_t memory_limit_in_bytes,
                                    cudnnConvolutionFwdAlgo_t* algo) {
    return Try([&] {
        if (handle == nullptr || x_desc == nullptr || w_desc == nullptr || conv_desc == nullptr ||
            y_desc == nullptr || algo == nullptr) {
            GetLogger()->info("cudnnGetConvolutionForwardAlgorithm param is null");
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuGetConvolutionForwardAlgorithm(Deref(handle),
                                         Deref(x_desc),
                                         Deref(w_desc),
                                         Deref(conv_desc),
                                         Deref(y_desc),
                                         preference,
                                         memory_limit_in_bytes,
                                         algo);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                       const cudnnTensorDescriptor_t x_desc,
                                       const cudnnFilterDescriptor_t w_desc,
                                       const cudnnConvolutionDescriptor_t conv_desc,
                                       const cudnnTensorDescriptor_t y_desc,
                                       const int requested_algo_count,
                                       int* returned_algo_count,
                                       cudnnConvolutionFwdAlgoPerf_t* perf_results) {}

cudnnStatus_t DNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle,
                                                                      int* count) {
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                        const cudnnTensorDescriptor_t x_desc,
                                        const cudnnFilterDescriptor_t w_desc,
                                        const cudnnConvolutionDescriptor_t conv_desc,
                                        const cudnnTensorDescriptor_t y_desc,
                                        cudnnConvolutionFwdAlgo_t algo,
                                        size_t* size_in_bytes) {
    return Try([&] {
        if (handle == nullptr || x_desc == nullptr || w_desc == nullptr || conv_desc == nullptr ||
            y_desc == nullptr || size_in_bytes == nullptr) {
            GetLogger()->info("cudnnGetConvolutionForwardWorkspaceSize param is null");
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        CuGetConvolutionForwardWorkspaceSize(Deref(handle),
                                             Deref(x_desc),
                                             Deref(w_desc),
                                             Deref(conv_desc),
                                             Deref(y_desc),
                                             algo,
                                             size_in_bytes);
    });
}

cudnnStatus_t DNNWINAPI cudnnConvolutionForward(cudnnHandle_t handle,
                                                  const void* alpha,
                                                  const cudnnTensorDescriptor_t x_desc,
                                                  const void* x,
                                                  const cudnnFilterDescriptor_t w_desc,
                                                  const void* w,
                                                  const cudnnConvolutionDescriptor_t conv_desc,
                                                  cudnnConvolutionFwdAlgo_t algo,
                                                  void* workspace,
                                                  size_t workspace_size_in_bytes,
                                                  const void* beta,
                                                  const cudnnTensorDescriptor_t y_desc,
                                                  void* y) {
    return Try([&] {
        if (x == nullptr || w == nullptr || y == nullptr || alpha == nullptr || beta == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuConvolutionForward(Deref(handle),
                             alpha,
                             Deref(x_desc),
                             x,
                             Deref(w_desc),
                             w,
                             Deref(conv_desc),
                             algo,
                             workspace,
                             workspace_size_in_bytes,
                             beta,
                             Deref(y_desc),
                             y);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                         const cudnnFilterDescriptor_t w_desc,
                                         const cudnnTensorDescriptor_t dy_desc,
                                         const cudnnConvolutionDescriptor_t conv_desc,
                                         const cudnnTensorDescriptor_t dx_desc,
                                         cudnnConvolutionBwdDataPreference_t preference,
                                         size_t memory_limit_in_bytes,
                                         cudnnConvolutionBwdDataAlgo_t* algo) {
    *algo = DNN_CONVOLUTION_BWD_DATA_ALGO_1;
    return DNN_STATUS_SUCCESS;
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
                                            const cudnnFilterDescriptor_t w_desc,
                                            const cudnnTensorDescriptor_t dy_desc,
                                            const cudnnConvolutionDescriptor_t conv_desc,
                                            const cudnnTensorDescriptor_t dx_desc,
                                            const int requested_algo_count,
                                            int* returned_algo_count,
                                            cudnnConvolutionBwdDataAlgoPerf_t* perf_results) {}

cudnnStatus_t DNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                                           int* count) {
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
                                             const cudnnFilterDescriptor_t w_desc,
                                             const cudnnTensorDescriptor_t dy_desc,
                                             const cudnnConvolutionDescriptor_t conv_desc,
                                             const cudnnTensorDescriptor_t dx_desc,
                                             cudnnConvolutionBwdDataAlgo_t algo,
                                             size_t* size_in_bytes) {
    return Try([&] {
        CHECK_EQ(
            Deref(dx_desc).GetDataType(), Deref(dy_desc).GetDataType(), DNN_STATUS_BAD_PARAM);
        CheckNull(size_in_bytes);

        CuGetConvolutionBackwardDataWorkspaceSize(Deref(handle),
                                                  Deref(w_desc),
                                                  Deref(dy_desc),
                                                  Deref(conv_desc),
                                                  Deref(dx_desc),
                                                  algo,
                                                  size_in_bytes);
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t x_desc,
                                           const cudnnTensorDescriptor_t dy_desc,
                                           const cudnnConvolutionDescriptor_t conv_desc,
                                           const cudnnFilterDescriptor_t dw_desc,
                                           cudnnConvolutionBwdFilterPreference_t preference,
                                           size_t memory_limit_in_bytes,
                                           cudnnConvolutionBwdFilterAlgo_t* algo) {
    *algo = DNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    return DNN_STATUS_SUCCESS;
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                              const cudnnTensorDescriptor_t x_desc,
                                              const cudnnTensorDescriptor_t dy_desc,
                                              const cudnnConvolutionDescriptor_t conv_desc,
                                              const cudnnFilterDescriptor_t dw_desc,
                                              const int requested_algo_count,
                                              int* returned_algo_count,
                                              cudnnConvolutionBwdFilterAlgoPerf_t* perf_results) {
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle,
                                                                             int* count) {
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                               const cudnnTensorDescriptor_t x_desc,
                                               const cudnnTensorDescriptor_t dy_desc,
                                               const cudnnConvolutionDescriptor_t conv_desc,
                                               const cudnnFilterDescriptor_t dw_desc,
                                               cudnnConvolutionBwdFilterAlgo_t algo,
                                               size_t* size_in_bytes) {
    return Try([&] {
        CHECK_EQ(Deref(x_desc).GetDataType(), Deref(dy_desc).GetDataType(), DNN_STATUS_BAD_PARAM);

        CuGetConvolutionBackwardFilterWorkspaceSize(Deref(handle),
                                                    Deref(x_desc),
                                                    Deref(dy_desc),
                                                    Deref(conv_desc),
                                                    Deref(dw_desc),
                                                    algo,
                                                    size_in_bytes);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t conv_desc,
                                                        int* group_count) {
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t conv_desc,
                                                      cudnnMathType_t* math_type) {
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t conv_desc,
                                int array_length_requested,
                                int* array_length,
                                int padA[],
                                int filter_strideA[],
                                int dilationA[],
                                cudnnConvolutionMode_t* mode,
                                cudnnDataType_t* data_type) {
    Try([&] {
        if (array_length == nullptr || padA == nullptr || filter_strideA == nullptr ||
            dilationA == nullptr || mode == nullptr || data_type == nullptr) {
            GetLogger()->info("cudnnGetConvolutionNdDescriptor coef is nullptr");
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        return Try([&] {
            Deref(conv_desc).Get(array_length_requested,
                                 array_length,
                                 padA,
                                 filter_strideA,
                                 dilationA,
                                 mode,
                                 data_type);
        });
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t conv_desc,
                                      const cudnnTensorDescriptor_t input_tensor_desc,
                                      const cudnnFilterDescriptor_t filter_desc,
                                      int nb_dims,
                                      int tensorOuputDimA[]) {
    return Try([&] {
        CuGetConvolutionNdForwardOutputDim(Deref(conv_desc),
                                           Deref(input_tensor_desc),
                                           Deref(filter_desc),
                                           nb_dims,
                                           tensorOuputDimA);
    });
}

cudnnStatus_t DNNWINAPI
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t conv_desc) {
    return Try([&] { delete conv_desc; });
}

cudnnStatus_t DNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                                       const void* alpha,
                                                       const cudnnTensorDescriptor_t dy_desc,
                                                       const void* dy,
                                                       const void* beta,
                                                       const cudnnTensorDescriptor_t db_desc,
                                                       void* db) {
    return Try([&] {
        if (alpha == nullptr || beta == nullptr || dy == nullptr || db == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        // if n,height,width of the output tensor is not 1,
        // the return status is DNN_STATUS_BAD_PARAM
        if (Deref(db_desc).GetN() != 1 || (Deref(db_desc).GetDim())[2] != 1 ||
            (Deref(db_desc).GetDim())[3] != 1) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        // if the numbers of feature maps of the input tensor and
        // output tensor is different, the return status is DNN_STATUS_BAD_PARAM
        // TODO(fbh): judge equality of numbers of feature maps of the input tensor

        // if the dataType of two tensor descriptors are different,
        // the return status is DNN_STATUS_BAD_PARAM
        if (Deref(dy_desc).GetDataType() != Deref(db_desc).GetDataType()) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuConvolutionBackwardBias(
            Deref(handle), alpha, Deref(dy_desc), dy, beta, Deref(db_desc), db);
    });
}

cudnnStatus_t DNNWINAPI cudnnConvolutionBackwardData(cudnnHandle_t handle,
                                                       const void* alpha,
                                                       const cudnnFilterDescriptor_t w_desc,
                                                       const void* w,
                                                       const cudnnTensorDescriptor_t dy_desc,
                                                       const void* dy,
                                                       const cudnnConvolutionDescriptor_t conv_desc,
                                                       cudnnConvolutionBwdDataAlgo_t algo,
                                                       void* workspace,
                                                       size_t work_space_size_in_bytes,
                                                       const void* beta,
                                                       const cudnnTensorDescriptor_t dx_desc,
                                                       void* dx) {
    return Try([&] {
        if (alpha == nullptr || beta == nullptr || dx == nullptr || dy == nullptr || w == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        CuConvolutionBackwardData(Deref(handle),
                                  alpha,
                                  Deref(w_desc),
                                  w,
                                  Deref(dy_desc),
                                  dy,
                                  Deref(conv_desc),
                                  algo,
                                  workspace,
                                  work_space_size_in_bytes,
                                  beta,
                                  Deref(dx_desc),
                                  dx);
    });
}

cudnnStatus_t DNNWINAPI
cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                               const void* alpha,
                               const cudnnTensorDescriptor_t x_desc,
                               const void* x,
                               const cudnnTensorDescriptor_t dy_desc,
                               const void* dy,
                               const cudnnConvolutionDescriptor_t conv_desc,
                               cudnnConvolutionBwdFilterAlgo_t algo,
                               void* workspace,
                               size_t work_space_size_in_bytes,
                               const void* beta,
                               const cudnnFilterDescriptor_t dw_desc,
                               void* dw) {
    return Try([&] {
        CheckNull(alpha, x, dy, beta, dw);

        CuConvolutionBackwardFilter(Deref(handle),
                                    alpha,
                                    Deref(x_desc),
                                    x,
                                    Deref(dy_desc),
                                    dy,
                                    Deref(conv_desc),
                                    algo,
                                    workspace,
                                    work_space_size_in_bytes,
                                    beta,
                                    Deref(dw_desc),
                                    dw);
    });
}

cudnnStatus_t DNNWINAPI cudnnConvolutionBiasActivationForward(cudnnHandle_t                       handle,
                                                                const void                         *alpha1,
                                                                const cudnnTensorDescriptor_t       xDesc,
                                                                const void                         *x,
                                                                const cudnnFilterDescriptor_t       wDesc,
                                                                const void                         *w,
                                                                const cudnnConvolutionDescriptor_t  convDesc,
                                                                cudnnConvolutionFwdAlgo_t           algo,
                                                                void                               *workSpace,
                                                                size_t                              workSpaceSizeInBytes,
                                                                const void                         *alpha2,
                                                                const cudnnTensorDescriptor_t       zDesc,
                                                                const void                         *z,
                                                                const cudnnTensorDescriptor_t       biasDesc,
                                                                const void                         *bias,
                                                                const cudnnActivationDescriptor_t   activationDesc,
                                                                const cudnnTensorDescriptor_t       yDesc,
                                                                void                               *y ) {
    return Try([&] {
        if (x == nullptr || w == nullptr || y == nullptr || alpha1 == nullptr || alpha2 == nullptr || z == nullptr || bias == nullptr) {
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        CuConvolutionBiasActivationForward(Deref(handle),
                                           alpha1,
                                           Deref(xDesc),
                                           x,
                                           Deref(wDesc),
                                           w,
                                           Deref(convDesc),
                                           algo,
                                           workSpace,
                                           workSpaceSizeInBytes,
                                           alpha2,
                                           Deref(zDesc),
                                           z,
                                           Deref(biasDesc),
                                           bias,
                                           Deref(yDesc),
                                           y);
    });
}

/** not support cudnnSetConvolutionReorderType in this version
 */
// cudnnStatus_t DNNWINAPI
// cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t
// conv_desc,
//                                                          cudnnReorderType_t
//                                                          reorderType)
//                                                          {}
}
