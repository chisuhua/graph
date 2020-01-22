/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_convolution_descriptor.h>
#include <cudnn/cudnn_exception.h>

namespace cudnn {
namespace impl {

CuConvolutionDescriptor::CuConvolutionDescriptor() {}

void CuConvolutionDescriptor::Set(int pad_h,
                                  int pad_w,
                                  int u,
                                  int v,
                                  int dilation_h,
                                  int dilation_w,
                                  cudnnConvolutionMode_t mode,
                                  cudnnDataType_t compute_type) {
    array_length_ = 2;
    mode_         = mode;
    data_type_    = compute_type;

    padA_[0] = pad_h;
    padA_[1] = pad_w;

    dilationA_[0] = dilation_h;
    dilationA_[1] = dilation_w;

    filter_strideA_[0] = u;
    filter_strideA_[1] = v;
}

void CuConvolutionDescriptor::Set(int group_count) { group_count_ = group_count; }

void CuConvolutionDescriptor::Set(int array_length,
                                  const int padA[],
                                  const int filterStrideA[],
                                  const int dilationA[],
                                  cudnnConvolutionMode_t mode,
                                  cudnnDataType_t data_type) {
    array_length_ = array_length;
    mode_         = mode;
    data_type_    = data_type;

    for (auto iter = 0; iter < array_length; ++iter) {
        padA_[iter]           = padA[iter];
        filter_strideA_[iter] = filterStrideA[iter];
        dilationA_[iter]      = dilationA[iter];
    }
}

void CuConvolutionDescriptor::Set(cudnnMathType_t mathType) { math_type_ = mathType; }

void CuConvolutionDescriptor::Get(int* pad_h,
                                  int* pad_w,
                                  int* u,
                                  int* v,
                                  int* dilation_h,
                                  int* dilation_w,
                                  cudnnConvolutionMode_t* mode,
                                  cudnnDataType_t* compute_type) const {
    *pad_h = padA_[0];
    *pad_w = padA_[1];

    *dilation_h = dilationA_[0];
    *dilation_w = dilationA_[1];

    *u            = filter_strideA_[0];
    *v            = filter_strideA_[1];
    *mode         = mode_;
    *compute_type = data_type_;
}

void CuConvolutionDescriptor::Get(int array_length_requested,
                                  int* array_length,
                                  int padA[],
                                  int filter_strideA[],
                                  int dilationA[],
                                  cudnnConvolutionMode_t* mode,
                                  cudnnDataType_t* data_type) const {
    *array_length = array_length_;
    *mode         = mode_;
    *data_type    = data_type_;

    for (auto iter = 0; iter < array_length_; ++iter) {
        padA[iter]           = padA_[iter];
        filter_strideA[iter] = filter_strideA_[iter];
        dilationA[iter]      = dilationA_[iter];
    }
}

/**
 * @brief this function serves as a heuristic for obtaining the best suited algorithm for
 * cudnnConvolutionForward for the given layer specifications
 */
void CuGetConvolutionForwardAlgorithm(const CuHandle& handle,
                                      const CuTensorDescriptor& x_desc,
                                      const CuFilterDescriptor& w_desc,
                                      const CuConvolutionDescriptor& conv_desc,
                                      const CuTensorDescriptor& y_desc,
                                      cudnnConvolutionFwdPreference_t preference,
                                      size_t memory_limit_in_bytes,
                                      cudnnConvolutionFwdAlgo_t* algo) {
    // TODO(fbh): need to support more conv algo
    switch (preference) {
    case DNN_CONVOLUTION_FWD_PREFER_FASTEST: {
        *algo = DNN_CONVOLUTION_FWD_ALGO_GEMM;
        break;
    }
    case DNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT: {
        *algo = DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        break;
    }
    case DNN_CONVOLUTION_FWD_NO_WORKSPACE: {
        *algo = DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        break;
    }
    }
}

void CuGetConvolutionForwardWorkspaceSize(const CuHandle& handle,
                                          const CuTensorDescriptor& x_desc,
                                          const CuFilterDescriptor& w_desc,
                                          const CuConvolutionDescriptor& conv_desc,
                                          const CuTensorDescriptor& y_desc,
                                          cudnnConvolutionFwdAlgo_t algo,
                                          size_t* size_in_bytes) {
    if (DNN_CONVOLUTION_FWD_ALGO_GEMM == algo) {
        *size_in_bytes = (x_desc.GetC() * w_desc.GetFilterW() * w_desc.GetFilterH()) *
                         x_desc.GetN() * y_desc.GetH() * y_desc.GetW() *
                         x_desc.GetDataTypeSizeInBytes();
    }
}

void CuGetConvolution2dForwardOutputDim(const CuConvolutionDescriptor& conv_desc,
                                        const CuTensorDescriptor& input_tensor_desc,
                                        const CuFilterDescriptor& filter_desc,
                                        int* n,
                                        int* c,
                                        int* h,
                                        int* w) {
    *n = input_tensor_desc.GetN();
    *c = filter_desc.GetOutC();
    *h = (input_tensor_desc.GetH() + 2 * conv_desc.GetPadA(0) -
          (conv_desc.GetDilationA(0) * (filter_desc.GetFilterDimA(2) - 1) + 1)) /
             conv_desc.GetFilterStrideA(0) +
         1;
    *w = (input_tensor_desc.GetW() + 2 * conv_desc.GetPadA(1) -
          (conv_desc.GetDilationA(1) * (filter_desc.GetFilterDimA(3) - 1) + 1)) /
             conv_desc.GetFilterStrideA(1) +
         1;
}

void CuGetConvolutionNdForwardOutputDim(const CuConvolutionDescriptor& conv_desc,
                                        const CuTensorDescriptor& input_tensor_desc,
                                        const CuFilterDescriptor& filter_desc,
                                        int nb_dims,
                                        int tensorOuputDimA[]) {
    tensorOuputDimA[0] = input_tensor_desc.GetN();
    tensorOuputDimA[1] = filter_desc.GetOutC();
    for (int idx = 2; idx < nb_dims; idx++) {
        tensorOuputDimA[idx] =
            (input_tensor_desc.GetDim(idx + 1) + 2 * conv_desc.GetPadA(idx - 2) -
             (conv_desc.GetDilationA(idx - 2) * (filter_desc.GetFilterDimA(idx) - 1) + 1)) /
                conv_desc.GetFilterStrideA(idx - 2) +
            1;
    }
}

void CuGetConvolutionBackwardFilterWorkspaceSize(const CuHandle&,
                                                 const CuTensorDescriptor& x_desc,
                                                 const CuTensorDescriptor& dy_desc,
                                                 const CuConvolutionDescriptor& conv_desc,
                                                 const CuFilterDescriptor& dw_desc,
                                                 cudnnConvolutionBwdFilterAlgo_t algo,
                                                 size_t* size_in_bytes) {
    // workspace_size = param_.batch_size * param_.kernel_height * param_.kernel_width *
    //                       param_.n_inputs * param_.out_height * param_.out_width * sizeof(float);

    int matrix_a_size = 0;
    int matrix_b_size = 0;

    *size_in_bytes = x_desc.GetDataTypeSizeInBytes();

    int tensor_dims = x_desc.GetNbDims();
    matrix_b_size   = (*size_in_bytes) * x_desc.GetN() * x_desc.GetC();
    matrix_b_size   = matrix_b_size * dy_desc.GetH() * dy_desc.GetW() * dw_desc.GetFilterH() *
                    dw_desc.GetFilterW();

    matrix_a_size =
        (*size_in_bytes) * dy_desc.GetN() * dy_desc.GetC() * dy_desc.GetH() * dy_desc.GetW();

    *size_in_bytes = static_cast<size_t>(matrix_b_size) + static_cast<size_t>(matrix_a_size);
}

// void CuGetConvolutionBackwardDataWorkspaceSize(const CuHandle& handle,
//                                                const CuFilterDescriptor& w_desc,
//                                                const CuTensorDescriptor& dy_desc,
//                                                const CuConvolutionDescriptor& conv_desc,
//                                                const CuTensorDescriptor& dx_desc,
//                                                cudnnConvolutionBwdDataAlgo_t algo,
//                                                size_t* size_in_bytes) {
//     int matrix_a_size = 0;
//     int matrix_b_size = 0;

//     *size_in_bytes = dx_desc.GetDataTypeSizeInBytes();

//     matrix_a_size = (*size_in_bytes) * w_desc.GetFilterH() * w_desc.GetFilterW() *
//                     w_desc.GetFilterDimA(0) * w_desc.GetFilterDimA(1);

//     matrix_b_size =
//         (*size_in_bytes) *
//         ((dy_desc.GetH() + (w_desc.GetFilterH() - 1) * 2) *
//          (dy_desc.GetW() + (w_desc.GetFilterW() - 1) * 2) * dy_desc.GetC() * dy_desc.GetN()) *
//         w_desc.GetFilterH() * w_desc.GetFilterW() * w_desc.GetFilterDimA(0);

//     *size_in_bytes = static_cast<size_t>(matrix_b_size) + static_cast<size_t>(matrix_a_size);
// }

void CuGetConvolutionBackwardDataWorkspaceSize(const CuHandle& handle,
                                               const CuFilterDescriptor& w_desc,
                                               const CuTensorDescriptor& dy_desc,
                                               const CuConvolutionDescriptor& conv_desc,
                                               const CuTensorDescriptor& dx_desc,
                                               cudnnConvolutionBwdDataAlgo_t algo,
                                               size_t* size_in_bytes) {
    int data_type_size = dx_desc.GetDataTypeSizeInBytes();

    // weight slides in height direction,the col matrix height
    int height_col = dy_desc.GetH();

    // weight slides in weight direction,the col matrix width
    int width_col = dy_desc.GetW();

    int workspace_size =
        data_type_size *
        (height_col * width_col * w_desc.GetInC() * w_desc.GetFilterH() * w_desc.GetFilterW()) *
        dx_desc.GetN();

    *size_in_bytes = static_cast<size_t>(workspace_size);
}

}  // namespace impl
}  // namespace cudnn
