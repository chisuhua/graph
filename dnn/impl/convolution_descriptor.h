/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_filter_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>

#include <tuple>
#include <vector>

// give cudnnConvolutionStruct a definition
struct cudnnConvolutionStruct {
    virtual ~cudnnConvolutionStruct() = default;
};

namespace cudnn {
namespace impl {

/**
 * @brief cudnnConvolutionStruct implementation class
 */
class CuConvolutionDescriptor : public cudnnConvolutionStruct {
 public:
    CuConvolutionDescriptor();
    ~CuConvolutionDescriptor() = default;

    /**
     * set a 2-dimension convolution descriptor, cudnnSetConvolution2dDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(int pad_h,
             int pad_w,
             int u,
             int v,
             int dilation_h,
             int dilation_w,
             cudnnConvolutionMode_t mode,
             cudnnDataType_t compute_type);

    /**
     * set the number of groups to be used in the associated convolution,
     * cudnnSetConvolutionGroupCount should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(int group_count);

    /**
     * set whether or not the use of tensor op is permitted in library routines associated with a
     * given convolution descriptor, cudnnSetConvolutionMathType should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(cudnnMathType_t math_type);

    /**
     * set an a n-D convolution descriptor , cudnnSetConvolutionNdDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(int array_length,
             const int padA[],
             const int filter_strideA[],
             const int dilationA[],
             cudnnConvolutionMode_t mode,
             cudnnDataType_t data_type);

    /**
     * get descriptor
     */
    void Get(int* pad_h,
             int* pad_w,
             int* u,
             int* v,
             int* dilation_h,
             int* dilation_w,
             cudnnConvolutionMode_t* mode,
             cudnnDataType_t* compute_type) const;

    /**
     * This function queries a previously initialized convolution descriptor object.
     */
    void Get(int array_length_requested,
             int* array_length,
             int padA[],
             int filter_strideA[],
             int dilationA[],
             cudnnConvolutionMode_t* mode,
             cudnnDataType_t* data_type) const;

    inline int GetGroupCount() const { return group_count_; }

    inline int GetPadH() const { return padA_[0]; }

    inline int GetPadW() const { return padA_[1]; }

    inline int GetPadA(int dim_id) const { return padA_[dim_id]; }

    inline std::vector<int> GetPadA() const { return padA_; }

    inline int GetFilterStrideH() const { return filter_strideA_[0]; }

    inline int GetFilterStrideW() const { return filter_strideA_[1]; }

    inline std::vector<int> GetFilterStrideA() const { return filter_strideA_; }

    inline int GetFilterStrideA(int dim_id) const { return filter_strideA_[dim_id]; }

    inline int GetDilationH() const { return dilationA_[0]; }

    inline int GetDilationW() const { return dilationA_[1]; }

    inline std::vector<int> GetDilationA() const { return dilationA_; }

    inline int GetDilationA(int dim_id) const { return dilationA_[dim_id]; }

    inline cudnnMathType_t GetMathType() const { return math_type_; }

    inline cudnnConvolutionMode_t GetMode() const { return mode_; }

    inline cudnnDataType_t GetDataType() const { return data_type_; }

 private:
    int array_length_                = 0;
    int group_count_                 = 1;
    std::vector<int> padA_           = std::vector<int>(DNN_DIM_MAX - 2);
    std::vector<int> filter_strideA_ = std::vector<int>(DNN_DIM_MAX - 2);
    std::vector<int> dilationA_      = std::vector<int>(DNN_DIM_MAX - 2);
    cudnnMathType_t math_type_       = static_cast<cudnnMathType_t>(0);
    cudnnConvolutionMode_t mode_     = static_cast<cudnnConvolutionMode_t>(0);
    cudnnDataType_t data_type_       = static_cast<cudnnDataType_t>(0);
};
REGIST_CONCRETE_OBJECT(cudnnConvolutionStruct, CuConvolutionDescriptor);

/**
 * brief@ Img2col concrete implementation
 */
void CuIm2Col(const CuHandle& handle,
              const CuTensorDescriptor& src_desc,
              const void* src_data,
              const CuFilterDescriptor& filter_desc,
              const CuConvolutionDescriptor& conv_desc,
              void* col_buffer);

/**
 * brief@ convBackwardBias concrete implementation
 */
void CuConvolutionBackwardBias(const CuHandle& handle,
                               const void* alpha,
                               const CuTensorDescriptor& dy_desc,
                               const void* dy,
                               const void* beta,
                               const CuTensorDescriptor& db_desc,
                               void* db);
/**
 * brief@ convforward concrete implementation
 */
void CuConvolutionForward(const CuHandle& handle,
                          const void* alpha,
                          const CuTensorDescriptor& x_desc,
                          const void* x,
                          const CuFilterDescriptor& w_desc,
                          const void* w,
                          const CuConvolutionDescriptor& conv_desc,
                          cudnnConvolutionFwdAlgo_t algo,
                          void* workSpace,
                          size_t workspace_size_in_bytes,
                          const void* beta,
                          const CuTensorDescriptor& y_desc,
                          void* y);

void CuGetConvolutionForwardAlgorithm(const CuHandle& handle,
                                      const CuTensorDescriptor& x_desc,
                                      const CuFilterDescriptor& w_desc,
                                      const CuConvolutionDescriptor& conv_desc,
                                      const CuTensorDescriptor& y_desc,
                                      cudnnConvolutionFwdPreference_t preference,
                                      size_t memory_limit_in_bytes,
                                      cudnnConvolutionFwdAlgo_t* algo);

void CuGetConvolutionForwardWorkspaceSize(const CuHandle& handle,
                                          const CuTensorDescriptor& x_desc,
                                          const CuFilterDescriptor& w_desc,
                                          const CuConvolutionDescriptor& conv_desc,
                                          const CuTensorDescriptor& y_desc,
                                          cudnnConvolutionFwdAlgo_t algo,
                                          size_t* size_in_bytes);

void CuGetConvolutionNdForwardOutputDim(const CuConvolutionDescriptor& conv_desc,
                                        const CuTensorDescriptor& input_tensor_desc,
                                        const CuFilterDescriptor& filter_desc,
                                        int nb_dims,
                                        int tensorOuputDimA[]);

void CuGetConvolution2dForwardOutputDim(const CuConvolutionDescriptor& conv_desc,
                                        const CuTensorDescriptor& input_tensor_desc,
                                        const CuFilterDescriptor& filter_desc,
                                        int* n,
                                        int* c,
                                        int* h,
                                        int* w);

void CuGetConvolutionBackwardFilterWorkspaceSize(const CuHandle&,
                                                 const CuTensorDescriptor& x_desc,
                                                 const CuTensorDescriptor& dy_desc,
                                                 const CuConvolutionDescriptor& conv_desc,
                                                 const CuFilterDescriptor& dw_desc,
                                                 cudnnConvolutionBwdFilterAlgo_t algo,
                                                 size_t* size_in_bytes);

void CuGetConvolutionBackwardDataWorkspaceSize(const CuHandle& handle,
                                               const CuFilterDescriptor& w_desc,
                                               const CuTensorDescriptor& dy_desc,
                                               const CuConvolutionDescriptor& conv_desc,
                                               const CuTensorDescriptor& dx_desc,
                                               cudnnConvolutionBwdDataAlgo_t algo,
                                               size_t* size_in_bytes);

void CuConvolutionBackwardData(const CuHandle& handle,
                               const void* alpha,
                               const CuFilterDescriptor& w_desc,
                               const void* w,
                               const CuTensorDescriptor& dy_desc,
                               const void* dy,
                               const CuConvolutionDescriptor& conv_desc,
                               cudnnConvolutionBwdDataAlgo_t algo,
                               void* workSpace,
                               size_t workSpaceSizeInBytes,
                               const void* beta,
                               const CuTensorDescriptor& dx_desc,
                               void* dx);

void CuConvolutionBackwardFilter(const CuHandle& handle,
                                 const void* alpha,
                                 const CuTensorDescriptor& x_desc,
                                 const void* x,
                                 const CuTensorDescriptor& dy_desc,
                                 const void* dy,
                                 const CuConvolutionDescriptor& conv_desc,
                                 cudnnConvolutionBwdFilterAlgo_t algo,
                                 void* workspace,
                                 size_t work_space_size_in_bytes,
                                 const void* beta,
                                 const CuFilterDescriptor& dw_desc,
                                 void* dw);

/**
 * brief@ lanuch img2col kernel function
 */
template <class T>
void LaunchIm2ColKernel(const CuTensorDescriptor& src_desc,
                        const T* src_data,
                        const CuFilterDescriptor& filter_desc,
                        const CuConvolutionDescriptor& conv_desc,
                        T* col_buffer);

/**
 * brief@ lanuch convolution matrixMul kernel function
 */
template <typename T1, typename T2>
void LaunchConvMatrixMulKernel(const T2* alpha,
                               const CuFilterDescriptor& w_desc,
                               const T1* w,
                               const CuConvolutionDescriptor& conv_desc,
                               cudnnConvolutionFwdAlgo_t algo,
                               const T1* workSpace,
                               size_t workspace_size_in_bytes,
                               const T2* beta,
                               const CuTensorDescriptor& y_desc,
                               T1* y);

/**
 * brief@ lanuch convolution gradient function with respect to the bias
 */
template <class T>
void LaunchConvBwdBiasKernel(const T* alpha,
                             const CuTensorDescriptor& dy_desc,
                             const T* dy,
                             const T* beta,
                             const CuTensorDescriptor& db_desc,
                             T* db);

/**
 * brief@ lanuch making matrix A used for convolution backward filter
 */
template <class T>
void LanunchConvFilterTransDy2AKernel(const CuTensorDescriptor& dy_desc,
                                      const T* dy,
                                      T* matrix_a,
                                      int matrix_a_work_space_size);

/**
 * brief@ lanuch making matrix B used for convolution backward filter
 */
template <class T>
void LanunchConvBwdFilterTransDy2BKernel(const CuTensorDescriptor& src_desc,
                                         const T* src_data,
                                         const CuTensorDescriptor& dy_desc,
                                         const CuFilterDescriptor& filter_desc,
                                         const CuConvolutionDescriptor& conv_desc,
                                         T* col_buffer);
/**
 * brief@ lanuch general matrix multiplication
 */
template <class T>
void LaunchMatrixMulKernel(const T* alpha,
                           const CuTensorDescriptor& x_desc,
                           const T* x,  // matrix b
                           const CuTensorDescriptor& dy_desc,
                           const T* dy,  // matrix a
                           const CuConvolutionDescriptor& conv_desc,
                           const T* beta,
                           const CuFilterDescriptor& dw_desc,
                           T* dw);

/**
 * brief@ lanuch making matrix A used for convolution backward data
 */
template <class T>
void LanunchConvBwdDataTransW2AKernel(const CuFilterDescriptor& w_desc,
                                      const T* w,
                                      T* matrix_a,
                                      int matrix_a_work_space_size);

/**
 * brief@ lanuch making matrix B used for convolution backward data
 */
template <class T>
void LanunchConvBwdDataTransDy2BKernel(const CuTensorDescriptor& dx_desc,
                                       const CuTensorDescriptor& dy_desc,
                                       const T* ptr_dy,
                                       const CuFilterDescriptor& w_desc,
                                       const CuConvolutionDescriptor& conv_desc,
                                       T* col_buffer);
/**
 * brief@ lanuch matrix multiply used for convolution backward data
 */
template <typename T1, typename T2>
void LaunchConvBwdDataMatrixMulKernel(const T2* alpha,
                                      const CuFilterDescriptor& w_desc,
                                      const T1* w,  // matrix a
                                      const CuTensorDescriptor& dy_desc,
                                      const T1* dy,  // matrix b
                                      const CuConvolutionDescriptor& conv_desc,
                                      const T2* beta,
                                      const CuTensorDescriptor& dx_desc,
                                      T1* dx);

/**
 * brief@ lanuch convolution activation bias matrixMul kernel function
 */
template <typename T1, typename T2>
void LaunchConvActBiasMatMulKernel(const T2* alpha1,
                                   const CuFilterDescriptor& w_desc,
                                   const T1* w,
                                   const CuConvolutionDescriptor& conv_desc,
                                   cudnnConvolutionFwdAlgo_t algo,
                                   const T1* workSpace,
                                   size_t workspace_size_in_bytes,
                                   const T2* alpha2,
                                   const T1* z,
                                   const T1* bias,
                                   const CuTensorDescriptor& y_desc,
                                   T1* y);

void CuConvolutionBiasActivationForward(const CuHandle& handle,
                                        const void* alpha1,
                                        const CuTensorDescriptor& x_desc,
                                        const void* x,
                                        const CuFilterDescriptor& w_desc,
                                        const void* w,
                                        const CuConvolutionDescriptor& conv_desc,
                                        cudnnConvolutionFwdAlgo_t algo,
                                        void* workSpace,
                                        size_t workspace_size_in_bytes,
                                        const void* alpha2,
                                        const CuTensorDescriptor& z_desc,
                                        const void* z,
                                        const CuTensorDescriptor& bias_desc,
                                        const void* bias,
                                        const CuTensorDescriptor& y_desc,
                                        void* y);

template <typename T1, typename T2>
void LaunchImplConvFwdKernel(const T2* alpha,
                             const CuTensorDescriptor& x_desc,
                             const T1* x,
                             const CuFilterDescriptor& w_desc,
                             const T1* w,
                             const CuConvolutionDescriptor& conv_desc,
                             const T2* beta,
                             const CuTensorDescriptor& y_desc,
                             T1* y);
}  // namespace impl
}  // namespace cudnn
