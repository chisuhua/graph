/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>

#include <memory>
#include <vector>

// give cudnnFilterStruct a definition
struct cudnnLRNStruct {
    virtual ~cudnnLRNStruct() = default;
};

namespace cudnn {
namespace impl {
/**
 * @brief cudnnLRNStruct implementation class
 *
 * The first dimension of the tensor defines the batch size n, and the second dimension defines the
 * number of features maps c.
 */
class CuLRNDescriptor : public cudnnLRNStruct {
 public:
    CuLRNDescriptor();
    ~CuLRNDescriptor() = default;

    /**
     * Set a LRN descriptor, cudnnSetLRNDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(unsigned lrn_n, double lrn_alpha, double lrn_beta, double lrn_k);

    /**
     * Get a LRN descriptor, cudnnGetLRNDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Get(unsigned* lrn_n, double* lrn_alpha, double* lrn_beta, double* lrn_k)
        const;


 private:
    std::shared_ptr<logger> logger_ = GetLogger();
    unsigned lrn_n_ = 5;
    double lrn_alpha_ = 1e-4;
    double lrn_beta_ = 0.75;
    double lrn_k_ = 2.0;
};

REGIST_CONCRETE_OBJECT(cudnnLRNStruct, CuLRNDescriptor);

void CuLRNCrossChannelForward(CuHandle handle,
                              const CuLRNDescriptor& norm_desc,
                              const void* alpha,
                              const CuTensorDescriptor& x_desc,
                              const void* x,
                              const void* beta,
                              const CuTensorDescriptor& y_desc,
                              void* y);

void CuLRNCrossChannelBackward(CuHandle handle,
                               const CuLRNDescriptor& norm_desc,
                               const void* alpha,
                               const CuTensorDescriptor& y_desc,
                               const void* y,
                               const CuTensorDescriptor& dy_desc,
                               const void* dy,
                               const CuTensorDescriptor& x_desc,
                               const void* x,
                               const void* beta,
                               const CuTensorDescriptor& dx_desc,
                               void* dx);

template <class T>
void LaunchLRNCrossChannelFwdKernel(const CuTensorDescriptor& x_desc,
                                    const T* x,
                                    const T lrn_alpha,
                                    const T lrn_beta,
                                    const T lrn_k,
                                    const int lrn_n,
                                    const float* alpha,
                                    const float* beta,
                                    const CuTensorDescriptor& y_desc,
                                    T* y);

template <class T>
void LaunchLRNCrossChannelBwdKernel(const T lrn_alpha,
                                    const T lrn_beta,
                                    const T lrn_k,
                                    const int lrn_n,
                                    const float* alpha,
                                    const CuTensorDescriptor& y_desc,
                                    const T* y,
                                    const T* dy,
                                    const float* beta,
                                    const CuTensorDescriptor& x_desc,
                                    const T* x,
                                    T* dx);
}  // namespace impl
}  // namespace cudnn
