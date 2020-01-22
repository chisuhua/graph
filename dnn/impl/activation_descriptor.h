/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/meta/cudnn_meta_activation.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <memory>

// give cudnnActivationStruct a definition
struct cudnnActivationStruct {
    virtual ~cudnnActivationStruct() = default;
};

namespace cudnn {
namespace impl {

/**
 * @brief cudnnActivationStruct implementation class
 */
class CuActivationDescriptor : public cudnnActivationStruct {
 public:
    /**
     * @brief default constructor
     */
    CuActivationDescriptor() = default;

    /**
     * @brief default destructor
     */
    ~CuActivationDescriptor() = default;

    /**
     * @brief set meta information
     *
     * @param[in]   mode            enumerant to specify the activaiton mode
     * @param[in]   relu_nan_opt    enumerant to specify the NAN propagation mode
     * @param[in]   coef            floating point number to specify the clipping threashod when the
     *                              mode is set to CLIPPED RELU or to specify the alpha coefficient
     *                              when the mode is ELU
     */
    void Set(cudnnActivationMode_t mode, cudnnNanPropagation_t relu_nan_opt, double coef);

    /**
     * @brief get activation mode
     *
     * @return activation mode
     */
    cudnnActivationMode_t GetMode() const;

    /**
     * @brief get nan propagation mode
     *
     * @return NAN propagation mode
     */
    cudnnNanPropagation_t GetNanPropagation() const;

    /**
     * @brief get floating point number of clipping threashod when activation mode is CLIPPED RELU
     * or alpha coefficient when mode is ELU
     *
     * @return the floating point number
     */
    double GetCoef() const;

    inline operator meta::CuMetaActivation() const { return *meta_; }

    /**
     * @brief activation forward
     *
     * @param[in]   handle  dnn context
     * @param[in]   alpha   pointer to scaling factor (in host memory) used to blend the computation
     *                      result with prior value in the output layer as follows:
     *                      dstValue = alpha[0]*result + beta[0]*priorDstValue.
     * @param[in]   x_meta  input tensor descriptor
     * @param[in]   x       data pointer to GPU memory associated with the tensor descriptor x_meta
     * @param[in]   beta    pointer to scaling factor. see above 'alpha'
     * @param[in]   y_meta  output tensor descriptor
     * @param[out]  y       data pointer to GPU memory associated with the output tensor descriptor
     *                      y_meta
     */
    void Forward(const CuHandle& handle,
                 const void* alpha,
                 const meta::CuMetaTensor& x_meta,
                 const void* x,
                 const void* beta,
                 const meta::CuMetaTensor& y_meta,
                 void* y) const;

    /**
     * @brief activation backward
     *
     * @param[in]   handle  dnn context
     * @param[in]   alpha   pointer to scaling factor (in host memory) used to blend the computation
     *                      result with prior value in the output layer as follows:
     *                      dstValue = alpha[0]*result + beta[0]*priorDstValue.
     * @param[in]   y_meta  output tensor descriptor
     * @param[out]  y       data pointer to GPU memory associated with the output tensor descriptor
     *                      y_meta
     * @param[in]   dy_meta output differential tensor descriptor
     * @param[out]  dy      data pointer to GPU memory associated with the output differential
     *                      tensor descriptor dy_meta
     * @param[in]   x_meta  input tensor descriptor
     * @param[in]   x       data pointer to GPU memory associated with the tensor descriptor x_meta
     * @param[in]   beta    pointer to scaling factor. see above 'alpha'
     * @param[in]   dx_meta input differential tensor descriptor
     * @param[out]  dx      data pointer to GPU memory associated with the tensor descriptor
     *                      dx_meta
     */
    void Backward(const CuHandle& handle,
                  const void* alpha,
                  const meta::CuMetaTensor& y_meta,
                  const void* y,
                  const meta::CuMetaTensor& dy_meta,
                  const void* dy,
                  const meta::CuMetaTensor& x_meta,
                  const void* x,
                  const void* beta,
                  const meta::CuMetaTensor& dx_meta,
                  void* dx) const;

 private:
    std::shared_ptr<meta::CuMetaActivation> meta_ = std::make_shared<meta::CuMetaActivation>();
};

REGIST_CONCRETE_OBJECT(cudnnActivationStruct, CuActivationDescriptor);

}  // namespace impl
}  // namespace cudnn
