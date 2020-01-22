/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited * Proprietary and
 * confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/meta/cudnn_meta_pooling.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <memory>
#include <vector>

// give cudnnPoolingStruct a definition
struct cudnnPoolingStruct {
    virtual ~cudnnPoolingStruct() = default;
};

namespace cudnn {
namespace impl {
/**
 * @class cudnnPoolingDescriptor_t implementation
 */
class CuPoolingDescriptor : public cudnnPoolingStruct {
 public:
    /*
     * @brief default constructor
     */
    CuPoolingDescriptor() = default;

    /*
     * @brief default destructor
     */
    ~CuPoolingDescriptor() = default;

    /*
     * @breif setting pooling descriptor properties
     *
     * @param[in]   mode        pooling mode
     * @param[in]   nan_opt     nan propagation mode
     * @param[in]   nb_dims     dimension of the pooling operation, must be greater than zero,
     *                          NOTE: currently only 2 and 3 are supported
     * @param[in]   windwo_dim  array of dimension nb_dims containing the window size for each
     *                          dimension. The value of array elements must be greater than zero.
     * @param[in]   padding     array of dimension nb_dims containing the padding size for each
     *                          dimension. Negative padding is allowd.
     * @param[in]   stride      array of dimension nb_dims containing the striding size for each
     *                          dimension. The value of array elements must be greater than zero.
     * @throw CuException with proper code is thrown when: DNN_STATUS_NOT_SUPPORTED, if nb_dims >
     * DNN_DIM_MAX - 2; DNN_STATUS_BAD_PARAM if either nb_dims, or at least one of the elements
     * of the arrays windwo_dim, or stride is negative, or mode or nan_opt has invalid enumerant
     * value.
     */
    void Set(cudnnPoolingMode_t mode,
             cudnnNanPropagation_t nan_opt,
             int nb_dims,
             const int window_dim[],
             const int padding[],
             const int stride[]);

    /*
     * @brief get pooling mode
     *
     * @return pooling mode
     */
    cudnnPoolingMode_t GetMode() const { return meta_->mode; }

    /*
     * @brief get nan propagation option
     *
     * @return NaN propagation option
     */
    cudnnNanPropagation_t GetNanPropagation() const { return meta_->nan_opt; }

    /*
     * @brief get dimension of the pooling operation
     *
     * @return dimension of the pooling operation
     */
    int GetNbDims() const { return meta_->nb_dims; }

    /*
     * @brief get window dimensions
     *
     * @return window dimension list
     */
    std::vector<int> GetWindowDim() const;

    /*
     * @brief get padding
     *
     * @return padding list
     */
    std::vector<int> GetPadding() const;

    /*
     * @brief get stride
     *
     * @return stride list
     */
    std::vector<int> GetStride() const;

    /*
     * @brief get underlying data
     *
     * @return CuDataPooling instance
     */
    operator meta::CuMetaPooling() const { return *meta_; }

    /*
     * @brief get the output dimension for pooling forward
     *
     * @param[in]   tensor_meta  input tensor meta
     * @param[in]   nb_dims      dimension number
     * @return output dimension list
     */
    std::vector<int> GetForwardOutputDim(const meta::CuMetaTensor& tensor_meta, int nb_dims) const;

    /*
     * @brief pooling forward
     *
     * @param[in]   handle      dnn handle
     * @param[in]   alpha       pointer to scaling factor in host memory used to blend the
     *                          computation result, with prior value in the output layer as follows:
     *                          dstValue = alpha[0] * result + beta[0] * priorDstValue
     * @param[in]   x_desc      input tensor descriptor
     * @param[in]   x           data pointer to GPU memory associated with the tensor descriptor
     *                          x_desc
     * @param[in]   beta        see "alpha"
     * @param[in]   y_desc      output tensor descriptor
     * @param[out]  y           data pointer to GPU memory associated with the output tensor
     *                          descriptor y_desc
     */
    void PoolingForward(const CuHandle& handle,
                        const void* alpha,
                        const meta::CuMetaTensor& x_desc,
                        const void* x,
                        const void* beta,
                        const meta::CuMetaTensor& y_desc,
                        void* y);

    /*
     * @brief pooling backward
     *
     * @param[in]   handle      dnn handle
     * @param[in]   alpha       pointer to scaling factor in host memory used to blend the
     *                          computation result, with prior value in the output layer as follows:
     *                          dstValue = alpha[0] * result + beta[0] * priorDstValue
     * @param[in]   y_desc      output tensor descriptor
     * @param[out]  y           data pointer to GPU memory associated with the output tensor
     *                          descriptor y_desc
     * @param[in]   dy_desc     output differential tensor descriptor
     * @param[out]  dy          data pointer to GPU memory associated with the output differential
     *                          tensor descriptor dy_desc
     * @param[in]   x_desc      input tensor descriptor
     * @param[in]   x           data pointer to GPU memory associated with the tensor descriptor
     *                          x_desc
     * @param[in]   beta        see "alpha"
     * @param[in]   dx_desc     input differential tensor descriptor
     * @param[out]  dx          data pointer to GPU memory associated with the differential tensor
     *                          descriptor dx_desc
     */
    void PoolingBackward(const CuHandle& handle,
                         const void* alpha,
                         const meta::CuMetaTensor& y_desc,
                         const void* y,
                         const meta::CuMetaTensor& dy_desc,
                         const void* dy,
                         const meta::CuMetaTensor& x_desc,
                         const void* x,
                         const void* beta,
                         const meta::CuMetaTensor& dx_desc,
                         void* dx);

 private:
    std::shared_ptr<meta::CuMetaPooling> meta_ =
        std::make_shared<meta::CuMetaPooling>(2,
                                              DNN_POOLING_MAX,
                                              DNN_NOT_PROPAGATE_NAN,
                                              std::vector<int>({0, 0}).data(),
                                              std::vector<int>({0, 0}).data(),
                                              std::vector<int>({0, 0}).data());
};

REGIST_CONCRETE_OBJECT(cudnnPoolingStruct, CuPoolingDescriptor);

}  // namespace impl
}  // namespace cudnn
