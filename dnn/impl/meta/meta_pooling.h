/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

namespace cudnn {
namespace impl {
namespace meta {

/** @strut
 * @brief underlying data structure for pooling
 *
 * NOTE: DON'T use in device code
 */
struct CuMetaPooling {
    /**
     * @brief constructor
     *
     * @param[in]   nb_dims         number of dimensions
     * @param[in]   mode            enumerant to specify the pooling mode
     * @param[in]   nan_opt         enumerant to specify the NaN propagation mode
     * @param[in]   window_dim      pointer to raw list of window dimension
     * @param[in]   padding         pointer to raw list of padding
     * @param[in]   stride          pointer to raw list of stride
     */
    CuMetaPooling(int pnb_dims,
                  cudnnPoolingMode_t pmode,
                  cudnnNanPropagation_t pnan_opt,
                  const int* window_dim,
                  const int* padding,
                  const int* stride)
        : nb_dims(pnb_dims),
          mode(pmode),
          nan_opt(pnan_opt),
          dim_d(nb_dims == 3 ? window_dim[0] : 1),
          dim_h(window_dim[nb_dims - 2]),
          dim_w(window_dim[nb_dims - 1]),
          padding_d(nb_dims == 3 ? padding[0] : 0),
          padding_h(padding[nb_dims - 2]),
          padding_w(padding[nb_dims - 1]),
          stride_d(nb_dims == 3 ? stride[0] : stride[0] * stride[1]),
          stride_h(stride[nb_dims - 2]),
          stride_w(stride[nb_dims - 1]) {}

    const int nb_dims;
    const cudnnPoolingMode_t mode;
    const cudnnNanPropagation_t nan_opt;
    const int dim_d     = 1;
    const int dim_h     = 1;
    const int dim_w     = 1;
    const int padding_d = 0;
    const int padding_h = 0;
    const int padding_w = 0;
    const int stride_d  = 0;
    const int stride_h  = 0;
    const int stride_w  = 0;
};

}  // namespace meta
}  // namespace impl
}  // namespace cudnn
