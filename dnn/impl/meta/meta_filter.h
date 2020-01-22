/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn.h>
#include <cudnn/impl/meta/cudnn_meta_fixed_array.h>

namespace cudnn {
namespace impl {
namespace meta {

/** @class
 * @brief underlying data structure for filter
 */
class CuMetaFilter {
 public:
    /**
     * @brief constructor
     *
     * @param[in]   nb_dims     number of dimensions
     * @param[in]   dim         raw list pointer to dimension sizes
     * @param[in]   data_type   enumerant to specify the data type
     * @param[in]   format      enumerant to sepcify the tensor format
     */
    CuMetaFilter(int nb_dims,
                 const int* dim,
                 const cudnnDataType_t data_type,
                 const cudnnTensorFormat_t format)
        : nb_dims_(nb_dims), dim_(nb_dims, dim), data_type_(data_type), format_(format) {}

    CuMetaFilter(const CuMetaFilter& other)
        : nb_dims_(other.nb_dims_),
          data_type_(other.data_type_),
          dim_(other.dim_),
          format_(other.format_) {}

    /**
     * @brief get number of dimensions
     *
     * @return number of dimensions
     */
    int GetNbDims() const { return nb_dims_; }

    /**
     * @brief get specified dimension size
     *
     * @param[in]   d   1-based dimension index
     * @return  dimension size
     */
    int GetDim(int d) const { return dim_.Get(d); }

    /**
     * @brief get data type
     *
     * @return data type
     */
    cudnnDataType_t GetDataType() const { return data_type_; }

    /**
     * @brief get tensor format
     *
     * @return tensor format
     */
    cudnnTensorFormat_t GetFormat() const { return format_; }

 private:
    const int nb_dims_;
    const meta::CuFixedArray<int> dim_;
    const cudnnDataType_t data_type_;
    const cudnnTensorFormat_t format_;
};
}  // namespace meta
}  // namespace impl
}  // namespace cudnn
