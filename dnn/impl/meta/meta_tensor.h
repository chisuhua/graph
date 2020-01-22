/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn.h>
#include <cudnn/impl/meta/cudnn_meta_fixed_array.h>

#include <utility>
#include <vector>

namespace cudnn {
namespace impl {
namespace meta {

/** @class
 * @brief underlying data structure for tensor
 */
class CuMetaTensor {
 public:
    /**
     * @brief constructor
     *
     * @param[in]   nb_dims     number of dimensions
     * @param[in]   dim         raw list pointer to dimension sizes
     * @param[in]   stride      raw list pointer to strides
     * @param[in]   data_type   enumerant to specify the data type
     * @param[in]   format      enumerant to sepcify the tensor format
     */
    __host__ __device__ CuMetaTensor(int nb_dims,
                                     const int* dim,
                                     const int* stride,
                                     const cudnnDataType_t data_type,
                                     const cudnnTensorFormat_t format)
        : nb_dims_(nb_dims),
          dim_(nb_dims, dim),
          stride_(nb_dims, stride),
          data_type_(data_type),
          format_(format) {}

    /**
     * @brief get number of dimensions
     *
     * @return number of dimensions
     */
    __host__ __device__ __forceinline__ int GetNbDims() const { return nb_dims_; }

    /**
     * @brief get specified dimension size
     *
     * @param[in]   d   1-based dimension index
     * @return  dimension size
     */
    __host__ __device__ __forceinline__ int GetDim(int d) const { return dim_.Get(d); }

    /**
     * @brief get specified stride
     *
     * @param[in]   d   1-based dimension index
     * @return stride
     */
    __host__ __device__ __forceinline__ int GetStride(int d) const { return stride_.Get(d); }

    /**
     * @brief get data type
     *
     * @return data type
     */
    __host__ __device__ __forceinline__ cudnnDataType_t GetDataType() const { return data_type_; }

    /**
     * @brief get tensor format
     *
     * @return tensor format
     */
    __host__ __device__ __forceinline__ cudnnTensorFormat_t GetFormat() const { return format_; }

    /**
     * @brief get total number of elements in spatial, for 4d tensor it's dim w
     * x dim h, for 5d tensor it's dim w x dim h x dim d
     *
     * @return number of elements in spatial
     */
    __host__ __device__ __forceinline__ int GetNbSpatialElements() const {
        return nb_dims_ == 4 ? dim_.Get(3) * dim_.Get(4) : dim_.Get(3) * dim_.Get(4) * dim_.Get(5);
    }

    /**
     * @brief get dimension list
     *
     * @return dimension list
     */
    inline std::vector<int> GetDim() const {
        std::vector<int> ret = {dim_[0], dim_[1], dim_[2], dim_[3]};
        if (nb_dims_ == 5) {
            ret.push_back(dim_[4]);
        }
        return std::move(ret);
    }

    /**
     * @brief get stride list
     *
     * @return stride list
     */
    inline std::vector<int> GetStride() const {
        std::vector<int> ret = {stride_[0], stride_[1], stride_[2], stride_[3]};
        if (nb_dims_ == 5) {
            ret.push_back(stride_[4]);
        }
        return std::move(ret);
    }

    /**
     * @brief get n dimension
     *
     * @return n dimension
     */
    inline int GetDimN() const { return dim_[0]; }

    /**
     * @brief get c dimension
     *
     * @return c dimension
     */
    inline int GetDimC() const { return dim_[1]; }

    /**
     * @brief get d dimension for 5d tensor, and 1 for 4d tensor
     *
     * @return d dimension
     */
    inline int GetDimD() const { return nb_dims_ == 4 ? 1 : dim_[2]; }

    /**
     * @brief get h dimension
     *
     * @return h dimension
     */
    inline int GetDimH() const { return dim_[nb_dims_ - 2]; }

    /**
     * @brief get w dimension
     *
     * @return w dimension
     */
    inline int GetDimW() const { return dim_[nb_dims_ - 1]; }

    /**
     * @brief get n stride
     *
     * @return n stride
     */
    inline int GetStrideN() const { return stride_[0]; }

    /**
     * @brief get c stride
     *
     * @return c stride
     */
    inline int GetStrideC() const { return stride_[1]; }

    /**
     * @brief get d stride for 5d tensor, stride w * stride h  for 4d tensor
     *
     * @return d stride
     */
    inline int GetStrideD() const { return nb_dims_ == 4 ? stride_[2] * stride_[3] : stride_[2]; }

    /**
     * @brief get c stride
     *
     * @return c stride
     */
    inline int GetStrideH() const { return stride_[nb_dims_ - 2]; }

    /**
     * @brief get w stride
     *
     * @return w stride
     */
    inline int GetStrideW() const { return stride_[nb_dims_ - 1]; }

 private:
    const int nb_dims_;
    const meta::CuFixedArray<int> dim_;
    const meta::CuFixedArray<int> stride_;
    const cudnnDataType_t data_type_;
    const cudnnTensorFormat_t format_;
};
}  // namespace meta
}  // namespace impl
}  // namespace cudnn
