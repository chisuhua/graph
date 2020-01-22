/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cuda_fp16.h>  // __half
#include <cudnn.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <stdint.h>  // int8_t, uint8_t, int32_t

namespace cudnn {
namespace impl {
namespace param {

template <class T>
class CuAccessor {
 public:
    const cudnn::impl::meta::CuMetaTensor meta_;
    const int nb_dims_;
    const int stride_n_;
    const int stride_c_;
    const int stride_d_;
    const int stride_h_;
    const int stride_w_;
    T* ptr_;

 public:
    __device__ __host__ __forceinline__ CuAccessor(const cudnn::impl::meta::CuMetaTensor desc,
                                                   T* ptr)
        : meta_(desc),
          nb_dims_(meta_.GetNbDims()),
          stride_n_(meta_.GetStride(1)),
          stride_c_(meta_.GetStride(2)),
          stride_d_(nb_dims_ == 4 ? 0 : meta_.GetStride(3)),
          stride_h_(meta_.GetStride(nb_dims_ - 1)),
          stride_w_(meta_.GetStride(nb_dims_)),
          ptr_(ptr) {}

    __device__ __forceinline__ T At(int n, int c, int d, int h, int w) const {
        return ptr_[n * stride_n_ + c * stride_c_ + d * stride_d_ + h * stride_h_ + w * stride_w_];
    }

    __device__ __forceinline__ T At(int idx) const { return ptr_[idx]; }

    __device__ __forceinline__ T& At(int n, int c, int d, int h, int w) {
        return ptr_[n * stride_n_ + c * stride_c_ + d * stride_d_ + h * stride_h_ + w * stride_w_];
    }

    __device__ __forceinline__ T& At(int idx) { return ptr_[idx]; }

    __device__ __forceinline__ T* Ptr(int n, int c, int d, int h, int w) {
        return &ptr_[n * stride_n_ + c * stride_c_ + d * stride_d_ + h * stride_h_ + w * stride_w_];
    }

    __device__ __forceinline__ T* Ptr(int idx) { return &ptr_[idx]; }

    __device__ __forceinline__ bool IsNull() const { return ptr_ == nullptr; }
};

template <class T>
class CuConstAccessor {
 public:
    const cudnn::impl::meta::CuMetaTensor meta_;
    const int nb_dims_;
    const int stride_n_;
    const int stride_c_;
    const int stride_d_;
    const int stride_h_;
    const int stride_w_;
    const T* ptr_;

 public:
    __device__ __host__ __forceinline__ CuConstAccessor(const cudnn::impl::meta::CuMetaTensor desc,
                                                        const T* ptr)
        : meta_(desc),
          nb_dims_(meta_.GetNbDims()),
          stride_n_(meta_.GetStride(1)),
          stride_c_(meta_.GetStride(2)),
          stride_d_(nb_dims_ == 4 ? 0 : meta_.GetStride(3)),
          stride_h_(meta_.GetStride(nb_dims_ - 1)),
          stride_w_(meta_.GetStride(nb_dims_)),
          ptr_(ptr) {}

    __device__ __forceinline__ T At(int n, int c, int d, int h, int w) const {
        return ptr_[n * stride_n_ + c * stride_c_ + d * stride_d_ + h * stride_h_ + w * stride_w_];
    }

    __device__ __forceinline__ T At(int idx) const { return ptr_[idx]; }

    __device__ __forceinline__ const T* Ptr(int n, int c, int d, int h, int w) const {
        return &ptr_[n * stride_n_ + c * stride_c_ + d * stride_d_ + h * stride_h_ + w * stride_w_];
    }

    __device__ __forceinline__ const T* Ptr(int idx) const { return &ptr_[idx]; }

    __device__ __forceinline__ bool IsNull() const { return ptr_ == nullptr; }
};

/** @class
 * @brief tensor parameters
 */
template <cudnnDataType_t>
struct CuParamTensor {
    __device__ __host__ __forceinline__ CuParamTensor(const cudnn::impl::meta::CuMetaTensor&,
                                                      void*) {
        assert(false);
    }
};

template <cudnnDataType_t>
struct CuParamConstTensor {
    __device__ __host__ __forceinline__ CuParamConstTensor(const cudnn::impl::meta::CuMetaTensor&,
                                                           void*) {
        assert(false);
    }
};

#define TENSOR_DEF_BODY                                                                         \
    __device__ __host__ __forceinline__ CuParamTensor(                                          \
        const cudnn::impl::meta::CuMetaTensor descriptor, void* tensor_ptr)                     \
        : CuAccessor(descriptor, reinterpret_cast<Type*>(tensor_ptr)) {}                        \
                                                                                                \
    __device__ __host__ __forceinline__ Type* GetPtr() { return ptr_; }                         \
                                                                                                \
    __device__ __host__ __forceinline__ bool IsNull() const { return ptr_ == nullptr; }         \
                                                                                                \
    __device__ __host__ __forceinline__ const cudnn::impl::meta::CuMetaTensor GetMeta() const { \
        return meta_;                                                                           \
    }

#define CONST_TENSOR_DEF_BODY                                                                   \
    __device__ __host__ __forceinline__ CuParamConstTensor(                                     \
        const cudnn::impl::meta::CuMetaTensor descriptor, const void* tensor_ptr)               \
        : CuConstAccessor(descriptor, reinterpret_cast<const Type*>(tensor_ptr)) {}             \
                                                                                                \
    __device__ __host__ __forceinline__ const Type* GetPtr() const { return ptr_; }             \
                                                                                                \
    __device__ __host__ __forceinline__ bool IsNull() const { return ptr_ == nullptr; }         \
                                                                                                \
    __device__ __host__ __forceinline__ const cudnn::impl::meta::CuMetaTensor GetMeta() const { \
        return meta_;                                                                           \
    }

// float type
template <>
struct CuParamTensor<DNN_DATA_FLOAT> : public CuAccessor<float> {
    typedef float Type;
    TENSOR_DEF_BODY
};

template <>
struct CuParamConstTensor<DNN_DATA_FLOAT> : public CuConstAccessor<float> {
    typedef float Type;
    CONST_TENSOR_DEF_BODY
};

// double type
template <>
struct CuParamTensor<DNN_DATA_DOUBLE> : public CuAccessor<double> {
    typedef double Type;
    TENSOR_DEF_BODY
};

template <>
struct CuParamConstTensor<DNN_DATA_DOUBLE> : public CuConstAccessor<double> {
    typedef double Type;
    CONST_TENSOR_DEF_BODY
};

// half type
template <>
struct CuParamTensor<DNN_DATA_HALF> : public CuAccessor<__half> {
    typedef __half Type;
    TENSOR_DEF_BODY
};

template <>
struct CuParamConstTensor<DNN_DATA_HALF> : public CuConstAccessor<__half> {
    typedef __half Type;
    CONST_TENSOR_DEF_BODY
};

// int8_t type
template <>
struct CuParamTensor<DNN_DATA_INT8> : public CuAccessor<int8_t> {
    typedef int8_t Type;
    TENSOR_DEF_BODY
};

template <>
struct CuParamConstTensor<DNN_DATA_INT8> : public CuConstAccessor<int8_t> {
    typedef int8_t Type;
    CONST_TENSOR_DEF_BODY
};

// uint8_t type
template <>
struct CuParamTensor<DNN_DATA_UINT8> : public CuAccessor<uint8_t> {
    typedef uint8_t Type;
    TENSOR_DEF_BODY
};

template <>
struct CuParamConstTensor<DNN_DATA_UINT8> : public CuConstAccessor<uint8_t> {
    typedef uint8_t Type;
    CONST_TENSOR_DEF_BODY
};

// int32_t type
template <>
struct CuParamTensor<DNN_DATA_INT32> : public CuAccessor<int32_t> {
    typedef int32_t Type;
    TENSOR_DEF_BODY
};

template <>
struct CuParamConstTensor<DNN_DATA_INT32> : public CuConstAccessor<int32_t> {
    typedef int32_t Type;
    CONST_TENSOR_DEF_BODY
};

}  // namespace param
}  // namespace impl
}  // namespace cudnn
