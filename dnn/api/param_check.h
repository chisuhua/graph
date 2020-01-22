/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

#include <map>

namespace cudnn {
namespace api {

#define CHECK_NULL(p)                                         \
    do {                                                      \
        if (p == nullptr) {                                   \
            cudnn::GetLogger()->info("{} is null", #p);       \
            throw cudnn::CuException(DNN_STATUS_BAD_PARAM); \
        }                                                     \
    } while (false)

#define CHECK_LOWER_BOUND(val, lower_bound, err_code)                                  \
    do {                                                                               \
        if (val < lower_bound) {                                                       \
            cudnn::GetLogger()->info("{}={} is less than {}", #val, val, lower_bound); \
            throw cudnn::CuException(err_code);                                        \
        }                                                                              \
    } while (false)

#define CHECK_UPPER_BOUND(val, upper_bound, err_code)                                     \
    do {                                                                                  \
        if (val > upper_bound) {                                                          \
            cudnn::GetLogger()->info("{}={} is greater than {}", #val, val, upper_bound); \
            throw cudnn::CuException(err_code);                                           \
        }                                                                                 \
    } while (false)

#define CHECK_RANGE(val, lower_bound, upper_bound, err_code) \
    do {                                                     \
        CHECK_LOWER_BOUND(val, lower_bound, err_code);       \
        CHECK_UPPER_BOUND(val, upper_bound, err_code);       \
    } while (false)

#define CHECK_EQ(target, val, err_code)                               \
    do {                                                              \
        if (target != val) {                                          \
            cudnn::GetLogger()->info("{} should be {}", val, target); \
            throw cudnn::CuException(err_code);                       \
        }                                                             \
    } while (false)

template <typename T>
inline void CheckNull(T v) {
    CHECK_NULL(v);
}

/**
 * @brief check given parameter pointers are null or not
 *
 * @param[in]   first   the first parameter pointer to check
 * @param[in]   args    rest parameter pointers to check
 * @throw CuException with DNN_STATUS_BAD_PARAM when one of parameter pointers is null
 */
template <typename T, typename... Args>
inline void CheckNull(T first, Args... args) {
    CheckNull(first);
    CheckNull(args...);
}

/**
 * check if the datatype is supported or not
 *
 * @param[in] data_type data_type to check
 * @param[in] err_code  error code to throw with when checking failed
 * @throw CuException with specified error code when the data type is not supported
 */
inline void CheckDataType(cudnnDataType_t data_type,
                          cudnnStatus_t err_code = DNN_STATUS_BAD_PARAM) {
    static std::map<cudnnDataType_t, bool> kSupportedDataTypes = {
        {DNN_DATA_FLOAT, true},
        {DNN_DATA_DOUBLE, false},  // DON'T support double
        {DNN_DATA_HALF, true},
        {DNN_DATA_INT8, true},
        {DNN_DATA_UINT8, true},
        {DNN_DATA_INT32, true},
        {DNN_DATA_INT8x4, true},
        {DNN_DATA_UINT8x4, true},
    };

    CHECK_RANGE(data_type, DNN_DATA_FLOAT, DNN_DATA_UINT8x4, err_code);

    if (!kSupportedDataTypes[data_type]) {
        GetLogger()->info("data type '{}' is not supported", data_type);
        throw CuException(err_code);
    }
}

/**
 * check if given tensor descriptor is 4d or 5d
 *
 * @param[in]   desc        tensor descriptor to check
 * @param[in]   err_code    error code set to CuException, default is DNN_STATUS_NOT_SUPPORTED
 * @throw CuException with error code
 */
inline void CheckIs4dOr5dTensor(const cudnn::impl::CuTensorDescriptor& desc,
                                cudnnStatus_t err_code = DNN_STATUS_NOT_SUPPORTED) {
    const auto nb_dims = desc.GetNbDims();
    if (nb_dims != 4 && nb_dims != 5) {
        GetLogger()->info("{} dimensions tensor is not supported", nb_dims);
        throw CuException(err_code);
    }
}

/**
 * check if two tensor descriptor have same strides
 *
 * @param[in]   x       1st tensor descriptor to compare
 * @param[in]   y       2nd tensor descriptor to compare
 * @param[in]   code    cudnnStatus_t code to be thrown with if any difference,
                        default is DNN_STATUS_BAD_PARAM
 * @throw CuException with 'code' if any difference in strides
 */
inline void CheckStrideDiffer(const cudnn::impl::CuTensorDescriptor& x,
                              const cudnn::impl::CuTensorDescriptor& y,
                              cudnnStatus_t code = DNN_STATUS_BAD_PARAM) {
    if (x.GetStride() != y.GetStride()) {
        GetLogger()->info("two tensor descriptor have different strides");
        throw CuException(code);
    }
}

/**
 * check if two tensor descriptor have same dimensions
 *
 * @param[in]   x       1st tensor descriptor to compare
 * @param[in]   y       2nd tensor descriptor to compare
 * @param[in]   code    cudnnStatus_t code to be thrown with if any difference,
 *                      default is DNN_STATUS_BAD_PARAM
 * @throw CuException with 'code' if any difference in dimensions
 */
inline void CheckDimensionDiffer(const cudnn::impl::CuTensorDescriptor& x,
                                 const cudnn::impl::CuTensorDescriptor& y,
                                 cudnnStatus_t code = DNN_STATUS_BAD_PARAM) {
    if (x.GetDim() != y.GetDim()) {
        GetLogger()->info("two tensor descriptor have different dimensions");
        throw CuException(code);
    }
}

/**
 * check if specified dimensions of two tensor descriptors differ.
 * NOTE: dimension is 1-base
 *
 * @param[in]   x       1st tensor descriptor to compare
 * @param[in]   y       2nd tensor descriptor to compare
 * @param[in]   dims    dimensions list to compare for
 * @param[in]   code    cudnnStatus_t code to be thrown with if any difference,
 *                      default is DNN_STATUS_BAD_PARAM
 * @throw CuException with 'code' if any difference in dimensions
 */
inline void CheckDimensionDiffer(const cudnn::impl::CuTensorDescriptor& x,
                                 const cudnn::impl::CuTensorDescriptor& y,
                                 std::initializer_list<int> dims,
                                 cudnnStatus_t code = DNN_STATUS_BAD_PARAM) {
    const auto dim_x = x.GetDim();
    const auto dim_y = y.GetDim();
    for (auto i : dims) {
        if (dim_x[i - 1] != dim_y[i - 1]) {
            GetLogger()->info("two tensor descriptor differ in {}-dimension, {} vs {}",
                              i,
                              dim_x[i - 1],
                              dim_y[i - 1]);
            throw CuException(code);
        }
    }
}

/**
 * check if two tensor descriptor have same data type
 * @param[in]   x       1st tensor descriptor to compare
 * @param[in]   y       2nd tensor descriptor to compare
 * @param[in]   code    cudnnStatus_t code to be thrown with if any difference,
 *                      default is DNN_STATUS_BAD_PARAM
 * @throw CuException with 'code' if data types are not same
 */
inline void CheckDataTypeDiffer(const cudnn::impl::CuTensorDescriptor& x,
                                const cudnn::impl::CuTensorDescriptor& y,
                                cudnnStatus_t code = DNN_STATUS_BAD_PARAM) {
    if (x.GetDataType() != y.GetDataType()) {
        GetLogger()->info("two tensor descriptor have different dimensions");
        throw CuException(code);
    }
}

/**
 * check if a tensor descript reprensents a fully-packed tensor
 *
 * @param[in]   desc    descriptor to be checked with
 * @param[in]   code    cudnnStatus_t code to be thrown with if any difference,
 *                      default is DNN_STATUS_BAD_PARAM
 * @throw CuException with 'code' if it's not a fully-packed tensor
 */
inline void CheckFullyPackedTensor(const cudnn::impl::CuTensorDescriptor& desc,
                                   cudnnStatus_t code = DNN_STATUS_BAD_PARAM) {
    for (int dim = desc.GetNbDims(); dim > 1; --dim) {
        if (desc.GetStride(dim - 1) != desc.GetDim(dim) * desc.GetStride(dim)) {
            throw CuException(code);
        }
    }
}

}  // namespace api
}  // namespace cudnn
