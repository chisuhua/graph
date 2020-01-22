/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>

#define DNN_MATH_FUNC_DEF __forceinline__ __device__

// TODO(Peter Han): all intrinsics/math functions need to disscuss with Driver/Compiler/HW team

namespace cudnn {
namespace impl {
namespace kernel {
/**
 * determine whether argument is a Nan
 *
 * @param[in]   x   value to check
 * @return true if NaN, false otherwise
 */
template <class T>
DNN_MATH_FUNC_DEF bool CuIsnan(T x) {
    return static_cast<T>(isnan(static_cast<float>(x)));
}

template <>
DNN_MATH_FUNC_DEF bool CuIsnan<float>(float x) {
    return isnan(x);
}

/**
 * calculates x's natural exponential function in round-to-nearest mode.
 *
 * @param[in]   x   value to calculate
 * @return  x's natural exponential
 */
template <class T>
DNN_MATH_FUNC_DEF T CuExp(T x) {
    return static_cast<T>(expf(static_cast<float>(x)));
}

template <>
DNN_MATH_FUNC_DEF float CuExp<float>(float x) {
    return expf(x);
}

/**
 * calculate the base e logarithm of the input argument
 *
 * @param[in]   x   value to calculate
 * @return  base e logarithm of the input x
 */
template <class T>
DNN_MATH_FUNC_DEF T CuLog(T x) {
    return static_cast<T>(log(static_cast<float>(x)));
}

template <>
DNN_MATH_FUNC_DEF float CuLog<float>(float x) {
    return log(x);
}

/**
 * determine the maximum numeric value of the arguments
 *
 * @param[in]   x   one of the values to be determined
 * @param[in]   y   the other value to be determined
 * @return  the maximum numeric vlaues of the arguments x and y.
 *          If both arguments are NaN, returns NaN
 *          If one argument is NaN, return the numeric argument.
 */
template <class T>
DNN_MATH_FUNC_DEF T CuFmax(T x, T y) {
    /*
    if (CuIsnan(x) && CuIsnan(y)) {
        return x;
    }
    if (IsIsnan(x)) {
        return y;
    }
    if (IsIsnan(y)) {
        return x;
    }
    */
    return static_cast<T>(fmaxf(static_cast<float>(x), static_cast<float>(y)));
}

template <>
DNN_MATH_FUNC_DEF float CuFmax(float x, float y) {
    return fmaxf(x, y);
}

/**
 * determin the minimumu numeric value of the arguments
 *
 * @param[in]   x   one of the values to be determined
 * @param[in]   y   the other value to be determined
 * @return  the minimum numeric vlaues of the arguments x and y.
 *          If both arguments are NaN, returns NaN
 *          If one argument is NaN, return the numeric argument.
 */
template <class T>
DNN_MATH_FUNC_DEF T CuFmin(T x, T y) {
    return static_cast<T>(fminf(static_cast<float>(x), static_cast<float>(y)));
}

template <>
DNN_MATH_FUNC_DEF float CuFmin(float x, float y) {
    return fminf(x, y);
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
