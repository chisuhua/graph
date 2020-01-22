/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cudnn.h>

#include "stdio.h"

namespace cudnn {
namespace impl {
namespace kernel {

template <class T>
__global__ void Gemm(T* c, const T* a, const T* b, int m, int k, int n) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n || y >= m) {
        return;
    }

    T res = static_cast<T>(0);
    for (int i = 0; i < k; ++i) {
        int idxa = k * y + i;
        int idxb = k * x + i;
        res += a[idxa] * b[idxb];
    }
    c[m * x + y] = res;
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
