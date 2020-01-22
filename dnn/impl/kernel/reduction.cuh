/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once

namespace cudnn {
namespace impl {
namespace kernel {

/*
 * @brief sum reduction for any size lenght shared memory
 *
 * @param[in]       tid         threadIdx.x
 * @param[in/out]   sdata       shared memory with data to sum
 * @param[in]       len         sdata length
 */
template <typename T>
__forceinline__ __device__ void ReductionSum(int tid, volatile T* sdata, int len) {
    auto pow2 = len;
    if (pow2 & (pow2 - 1)) {
        while (pow2 & (pow2 - 1)) {
            pow2 &= (pow2 - 1);
        }
        if (tid >= pow2) {
            sdata[tid - pow2] = sdata[tid - pow2] + sdata[tid];
        }
        __syncthreads();
    }

    if (pow2 == 1024) {
        if (tid < 512) {
            sdata[tid] = sdata[tid] + sdata[tid + 512];
        }
        __syncthreads();
    }

    if (pow2 >= 512) {
        if (tid < 256) {
            sdata[tid] = sdata[tid] + sdata[tid + 256];
        }
        __syncthreads();
    }

    if (pow2 >= 256) {
        if (tid < 128) {
            sdata[tid] = sdata[tid] + sdata[tid + 128];
        }
        __syncthreads();
    }

    if (pow2 >= 128) {
        if (tid < 64) {
            sdata[tid] = sdata[tid] + sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (pow2 >= 64 && tid < 32) {
            sdata[tid] = sdata[tid] + sdata[tid + 32];
        }

        if (pow2 >= 32 && tid < 16) {
            sdata[tid] = sdata[tid] + sdata[tid + 16];
        }

        if (pow2 >= 16 && tid < 8) {
            sdata[tid] = sdata[tid] + sdata[tid + 8];
        }

        if (pow2 >= 8 && tid < 4) {
            sdata[tid] = sdata[tid] + sdata[tid + 4];
        }

        if (pow2 >= 4 && tid < 2) {
            sdata[tid] = sdata[tid] + sdata[tid + 2];
        }

        if (pow2 >= 2 && tid < 1) {
            sdata[tid] = sdata[tid] + sdata[tid + 1];
        }
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
