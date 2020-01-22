/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn/impl/kernel/rand_generator.h>

namespace cudnn {
namespace impl {
namespace kernel {
    __global__ void randstate_init_kernel(void* states, unsigned long long seed) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        ixrand_state_xorwow* current_state = static_cast<ixrand_state_xorwow*>(states) + tid;
        ixrand_init(seed, tid, 0, current_state);
    }

    template<class T>
    __global__ void dropout_fwd_kernel(int nthreads,
                                       const T* x,
                                       T* y,
                                       float* reserveSpace,
                                       void* states,
                                       float dropout_ratio) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        float scale_value = 1.f / (1.f - dropout_ratio);
        ixrand_state_xorwow* current_state = static_cast<ixrand_state_xorwow*>(states) + tid;
        CUDA_KERNEL_LOOP(index, nthreads) {
            unsigned int mask = ixrand_rand(current_state);
            float mask_result = ixrand_uniform_distribution(mask);
            reserveSpace[index] = mask_result;
            y[index] = (mask_result > dropout_ratio) ? x[index] * scale_value : 0;
        }
    }

    template<class T>
    __global__ void dropout_bwd_kernel(int nthreads,
                                       const T* dy,
                                       T* dx,
                                       float* reserveSpace,
                                       float dropout_ratio) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            float mask_result = reserveSpace[index];
            T input = dy[index];
            T result = (mask_result > dropout_ratio) ? input : 0;
            dx[index] = result;
        }
    }

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
