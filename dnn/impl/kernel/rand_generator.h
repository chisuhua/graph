/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif

#include "ixrand_precalc.h"

#define SKIPAHEAD_BLOCKSIZE (4)
#define SKIPAHEAD_MASK ((1 << SKIPAHEAD_BLOCKSIZE) - 1)
#define XORWOW_N (5)

#define IXRAND_2POW32_INV (2.3283064e-10f)

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

namespace cudnn {
namespace impl {
namespace kernel {

namespace detail {
FQUALIFIERS void copy_vec(unsigned int* dst, const unsigned int* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

FQUALIFIERS void copy_mat(unsigned int* matrix, unsigned int* matrixA, int n) {
    for (int i = 0; i < n * n * 32; i++) {
        matrix[i] = matrixA[i];
    }
}

FQUALIFIERS void
__rand_matvec(unsigned int* vector, unsigned int* matrix, unsigned int* result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 32; j++) {
            if (vector[i] & (1 << j)) {
                for (int k = 0; k < n; k++) {
                    result[k] ^= matrix[n * (i * 32 + j) + k];
                }
            }
        }
    }
    copy_vec(vector, result, XORWOW_N);
}

FQUALIFIERS void __rand_matmat(unsigned int* matrixA, unsigned int* matrixB, int n) {
    unsigned int result[XORWOW_N];
    for (int i = 0; i < n * 32; i++) {
        __rand_matvec(matrixA + i * n, matrixB, result, n);
        for (int j = 0; j < n; j++) {
            matrixA[i * n + j] = result[j];
        }
    }
}

}  // end namespace detail

// xorwow algorithm
class rand_engine {
 public:
    struct xorwow_state {
        unsigned int d, v[5];
        /*
        int boxmuller_flag;
        int boxmuller_flag_double;
        float boxmuller_extra;
        double boxmuller_extra_double;
        */
    };

    FQUALIFIERS rand_engine() : rand_engine(0ULL, 0, 0) {}
    FQUALIFIERS rand_engine(const unsigned long long seed,
                            const unsigned long long subsequence,
                            const unsigned long long offset) {
        unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
        unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;

        unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
        // Simple multiplication to mix up bits
        // Constants are arbitrary odd values
        unsigned int t0 = 1099087573UL * s0;
        unsigned int t1 = 2591861531UL * s1;
        m_state.d       = 6615241 + t1 + t0;
        m_state.v[0]    = 123456789UL + t0;
        m_state.v[1]    = 362436069UL ^ t0;
        m_state.v[2]    = 521288629UL + t1;
        m_state.v[3]    = 88675123UL ^ t1;
        m_state.v[4]    = 5783321UL + t0;

        discard_subsequence(subsequence, scratch);
        discard(offset, scratch);
        /*
            m_state.boxmuller_flag = 0;
            m_state.boxmuller_flag_double = 0;
            m_state.boxmuller_extra = 0.f;
            m_state.boxmuller_extra_double = 0.;
        */
    }
    FQUALIFIERS void discard_subsequence(unsigned long long subsequence, unsigned int* scratch) {
        jump(subsequence, scratch, XORWOW_N);
    }

    FQUALIFIERS void discard(unsigned long long offset, unsigned int* scratch) {
        jump(offset, scratch, XORWOW_N);
        m_state.d += static_cast<unsigned int>(offset) * 362437;
    }

    FQUALIFIERS
    unsigned int operator()() { return next(); }

    FQUALIFIERS
    unsigned int next() {
        const unsigned int t = m_state.v[0] ^ (m_state.v[0] >> 2);
        m_state.v[0]         = m_state.v[1];
        m_state.v[1]         = m_state.v[2];
        m_state.v[2]         = m_state.v[3];
        m_state.v[3]         = m_state.v[4];
        m_state.v[4]         = (m_state.v[4] ^ (m_state.v[4] << 4)) ^ (t ^ (t << 1));

        m_state.d += 362437;

        return m_state.d + m_state.v[4];
    }

 protected:
    xorwow_state m_state;
    FQUALIFIERS void jump(unsigned long long x, unsigned int* scratch, int n) {
        // unsigned int matrix[n * n * 32];
        unsigned int* matrix = scratch;
        // unsigned int matrixA[n * n * 32];
        unsigned int* matrixA = scratch + (n * n * 32);
        // unsigned int vector[n];
        unsigned int* vector = scratch + (n * n * 32) + (n * n * 32);
        // unsigned int result[n];
        unsigned int* result = scratch + (n * n * 32) + (n * n * 32) + n;
        unsigned long long p = x;
        for (int i = 0; i < n; i++) {
            vector[i] = m_state.v[i];
        }
        int matrix_num = 0;
        while (p && matrix_num < PRECALC_NUM_MATRICES - 1) {
            for (unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
                detail::__rand_matvec(vector, precalc_xorwow_matrix[matrix_num], result, n);
                // __curand_veccopy(vector, result, n);
            }
            p >>= PRECALC_BLOCK_SIZE;
            matrix_num++;
        }
        if (p) {
            detail::copy_mat(matrix, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
            detail::copy_mat(matrixA, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
        }
        while (p) {
            for (unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
                detail::__rand_matvec(vector, matrixA, result, n);
                // __curand_veccopy(vector, result, n);
            }
            p >>= SKIPAHEAD_BLOCKSIZE;
            if (p) {
                for (int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
                    detail::__rand_matmat(matrix, matrixA, n);
                    detail::copy_mat(matrixA, matrix, n);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            m_state.v[i] = vector[i];
        }
    }
};

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn

typedef cudnn::impl::kernel::rand_engine ixrand_state_xorwow;
FQUALIFIERS
void ixrand_init(const unsigned long long seed,
                 const unsigned long long subsequence,
                 const unsigned long long offset,
                 ixrand_state_xorwow* state) {
    *state = ixrand_state_xorwow(seed, subsequence, offset);
}

FQUALIFIERS
unsigned int ixrand_rand(ixrand_state_xorwow* state) { return state->next(); }

FQUALIFIERS
float ixrand_uniform_distribution(unsigned int v) {
    return IXRAND_2POW32_INV + (v * IXRAND_2POW32_INV);
}
