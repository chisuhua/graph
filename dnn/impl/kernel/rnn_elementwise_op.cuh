/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>

#include "stdio.h"  // debug only

namespace {
template <class T>
__forceinline__ __device__ T Sigmoid(T in) {
    return 1.f / (1.f + expf(-in));
}
}  // namespace

namespace cudnn {
namespace impl {
namespace kernel {

/*
 * @brief ReLU forward / training forward kernel
 * theta = wx + rh + bw + br
 * h = ReLU(theta)
 * if training, theta should be saved to rspace_lin
 *              hidden should be also saved to rspace_hidden
 */
template <class T>
__global__ void FusedReLUForward(const T* wx,
                                 const T* rh,
                                 const T* bw,
                                 const T* br,
                                 T* hidden,
                                 T* hy,
                                 T* reserve_hidden,
                                 T* reserve_lin,
                                 int nc,
                                 int batch_size,
                                 int bi) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * nc) {
        return;
    }
    const int bias_idx = idx % nc;
    const int batch    = idx / nc;

    T lin                              = wx[idx] + rh[idx] + bw[bias_idx] + br[bias_idx];
    T tmp_h                            = (lin > 0) * lin;
    hy[idx]                            = tmp_h;
    hidden[batch * nc * bi + bias_idx] = tmp_h;

    if (reserve_hidden != nullptr) {
        reserve_hidden[idx] = tmp_h;
        reserve_lin[idx]    = lin;
    }
}

/*
 * @brief TANH forward / training forward kernel
 * theta = wx + rh + bw + br
 * h = tanh(theta)
 * if training, theta should be saved to rspace_lin
 *              hidden should be also saved to rspace_hidden
 */
template <class T>
__global__ void FusedTanhForward(const T* wx,
                                 const T* rh,
                                 const T* bw,
                                 const T* br,
                                 T* hidden,
                                 T* hy,
                                 T* reserve_hidden,
                                 T* reserve_hidden2,
                                 int nc,
                                 int batch_size,
                                 int bi) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * nc) {
        return;
    }
    const int bias_idx = idx % nc;
    const int batch    = idx / nc;

    T lin                              = wx[idx] + rh[idx] + bw[bias_idx] + br[bias_idx];
    T tmp_h                            = tanhf(lin);
    hy[idx]                            = tmp_h;
    hidden[batch * nc * bi + bias_idx] = tmp_h;

    if (reserve_hidden != nullptr) {
        reserve_hidden[idx]                         = tmp_h;
        reserve_hidden2[batch * nc * bi + bias_idx] = tmp_h;
    }
}

template <class T>
__global__ void FusedLSTMForward(const T* wx,
                                 const T* rh,
                                 const T* bw,
                                 const T* br,
                                 const T* in_cy,
                                 T* hidden,
                                 T* hy,
                                 T* cy,
                                 T* rhidden,
                                 T* rcell,
                                 T* rgates,
                                 bool skip_input,
                                 int nc,
                                 int batch_size,
                                 int bi) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * nc) {
        return;
    }
    const int bias_idx = idx % nc;
    const int batch    = idx / nc;
    const int base     = nc * batch * 4;

    T gates[4];
#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
        if (!skip_input) {
            gates[i] = wx[base + nc * i + bias_idx] + rh[base + nc * i + bias_idx] +
                       bw[nc * i + bias_idx] + br[nc * i + bias_idx];
        } else {
            gates[i] = wx[nc * batch + bias_idx] + rh[base + nc * i + bias_idx] +
                       bw[nc * i + bias_idx] + br[nc * i + bias_idx];
        }
    }

    gates[0] = Sigmoid(gates[0]);
    gates[1] = Sigmoid(gates[1]);
    gates[2] = tanhf(gates[2]);
    gates[3] = Sigmoid(gates[3]);

    T cell_state = gates[1] * in_cy[nc * batch + bias_idx] + gates[0] * gates[2];
    T tmp_h      = gates[3] * tanhf(cell_state);

    hy[idx]                            = tmp_h;
    hidden[nc * batch * bi + bias_idx] = tmp_h;
    cy[nc * batch + bias_idx]          = cell_state;

    if (rhidden != nullptr) {
        rhidden[idx]                 = tmp_h;
        rcell[nc * batch + bias_idx] = cell_state;
    }
}

template <class T>
__global__ void FusedGRUForward(const T* wx,
                                const T* rh,
                                const T* bw,
                                const T* br,
                                T* hidden,
                                T* hy,
                                T* rhidden,
                                T* rgates,
                                bool skip_input,
                                int nc,
                                int batch_size,
                                int bi) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * nc) {
        return;
    }
    const int bias_idx = idx % nc;
    const int batch    = idx / nc;
    const int base     = nc * batch * 3;

    T gates[3];
#pragma unroll 2
    for (int i = 0; i < 2; ++i) {
        if (!skip_input) {
            gates[i] = wx[base + nc * i + bias_idx] + rh[base + nc * i + bias_idx] +
                       bw[nc * i + bias_idx] + br[nc * i + bias_idx];
        } else {
            gates[i] = wx[nc * batch + bias_idx] + rh[base + nc * i + bias_idx] +
                       bw[nc * i + bias_idx] + br[nc * i + bias_idx];
        }
    }

    gates[0] = Sigmoid(gates[0]);
    gates[1] = Sigmoid(gates[1]);

    T wxbw;
    T rhbr = rh[base + nc * 2 + bias_idx] + br[nc * 2 + bias_idx];
    if (!skip_input) {
        wxbw = wx[base + nc * 2 + bias_idx] + bw[nc * 2 + bias_idx];
    } else {
        wxbw = wx[nc * batch + bias_idx] + bw[nc * 2 + bias_idx];
    }

    gates[2] = tanhf(wxbw + gates[0] * rhbr);

    T tmp_h                            = (1 - gates[1]) * gates[2] + gates[1] * hy[idx];
    hidden[nc * batch * bi + bias_idx] = tmp_h;
    hy[idx]                            = tmp_h;

    if (rhidden != nullptr) {
        rhidden[idx]                               = tmp_h;
        rgates[nc * batch * 6 + bias_idx]          = gates[0];
        rgates[nc * batch * 6 + nc + bias_idx]     = gates[1];
        rgates[nc * batch * 6 + nc * 2 + bias_idx] = wxbw;
        rgates[nc * batch * 6 + nc * 3 + bias_idx] = rhbr;
    }
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
