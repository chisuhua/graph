/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn.h>
#include <cudnn/impl/cudnn_rnn_params.h>
#include <cudnn/impl/cudnn_rnn_reserve_space.h>
#include <cudnn/impl/cudnn_rnn_work_space_forward.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/kernel/cudnn_gemm.cuh>
#include <cudnn/impl/kernel/cudnn_rnn_elementwise_op.cuh>
#include <cudnn/impl/meta/cudnn_meta_filter.h>
#include <cudnn/impl/meta/cudnn_meta_rnn.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <utility>
#include <vector>

namespace {
using cudnn::GetLogger;
using cudnn::impl::kernel::Gemm;

constexpr int BLOCK_DIM = 256;

template <class T>
void RnnGemm(cudaStream_t stream, const T* a, const T* b, T* c, int m, int k, int n) {
    dim3 block;
    block.x = 32;
    block.y = 32;
    block.z = 1;
    dim3 grid;
    grid.x = (n + block.x - 1) / block.x;
    grid.y = (m + block.y - 1) / block.y;
    grid.z = 1;
    cudnn::GetLogger()->debug("m={}, k={}, n={}", m, k, n);
    Gemm<T><<<grid, block, 0, stream>>>(c, a, b, m, k, n);
}

#define CUDA_FUNC_CALL_AND_THROW(func, ...)                               \
    do {                                                                  \
        auto code = func(__VA_ARGS__);                                    \
        if (code != cudaSuccess) {                                        \
            GetLogger()->warn("{}: {} return {}", __LINE__, #func, code); \
            throw cudnn::CuException(DNN_STATUS_EXECUTION_FAILED);      \
        }                                                                 \
    } while (false)

#define CUDA_FUNC_CALL_QUIET(func, ...)                                   \
    do {                                                                  \
        auto code = func(__VA_ARGS__);                                    \
        if (code != cudaSuccess) {                                        \
            GetLogger()->warn("{}: {} return {}", __LINE__, #func, code); \
        }                                                                 \
    } while (false)

int ComputeTotalBatchSize(const std::vector<cudnn::impl::meta::CuMetaTensor>& x_metas) {
    int total = 0;
    for (int i = 0; i < x_metas.size(); ++i) {
        total += x_metas[i].GetDim(1);
    }
    return total;
}

}  // namespace

namespace cudnn {
namespace impl {

using cudnn::impl::CuDropoutDescriptor;
using cudnn::impl::CuRnnParams;
using cudnn::impl::CuRnnReserveSpace;
using cudnn::impl::CuRnnWorkSpaceForward;
using cudnn::impl::meta::CuMetaDropout;
using cudnn::impl::meta::CuMetaFilter;
using cudnn::impl::meta::CuMetaRnn;
using cudnn::impl::meta::CuMetaTensor;

template <class T>
void CuRnnForward(
    const cudaStream_t fwd_stream,             // default stream used for first to last path
    const CuMetaRnn& rnn_meta,                 // rnn meta
    CuDropoutDescriptor* dropout_desc,         // dropout descriptor
    const std::vector<CuMetaTensor>& x_metas,  // input tensors of each time step
                                               // dimension : {batch size, input size, 1}
    const T* x,                                //
    const CuMetaTensor& hx_meta,               // init hiden state, if null, init with all 0
                                               // Unidirectional mode :
                                               // dimension : {layers, batch size, hidden size}
                                               // Bidirectional mode :
                                               // dimension : {layers * 2, batch size, hidden size}
    const T* hx,                               //
    const CuMetaTensor& cx_meta,               // init cell state for LSTM, if null, init with all 0
                                               // Unidirectional mode :
                                               // dimension : {layers, batch size, hidden size}
                                               // Bidirectional mode :
                                               // dimension : {layers * 2, batch size, hidden size}
    const T* cx,                               //
    const CuMetaFilter& w_meta,                // filter
    const T* w,                                //
    const std::vector<CuMetaTensor>& y_metas,  // output tensors of each time step
                                               // Unidirectional mode :
                                               // dimension : {batch size, hidden size, 1}
                                               // Bidirectional mode :
                                               // dimension : {batch size * 2, hidden size, 1}
    T* y,                                      //
    const CuMetaTensor& hy_meta,               // final hidden state, if null, then not save out
                                               // Unidirectional mode :
                                               // dimension : {layers, batch size, hidden size}
                                               // Bidirectional mode :
                                               // dimension : {layers * 2, batch size, hidden size}
    T* hy,                                     //
    const CuMetaTensor& cy_meta,               // final cell state, if null, then not save out
                                               // Unidirectional mode :
                                               // dimension : {layers, batch size, hidden size}
                                               // Bidirectional mode :
                                               // dimension : {layers * 2, batch size, hidden size}
    T* cy,                                     //
    T* workspace,                              // workspace
    size_t workspace_size_in_bytes,            //
    T* reserve_space,                          // reserve space
    size_t reserve_space_in_bytes) {
    const auto training         = reserve_space != nullptr;
    const auto mode             = rnn_meta.GetMode();
    const auto nb_layers        = rnn_meta.GetNbLayers();
    const auto nb_ffnn          = rnn_meta.GetNbFfnn();
    const auto bi_mode          = rnn_meta.GetDirectionMode() == DNN_BIDIRECTIONAL;
    const auto bi               = rnn_meta.GetDirectionMode() == DNN_UNIDIRECTIONAL ? 1 : 2;
    const auto nc               = rnn_meta.GetHiddenSize();
    const auto nr               = rnn_meta.GetRecProjSize();
    const auto skip_input       = rnn_meta.GetInputMode() == DNN_SKIP_INPUT;
    const auto seq_len          = x_metas.size();
    const auto total_batch_size = ComputeTotalBatchSize(x_metas);
    const auto max_batch_size   = x_metas[0].GetDim(1);
    const auto ni               = x_metas[0].GetDim(2);
    const auto params = CuRnnParams(rnn_meta, x_metas[0], reinterpret_cast<const void*>(w));

    auto wspace = CuRnnWorkSpaceForward<T>(rnn_meta, x_metas, workspace, x, y, hx, cx, fwd_stream);
    auto rspace = CuRnnReserveSpace<T>(rnn_meta, x_metas, reserve_space, y);
    if (wspace.GetSizeInBytes() < workspace_size_in_bytes) {
        GetLogger()->info("workspace passed in is too small");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }
    if (training && rspace.GetSizeInBytes() < reserve_space_in_bytes) {
        GetLogger()->info("reserve space passed in is too small");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }

    cudaStream_t bwd_stream = nullptr;
    if (bi_mode) {
        bwd_stream = fwd_stream;
        CUDA_FUNC_CALL_QUIET(cudaStreamCreate, &bwd_stream);
    }

    for (int layer_id = 0; layer_id < nb_layers; ++layer_id) {
        const auto fwd_layer_id = layer_id * bi;
        const auto bwd_layer_id = layer_id * bi + 1;

        // compute Wx
        if (layer_id == 0) {
            if (!skip_input) {
                RnnGemm(fwd_stream,
                        params.GetBatchedW<T>(fwd_layer_id),
                        x,
                        wspace.GetWx(fwd_layer_id),
                        nc * nb_ffnn,
                        ni,
                        total_batch_size);
#ifndef NDEBUG
                wspace.Dump("WX FWD");
#endif
                if (bi_mode) {
                    RnnGemm(bwd_stream,
                            params.GetBatchedW<T>(bwd_layer_id),
                            x,
                            wspace.GetWx(bwd_layer_id),
                            nc * nb_ffnn,
                            ni,
                            total_batch_size);
#ifndef NDEBUG
                    wspace.Dump("WX BWD");
#endif
                }
            }
        } else {
            RnnGemm(fwd_stream,
                    params.GetBatchedW<T>(fwd_layer_id),
                    wspace.GetHidden(fwd_layer_id - bi),
                    wspace.GetWx(fwd_layer_id),
                    nc * nb_ffnn,
                    nr * bi,
                    total_batch_size);
#ifndef NDEBUG
            wspace.Dump("WX FWD");
#endif
            if (bi_mode) {
                RnnGemm(bwd_stream,
                        params.GetBatchedW<T>(bwd_layer_id),
                        wspace.GetHidden(fwd_layer_id - bi),
                        wspace.GetWx(bwd_layer_id),
                        nc * nb_ffnn,
                        nr * bi,
                        total_batch_size);
#ifndef NDEBUG
                wspace.Dump("WX BWD");
#endif
            }
        }

        for (int fwd_ts = 0, bwd_ts = seq_len - 1; fwd_ts < seq_len; ++fwd_ts, --bwd_ts) {
            const int fwd_batch_size = x_metas[fwd_ts].GetDim(1);
            const int bwd_batch_size = x_metas[bwd_ts].GetDim(1);

            // compute Rh
            RnnGemm(fwd_stream,
                    params.GetBatchedR<T>(fwd_layer_id),
                    fwd_ts != 0 ? wspace.GetHy(fwd_layer_id) : wspace.GetHx(fwd_layer_id),
                    wspace.GetRh(fwd_layer_id),
                    nc * nb_ffnn,
                    nc,
                    fwd_batch_size);
#ifndef NDEBUG
            wspace.Dump("RH FWD");
#endif

            if (bi_mode) {
                RnnGemm(bwd_stream,
                        params.GetBatchedR<T>(bwd_layer_id),
                        fwd_ts != 0 ? wspace.GetHy(bwd_layer_id) : wspace.GetHx(bwd_layer_id),
                        wspace.GetRh(bwd_layer_id),
                        nc * nb_ffnn,
                        nc,
                        bwd_batch_size);
#ifndef NDEBUG
                wspace.Dump("RH BWD");
#endif
            }

            if (mode == DNN_RNN_RELU) {
                int grid = (fwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                kernel::FusedReLUForward<T><<<grid, BLOCK_DIM, 0, fwd_stream>>>(
                    wspace.GetWx(fwd_layer_id, fwd_ts),
                    wspace.GetRh(fwd_layer_id),
                    params.GetBatchedBw<T>(fwd_layer_id),
                    params.GetBatchedBr<T>(fwd_layer_id),
                    wspace.GetHidden(fwd_layer_id, fwd_ts),
                    wspace.GetHy(fwd_layer_id),
                    training ? rspace.GetHidden(fwd_layer_id, fwd_ts) : nullptr,
                    training ? rspace.GetReLULin(fwd_layer_id, fwd_ts) : nullptr,
                    nc,
                    fwd_batch_size,
                    bi);
#ifndef NDEBUG
                wspace.Dump("RELU OP FWD");
#endif

                if (bi_mode) {
                    int grid = (bwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                    kernel::FusedReLUForward<T><<<grid, BLOCK_DIM, 0, bwd_stream>>>(
                        wspace.GetWx(bwd_layer_id, bwd_ts),
                        wspace.GetRh(bwd_layer_id),
                        params.GetBatchedBw<T>(bwd_layer_id),
                        params.GetBatchedBr<T>(bwd_layer_id),
                        wspace.GetHidden(bwd_layer_id, bwd_ts),
                        wspace.GetHy(bwd_layer_id),
                        training ? rspace.GetHidden(bwd_layer_id, bwd_ts) : nullptr,
                        training ? rspace.GetReLULin(bwd_layer_id, bwd_ts) : nullptr,
                        nc,
                        bwd_batch_size,
                        bi);
#ifndef NDEBUG
                    wspace.Dump("RELU OP BWD");
#endif
                }
            } else if (mode == DNN_RNN_TANH) {
                int grid = (fwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                kernel::FusedTanhForward<T><<<grid, BLOCK_DIM, 0, fwd_stream>>>(
                    wspace.GetWx(fwd_layer_id, fwd_ts),
                    wspace.GetRh(fwd_layer_id),
                    params.GetBatchedBw<T>(fwd_layer_id),
                    params.GetBatchedBr<T>(fwd_layer_id),
                    wspace.GetHidden(fwd_layer_id, fwd_ts),
                    wspace.GetHy(fwd_layer_id),
                    training ? rspace.GetHidden(fwd_layer_id, fwd_ts) : nullptr,
                    training ? rspace.GetTanhHidden(fwd_layer_id, fwd_ts) : nullptr,
                    nc,
                    fwd_batch_size,
                    bi);
#ifndef NDEBUG
                wspace.Dump("TANH OP FWD");
#endif

                if (bi_mode) {
                    int grid = (bwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                    kernel::FusedTanhForward<T><<<grid, BLOCK_DIM, 0, bwd_stream>>>(
                        wspace.GetWx(bwd_layer_id, bwd_ts),
                        wspace.GetRh(bwd_layer_id),
                        params.GetBatchedBw<T>(bwd_layer_id),
                        params.GetBatchedBr<T>(bwd_layer_id),
                        wspace.GetHidden(bwd_layer_id, bwd_ts),
                        wspace.GetHy(bwd_layer_id),
                        training ? rspace.GetHidden(bwd_layer_id, bwd_ts) : nullptr,
                        training ? rspace.GetTanhHidden(bwd_layer_id, bwd_ts) : nullptr,
                        nc,
                        bwd_batch_size,
                        bi);
#ifndef NDEBUG
                    wspace.Dump("TANH OP BWD");
#endif
                }
            } else if (mode == DNN_LSTM) {
                int grid = (fwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                kernel::FusedLSTMForward<T><<<grid, BLOCK_DIM, 0, fwd_stream>>>(
                    wspace.GetWx(fwd_layer_id, fwd_ts),
                    wspace.GetRh(fwd_layer_id),
                    params.GetBatchedBw<T>(fwd_layer_id),
                    params.GetBatchedBr<T>(fwd_layer_id),
                    fwd_ts != 0 ? wspace.GetCy(fwd_layer_id) : wspace.GetCx(fwd_layer_id),
                    wspace.GetHidden(fwd_layer_id, fwd_ts),
                    wspace.GetHy(fwd_layer_id),
                    wspace.GetCy(fwd_layer_id),
                    training ? rspace.GetHidden(fwd_layer_id, fwd_ts) : nullptr,
                    training ? rspace.GetCellState(fwd_layer_id, fwd_ts) : nullptr,
                    training ? rspace.GetLSTMGateResult(fwd_layer_id, fwd_ts) : nullptr,
                    layer_id == 0 && skip_input,
                    nc,
                    fwd_batch_size,
                    bi);
#ifndef NDEBUG
                wspace.Dump("LSTM OP FWD");
#endif

                if (bi_mode) {
                    int grid = (bwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                    kernel::FusedLSTMForward<T><<<grid, BLOCK_DIM, 0, bwd_stream>>>(
                        wspace.GetWx(bwd_layer_id, bwd_ts),
                        wspace.GetRh(bwd_layer_id),
                        params.GetBatchedBw<T>(bwd_layer_id),
                        params.GetBatchedBr<T>(bwd_layer_id),
                        fwd_ts != 0 ? wspace.GetCy(bwd_layer_id) : wspace.GetCx(bwd_layer_id),
                        wspace.GetHidden(bwd_layer_id, bwd_ts),
                        wspace.GetHy(bwd_layer_id),
                        wspace.GetCy(bwd_layer_id),
                        training ? rspace.GetHidden(bwd_layer_id, bwd_ts) : nullptr,
                        training ? rspace.GetCellState(bwd_layer_id, bwd_ts) : nullptr,
                        training ? rspace.GetLSTMGateResult(bwd_layer_id, bwd_ts) : nullptr,
                        layer_id == 0 && skip_input,
                        nc,
                        bwd_batch_size,
                        bi);
#ifndef NDEBUG
                    wspace.Dump("LSTM OP BWD");
#endif
                }
            } else {
                int grid = (fwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                kernel::FusedGRUForward<T><<<grid, BLOCK_DIM, 0, fwd_stream>>>(
                    wspace.GetWx(fwd_layer_id, fwd_ts),
                    wspace.GetRh(fwd_layer_id),
                    params.GetBatchedBw<T>(fwd_layer_id),
                    params.GetBatchedBr<T>(fwd_layer_id),
                    wspace.GetHidden(fwd_layer_id, fwd_ts),
                    wspace.GetHy(fwd_layer_id),
                    training ? rspace.GetHidden(fwd_layer_id, fwd_ts) : nullptr,
                    training ? rspace.GetGRUGateResult(fwd_layer_id, fwd_ts) : nullptr,
                    layer_id == 0 && skip_input,
                    nc,
                    fwd_batch_size,
                    bi);
#ifndef NDEBUG
                wspace.Dump("GRU OP FWD");
#endif

                if (bi_mode) {
                    int grid = (bwd_batch_size * nc + BLOCK_DIM - 1) / BLOCK_DIM;
                    kernel::FusedGRUForward<T><<<grid, BLOCK_DIM, 0, bwd_stream>>>(
                        wspace.GetWx(bwd_layer_id, bwd_ts),
                        wspace.GetRh(bwd_layer_id),
                        params.GetBatchedBw<T>(bwd_layer_id),
                        params.GetBatchedBr<T>(bwd_layer_id),
                        wspace.GetHidden(bwd_layer_id, bwd_ts),
                        wspace.GetHy(bwd_layer_id),
                        training ? rspace.GetHidden(bwd_layer_id, bwd_ts) : nullptr,
                        training ? rspace.GetGRUGateResult(bwd_layer_id, bwd_ts) : nullptr,
                        layer_id == 0 && skip_input,
                        nc,
                        bwd_batch_size,
                        bi);
#ifndef NDEBUG
                    wspace.Dump("GRU OP BWD");
#endif
                }

                CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, fwd_stream);
                if (bi_mode) {
                    CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, bwd_stream);
                }
            }
        }

        if (training && layer_id < nb_layers - 1) {
            // TODO(Peter Han): apply dropout forward
        }
    }

    if (hy != nullptr) {
        CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                 hy,
                                 wspace.GetHy(0),
                                 max_batch_size * nb_layers * bi * nc * sizeof(T),
                                 cudaMemcpyDeviceToDevice,
                                 fwd_stream);
    }
    if (mode == DNN_LSTM && cy != nullptr) {
        CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                 cy,
                                 wspace.GetCy(0),
                                 max_batch_size * nb_layers * bi * nc * sizeof(T),
                                 cudaMemcpyDeviceToDevice,
                                 fwd_stream);
    }

    // finalize
    if (bi_mode && fwd_stream != bwd_stream) {
        CUDA_FUNC_CALL_QUIET(cudaStreamDestroy, bwd_stream);
    }
}
}  // namespace impl
}  // namespace cudnn
