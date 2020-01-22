/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <bits/stdc++.h>
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_dropout_descriptor.h>
#include <cudnn/impl/cudnn_filter_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_rnn.h>
#include <cudnn/impl/cudnn_rnn_descriptor.h>
#include <cudnn/impl/cudnn_rnn_forward.cuh>
#include <cudnn/impl/cudnn_rnn_params.h>
#include <cudnn/impl/cudnn_rnn_reserve_space.h>
#include <cudnn/impl/cudnn_rnn_work_space_forward.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/cudnn_tensor_type.h>
#include <cudnn/impl/meta/cudnn_meta_filter.h>
#include <cudnn/impl/meta/cudnn_meta_rnn.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

#include <iostream>
#include <vector>

using cudnn::GetLogger;
using cudnn::Try;
using cudnn::api::CheckDataTypeDiffer;
using cudnn::api::CheckDimensionDiffer;
using cudnn::api::CheckFullyPackedTensor;
using cudnn::api::CheckIs4dOr5dTensor;
using cudnn::api::CheckNull;
using cudnn::impl::Deref;
using cudnn::impl::CuDropoutDescriptor;
using cudnn::impl::CuRnnDescriptor;
using cudnn::impl::CuRnnForward;
using cudnn::impl::CuRnnParams;
using cudnn::impl::CuRnnReserveSpace;
using cudnn::impl::CuTensorDescriptor;
using cudnn::impl::CuTensorType;
using cudnn::impl::meta::CuMetaFilter;
using cudnn::impl::meta::CuMetaRnn;
using cudnn::impl::meta::CuMetaTensor;

namespace {
using cudnn::CuException;

/**
 * @brief check the validity of a sequence tensor descriptors passed to
 * cudnnRNNForwardInference
 *
 * @param[in]   seq_descs       vector of CuMetaTensor to be checked
 * @param[in]   bidirectional   coef for dim2 to be multiplied with, 1 or 2 (for y/dy when
 *                              DNN_BIDIRECTIONAL mode), default is 1, x/dx could always use
 * 								default value
 * @throw CuException when one of below criteria meet:
 *   1. at least one tensor isn't fully packed (covered by No.5)
 *   2. at least one tensor isn't 3-d tensor
 *   3. at least one tensor's first dimension (batch size) increase
 *   4. at least one tensor's dimension doesn't match [batch size, input size * bindirectional, 1]
 *   5. at least one tensor's stride doesn't match [dim2 * dim3, dim2, 1]
 */
inline void CheckSequenceTensorDescriptors(const std::vector<CuMetaTensor>& tensor_metas,
                                           int dim2,
                                           int bidirectional = 1) {
    int pre_batch_size = INT_MAX;
    for (int i = 0; i < tensor_metas.size(); ++i) {
        auto tensor_meta = tensor_metas[i];
        auto nb_dims     = tensor_meta.GetNbDims();
        if (nb_dims != 3) {
            GetLogger()->info("{}-th x_desc isn't a 3-dimension tensor descriptor", i);
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        const auto dim_1 = tensor_meta.GetDim(1);
        if (dim_1 > pre_batch_size) {
            GetLogger()->info("batch size increased from {}-th x_desc", i);
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
        pre_batch_size = dim_1;

        const auto dim_2 = tensor_meta.GetDim(2);
        const auto dim_3 = tensor_meta.GetDim(3);
        if (dim_2 != dim2 * bidirectional || dim_3 != 1) {
            GetLogger()->info(
                "dimensions of {}-th descriptor in sequence tensor descriptors is not valid", i);
            throw CuException(DNN_STATUS_BAD_PARAM);
        }

        const auto stride_1 = tensor_meta.GetStride(1);
        const auto stride_2 = tensor_meta.GetStride(2);
        const auto stride_3 = tensor_meta.GetStride(3);
        if (stride_1 != dim_2 * dim_3 || stride_2 != dim_3 || stride_3 != 1) {
            GetLogger()->info("Strides of {}-th x_desc isn't valid", i);
            throw CuException(DNN_STATUS_BAD_PARAM);
        }
    }
}

/**
 * @brief check validity of a state tensor descriptor passed to cudnnRNNForwardTraining
 *
 * @param[in]   state_desc  state tensor descriptor to be checked
 * @param[in]   rnn_desc    RNN descriptor
 * @param[in]   seq_descs   vector of CuTensorDescriptor represents sequence descriptor vector
 * @throw CuException when one of below criteria meet:
 *      1. dimension isn't of [nb layers * bidirectional, batch size, hidden size]
 *      2. stride isn't of [dim3 * dim2, dim2, 1]
 */
void CheckStateDescriptor(const CuMetaTensor& state_meta,
                          const CuMetaRnn& rnn_meta,
                          const std::vector<CuMetaTensor>& tensor_metas) {
    const auto hidden_size = rnn_meta.GetHiddenSize();
    const auto batch_size  = tensor_metas[0].GetDim(1);
    const auto directional = rnn_meta.GetDirectionMode() == DNN_UNIDIRECTIONAL ? 1 : 2;
    const auto nb_layers   = rnn_meta.GetNbLayers();

    const auto dim_1 = state_meta.GetDim(1);
    const auto dim_2 = state_meta.GetDim(2);
    const auto dim_3 = state_meta.GetDim(3);

    if (dim_1 != nb_layers * directional || dim_2 != batch_size || dim_3 != hidden_size) {
        GetLogger()->info("dimensions of state descriptor is not valid");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }

    const auto stride_1 = state_meta.GetStride(1);
    const auto stride_2 = state_meta.GetStride(2);
    const auto stride_3 = state_meta.GetStride(3);
    if (stride_1 != dim_2 * dim_3 || stride_2 != dim_3 || stride_3 != 1) {
        GetLogger()->info("stride of state descriptor is not valid");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }
}

void CheckProjetionEnablingParameters(const CuMetaRnn&& rnn_meta,
                                      int rec_proj_size,
                                      int out_proj_size) {
    if (rnn_meta.GetMode() != DNN_LSTM) {
        GetLogger()->info("Projection can only be enabled in LSTM mode");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }

    if (rnn_meta.GetAlgo() != DNN_RNN_ALGO_STANDARD) {
        GetLogger()->info("Projection can only be enabled in RNN_ALGO_STANDARD mode");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }

    CHECK_LOWER_BOUND(rec_proj_size, 0, DNN_STATUS_BAD_PARAM);

    if (out_proj_size != 0) {
        cudnn::GetLogger()->info("cudnn doesn't support ouput projection in current version");
        throw CuException(DNN_STATUS_NOT_SUPPORTED);
    }

    if (rec_proj_size > rnn_meta.GetHiddenSize()) {
        cudnn::GetLogger()->info(
            "cudnn doesn't support recurrent projection size that is larger than hidden size");
        throw CuException(DNN_STATUS_NOT_SUPPORTED);
    }
}

void CheckDataType(const cudnn::impl::meta::CuMetaTensor& x_meta, cudnnDataType_t data_type) {
    // TODO(Peter Han): There isn't detail spec about "The combination of dataType and tensor
    // descriptor data type is invalid". Tensorflow simply give same data_type value to both
    // data_type and the parameter in setting x_desc, so here we just check if the two are same
    if (data_type != x_meta.GetDataType()) {
        cudnn::GetLogger()->info(
            "The combination of data type {} in x_desc and data_type {} is invalid",
            x_meta.GetDataType(),
            data_type);
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }
}

void CheckFullPacked(const cudnn::impl::meta::CuMetaTensor& x_meta) {
    // TODO(Peter Han): Don't know what's is so called "fully-packed" in RNN.
    // Below check points are gathered from many tries
    if (x_meta.GetStride(2) != 1 || x_meta.GetStride(3) != 1 ||
        x_meta.GetStride(1) != x_meta.GetDim(2)) {
        cudnn::GetLogger()->info(
            "input tensor should have stride 1 for both 2nd and 3rd dimension");
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }
    if (x_meta.GetDim(3) != 1) {
        cudnn::GetLogger()->info("lowest dimension of input tensor isn't 1");
        throw cudnn::CuException(DNN_STATUS_BAD_PARAM);
    }
}
}  // namespace

extern "C" {
cudnnStatus_t DNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnn_desc) {
    return Try([&] {
        CheckNull(rnn_desc);

        *rnn_desc = new CuRnnDescriptor();
    });
}

cudnnStatus_t DNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnn_desc) {
    return Try([&] { delete rnn_desc; });
}

cudnnStatus_t DNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnn_desc, int* count) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
                                        const cudnnRNNDescriptor_t rnn_desc,
                                        const int seq_length,
                                        const cudnnTensorDescriptor_t* x_desc,
                                        const void* x,
                                        const cudnnTensorDescriptor_t hx_desc,
                                        const void* hx,
                                        const cudnnTensorDescriptor_t cx_desc,
                                        const void* cx,
                                        const cudnnFilterDescriptor_t w_desc,
                                        const void* w,
                                        const cudnnTensorDescriptor_t* y_desc,
                                        void* y,
                                        const cudnnTensorDescriptor_t hy_desc,
                                        void* hy,
                                        const cudnnTensorDescriptor_t cy_desc,
                                        void* cy,
                                        const float find_intensity,
                                        const int requested_algo_count,
                                        int* returned_algo_count,
                                        cudnnAlgorithmPerformance_t* perf_results,
                                        void* workspace,
                                        size_t workspace_size_in_bytes) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnn_desc, int* count) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnn_desc,
                                       const int seq_length,
                                       const cudnnTensorDescriptor_t* x_desc,
                                       const void* x,
                                       const cudnnTensorDescriptor_t hx_desc,
                                       const void* hx,
                                       const cudnnTensorDescriptor_t cx_desc,
                                       const void* cx,
                                       const cudnnFilterDescriptor_t w_desc,
                                       const void* w,
                                       const cudnnTensorDescriptor_t* y_desc,
                                       void* y,
                                       const cudnnTensorDescriptor_t hy_desc,
                                       void* hy,
                                       const cudnnTensorDescriptor_t cy_desc,
                                       void* cy,
                                       const float find_intensity,
                                       const int requested_algo_count,
                                       int* returned_algo_count,
                                       cudnnAlgorithmPerformance_t* perf_results,
                                       void* workspace,
                                       size_t workspace_size_in_bytes,
                                       void* reserve_space,
                                       size_t reserve_space_size_in_bytes) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnn_desc, int* count) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t rnn_desc,
                                    const int seq_length,
                                    const cudnnTensorDescriptor_t* y_desc,
                                    const void* y,
                                    const cudnnTensorDescriptor_t* dy_desc,
                                    const void* dy,
                                    const cudnnTensorDescriptor_t dhy_desc,
                                    const void* dhy,
                                    const cudnnTensorDescriptor_t dcy_desc,
                                    const void* dcy,
                                    const cudnnFilterDescriptor_t w_desc,
                                    const void* w,
                                    const cudnnTensorDescriptor_t hx_desc,
                                    const void* hx,
                                    const cudnnTensorDescriptor_t cx_desc,
                                    const void* cx,
                                    const cudnnTensorDescriptor_t* dx_desc,
                                    void* dx,
                                    const cudnnTensorDescriptor_t dhx_desc,
                                    void* dhx,
                                    const cudnnTensorDescriptor_t dcx_desc,
                                    void* dcx,
                                    const float find_intensity,
                                    const int requested_algo_count,
                                    int* returned_algo_count,
                                    cudnnAlgorithmPerformance_t* perf_results,
                                    void* workspace,
                                    size_t workspace_size_in_bytes,
                                    void* reserve_space,
                                    size_t reserve_space_size_in_bytes) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnn_desc, int* count) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI
cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnn_desc,
                                       const int seq_length,
                                       const cudnnTensorDescriptor_t* x_desc,
                                       const void* x,
                                       const cudnnTensorDescriptor_t hx_desc,
                                       const void* hx,
                                       const cudnnTensorDescriptor_t* y_desc,
                                       const void* y,
                                       const float find_intensity,
                                       const int requested_algo_count,
                                       int* returned_algo_count,
                                       cudnnAlgorithmPerformance_t* perf_results,
                                       const void* workspace,
                                       size_t workspace_size_in_bytes,
                                       const cudnnFilterDescriptor_t dw_desc,
                                       void* dw,
                                       const void* reserve_space,
                                       size_t reserve_space_size_in_bytes) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

/* Expensive. Creates the plan for the specific settings. */
cudnnStatus_t DNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnn_desc,
                                                       const int minibatch,
                                                       const cudnnDataType_t data_type,
                                                       cudnnPersistentRNNPlan_t* plan) {
    return Try([&] {
        // TBD
    });
}

/* Attaches the plan to the descriptor. */
cudnnStatus_t DNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnn_desc,
                                                    cudnnPersistentRNNPlan_t plan) {
    return Try([&] {
        // TBD
    });
}

cudnnStatus_t DNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
    return Try([&] {
        // TBD
    });
}

cudnnStatus_t DNNWINAPI cudnnSetRNNDescriptor(cudnnHandle_t handle,
                                                cudnnRNNDescriptor_t rnn_desc,
                                                const int hidden_size,
                                                const int num_layers,
                                                cudnnDropoutDescriptor_t dropout_desc,
                                                cudnnRNNInputMode_t input_mode,
                                                cudnnDirectionMode_t direction,
                                                cudnnRNNMode_t mode,
                                                cudnnRNNAlgo_t algo,
                                                cudnnDataType_t data_type) {
    return Try([&] {
        CHECK_LOWER_BOUND(hidden_size, 1, DNN_STATUS_BAD_PARAM);
        CHECK_LOWER_BOUND(num_layers, 1, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(input_mode, DNN_LINEAR_INPUT, DNN_SKIP_INPUT, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(direction, DNN_UNIDIRECTIONAL, DNN_BIDIRECTIONAL, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(mode, DNN_RNN_RELU, DNN_GRU, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(
            algo, DNN_RNN_ALGO_STANDARD, DNN_RNN_ALGO_PERSIST_DYNAMIC, DNN_STATUS_BAD_PARAM);
        CHECK_RANGE(data_type, DNN_DATA_FLOAT, DNN_DATA_HALF, DNN_STATUS_BAD_PARAM);

        Deref(rnn_desc).Set(hidden_size,
                            num_layers,
                            num_layers == 1 ? nullptr
                                            : dynamic_cast<CuDropoutDescriptor*>(dropout_desc),
                            input_mode,
                            direction,
                            mode,
                            algo,
                            data_type);
    });
}

cudnnStatus_t DNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                                                      cudnnRNNDescriptor_t rnn_desc,
                                                      const int rec_proj_size,
                                                      const int out_proj_size) {
    return Try([&] {
        CheckProjetionEnablingParameters(Deref(rnn_desc), rec_proj_size, out_proj_size);
        Deref(rnn_desc).SetProjectionLayers(rec_proj_size, out_proj_size);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                                                      const cudnnRNNDescriptor_t rnn_desc,
                                                      int* rec_proj_size,
                                                      int* out_proj_size) {
    return Try([&] {
        CheckNull(rec_proj_size, out_proj_size);
        std::tie(*rec_proj_size, *out_proj_size) = Deref(rnn_desc).GetProjectionLayers();
    });
}

cudnnStatus_t DNNWINAPI cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle,
                                                         cudnnRNNDescriptor_t rnn_desc,
                                                         cudnnAlgorithmDescriptor_t algo_desc) {
    GetLogger()->error("{} is not supported by this version", __func__);
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetRNNDescriptor(cudnnHandle_t handle,
                                                cudnnRNNDescriptor_t rnn_desc,
                                                int* hidden_size,
                                                int* num_layers,
                                                cudnnDropoutDescriptor_t* dropout_desc,
                                                cudnnRNNInputMode_t* input_mode,
                                                cudnnDirectionMode_t* direction,
                                                cudnnRNNMode_t* mode,
                                                cudnnRNNAlgo_t* algo,
                                                cudnnDataType_t* data_type) {
    return Try([&] {
        CheckNull(hidden_size, num_layers, input_mode, direction, mode, algo, data_type);

        CuDropoutDescriptor* dropout = nullptr;
        std::tie(
            *hidden_size, *num_layers, dropout, *input_mode, *direction, *mode, *algo, *data_type) =
            Deref(rnn_desc).Get();
        *dropout_desc = dropout;
    });
}

cudnnStatus_t DNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnn_desc,
                                                    cudnnMathType_t m_type) {
    return Try([&] {
        CHECK_RANGE(m_type,
                    DNN_DEFAULT_MATH,
                    DNN_TENSOR_OP_MATH,  // DNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
                    DNN_STATUS_BAD_PARAM);

        Deref(rnn_desc).SetMatrixMathType(m_type);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnn_desc,
                                                    cudnnMathType_t* m_type) {
    return Try([&] {
        CheckNull(m_type);

        *m_type = Deref(rnn_desc).GetMatrixMathType();
    });
}

/* dataType in the RNN descriptor is used to determine math precision */
/* dataType in weight descriptors and input descriptors is used to describe storage */
cudnnStatus_t DNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t /*handle*/,
                                                   const cudnnRNNDescriptor_t rnn_desc,
                                                   const int seq_length,
                                                   const cudnnTensorDescriptor_t* x_desc,
                                                   size_t* size_in_bytes) {
    return Try([&] {
        CheckNull(x_desc, size_in_bytes);
        std::vector<CuMetaTensor> x_metas;
        for (int i = 0; i < seq_length; ++i) {
            x_metas.push_back(std::move(static_cast<CuMetaTensor>(Deref(x_desc[i]))));
        }

        CuMetaRnn meta       = Deref(rnn_desc);
        const auto data_type = meta.GetMathPrec();
        if (data_type == DNN_DATA_FLOAT) {
            auto fwd_space = cudnn::impl::CuRnnWorkSpaceForward<float>(meta, x_metas);
            *size_in_bytes = fwd_space.GetSizeInBytes();
        } else if (data_type == DNN_DATA_DOUBLE) {
            auto fwd_space = cudnn::impl::CuRnnWorkSpaceForward<double>(meta, x_metas);
            *size_in_bytes = fwd_space.GetSizeInBytes();
        }
    });
}

cudnnStatus_t DNNWINAPI cudnnGetRNNTrainingReserveSize(cudnnHandle_t /*handle*/,
                                                         const cudnnRNNDescriptor_t rnn_desc,
                                                         const int seq_length,
                                                         const cudnnTensorDescriptor_t* x_desc,
                                                         size_t* size_in_bytes) {
    return Try([&] {
        CheckNull(size_in_bytes);

        const auto rnn_meta = static_cast<cudnn::impl::meta::CuMetaRnn>(Deref(rnn_desc));
        std::vector<CuMetaTensor> x_metas;
        for (int i = 0; i < seq_length; ++i) {
            x_metas.push_back(std::move(static_cast<CuMetaTensor>(Deref(x_desc[i]))));
        }

        auto data_type = x_metas[0].GetDataType();
        if (data_type == DNN_DATA_FLOAT) {
            *size_in_bytes = CuRnnReserveSpace<float>(rnn_meta, x_metas).GetSizeInBytes();
        } else if (data_type == DNN_DATA_DOUBLE) {
            *size_in_bytes = CuRnnReserveSpace<double>(rnn_meta, x_metas).GetSizeInBytes();
        } else if (data_type == DNN_DATA_HALF) {
            *size_in_bytes = CuRnnReserveSpace<__half>(rnn_meta, x_metas).GetSizeInBytes();
        }
    });
}

cudnnStatus_t DNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t /*handle*/,
                                                const cudnnRNNDescriptor_t rnn_desc,
                                                const cudnnTensorDescriptor_t x_desc,
                                                size_t* size_in_bytes,
                                                cudnnDataType_t data_type) {
    return Try([&] {
        const CuMetaTensor x_meta = Deref(x_desc);
        CheckDataType(x_meta, data_type);
        CheckFullPacked(x_meta);
        CheckNull(size_in_bytes);

        *size_in_bytes = CuRnnParams(Deref(rnn_desc), x_meta, nullptr).GetParamsSize();
    });
}

cudnnStatus_t DNNWINAPI
cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
                                const cudnnRNNDescriptor_t rnn_desc,
                                const int pseudo_layer,
                                const cudnnTensorDescriptor_t x_desc,
                                const cudnnFilterDescriptor_t w_desc,
                                const void* w,
                                const int lin_layer_id,
                                cudnnFilterDescriptor_t lin_layer_mat_desc,
                                void** lin_layer_mat) {
    return Try([&] {
        CheckNull(w, lin_layer_mat);
        // TODO(Peter Han): other parameter check need to be done

        const auto data_type = Deref(x_desc).GetDataType();
        auto filter_ptr =
            CuRnnParams(Deref(rnn_desc), Deref(x_desc), w).GetWeight(pseudo_layer, lin_layer_id);

        std::vector<int> dim(0);
        int nb_dims = filter_ptr.first.GetNbDims();
        for (int i = 0; i < nb_dims; ++i) {
            dim.push_back(filter_ptr.first.GetDim(i + 1));
        }
        Deref(lin_layer_mat_desc).Set(data_type, DNN_TENSOR_NCHW, nb_dims, dim.data());
        *lin_layer_mat = const_cast<void*>(filter_ptr.second);
    });
}

cudnnStatus_t DNNWINAPI cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
                                                        const cudnnRNNDescriptor_t rnn_desc,
                                                        const int pseudo_layer,
                                                        const cudnnTensorDescriptor_t x_desc,
                                                        const cudnnFilterDescriptor_t w_desc,
                                                        const void* w,
                                                        const int lin_layer_id,
                                                        cudnnFilterDescriptor_t lin_layer_bias_desc,
                                                        void** lin_layer_bias) {
    return Try([&] {
        CheckNull(w, lin_layer_bias);
        // TODO(Peter Han): other parameter check need to be done

        const auto data_type = Deref(x_desc).GetDataType();

        auto filter_ptr =
            CuRnnParams(Deref(rnn_desc), Deref(x_desc), w).GetBias(pseudo_layer, lin_layer_id);

        int nb_dims = filter_ptr.first.GetNbDims();
        std::vector<int> dim(0);
        for (int i = 0; i < nb_dims; ++i) {
            dim.push_back(filter_ptr.first.GetDim(i + 1));
        }
        Deref(lin_layer_bias_desc).Set(data_type, DNN_TENSOR_NCHW, nb_dims, dim.data());
        *lin_layer_bias = const_cast<void*>(filter_ptr.second);
    });
}

#define INFERENCE(T)                                        \
    CuRnnForward<T>(stream,                                 \
                    rnn_meta,                               \
                    Deref(rnn_desc).GetDropoutDescriptor(), \
                    x_metas,                                \
                    reinterpret_cast<const T*>(x),          \
                    hx_meta,                                \
                    reinterpret_cast<const T*>(hx),         \
                    cx_meta,                                \
                    reinterpret_cast<const T*>(cx),         \
                    w_meta,                                 \
                    reinterpret_cast<const T*>(w),          \
                    y_metas,                                \
                    reinterpret_cast<T*>(y),                \
                    hy_meta,                                \
                    reinterpret_cast<T*>(hy),               \
                    cy_meta,                                \
                    reinterpret_cast<T*>(cy),               \
                    reinterpret_cast<T*>(workspace),        \
                    workspace_size_in_bytes,                \
                    nullptr,                                \
                    0);
cudnnStatus_t DNNWINAPI cudnnRNNForwardInference(cudnnHandle_t handle,
                                                   const cudnnRNNDescriptor_t rnn_desc,
                                                   const int seq_length,
                                                   const cudnnTensorDescriptor_t* x_desc,
                                                   const void* x,
                                                   const cudnnTensorDescriptor_t hx_desc,
                                                   const void* hx,
                                                   const cudnnTensorDescriptor_t cx_desc,
                                                   const void* cx,
                                                   const cudnnFilterDescriptor_t w_desc,
                                                   const void* w,
                                                   const cudnnTensorDescriptor_t* y_desc,
                                                   void* y,
                                                   const cudnnTensorDescriptor_t hy_desc,
                                                   void* hy,
                                                   const cudnnTensorDescriptor_t cy_desc,
                                                   void* cy,
                                                   void* workspace,
                                                   size_t workspace_size_in_bytes) {
    return Try([&] {
        std::vector<CuMetaTensor> x_metas;
        std::vector<CuMetaTensor> y_metas;
        for (int i = 0; i < seq_length; ++i) {
            x_metas.push_back(std::move(static_cast<CuMetaTensor>(Deref(x_desc[i]))));
            y_metas.push_back(std::move(static_cast<CuMetaTensor>(Deref(y_desc[i]))));
        }

        const auto rnn_meta    = static_cast<CuMetaRnn>(Deref(rnn_desc));
        const auto directional = rnn_meta.GetDirectionMode() == DNN_UNIDIRECTIONAL ? 1 : 2;
        CheckSequenceTensorDescriptors(x_metas, x_metas[0].GetDim(2));
        CheckSequenceTensorDescriptors(y_metas, rnn_meta.GetHiddenSize(), directional);

        const CuMetaTensor hx_meta = Deref(hx_desc);
        const CuMetaTensor cx_meta = Deref(cx_desc);
        const CuMetaTensor hy_meta = Deref(hy_desc);
        const CuMetaTensor cy_meta = Deref(cy_desc);

        CheckStateDescriptor(hx_meta, rnn_meta, x_metas);
        CheckStateDescriptor(cx_meta, rnn_meta, x_metas);
        CheckStateDescriptor(hy_meta, rnn_meta, x_metas);
        CheckStateDescriptor(cy_meta, rnn_meta, x_metas);
        // FIXME(Peter Han): check workspace_size_in_bytes valid or not

        auto stream               = Deref(handle).GetStream();
        const auto dropout_desc   = Deref(rnn_desc).GetDropoutDescriptor();
        const CuMetaFilter w_meta = Deref(w_desc);
        const auto data_type      = x_metas[0].GetDataType();

        switch (data_type) {
        case DNN_DATA_FLOAT: INFERENCE(CuTensorType<DNN_DATA_FLOAT>::Type); break;
        case DNN_DATA_DOUBLE: INFERENCE(typename CuTensorType<DNN_DATA_DOUBLE>::Type); break;
        // case DNN_DATA_HALF: INFERENCE(typename CuTensorType<DNN_DATA_HALF>::Type); break;
        case DNN_DATA_INT8:
        case DNN_DATA_UINT8:
        case DNN_DATA_INT32:
        case DNN_DATA_INT8x4:
        case DNN_DATA_UINT8x4:
        default: break;
        }
    });
}

cudnnStatus_t DNNWINAPI cudnnRNNForwardTraining(cudnnHandle_t handle,
                                                  const cudnnRNNDescriptor_t rnn_desc,
                                                  const int seq_length,
                                                  const cudnnTensorDescriptor_t* x_desc,
                                                  const void* x,
                                                  const cudnnTensorDescriptor_t hx_desc,
                                                  const void* hx,
                                                  const cudnnTensorDescriptor_t cx_desc,
                                                  const void* cx,
                                                  const cudnnFilterDescriptor_t w_desc,
                                                  const void* w,
                                                  const cudnnTensorDescriptor_t* y_desc,
                                                  void* y,
                                                  const cudnnTensorDescriptor_t hy_desc,
                                                  void* hy,
                                                  const cudnnTensorDescriptor_t cy_desc,
                                                  void* cy,
                                                  void* workspace,
                                                  size_t workspace_size_in_bytes,
                                                  void* reservespace,
                                                  size_t reservespace_size_in_bytes) {
    return Try([&] {});
}

cudnnStatus_t DNNWINAPI cudnnRNNBackwardData(cudnnHandle_t handle,
                                               const cudnnRNNDescriptor_t rnn_desc,
                                               const int seq_length,
                                               const cudnnTensorDescriptor_t* y_desc,
                                               const void* y,
                                               const cudnnTensorDescriptor_t* dy_desc,
                                               const void* dy,
                                               const cudnnTensorDescriptor_t dhy_desc,
                                               const void* dhy,
                                               const cudnnTensorDescriptor_t dcy_desc,
                                               const void* dcy,
                                               const cudnnFilterDescriptor_t w_desc,
                                               const void* w,
                                               const cudnnTensorDescriptor_t hx_desc,
                                               const void* hx,
                                               const cudnnTensorDescriptor_t cx_desc,
                                               const void* cx,
                                               const cudnnTensorDescriptor_t* dx_desc,
                                               void* dx,
                                               const cudnnTensorDescriptor_t dhx_desc,
                                               void* dhx,
                                               const cudnnTensorDescriptor_t dcx_desc,
                                               void* dcx,
                                               void* workspace,
                                               size_t workspace_size_in_bytes,
                                               void* reservespace,
                                               size_t reservespace_size_in_bytes) {
    return Try([&] {
        // CuRNNBackwardData(Deref(handle),
        //                   Deref(rnn_desc,
        //                   const int seq_length,
        //                   const cudnnTensorDescriptor_t* y_desc,
        //                   const void* y,
        //                   const cudnnTensorDescriptor_t* dy_desc,
        //                   const void* dy,
        //                   const cudnnTensorDescriptor_t dhy_desc,
        //                   const void* dhy,
        //                   const cudnnTensorDescriptor_t dcy_desc,
        //                   const void* dcy,
        //                   const cudnnFilterDescriptor_t w_desc,
        //                   const void* w,
        //                   const cudnnTensorDescriptor_t hx_desc,
        //                   const void* hx,
        //                   const cudnnTensorDescriptor_t cx_desc,
        //                   const void* cx,
        //                   const cudnnTensorDescriptor_t* dx_desc,
        //                   void* dx,
        //                   const cudnnTensorDescriptor_t dhx_desc,
        //                   void* dhx,
        //                   const cudnnTensorDescriptor_t dcx_desc,
        //                   void* dcx,
        //                   void* workspace,
        //                   size_t workspace_size_in_bytes,
        //                   void* reservespace,
        //                   size_t reservespace_size_in_bytes);
    });
}

cudnnStatus_t DNNWINAPI cudnnRNNBackwardWeights(cudnnHandle_t handle,
                                                  const cudnnRNNDescriptor_t rnn_desc,
                                                  const int seq_length,
                                                  const cudnnTensorDescriptor_t* x_desc,
                                                  const void* x,
                                                  const cudnnTensorDescriptor_t hx_desc,
                                                  const void* hx,
                                                  const cudnnTensorDescriptor_t* y_desc,
                                                  const void* y,
                                                  const void* workspace,
                                                  size_t workspace_size_in_bytes,
                                                  const cudnnFilterDescriptor_t dw_desc,
                                                  void* dw,
                                                  const void* reservespace,
                                                  size_t reservespace_size_in_bytes) {
    return Try([&] {});
}

/* DEPRECATED routines to be removed next release :
   User should use the non-suffixed version (which has the API and functionality of _v6 version)
   Routines with _v5 suffix has the functionality of the non-suffixed routines in the DNN V6
 */

cudnnStatus_t DNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
                                                   cudnnRNNDescriptor_t rnn_desc,
                                                   const int hidden_size,
                                                   const int num_layers,
                                                   cudnnDropoutDescriptor_t dropout_desc,
                                                   cudnnRNNInputMode_t input_mode,
                                                   cudnnDirectionMode_t direction,
                                                   cudnnRNNMode_t mode,
                                                   cudnnRNNAlgo_t algo,
                                                   cudnnDataType_t data_type) {
    return cudnnSetRNNDescriptor(handle,
                                 rnn_desc,
                                 hidden_size,
                                 num_layers,
                                 dropout_desc,
                                 input_mode,
                                 direction,
                                 mode,
                                 algo,
                                 data_type);
}

cudnnStatus_t DNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnn_desc,
                                                   int hidden_size,
                                                   int num_layers,
                                                   cudnnDropoutDescriptor_t dropout_desc,
                                                   cudnnRNNInputMode_t input_mode,
                                                   cudnnDirectionMode_t direction,
                                                   cudnnRNNMode_t mode,
                                                   cudnnDataType_t data_type) {
    return cudnnSetRNNDescriptor(nullptr,
                                 rnn_desc,
                                 hidden_size,
                                 num_layers,
                                 dropout_desc,
                                 input_mode,
                                 direction,
                                 mode,
                                 DNN_RNN_ALGO_STANDARD,
                                 data_type);
}
}  // namespace
