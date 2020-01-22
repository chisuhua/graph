/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn/impl/kernel/cudnn_reduction.cuh>
#include <cudnn/impl/param/cudnn_param_scaling.h>
#include <cudnn/impl/param/cudnn_param_tensor.h>

#include <stdio.h>

namespace cudnn {
namespace impl {
namespace kernel {
using cudnn::impl::param::CuParamConstTensor;
using cudnn::impl::param::CuParamScaling;
using cudnn::impl::param::CuParamTensor;

static constexpr int kNbThreadsPerBlockTrainingForward = 256;
/**
 * BN forward per activation mode needs 4 steps which each step is a standalone kernel launch:
 *  0 : adjust previous mean and variance by multiplying (1-factor)
 *  1 : compute mean
 *  2 : compute variance
 *  3 : normalize, scale and shift
 */
template <cudnnDataType_t DataType>
__global__ void BatchNormTraiingForwardSpatialFactorPreviousMeanVariance(
    CuParamTensor<DataType> tensor_running_mean,
    CuParamTensor<DataType> tensor_running_variance,
    float factor) {

    int c = blockDim.x * blockIdx.x + threadIdx.x;

    tensor_running_mean.At(0, c, 0, 0, 0) = tensor_running_mean.At(0, c, 0, 0, 0) * (1 - factor);
    tensor_running_variance.At(0, c, 0, 0, 0) =
        tensor_running_variance.At(0, c, 0, 0, 0) * (1 - factor);
}

template <cudnnDataType_t DataType>
__global__ void BatchNormTrainingForwardSpatialCalcMean(const CuParamConstTensor<DataType> tensor_x,
                                                        float factor,
                                                        CuParamTensor<DataType> tensor_running_mean,
                                                        CuParamTensor<DataType> tensor_save_mean) {
    using TensorType = typename param::CuParamTensor<DataType>::Type;

    const auto meta_x  = tensor_x.GetMeta();
    const auto nb_dims = meta_x.GetNbDims();

    const auto dim_w                = meta_x.GetDim(nb_dims);
    const auto dim_h                = meta_x.GetDim(nb_dims - 1);
    const auto dim_d                = nb_dims == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_c                = meta_x.GetDim(2);
    const auto dim_n                = meta_x.GetDim(1);
    const auto dim_hw               = dim_h * dim_w;
    const auto total_nb_per_channel = dim_d * dim_h * dim_w;

    // load input x from global memory to shared memroy
    __shared__ TensorType sdata[kNbThreadsPerBlockTrainingForward];
    const int tid = threadIdx.x;
    sdata[tid]    = static_cast<TensorType>(0);  // important

    const int spatial_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (spatial_idx >= total_nb_per_channel) {
        return;
    }

    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int d = spatial_idx / dim_hw;
    const int h = spatial_idx % dim_hw / dim_w;
    const int w = spatial_idx % dim_w;

    sdata[tid] = tensor_x.At(n, c, d, h, w);
    __syncthreads();

    ReductionSum<TensorType>(tid, sdata, blockDim.x);

    if (tid == 0) {
        TensorType* ptr_running_mean = tensor_running_mean.Ptr(0, c, 0, 0, 0);
        TensorType updated_mean      = sdata[0] / (dim_n * total_nb_per_channel);

        // NOTE: factor only applied on running mean & variance
        atomicAdd(ptr_running_mean, updated_mean * factor);

        // when save pointer is not null, save the intermediate mean
        if (!tensor_save_mean.IsNull()) {
            TensorType* ptr_save_mean = tensor_save_mean.Ptr(0, c, 0, 0, 0);
            atomicAdd(ptr_save_mean, updated_mean);
        }
    }
    __syncthreads();
}

template <cudnnDataType_t DataType>
__global__ void
BatchNormTrainingForwardSpatialCalcVariance(const CuParamConstTensor<DataType> tensor_x,
                                            float factor,
                                            const CuParamConstTensor<DataType> tensor_running_mean,
                                            CuParamTensor<DataType> tensor_running_variance) {
    using TensorType = typename param::CuParamTensor<DataType>::Type;

    const auto meta_x  = tensor_x.GetMeta();
    const auto nb_dims = meta_x.GetNbDims();

    const auto dim_w                = meta_x.GetDim(nb_dims);
    const auto dim_h                = meta_x.GetDim(nb_dims - 1);
    const auto dim_d                = nb_dims == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_hw               = dim_h * dim_w;
    const auto dim_n                = meta_x.GetDim(1);
    const auto total_nb_per_channel = dim_d * dim_h * dim_w;

    // load input x from global memory to shared memroy
    __shared__ TensorType sdata[kNbThreadsPerBlockTrainingForward];
    const int tid = threadIdx.x;
    sdata[tid]    = static_cast<TensorType>(0);  // important

    const int n           = blockIdx.z;
    const int c           = blockIdx.y;
    const int spatial_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (spatial_idx >= total_nb_per_channel) {
        return;
    }

    const int d = spatial_idx / dim_hw;
    const int h = spatial_idx % dim_hw / dim_w;
    const int w = spatial_idx % dim_w;

    TensorType running_mean = tensor_running_mean.At(0, c, 0, 0, 0);
    sdata[tid]              = tensor_x.At(n, c, d, h, w);

    sdata[tid] = sdata[tid] - running_mean;

    sdata[tid] = sdata[tid] * sdata[tid] / (total_nb_per_channel * dim_n - 1);

    __syncthreads();

    ReductionSum<TensorType>(tid, sdata, blockDim.x);

    if (tid == 0) {
        TensorType* ptr_running_variance = tensor_running_variance.Ptr(0, c, 0, 0, 0);
        atomicAdd(ptr_running_variance, sdata[0] * factor);
    }
    __syncthreads();
}

template <cudnnDataType_t DataType>
__global__ void BatchNormTrainingForwardSpatialNormalScaleShift(
    const CuParamScaling<DataType> scaling,
    const CuParamConstTensor<DataType> tensor_x,
    CuParamTensor<DataType> tensor_y,
    const CuParamConstTensor<DataType> tensor_bn_scale,
    const CuParamConstTensor<DataType> tensor_bn_bias,
    const CuParamConstTensor<DataType> tensor_running_mean,
    const CuParamConstTensor<DataType> tensor_running_variance,
    float epsilon,
    CuParamTensor<DataType> tensor_save_inv_variance) {
    using TensorType = typename param::CuParamTensor<DataType>::Type;

    const auto meta_x  = tensor_x.GetMeta();
    const auto nb_dims = meta_x.GetNbDims();

    const auto dim_w  = meta_x.GetDim(nb_dims);
    const auto dim_h  = meta_x.GetDim(nb_dims - 1);
    const auto dim_d  = nb_dims == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_hw = dim_h * dim_w;

    const auto total_nb_per_channel = dim_d * dim_h * dim_w;
    const auto m                    = meta_x.GetDim(1) * total_nb_per_channel;

    const int n           = blockIdx.z;
    const int c           = blockIdx.y;
    const int spatial_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (spatial_idx >= total_nb_per_channel) {
        return;
    }

    const int d = spatial_idx / dim_hw;
    const int h = spatial_idx % dim_hw / dim_w;
    const int w = spatial_idx % dim_w;

    TensorType x     = tensor_x.At(n, c, d, h, w);
    TensorType mean  = tensor_running_mean.At(0, c, 0, 0, 0);
    TensorType var   = tensor_running_variance.At(0, c, 0, 0, 0);
    TensorType scale = tensor_bn_scale.At(0, c, 0, 0, 0);
    TensorType bias  = tensor_bn_bias.At(0, c, 0, 0, 0);

    // nomalize
    var                = (m - 1) * var / m;
    TensorType inv_var = 1 / sqrt(epsilon + var);
    TensorType x_hat   = (x - mean) * inv_var;

    // when save pointer is not null, save the intermediate variance
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (!tensor_save_inv_variance.IsNull()) {
            tensor_save_inv_variance.At(0, c, 0, 0, 0) = inv_var;
        }
    }

    // batch norm scale and shift
    TensorType y = scale * x_hat + bias;

    // cudnn scale and shift
    if (scaling.beta != 0) {
        y = scaling.alpha * y + scaling.beta * tensor_y.At(n, c, d, h, w);
    }
    tensor_y.At(n, c, d, h, w) = y;
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
