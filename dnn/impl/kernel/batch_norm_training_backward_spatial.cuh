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

template <cudnnDataType_t DataType>
__global__ void
BatchNormTrainingBackwardSpatialFactorScaleDiffBiasDiff(const CuParamScaling<DataType> scaling,
                                                        CuParamTensor<DataType> dbn_scale_result,
                                                        CuParamTensor<DataType> dbn_bias_result) {
    const auto beta                      = scaling.beta;
    const auto tid                       = threadIdx.x;
    dbn_scale_result.At(0, tid, 0, 0, 0) = dbn_scale_result.At(0, tid, 0, 0, 0) * beta;
    dbn_bias_result.At(0, tid, 0, 0, 0)  = dbn_bias_result.At(0, tid, 0, 0, 0) * beta;
}

template <cudnnDataType_t DataType>
__global__ void BatchNormTrainingBackwardSpatialDiffVarianceMeanScaleBias(
    const CuParamConstTensor<DataType> tensor_x,
    const CuParamConstTensor<DataType> tensor_dy,
    const CuParamConstTensor<DataType> tensor_bn_scale,
    const CuParamConstTensor<DataType> tensor_saved_mean,
    const CuParamConstTensor<DataType> tensor_saved_inv_variance,
    const CuParamScaling<DataType> scaling,
    CuParamTensor<DataType> tensor_dmean,
    CuParamTensor<DataType> tensor_dvariance,
    CuParamTensor<DataType> tensor_dscale,
    CuParamTensor<DataType> tensor_dbias) {
    using TensorType = typename CuParamTensor<DataType>::Type;

    const auto meta_x              = tensor_x.GetMeta();
    const auto nb_dims             = meta_x.GetNbDims();
    const auto dim_n               = meta_x.GetDim(1);
    const auto dim_d               = meta_x.GetNbDims() == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_h               = meta_x.GetDim(nb_dims - 1);
    const auto dim_w               = meta_x.GetDim(nb_dims);
    const auto dim_hw              = dim_h * dim_w;
    const auto nb_spatial_elements = dim_d * dim_h * dim_w;

    const auto n           = blockIdx.z;
    const auto c           = blockIdx.y;
    const auto grid_size_x = blockDim.x * gridDim.x;
    const auto tid         = threadIdx.x;
    const auto global_tid  = blockDim.x * blockIdx.x + tid;

    const auto mean    = tensor_saved_mean.At(0, c, 0, 0, 0);
    const auto inv_var = tensor_saved_inv_variance.At(0, c, 0, 0, 0);
    const auto scale   = tensor_bn_scale.At(0, c, 0, 0, 0);

    extern __shared__ TensorType sdata[];
    TensorType* sdata_dmean     = &sdata[blockDim.x * 0];
    TensorType* sdata_dvariance = &sdata[blockDim.x * 1];
    TensorType* sdata_dscale    = &sdata[blockDim.x * 2];
    TensorType* sdata_dbias     = &sdata[blockDim.x * 3];
    // IMPORTANT: all slots need to be initialized to 0, otherwise you need to
    // pass a proper 'len' to the 3rd parameter of ReductionSum function
    // instead of a simple maximum value which is blockDim.x
    sdata_dmean[tid]     = 0;
    sdata_dvariance[tid] = 0;
    sdata_dscale[tid]    = 0;
    sdata_dbias[tid]     = 0;
    if (global_tid >= nb_spatial_elements) {
        return;
    }
    __syncthreads();

    if (global_tid + grid_size_x < nb_spatial_elements) {
        const auto global_tid2 = global_tid + grid_size_x;
        const auto d1          = global_tid / dim_hw;
        const auto d2          = global_tid2 / dim_hw;
        const auto h1          = global_tid % dim_hw / dim_w;
        const auto h2          = global_tid2 % dim_hw / dim_w;
        const auto w1          = global_tid % dim_w;
        const auto w2          = global_tid2 % dim_w;
        const auto x1          = tensor_x.At(n, c, d1, h1, w1);
        const auto x2          = tensor_x.At(n, c, d2, h2, w2);
        const auto dy1         = tensor_dy.At(n, c, d1, h1, w1);
        const auto dy2         = tensor_dy.At(n, c, d2, h2, w2);
        const auto dx1_hat     = dy1 * scale;
        const auto dx2_hat     = dy2 * scale;
        const auto x1_hat      = (x1 - mean);
        const auto x2_hat      = (x2 - mean);
        sdata_dmean[tid]       = (dx1_hat + dx2_hat) * -1 * inv_var;
        sdata_dvariance[tid] =
            (dx1_hat * (x1 - mean) + dx2_hat * (x2 - mean)) * -0.5 * inv_var * inv_var * inv_var;
        sdata_dscale[tid] = dy1 * x1_hat + dy2 * x2_hat;
        sdata_dbias[tid]  = dy1 + dy2;
    } else {
        const auto d         = global_tid / dim_hw;
        const auto h         = global_tid % dim_hw / dim_w;
        const auto w         = global_tid % dim_w;
        const auto x         = tensor_x.At(n, c, d, h, w);
        const auto dy        = tensor_dy.At(n, c, d, h, w);
        const auto dx_hat    = dy * scale;
        const auto x_hat     = (x - mean);
        sdata_dmean[tid]     = -1 * dx_hat * inv_var;
        sdata_dvariance[tid] = dx_hat * (x - mean) * -0.5 * inv_var * inv_var * inv_var;
        sdata_dscale[tid]    = dy * x_hat;
        sdata_dbias[tid]     = dy;
    }
    __syncthreads();

    ReductionSum(tid, sdata_dmean, blockDim.x);
    ReductionSum(tid, sdata_dvariance, blockDim.x);
    ReductionSum(tid, sdata_dscale, blockDim.x);
    ReductionSum(tid, sdata_dbias, blockDim.x);

    if (tid == 0) {
        TensorType* dvar_ptr = tensor_dvariance.Ptr(0, c, 0, 0, 0);
        atomicAdd(dvar_ptr, sdata_dvariance[0]);

        TensorType* dmean_ptr = tensor_dmean.Ptr(0, c, 0, 0, 0);
        atomicAdd(dmean_ptr, sdata_dmean[0]);

        TensorType* dscale_ptr = tensor_dscale.Ptr(0, c, 0, 0, 0);
        if (scaling.beta != 0) {
            atomicAdd(dscale_ptr, sdata_dscale[0] * inv_var * scaling.alpha);
        } else {
            atomicAdd(dscale_ptr, sdata_dscale[0] * inv_var);
        }

        TensorType* dbias_ptr = tensor_dbias.Ptr(0, c, 0, 0, 0);
        if (scaling.beta != 0) {
            atomicAdd(dbias_ptr, sdata_dbias[0] * scaling.alpha);
        } else {
            atomicAdd(dbias_ptr, sdata_dbias[0]);
        }
    }
}

template <cudnnDataType_t DataType>
__global__ void BatchNormTrainingBackwardSpatialDiffData(
    const CuParamConstTensor<DataType> tensor_x,
    const CuParamConstTensor<DataType> tensor_dy,
    const CuParamConstTensor<DataType> tensor_bn_scale,
    const CuParamConstTensor<DataType> tensor_saved_mean,
    const CuParamConstTensor<DataType> tensor_saved_inv_variance,
    const CuParamConstTensor<DataType> tensor_dmean,
    const CuParamConstTensor<DataType> tensor_dvariance,
    const CuParamScaling<DataType> scaling,
    CuParamTensor<DataType> tensor_dx) {
    using TensorType = typename CuParamTensor<DataType>::Type;

    const auto meta_x            = tensor_x.GetMeta();
    const auto nb_dims           = meta_x.GetNbDims();
    const auto dim_h             = meta_x.GetDim(nb_dims - 1);
    const auto dim_w             = meta_x.GetDim(nb_dims);
    const auto dim_d             = nb_dims == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_hw            = dim_h * dim_w;
    const auto nb_total_elements = meta_x.GetDim(1) * dim_d * dim_h * dim_w;

    const auto global_tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_tid >= dim_d * dim_h * dim_w) {
        return;
    }

    const auto n = blockIdx.z;
    const auto c = blockIdx.y;
    const auto d = global_tid / dim_hw;
    const auto h = global_tid % dim_hw / dim_w;
    const auto w = global_tid % dim_w;

    TensorType dx = tensor_dy.At(n, c, d, h, w) * tensor_bn_scale.At(0, c, 0, 0, 0) *
                    tensor_saved_inv_variance.At(0, c, 0, 0, 0);
    dx += tensor_dvariance.At(0, c, 0, 0, 0) * 2 *
          (tensor_x.At(n, c, d, h, w) - tensor_saved_mean.At(0, c, 0, 0, 0)) / nb_total_elements;
    dx += tensor_dmean.At(0, c, 0, 0, 0) / nb_total_elements;

    if (scaling.beta != 0) {
        dx = dx * scaling.alpha + tensor_dx.At(n, c, d, h, w) * scaling.beta;
    }
    tensor_dx.At(n, c, d, h, w) = dx;
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
