/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#pragma once
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn/impl/kernel/cudnn_reduction.cuh>
#include <cudnn/impl/param/cudnn_param_tensor.h>

#include <stdio.h>

namespace cudnn {
namespace impl {
namespace kernel {

// launch config for both mean and inv variance calculation:
//    block is one dimension, and combined with grid x direction handle spatial elements
//    grid y direction handle channel
//    grid z direction handle batch size

template <cudnnDataType_t DataType>
__global__ void BatchNormTrainingCalcMean(const CuParamConstTensor<DataType> tensor_x,
                                          CuParamTensor<DataType> tensor_mean) {
    using TensorType = typename CuParamTensor<DataType>::Type;

    const auto meta_x              = tensor_x.GetMeta();
    const auto nb_dims             = meta_x.GetNbDims();
    const auto dim_n               = meta_x.GetDim(1);
    const auto dim_d               = nb_dims == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_h               = meta_x.GetDim(nb_dims - 1);
    const auto dim_w               = meta_x.GetDim(nb_dims);
    const auto dim_hw              = dim_h * dim_w;
    const auto nb_spatial_elements = dim_d * dim_h * dim_w;

    const auto n                 = blockIdx.z;
    const auto c                 = blockIdx.y;
    const auto nb_total_elements = nb_spatial_elements * dim_n;
    const auto grid_size_x       = blockDim.x * gridDim.x;
    const auto tid               = threadIdx.x;
    const auto global_tid        = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ TensorType sdata_mean[];
    sdata_mean[tid] = 0;  // IMPORTANT: all slots need to be initialized to 0, otherwise you need to
                          // pass a proper 'len' to the 3rd parameter of ReductionSum function
                          // instead of a simple maximum value which is blockDim.x
    if (global_tid >= nb_spatial_elements) {
        return;
    }
    __syncthreads();

    if (global_tid + grid_size_x < nb_spatial_elements) {
        const auto global_tid1 = global_tid;
        const auto global_tid2 = global_tid1 + grid_size_x;
        const auto d1          = global_tid1 / dim_hw;
        const auto d2          = global_tid2 / dim_hw;
        const auto h1          = global_tid1 % dim_hw / dim_w;
        const auto h2          = global_tid2 % dim_hw / dim_w;
        const auto w1          = global_tid1 % dim_w;
        const auto w2          = global_tid2 % dim_w;
        sdata_mean[tid]        = tensor_x.At(n, c, d1, h1, w1) + tensor_x.At(n, c, d2, h2, w2);
    } else {
        const auto d    = global_tid / dim_hw;
        const auto h    = global_tid % dim_hw / dim_w;
        const auto w    = global_tid % dim_w;
        sdata_mean[tid] = tensor_x.At(n, c, d, h, w);
    }
    __syncthreads();

    ReductionSum(tid, sdata_mean, blockDim.x);

    if (tid == 0) {
        TensorType* ptr = tensor_mean.Ptr(0, c, 0, 0, 0);
        atomicAdd(ptr, sdata_mean[0] / nb_total_elements);
    }
}

template <cudnnDataType_t DataType>
__global__ void BatchNormTrainingCalcVariance(const CuParamConstTensor<DataType> tensor_x,
                                              const CuParamConstTensor<DataType> tensor_mean,
                                              CuParamTensor<DataType> tensor_variance) {
    using TensorType = typename CuParamTensor<DataType>::Type;

    const auto meta_x              = tensor_x.GetMeta();
    const auto nb_dims             = meta_x.GetNbDims();
    const auto dim_n               = meta_x.GetDim(1);
    const auto dim_d               = meta_x.GetNbDims() == 4 ? 1 : meta_x.GetDim(3);
    const auto dim_h               = meta_x.GetDim(nb_dims - 1);
    const auto dim_w               = meta_x.GetDim(nb_dims);
    const auto dim_hw              = dim_h * dim_w;
    const auto nb_spatial_elements = dim_d * dim_h * dim_w;
    const auto nb_total_elements   = nb_spatial_elements * dim_n;

    const auto n           = blockIdx.z;
    const auto c           = blockIdx.y;
    const auto grid_size_x = blockDim.x * gridDim.x;
    const auto tid         = threadIdx.x;
    const auto global_tid  = blockDim.x * blockIdx.x + tid;

    extern __shared__ TensorType sdata_variance[];
    sdata_variance[tid] = 0;  // IMPORTANT: all slots need to be initialized to 0, otherwise you
                              // need to pass a proper 'len' to the 3rd parameter of ReductionSum
                              // function instead of a simple maximum value which is blockDim.x
    if (global_tid >= nb_spatial_elements) {
        return;
    }
    __syncthreads();

    if (global_tid + grid_size_x < nb_spatial_elements) {
        const auto global_tid1 = global_tid;
        const auto global_tid2 = grid_size_x + global_tid1;
        const auto d1          = global_tid1 / dim_hw;
        const auto d2          = global_tid2 / dim_hw;
        const auto h1          = global_tid1 % dim_hw / dim_w;
        const auto h2          = global_tid2 % dim_hw / dim_w;
        const auto w1          = global_tid1 % dim_w;
        const auto w2          = global_tid2 % dim_w;
        const auto mean        = tensor_mean.At(0, c, 0, 0, 0);
        const auto x1          = tensor_x.At(n, c, d1, h1, w1) - mean;
        const auto x2          = tensor_x.At(n, c, d2, h2, w2) - mean;
        sdata_variance[tid]    = x1 * x1 + x2 * x2;
    } else {
        const auto d        = global_tid / dim_hw;
        const auto h        = global_tid % dim_hw / dim_w;
        const auto w        = global_tid % dim_w;
        const auto mean     = tensor_mean.At(0, c, 0, 0, 0);
        const auto x        = tensor_x.At(n, c, d, h, w) - mean;
        sdata_variance[tid] = x * x;
    }
    __syncthreads();

    ReductionSum(tid, sdata_variance, blockDim.x);

    if (tid == 0) {
        TensorType* ptr = tensor_variance.Ptr(0, c, 0, 0, 0);
        atomicAdd(ptr, sdata_variance[0] / nb_total_elements);
    }
}

template <cudnnDataType_t DataType>
__global__ void
BatchNormTrainingInvertVariance(int m, CuParamTensor<DataType> tensor_inv_variance, float epsilon) {
    const auto channel                          = threadIdx.x;
    auto var                                    = tensor_inv_variance.At(0, channel, 0, 0, 0);
    tensor_inv_variance.At(0, channel, 0, 0, 0) = 1 / sqrt(var + epsilon);
}

}  // namespace kernel
}  // namespace impl
}  // namespace cudnn
