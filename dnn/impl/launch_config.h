/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <tuple>

#include <iostream>

namespace cudnn {
namespace impl {

// most widely used launch configuration, block has one dimension which combined by grid's first
// dimension to span one image's spatial elements of one channel
// grid has 3 dimensions, the second dimension spans channel and thrid dimension spans batch size
inline std::tuple<dim3, dim3>
CalcLaunchConfig(const cudnn::impl::meta::CuMetaTensor& input_desc,
                 unsigned int tile_size,
                 unsigned int block_x = cudnn::impl::kMaxThreadNbPerBlock) {
    const unsigned int nb_dims             = input_desc.GetNbDims();
    const unsigned int dim_w               = input_desc.GetDim(nb_dims);
    const unsigned int dim_h               = input_desc.GetDim(nb_dims - 1);
    const unsigned int dim_n               = input_desc.GetDim(1);
    const unsigned int dim_c               = input_desc.GetDim(2);
    const unsigned int dim_d               = nb_dims == 4 ? 1 : input_desc.GetDim(3);
    const unsigned int nb_spatial_elements = dim_d * dim_h * dim_w;

    const unsigned int nb_tiles = (nb_spatial_elements + tile_size - 1) / tile_size;

    const dim3 block = {block_x, 1, 1};
    const dim3 grid  = {(nb_tiles + block.x - 1) / block.x, dim_c, dim_n};
    return std::make_tuple(grid, block);
}

// will stretch all tensor into one list, and divid them into block and grid
inline std::tuple<dim3, dim3>
CalcLaunchConfigOneDimension(const cudnn::impl::meta::CuMetaTensor& input_desc,
                             unsigned int tile_size,
                             unsigned int block_x = cudnn::impl::kMaxThreadNbPerBlock) {
    const unsigned int nb_dims     = input_desc.GetNbDims();
    const unsigned int dim_w       = input_desc.GetDim(nb_dims);
    const unsigned int dim_h       = input_desc.GetDim(nb_dims - 1);
    const unsigned int dim_n       = input_desc.GetDim(1);
    const unsigned int dim_c       = input_desc.GetDim(2);
    const unsigned int dim_d       = nb_dims == 4 ? 1 : input_desc.GetDim(3);
    const unsigned int nb_elements = dim_n * dim_c * dim_d * dim_h * dim_w;

    const unsigned int nb_tiles = (nb_elements + tile_size - 1) / tile_size;

    const dim3 block = {block_x, 1, 1};
    const dim3 grid  = {(nb_tiles + block.x - 1) / block.x, 1, 1};
    return std::make_tuple(grid, block);
}

// used in sum like operation, such as compute mean and variance
// block: one dimension, combined with grid.x dimension handle spatial space
// grid.y handle channel space, grid.z handle batch size space
// Not like above CalcLaunchConfig function, there's no 'tile_size' parameter for no gain from
// unrolling inside kernel.
// sm_size_per_thread represents the unit memory slot size for a single kernel thread
inline std::tuple<dim3, dim3, int>
CalcLaunchConfigReduction(const cudnn::impl::meta::CuMetaTensor& input_desc,
                          int sm_size_per_thread) {
    const unsigned int nb_dims             = input_desc.GetNbDims();
    const unsigned int dim_w               = input_desc.GetDim(nb_dims);
    const unsigned int dim_h               = input_desc.GetDim(nb_dims - 1);
    const unsigned int dim_n               = input_desc.GetDim(1);
    const unsigned int dim_c               = input_desc.GetDim(2);
    const unsigned int dim_d               = nb_dims == 4 ? 1 : input_desc.GetDim(3);
    const unsigned int nb_spatial_elements = dim_d * dim_h * dim_w;

    const auto max_block_size = 256;
    unsigned int block_size =
        nb_spatial_elements < max_block_size ? nb_spatial_elements : max_block_size;
    while (block_size * sm_size_per_thread > kMaxSharedMemSizePerBlock) {
        block_size /= 2;
    }

    dim3 block = {block_size, 1, 1};

    dim3 grid = {((nb_spatial_elements + block.x - 1) / block.x + 1) / 2, dim_c, dim_n};

    return std::make_tuple(grid, block, block.x * sm_size_per_thread);
}

}  // namespace impl
}  // namespace cudnn
