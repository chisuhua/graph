/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_dropout_descriptor.h>
#include <cudnn/impl/meta/cudnn_meta_filter.h>
#include <cudnn/impl/meta/cudnn_meta_rnn.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <vector>

namespace cudnn {
namespace impl {

size_t CuRnnGetWorkspaceSize(const meta::CuMetaRnn& rnn_meta,
                             int seq_length,
                             const std::vector<meta::CuMetaTensor>& x_metas);

}  // namespace impl
}  // namespace cudnn
