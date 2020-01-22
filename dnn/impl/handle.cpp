/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_handle.h>

namespace cudnn {
namespace impl {

CuHandle::CuHandle() {}

void CuHandle::SetStream(cudaStream_t stream) { stream_ = stream; }

cudaStream_t CuHandle::GetStream() const { return stream_; }

}  // namespace impl
}  // namespace cudnn
