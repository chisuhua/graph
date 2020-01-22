/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>

// give cudnnContext a definition
struct cudnnContext {
    virtual ~cudnnContext() = default;
};

namespace cudnn {
namespace impl {

/**
 * @brief cudnnHandle_t implementation class
 */
class CuHandle : public cudnnContext {
 public:
    CuHandle();
    ~CuHandle() = default;

    void SetStream(cudaStream_t stream);
    cudaStream_t GetStream() const;

 private:
    cudaStream_t stream_ = nullptr;
};

REGIST_CONCRETE_OBJECT(cudnnContext, CuHandle);

}  // namespace impl
}  // namespace cudnn
