/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cuda.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

namespace cudnn {
namespace impl {

template <typename Type>
struct CuDeviceMemory {
    explicit CuDeviceMemory(int len, cudaStream_t stream = nullptr) : stream_(stream), len_(len) {
        auto ret = cudaMalloc(&pointer_, len_ * sizeof(Type));
        if (ret != cudaSuccess) {
            GetLogger()->error("failed to alloc device memory of {} bytes", len * sizeof(Type));
            throw CuException(DNN_STATUS_ALLOC_FAILED);
        }
    }

    void SyncToDevice(Type* src, int len) {
        assert(len <= len_);
        auto ret =
            cudaMemcpyAsync(pointer_, src, sizeof(Type) * len, cudaMemcpyHostToDevice, stream_);
        if (ret != cudaSuccess) {
            GetLogger()->error("failed to copy memory of {} bytes to device", len * sizeof(Type));
            throw CuException(DNN_STATUS_ALLOC_FAILED);
        }
    }

    void SyncToDevice(Type* src) { SyncToDevice(src, len_); }

    ~CuDeviceMemory() {
        GetLogger()->info(">>>> Device memory get destroied");
        auto ret = cudaFree(pointer_);
        if (ret != cudaSuccess) {
            GetLogger()->error("failed to free device memory of {} bytes", len_ * sizeof(Type));
        }
    }

    Type* operator&() { return pointer_; }

    const Type* operator&() const { return pointer_; }

 private:
    cudaStream_t stream_ = nullptr;
    int len_             = 0;
    Type* pointer_       = nullptr;
};

}  // namespace impl
}  // namespace cudnn
