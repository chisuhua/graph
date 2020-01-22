/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/meta/cudnn_meta_dropout.h>

#include <memory>
#include <tuple>

// give cudnnDropoutStruct a definition
struct cudnnDropoutStruct {
    virtual ~cudnnDropoutStruct() = default;
};

namespace cudnn {
namespace impl {

using DropoutProperties = std::tuple<float, void*, size_t, unsigned long long>;

/**
 * @brief cudnnDropoutStruct implementation class
 */
class CuDropoutDescriptor : public cudnnDropoutStruct {
 public:
    CuDropoutDescriptor();
    ~CuDropoutDescriptor() = default;

    void Set(CuHandle handle,
             float dropout,
             void* states,
             int state_size_in_bytes,
             unsigned long long seed);
    void Restore(CuHandle handle,
                 float dropout,
                 void* states,
                 int state_size_in_bytes,
                 unsigned long long seed);

    DropoutProperties Get() const;

    void Get(void** rand_states, float* dropout) const;

    operator meta::CuMetaDropout() const;

    /*
     * @brief set dropout descriptor from meta
     *
     * Difference from Set() method is that this will not actually trigger any action in GPU side
     */
    void operator=(const meta::CuMetaDropout& meta);

 private:
    std::shared_ptr<meta::CuMetaDropout> meta_ = nullptr;
    float dropout;
    unsigned long long seed;
    void* rand_states;
};

REGIST_CONCRETE_OBJECT(cudnnDropoutStruct, CuDropoutDescriptor);

void randstate_init(void* states, unsigned long long seed);

template <class T>
void LaunchDropoutFwdKernel(cudaStream_t& stream,
                            const T* x,
                            T* y,
                            float* reserveSpace,
                            void* rand_states,
                            int tensorLen,
                            float dropout_ratio);

void CuDropoutForward(CuHandle& handle,
                      const CuTensorDescriptor& x_desc,
                      const CuDropoutDescriptor& dropout_desc,
                      const void* in_x,
                      const CuTensorDescriptor& y_desc,
                      void* out_y,
                      void* reserveSpace,
                      size_t reserveSpaceSizeInBytes);

template <typename T>
void LaunchDropoutBwdKernel(
    cudaStream_t& stream, const T* dy, T* dx, float* reserveSpace, int tensorLen, float dropout);

void CuDropoutBackward(CuHandle& handle,
                       const CuDropoutDescriptor& dropout_desc,
                       const CuTensorDescriptor& dy_desc,
                       const void* in_dy,
                       const CuTensorDescriptor& dx_desc,
                       void* out_dx,
                       void* reserveSpace,
                       size_t reserveSpaceSizeInBytes);
}  // namespace impl
}  // namespace cudnn
