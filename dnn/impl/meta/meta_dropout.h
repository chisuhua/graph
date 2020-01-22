/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn.h>

namespace cudnn {
namespace impl {
namespace meta {

/** @class
 * @brief underlying data structure for dropout
 */
class CuMetaDropout {
 public:
    CuMetaDropout(float dropout, void* states, size_t state_size_in_bytes, unsigned long long seed)
        : dropout_(dropout),
          states_(states),
          state_size_in_bytes_(state_size_in_bytes),
          seed_(seed) {}

    CuMetaDropout(const CuMetaDropout& other) = default;

    float GetDropout() const { return dropout_; }

    void* GetStates() const { return states_; }

    size_t GetStateSizeInBytes() const { return state_size_in_bytes_; }

    unsigned long long GetSeed() const { return seed_; }

 private:
    float dropout_;
    void* states_ = nullptr;
    size_t state_size_in_bytes_;
    unsigned long long seed_;
};

}  // namespace meta
}  // namespace impl
}  // namespace cudnn
