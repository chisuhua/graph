/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_dropout_descriptor.h>

namespace cudnn {
namespace impl {

using meta::CuMetaDropout;

CuDropoutDescriptor::CuDropoutDescriptor() {}

void CuDropoutDescriptor::Set(CuHandle handle,
                              float dropout_ratio,
                              void* states,
                              int state_size_in_bytes,
                              unsigned long long seed) {
    if (meta_ == nullptr) {
        meta_ = std::make_shared<CuMetaDropout>(dropout_ratio, states, state_size_in_bytes, seed);
    } else {
        meta_.reset(new CuMetaDropout(dropout_ratio, states, state_size_in_bytes, seed));
    }
    if (NULL != states) {
        randstate_init(states, seed);
    }
    rand_states = states;
    dropout     = dropout_ratio;
}

void CuDropoutDescriptor::Restore(CuHandle handle,
                                  float dropout,
                                  void* states,
                                  int state_size_in_bytes,
                                  unsigned long long seed) {
    // TODO(Peter Han): GPU action
}

void CuDropoutDescriptor::Get(void** states, float* dropout_ratio) const {
    *states        = rand_states;
    *dropout_ratio = dropout;
}

DropoutProperties CuDropoutDescriptor::Get() const {
    return std::make_tuple(
        meta_->GetDropout(), meta_->GetStates(), meta_->GetStateSizeInBytes(), meta_->GetSeed());
}

CuDropoutDescriptor::operator CuMetaDropout() const { return *meta_; }

void CuDropoutDescriptor::operator=(const CuMetaDropout& meta) {
    if (meta_ == nullptr) {
        meta_ = std::make_shared<CuMetaDropout>(
            meta.GetDropout(), meta.GetStates(), meta.GetStateSizeInBytes(), meta.GetSeed());
    } else {
        meta_.reset(new CuMetaDropout(
            meta.GetDropout(), meta.GetStates(), meta.GetStateSizeInBytes(), meta.GetSeed()));
    }
}

}  // namespace impl
}  // namespace cudnn
