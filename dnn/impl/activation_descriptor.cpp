/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_activation_descriptor.h>

namespace cudnn {
namespace impl {

void CuActivationDescriptor::Set(cudnnActivationMode_t mode,
                                 cudnnNanPropagation_t relu_nan_opt,
                                 double coef) {
    meta_.reset(new meta::CuMetaActivation({mode, relu_nan_opt, coef}));
}

cudnnActivationMode_t CuActivationDescriptor::GetMode() const { return meta_->mode; }

cudnnNanPropagation_t CuActivationDescriptor::GetNanPropagation() const { return meta_->nan_opt; }

double CuActivationDescriptor::GetCoef() const { return meta_->coef; }

}  // namespace impl
}  // namespace cudnn
