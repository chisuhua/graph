/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_lrn_descriptor.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/cudnn_exception.h>

namespace cudnn {
namespace impl {

CuLRNDescriptor::CuLRNDescriptor() {}

void CuLRNDescriptor::Set(unsigned lrn_n,
                          double lrn_alpha,
                          double lrn_beta,
                          double lrn_k) {
    lrn_n_     = lrn_n;
    lrn_alpha_ = lrn_alpha;
    lrn_beta_  = lrn_beta;
    lrn_k_     = lrn_k;
    }

void CuLRNDescriptor::Get(unsigned* lrn_n,
                          double* lrn_alpha,
                          double* lrn_beta,
                          double* lrn_k) const {
    *lrn_n     = lrn_n_;
    *lrn_alpha = lrn_alpha_;
    *lrn_beta  = lrn_beta_;
    *lrn_k     = lrn_k_;
    }

}  // namespace impl
}  // namespace cudnn


