/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

#include <cudnn.h>

namespace cudnn {
namespace impl {
namespace meta {

/** @struct
 * @brief underlying data structure for activation
 */
struct CuMetaActivation {
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t nan_opt;
    double coef;
};

}  // namespace meta
}  // namespace impl
}  // namespace cudnn
