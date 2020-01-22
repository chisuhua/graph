/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once

namespace cudnn {
namespace impl {

/** @struct
 * @breif a structure to describe a matrix position, rows, columns and len in workspace or reserve
 * space for RNN inference and training
 */
struct CuRnnMatrix {
    int start; /* starting position in workspace / reserve space */
    int rows;  /* number of rows */
    int cols;  /* number of columns*/
};

}  // namespace impl
}  // namespace cudnn
