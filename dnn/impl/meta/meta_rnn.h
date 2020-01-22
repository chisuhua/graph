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
 * @brief underlying data structure for RNN
 */
class CuMetaRnn {
 public:
    CuMetaRnn(int hidden_size,
              int nb_layers,
              cudnnRNNInputMode_t input_mode,
              cudnnDirectionMode_t direction_mode,
              cudnnRNNMode_t mode,
              cudnnRNNAlgo_t algo,
              cudnnDataType_t math_prec)
        : hidden_size_(hidden_size),
          nb_layers_(nb_layers),
          input_mode_(input_mode),
          direction_mode_(direction_mode),
          mode_(mode),
          algo_(algo),
          math_prec_(math_prec),
          nb_ffnn_(mode_ == DNN_GRU ? 3 : (mode_ == DNN_LSTM ? 4 : 1)),
          matrix_math_type_(DNN_DEFAULT_MATH),
          rec_proj_size_(hidden_size_),
          out_proj_size_(0) {}

    int GetHiddenSize() const { return hidden_size_; }

    int GetNbLayers() const { return nb_layers_; }

    cudnnRNNInputMode_t GetInputMode() const { return input_mode_; }

    cudnnDirectionMode_t GetDirectionMode() const { return direction_mode_; }

    cudnnRNNMode_t GetMode() const { return mode_; }

    cudnnRNNAlgo_t GetAlgo() const { return algo_; }

    cudnnDataType_t GetMathPrec() const { return math_prec_; }

    int GetNbFfnn() const { return nb_ffnn_; }

    cudnnMathType_t GetMatrixMathType() const { return matrix_math_type_; }

    void SetMatrixMathType(cudnnMathType_t matrix_math_type) {
        matrix_math_type_ = matrix_math_type;
    }

    int GetRecProjSize() const { return rec_proj_size_; }

    void SetRecProjSize(int rec_proj_size) { rec_proj_size_ = rec_proj_size; }

    int GetOutProjSize() const { return out_proj_size_; }

    void SetOutProjSize(int out_proj_size) { out_proj_size_ = out_proj_size; }

    bool IsRecProjEnabled() const { return mode_ == DNN_LSTM && rec_proj_size_ < hidden_size_; }

 private:
    const int hidden_size_;
    const int nb_layers_;
    const cudnnRNNInputMode_t input_mode_;
    const cudnnDirectionMode_t direction_mode_;
    const cudnnRNNMode_t mode_;
    const cudnnRNNAlgo_t algo_;
    const cudnnDataType_t math_prec_;
    const int nb_ffnn_;
    cudnnMathType_t matrix_math_type_;
    int rec_proj_size_;
    int out_proj_size_;
};

}  // namespace meta
}  // namespace impl
}  // namespace cudnn
