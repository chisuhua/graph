/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/meta/cudnn_meta_filter.h>
#include <cudnn/impl/meta/cudnn_meta_rnn.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <utility>
#include <vector>

namespace cudnn {
namespace impl {

/** @struct
 * @brief a struct for conveniently accessing RNN weight and bias.
 */
struct CuRnnParams {
    CuRnnParams(const meta::CuMetaRnn& meta, const meta::CuMetaTensor& x_meta, const void* ptr);

    ~CuRnnParams() = default;

    size_t GetParamsSize() const;

    std::pair<meta::CuMetaFilter, const void*> GetWeight(int pseudo_layer, int lin_layer_id);

    std::pair<meta::CuMetaFilter, const void*> GetBias(int pseudo_layer, int lin_layer_id);

    int GetBiasBaseIndex() const;

    template <class T>
    const T* GetBatchedW(int pseudo_layer) const {
        auto item = weight_[pseudo_layer * nb_lin_layers_];
        return reinterpret_cast<const T*>(ptr_) + item.second;
    }

    template <class T>
    const T* GetBatchedR(int pseudo_layer) const {
        auto item = weight_[pseudo_layer * nb_lin_layers_ + nb_ffnn_];
        return reinterpret_cast<const T*>(ptr_) + item.second;
    }

    template <class T>
    const T* GetBatchedBw(int pseudo_layer) const {
        auto item = bias_[pseudo_layer * nb_ffnn_ * 2];
        return reinterpret_cast<const T*>(ptr_) + item.second;
    }

    template <class T>
    const T* GetBatchedBr(int pseudo_layer) const {
        auto item = bias_[pseudo_layer * nb_ffnn_ * 2 + nb_ffnn_];
        return reinterpret_cast<const T*>(ptr_) + item.second;
    }

 private:
    void BuildWeightBias();

 private:
    const cudnnDataType_t data_type_;
    const meta::CuMetaRnn rnn_meta_;
    const int nb_layers_;
    const int nb_ffnn_;
    const int nb_lin_layers_;
    const int bias_multiplier_;
    const int bi_;
    const int nc_;
    const int ni_;
    const int nr_;
    const void* ptr_;
    int weights_len_;
    int bias_len_;

    std::vector<std::pair<meta::CuMetaFilter, size_t>> weight_;
    std::vector<std::pair<meta::CuMetaFilter, size_t>> bias_;
};

}  // namespace impl
}  // namespace cudnn
