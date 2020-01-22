/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_rnn_params.h>

#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace {
constexpr int kBiasMultiplier[] = {0, 1, 1, 2};
}  // namespace

namespace cudnn {
namespace impl {

using cudnn::impl::meta::CuMetaFilter;
using cudnn::impl::meta::CuMetaRnn;
using cudnn::impl::meta::CuMetaTensor;
using std::accumulate;
using std::make_pair;
using std::make_tuple;
using std::next;
using std::pair;
using std::vector;

CuRnnParams::CuRnnParams(const CuMetaRnn& meta, const CuMetaTensor& x_meta, const void* ptr)
    : data_type_(meta.GetMathPrec()),
      rnn_meta_(meta),
      nb_layers_(meta.GetNbLayers()),
      nb_ffnn_(meta.GetNbFfnn()),
      nb_lin_layers_(nb_ffnn_ * 2 + (meta.IsRecProjEnabled() ? 1 : 0)),
      // bias_multiplier_(kBiasMultiplier[meta.GetBiasMode()]),
      bias_multiplier_(2),
      bi_(meta.GetDirectionMode() == DNN_UNIDIRECTIONAL ? 1 : 2),
      nc_(meta.GetHiddenSize()),
      ni_(meta.GetInputMode() == DNN_SKIP_INPUT ? 0 : x_meta.GetDim(2)),
      nr_(meta.GetRecProjSize()),
      ptr_(ptr),
      weights_len_(0),
      bias_len_(0) {
    BuildWeightBias();
}

size_t CuRnnParams::GetParamsSize() const {
    return (weights_len_ + bias_len_) * kUnit.at(data_type_);
}

int CuRnnParams::GetBiasBaseIndex() const { return weights_len_; }

std::pair<CuMetaFilter, const void*> CuRnnParams::GetWeight(int pseudo_layer, int lin_layer_id) {
    auto item = weight_[pseudo_layer * nb_lin_layers_ + lin_layer_id];
    return make_pair(item.first,
                     reinterpret_cast<const void*>(reinterpret_cast<const char*>(ptr_) +
                                                   item.second * kUnit.at(data_type_)));
}

std::pair<CuMetaFilter, const void*> CuRnnParams::GetBias(int pseudo_layer, int lin_layer_id) {
    auto item = bias_[pseudo_layer * nb_ffnn_ * 2 + lin_layer_id];
    return make_pair(item.first,
                     reinterpret_cast<const void*>(reinterpret_cast<const char*>(ptr_) +
                                                   item.second * kUnit.at(data_type_)));
}

void CuRnnParams::BuildWeightBias() {
    const auto nb_pseudo_layers             = nb_layers_ * bi_;
    const vector<int> dim_hidden_weights    = {1, nc_, nr_};
    const vector<int> dim_hidden_weights_bi = {1, nc_, nr_ * 2};
    const vector<int> dim_proj_weights      = {1, nr_, nc_};
    const vector<int> dim_bias              = {1, nc_, 1};
    std::vector<int> dim_input_weights(0);
    if (ni_ != 0) {
        dim_input_weights = {1, nc_, ni_};
    }

    const auto input_filter = CuMetaFilter(
        dim_input_weights.size(), dim_input_weights.data(), data_type_, DNN_TENSOR_NCHW);
    const auto hidden_filter = CuMetaFilter(
        dim_hidden_weights.size(), dim_hidden_weights.data(), data_type_, DNN_TENSOR_NCHW);
    const auto hidden_filter_bi = CuMetaFilter(
        dim_hidden_weights_bi.size(), dim_hidden_weights_bi.data(), data_type_, DNN_TENSOR_NCHW);
    const auto proj_filter = CuMetaFilter(
        dim_proj_weights.size(), dim_proj_weights.data(), data_type_, DNN_TENSOR_NCHW);
    const auto bias_filter =
        CuMetaFilter(dim_bias.size(), dim_bias.data(), data_type_, DNN_TENSOR_NCHW);

    for (int layer_id = 0; layer_id < nb_pseudo_layers; ++layer_id) {
        // input
        if (layer_id == 0 || (bi_ == 2 && layer_id == 1)) {
            for (int ffnn_id = 0; ffnn_id < nb_ffnn_; ++ffnn_id) {
                weight_.push_back(make_pair(input_filter, weights_len_));
                if (ni_ != 0) {
                    weights_len_ += dim_input_weights[2] * dim_input_weights[1];
                }
            }
        } else {
            for (int ffnn_id = 0; ffnn_id < nb_ffnn_; ++ffnn_id) {
                if (bi_ == 1) {
                    weight_.push_back(make_pair(hidden_filter, weights_len_));
                    weights_len_ += dim_hidden_weights[2] * dim_hidden_weights[1];
                } else {
                    weight_.push_back(make_pair(hidden_filter_bi, weights_len_));
                    weights_len_ += dim_hidden_weights_bi[2] * dim_hidden_weights_bi[1];
                }
            }
        }
        // hidden
        for (int ffnn_id = 0; ffnn_id < nb_ffnn_; ++ffnn_id) {
            weight_.push_back(make_pair(hidden_filter, weights_len_));
            weights_len_ += dim_hidden_weights[2] * dim_hidden_weights[1];
        }
        // rec proj
        if (rnn_meta_.IsRecProjEnabled()) {
            weight_.push_back(make_pair(proj_filter, weights_len_));
            weights_len_ += dim_proj_weights[2] * dim_proj_weights[1];
        }
    }

    for (int layer_id = 0; layer_id < nb_pseudo_layers; ++layer_id) {
        // bias
        for (int ffnn_id = 0; ffnn_id < nb_ffnn_ * bias_multiplier_; ++ffnn_id) {
            bias_.push_back(make_pair(bias_filter, weights_len_ + bias_len_));
            bias_len_ += dim_bias[2] * dim_bias[1];
        }
    }
}

}  // namespace impl
}  // namespace cudnn
