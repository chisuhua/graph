/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_rnn_matrix.h>
#include <cudnn/impl/meta/cudnn_meta_rnn.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <memory>
#include <numeric>
#include <vector>

namespace cudnn {
namespace impl {

template <class T>
struct CuRnnReserveSpace {
    CuRnnReserveSpace(const meta::CuMetaRnn& meta,
                      const std::vector<meta::CuMetaTensor>& x_metas,
                      void* reserve_space = nullptr,
                      T* y                = nullptr)
        : mode_(meta.GetMode()),
          bi_mode_(meta.GetDirectionMode() == DNN_BIDIRECTIONAL),
          bi_(meta.GetDirectionMode() == DNN_UNIDIRECTIONAL ? 1 : 2),
          nb_ffnn_(meta.GetNbFfnn()),
          nb_layers_(meta.GetNbLayers()),
          nc_(meta.GetHiddenSize()),
          nr_(meta.GetRecProjSize()),
          ni_(x_metas[0].GetDim(2)),
          seq_len_(x_metas.size()),
          max_batch_size_(x_metas[0].GetDim(1)),
          ptr_(reinterpret_cast<T*>(reserve_space)),
          y_(y) {
        for (int i = 0; i < x_metas.size(); ++i) {
            const auto bs = x_metas[i].GetDim(1);
            acc_batches_.push_back(total_batch_size_);
            total_batch_size_ += bs;
        }

        auto round        = [](int v, int to = 4) { return v % to == 0 ? v : (v / to + 1) * to; };
        hidden_state_len_ = round(bi_ * nc_ * total_batch_size_ * (nb_layers_ - 1));

        if (mode_ == DNN_RNN_RELU || mode_ == DNN_RNN_TANH) {
            size_in_bytes_ =
                (hidden_state_len_ + round(bi_ * nc_ * total_batch_size_ * nb_layers_)) * sizeof(T);
        } else if (mode_ == DNN_LSTM) {
            cell_state_len_ = round(bi_ * nc_ * total_batch_size_ * nb_layers_);
            int gates_len   = round(bi_ * nc_ * total_batch_size_ * nb_ffnn_ * nb_layers_);
            size_in_bytes_  = (hidden_state_len_ + cell_state_len_ + gates_len) * sizeof(T);
        } else {
            size_in_bytes_ =
                (hidden_state_len_ + round(nc_ * 6 * total_batch_size_ * nb_layers_ * bi_)) *
                sizeof(T);
        }
    }

    ~CuRnnReserveSpace() = default;

    size_t GetSizeInBytes() const { return size_in_bytes_; }

    /*NOTE: timestamp is 0-based index*/
    inline T* GetHidden(int pseudo_layer_id, int ts) const {
        assert(0 <= ts && ts < seq_len_);

        const int physical_layer_id = pseudo_layer_id / bi_;
        assert(physical_layer_id < nb_layers_);

        int idx = acc_batches_[ts] * bi_ * nc_;
        idx += (pseudo_layer_id % bi_) * nc_;

        if (physical_layer_id == (nb_layers_ - 1)) {
            return y_ + idx;
        } else {
            int layer_base = physical_layer_id * total_batch_size_ * bi_ * nc_;
            return ptr_ + layer_base + idx;
        }
    }

    /*NOTE: timestamp is 0-based index*/
    inline T* GetReLULin(int pseudo_layer_id, int ts) const {
        assert(mode_ == DNN_RNN_RELU);
        assert(0 <= ts && ts < seq_len_);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_;
        idx += acc_batches_[ts] * nc_;

        return ptr_ + idx;
    }

    /*NOTE: timestamp is 0-based index*/
    inline T* GetTanhHidden(int pseudo_layer_id, int ts) const {
        assert(mode_ == DNN_RNN_TANH);
        assert(0 <= ts && ts < seq_len_);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_;
        idx += acc_batches_[ts] * nc_;

        return ptr_ + idx;
    }

    inline T* GetCellState(int pseudo_layer_id, int ts) const {
        assert(mode_ == DNN_LSTM);
        assert(0 <= ts && ts < seq_len_);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_;
        idx += acc_batches_[ts] * nc_;

        return ptr_ + idx;
    }

    /* gate: 0, input; 1, forget; 2 cell prime; 3 output */
    inline T* GetLSTMGateResult(int pseudo_layer_id, int ts, int gate = 0) const {
        assert(mode_ == DNN_LSTM);
        assert(0 <= ts && ts < seq_len_);
        assert(0 <= gate && gate < 4);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_ * 4;
        idx += acc_batches_[ts] * nc_ * 4;
        idx += nc_ * gate;

        return ptr_ + idx;
    }

    inline T* GetGRUGateResult(int pseudo_layer_id, int ts, int gate = 0) const {
        assert(mode_ == DNN_GRU);
        assert(0 <= ts && ts < seq_len_);
        assert(0 <= gate && gate < 2);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_ * 6;
        idx += acc_batches_[ts] * nc_ * 6;
        idx += nc_ * gate;

        return ptr_ + idx;
    }

    inline T* GetGRUWxBw(int pseudo_layer_id, int ts, int gate = 0) const {
        assert(mode_ == DNN_GRU);
        assert(0 <= ts && ts < seq_len_);
        assert(0 <= gate && gate < 2);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_ * 6;
        idx += acc_batches_[ts] * nc_ * 6;
        idx += nc_ * 2;

        return ptr_ + idx;
    }

    inline T* GetGRURhBr(int pseudo_layer_id, int ts, int gate = 0) const {
        assert(mode_ == DNN_GRU);
        assert(0 <= ts && ts < seq_len_);
        assert(0 <= gate && gate < 2);

        int idx = hidden_state_len_ + pseudo_layer_id * total_batch_size_ * nc_ * 6;
        idx += acc_batches_[ts] * nc_ * 6;
        idx += nc_ * 3;

        return ptr_ + idx;
    }

 private:
    const cudnnRNNMode_t mode_;
    const bool bi_mode_;
    const int bi_;
    const int nb_ffnn_;
    const int nb_layers_;
    const int nc_;
    const int nr_;
    const int ni_;
    const int seq_len_;
    const int max_batch_size_;
    T* ptr_ = nullptr;
    T* y_   = nullptr;

    std::vector<int> acc_batches_ = std::vector<int>(0);
    int total_batch_size_         = 0;
    int hidden_state_len_         = 0;
    int cell_state_len_           = 0;
    int size_in_bytes_            = 0;
};

}  // namespace impl
}  // namespace cudnn
