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
#include <string>
#include <vector>

#ifndef NDEBUG
#include <iostream>
#endif

namespace {

#define CUDA_FUNC_CALL_AND_THROW(func, ...)                               \
    do {                                                                  \
        auto code = func(__VA_ARGS__);                                    \
        if (code != cudaSuccess) {                                        \
            GetLogger()->warn("{}: {} return {}", __LINE__, #func, code); \
            throw cudnn::CuException(DNN_STATUS_EXECUTION_FAILED);      \
        }                                                                 \
    } while (false)

}  // namespace

namespace cudnn {
namespace impl {

/* workspace forward layout
| hy part                         |
|---------------------------------|
| pseudo layer#1 bs#1 hx, bs#2 hx |
| pseudo layer#2 bs#1 hx, bs#2 hx |
| pseudo layer#3 bs#1 hx, bs#2 hx |
| pseudo layer#4 bs#1 hx, bs#2 hx |
|---------------------------------|
| cy part                         |
|---------------------------------|
| pseudo layer#1 bs#1 cx, bs#2 cx |
| pseudo layer#2 bs#1 cx, bs#2 cx |
| pseudo layer#3 bs#1 cx, bs#2 cx |
| pseudo layer#4 bs#1 cx, bs#2 cx |
|---------------------------------|
| Wx part                         |
|---------------------------------|
| ts#1 bs#1 f2l, bs#2 f2l         |
| ts#2 bs#1 f2l, bs#2 f2l         |
| ts#1 bs#1 l2f, bs#2 l2f         |
| ts#2 bs#2 l2f, bs#2 l2f         |
|---------------------------------|
| Rh part                         |
|---------------------------------|
| bs#1 f2l, bs#2 f2l              |
| bs#1 l2f, bs#2 l2f              |
|---------------------------------|
*/
template <class T>
struct CuRnnWorkSpaceForward {
    CuRnnWorkSpaceForward(const meta::CuMetaRnn& meta,
                          const std::vector<meta::CuMetaTensor>& x_metas,
                          void* workspace     = nullptr,
                          const void* x       = nullptr,
                          void* y             = nullptr,
                          const void* hx      = nullptr,
                          const void* cx      = nullptr,
                          cudaStream_t stream = nullptr)
        : mode_(meta.GetMode()),
          skip_input_(meta.GetInputMode() == DNN_SKIP_INPUT),
          bi_mode_(meta.GetDirectionMode() == DNN_BIDIRECTIONAL),
          bi_(meta.GetDirectionMode() == DNN_UNIDIRECTIONAL ? 1 : 2),
          nb_ffnn_(meta.GetNbFfnn()),
          nb_layers_(meta.GetNbLayers()),
          nc_(meta.GetHiddenSize()),
          nr_(meta.GetRecProjSize()),
          max_batch_size_(x_metas[0].GetDim(1)),
          seq_len_(x_metas.size()),
          ptr_(reinterpret_cast<T*>(workspace)),
          x_(reinterpret_cast<const T*>(x)),
          y_(reinterpret_cast<T*>(y)),
          hx_(reinterpret_cast<const T*>(hx)),
          cx_(reinterpret_cast<const T*>(cx))
#ifndef NDEBUG
          ,
          stream_(stream)
#endif
    {
        for (int i = 0; i < x_metas.size(); ++i) {
            const auto bs = x_metas[i].GetDim(1);
            batches_.push_back(bs);
            acc_batches_.push_back(total_batch_size_);
            total_batch_size_ += bs;
        }

        auto round = [](int v, int to = 4) { return v % to == 0 ? v : (v / to + 1) * to; };

        const int hy_size = round(nb_layers_ * bi_ * nc_ * max_batch_size_);
        const int cy_size = mode_ == DNN_LSTM ? hy_size : 0;
        const int wx_size = round(total_batch_size_ * nc_ * bi_ * nb_ffnn_);
        const int rh_size = round(nc_ * bi_ * max_batch_size_ * nb_ffnn_);

        cy_base_       = hy_size;
        wx_base_       = cy_base_ + cy_size;
        rh_base_       = wx_base_ + wx_size;
        size_in_bytes_ = (hy_size + cy_size + wx_size + rh_size) * sizeof(T);

        if (workspace == nullptr) {
            return;
        }

        const int hx_cx_size = nb_layers_ * bi_ * nc_ * max_batch_size_ * sizeof(T);
        if (hx == nullptr) {
            CUDA_FUNC_CALL_AND_THROW(cudaMemsetAsync, ptr_, 0, hx_cx_size, stream);
        } else {
            CUDA_FUNC_CALL_AND_THROW(
                cudaMemcpyAsync, ptr_, hx, hx_cx_size, cudaMemcpyDeviceToDevice, stream);
        }

        if (mode_ == DNN_LSTM) {
            if (cx == nullptr) {
                CUDA_FUNC_CALL_AND_THROW(cudaMemsetAsync, ptr_ + cy_base_, 0, hx_cx_size, stream);
            } else {
                CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                         ptr_ + cy_base_,
                                         cx,
                                         hx_cx_size,
                                         cudaMemcpyDeviceToDevice,
                                         stream);
            }
        }
    }

    ~CuRnnWorkSpaceForward() = default;

    size_t GetSizeInBytes() const { return size_in_bytes_; }

    inline const T* GetHx(int pseudo_layer_id) const {
        int idx = nc_ * max_batch_size_ * pseudo_layer_id;
        if (hx_ != nullptr) {
            return hx_ + idx;
        } else {
            return ptr_ + idx;  // have already been initialized with 0
        }
    }

    inline T* GetHy(int pseudo_layer_id) const {
        int idx = nc_ * max_batch_size_ * pseudo_layer_id;
        return ptr_ + idx;
    }

    inline const T* GetCx(int pseudo_layer_id) const {
        int idx = nc_ * max_batch_size_ * pseudo_layer_id;
        if (cx_ != nullptr) {
            return cx_ + idx;
        } else {
            return ptr_ + cy_base_ + idx;  // have already been initialized with 0
        }
    }

    inline T* GetCy(int pseudo_layer_id) const {
        int idx = nc_ * max_batch_size_ * pseudo_layer_id;
        return ptr_ + cy_base_ + idx;
    }

    inline T* GetWx(int pseudo_layer_id, int ts = 0) const {
        if (skip_input_ && (pseudo_layer_id / bi_ == 0)) {
            int idx = nc_ * acc_batches_[ts];
            return const_cast<T*>(x_ + idx);
        } else {
            int idx = wx_base_ + total_batch_size_ * nc_ * nb_ffnn_ * (pseudo_layer_id % bi_);
            idx += acc_batches_[ts] * nc_ * nb_ffnn_;
            return ptr_ + idx;
        }
    }

    inline T* GetRh(int pseudo_layer_id) const {
        int idx = rh_base_ + max_batch_size_ * nc_ * nb_ffnn_ * (pseudo_layer_id % bi_);
        return ptr_ + idx;
    }

    inline T* GetHidden(int pseudo_layer_id, int ts = 0) const {
        int skip = (bi_mode_ && pseudo_layer_id % 2) ? nc_ : 0;
        int idx  = nc_ * bi_ * acc_batches_[ts];
        idx += skip;
        return y_ + idx;
    }

#ifndef NDEBUG
    void Dump(const std::string& info) const {
        const int hy_size = nb_layers_ * bi_ * nc_ * max_batch_size_;
        const int cy_size = nb_layers_ * bi_ * nc_ * max_batch_size_;
        const int wx_size = total_batch_size_ * nc_ * bi_ * nb_ffnn_;
        const int rh_size = nc_ * bi_ * max_batch_size_ * nb_ffnn_;
        const int y_size  = nc_ * bi_ * total_batch_size_;

        std::shared_ptr<std::vector<T>> hy = std::make_shared<std::vector<T>>(hy_size);
        CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                 hy->data(),
                                 ptr_,
                                 hy_size * sizeof(T),
                                 cudaMemcpyDeviceToHost,
                                 stream_);
        CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, stream_);
        std::cout << "------Workspace snapshot " << info << "--------------------" << std::endl;
        std::cout << "Hx/Hy part" << std::endl;
        int idx = 0;
        for (int pl = 0; pl < nb_layers_ * bi_; ++pl) {
            for (int b = 0; b < max_batch_size_; ++b) {
                std::cout << "pseudo layer #" << pl << " batch#" << b << " :";
                for (int i = 0; i < nc_; ++i) {
                    std::cout << hy->at(idx++) << " ";
                }
                std::cout << std::endl;
            }
        }

        if (mode_ == DNN_LSTM) {
            std::shared_ptr<std::vector<T>> cy = std::make_shared<std::vector<T>>(cy_size);
            CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                     cy->data(),
                                     ptr_ + cy_base_,
                                     cy_size * sizeof(T),
                                     cudaMemcpyDeviceToHost,
                                     stream_);
            CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, stream_);
            std::cout << "Cx/Cy part" << std::endl;
            idx = 0;
            for (int pl = 0; pl < nb_layers_ * bi_; ++pl) {
                for (int b = 0; b < max_batch_size_; ++b) {
                    std::cout << "pseudo layer #" << pl << " batch#" << b << " :";
                    for (int i = 0; i < nc_; ++i) {
                        std::cout << cy->at(idx++) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::shared_ptr<std::vector<T>> wx = std::make_shared<std::vector<T>>(wx_size);
        CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                 wx->data(),
                                 ptr_ + wx_base_,
                                 wx_size * sizeof(T),
                                 cudaMemcpyDeviceToHost,
                                 stream_);
        CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, stream_);
        std::cout << "Wx part" << std::endl;
        idx = 0;
        for (int dir = 0; dir < bi_; ++dir) {
            for (int ts = 0; ts < seq_len_; ++ts) {
                for (int b = 0; b < max_batch_size_; ++b) {
                    if (dir == 0) {
                        std::cout << "fwd ";
                    } else {
                        std::cout << "bwd ";
                    }
                    std::cout << "ts #" << ts << " batch#" << b << " :";
                    for (int i = 0; i < nc_; ++i) {
                        std::cout << wx->at(idx++) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::shared_ptr<std::vector<T>> rh = std::make_shared<std::vector<T>>(rh_size);
        CUDA_FUNC_CALL_AND_THROW(cudaMemcpyAsync,
                                 rh->data(),
                                 ptr_ + rh_base_,
                                 rh_size * sizeof(T),
                                 cudaMemcpyDeviceToHost,
                                 stream_);
        CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, stream_);
        std::cout << "Rh part" << std::endl;
        idx = 0;
        for (int dir = 0; dir < bi_; ++dir) {
            for (int b = 0; b < max_batch_size_; ++b) {
                if (dir == 0) {
                    std::cout << "fwd ";
                } else {
                    std::cout << "bwd ";
                }
                std::cout << "batch#" << b << " :";
                for (int i = 0; i < nc_; ++i) {
                    std::cout << rh->at(idx++) << " ";
                }
                std::cout << std::endl;
            }
        }

        std::shared_ptr<std::vector<T>> y = std::make_shared<std::vector<T>>(y_size);
        CUDA_FUNC_CALL_AND_THROW(
            cudaMemcpyAsync, y->data(), y_, y_size * sizeof(T), cudaMemcpyDeviceToHost, stream_);
        CUDA_FUNC_CALL_AND_THROW(cudaStreamSynchronize, stream_);
        std::cout << "Y part" << std::endl;
        idx = 0;
        for (int ts = 0; ts < seq_len_; ++ts) {
            for (int b = 0; b < batches_[ts]; ++b) {
                for (int dir = 0; dir < bi_; ++dir) {
                    std::cout << "ts#" << ts;
                    if (dir == 0) {
                        std::cout << " fwd ";
                    } else {
                        std::cout << " bwd ";
                    }
                    std::cout << "batch#" << b << " :";
                    for (int i = 0; i < nc_; ++i) {
                        std::cout << y->at(idx++) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "----------------------------------------------------" << std::endl;
    }
#endif

 private:
    const cudnnRNNMode_t mode_;
    const bool skip_input_;
    const bool bi_mode_;
    const int bi_;
    const int nb_ffnn_;
    const int nb_layers_;
    const int nc_;
    const int nr_;
    const int max_batch_size_;
    const int seq_len_;
    T* ptr_      = nullptr;
    const T* x_  = nullptr;
    T* y_        = nullptr;
    const T* hx_ = nullptr;
    const T* cx_ = nullptr;

    std::vector<int> batches_     = std::vector<int>(0);
    std::vector<int> acc_batches_ = std::vector<int>(0);
    int total_batch_size_         = 0;

    int hy_len_      = 0;
    int cy_base_     = 0;
    int wx_base_     = 0;
    int rh_base_     = 0;
    int hidden_base_ = 0;

    int size_in_bytes_ = 0;
#ifndef NDEBUG
    cudaStream_t stream_;
#endif
};

}  // namespace impl
}  // namespace cudnn
