/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn/impl/cudnn_rnn_descriptor.h>

#include <cudnn/impl/cudnn_common_def.h>
#include <cudnn/impl/cudnn_rnn_params.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>
#include <tuple>

namespace cudnn {
namespace impl {

using cudnn::impl::meta::CuMetaRnn;
using std::make_shared;

void CuRnnDescriptor::Set(int hidden_size,
                          int num_layers,
                          CuDropoutDescriptor* dropout,
                          cudnnRNNInputMode_t input_mode,
                          cudnnDirectionMode_t direction_mode,
                          cudnnRNNMode_t mode,
                          cudnnRNNAlgo_t algo,
                          cudnnDataType_t math_prec) {
    if (meta_ != nullptr) {
        meta_.reset(new CuMetaRnn(
            hidden_size, num_layers, input_mode, direction_mode, mode, algo, math_prec));
        dropout_ = dropout;
    } else {
        meta_ = make_shared<CuMetaRnn>(
            hidden_size, num_layers, input_mode, direction_mode, mode, algo, math_prec);
        dropout_ = dropout;
    }

    initialized = true;
}

void CuRnnDescriptor::SetMatrixMathType(cudnnMathType_t math_type) {
    CheckInitialized();
    meta_->SetMatrixMathType(math_type);
}

void CuRnnDescriptor::SetProjectionLayers(int rec_proj_size, int out_proj_size) {
    CheckInitialized();
    meta_->SetRecProjSize(rec_proj_size);
    meta_->SetOutProjSize(out_proj_size);
}

RnnProperties CuRnnDescriptor::Get() const {
    CheckInitialized();
    return std::make_tuple(meta_->GetHiddenSize(),
                           meta_->GetNbLayers(),
                           dropout_,
                           meta_->GetInputMode(),
                           meta_->GetDirectionMode(),
                           meta_->GetMode(),
                           meta_->GetAlgo(),
                           meta_->GetMathPrec());
}

cudnnMathType_t CuRnnDescriptor::GetMatrixMathType() const {
    CheckInitialized();
    return meta_->GetMatrixMathType();
}

std::tuple<int, int> CuRnnDescriptor::GetProjectionLayers() const {
    CheckInitialized();
    return std::make_tuple<int, int>(meta_->GetRecProjSize(), meta_->GetOutProjSize());
}

CuRnnDescriptor::operator meta::CuMetaRnn() const {
    CheckInitialized();
    return *meta_;
}

CuDropoutDescriptor* CuRnnDescriptor::GetDropoutDescriptor() {
    CheckInitialized();
    return dropout_;
}

void CuRnnDescriptor::CheckInitialized() const {
    if (!initialized) {
        GetLogger()->info("RNN descriptor is still uninitialized, call SetRNNDescriptor API first");
        throw CuException(DNN_STATUS_BAD_PARAM);
    }
}

}  // namespace impl
}  // namespace cudnn
