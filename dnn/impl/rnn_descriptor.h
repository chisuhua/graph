/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_dropout_descriptor.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/meta/cudnn_meta_rnn.h>

#include <memory>
#include <tuple>

// give cudnnRNNStruct a definition
struct cudnnRNNStruct {
    virtual ~cudnnRNNStruct() = default;
};

namespace cudnn {
namespace impl {

using RnnProperties = std::tuple<int,
                                 int,
                                 CuDropoutDescriptor*,
                                 cudnnRNNInputMode_t,
                                 cudnnDirectionMode_t,
                                 cudnnRNNMode_t,
                                 cudnnRNNAlgo_t,
                                 cudnnDataType_t>;

class CuRnnDescriptor : public cudnnRNNStruct {
 public:
    CuRnnDescriptor()          = default;
    virtual ~CuRnnDescriptor() = default;

    void Set(int hidden_size,
             int num_layers,
             CuDropoutDescriptor* dropout,
             cudnnRNNInputMode_t input_mode,
             cudnnDirectionMode_t direction,
             cudnnRNNMode_t mode,
             cudnnRNNAlgo_t algo,
             cudnnDataType_t math_prec);
    RnnProperties Get() const;

    void SetMatrixMathType(cudnnMathType_t math_type);
    cudnnMathType_t GetMatrixMathType() const;

    void SetProjectionLayers(int rec_proj_size, int out_proj_size);
    std::tuple<int, int> GetProjectionLayers() const;

    CuDropoutDescriptor* GetDropoutDescriptor();

    operator meta::CuMetaRnn() const;

 private:
    void CheckInitialized() const;

 private:
    bool initialized                       = false;
    std::shared_ptr<meta::CuMetaRnn> meta_ = nullptr;
    CuDropoutDescriptor* dropout_          = nullptr;
};

REGIST_CONCRETE_OBJECT(cudnnRNNStruct, CuRnnDescriptor);

}  // namespace impl
}  // namespace cudnn
