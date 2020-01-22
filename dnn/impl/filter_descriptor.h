/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/cudnn_tensor_descriptor.h>
#include <cudnn/impl/meta/cudnn_meta_filter.h>

#include <memory>
#include <vector>

// give cudnnFilterStruct a definition
struct cudnnFilterStruct {
    virtual ~cudnnFilterStruct() = default;
};

namespace cudnn {
namespace impl {
/**
 * @brief cudnnFilterStruct implementation class
 *
 * The first dimension of the tensor defines the batch size n, and the second dimension defines the
 * number of features maps c.
 */
class CuFilterDescriptor : public cudnnFilterStruct {
 public:
    CuFilterDescriptor();
    ~CuFilterDescriptor() = default;

    /**
     * Set an a 4-D filter descriptor , cudnnSetFilter4dDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w);

    /**
     * Set an a N-D filter descriptor , cudnnSetFilterNdDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void
    Set(cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]);

    /**
     * Get an a 4-D filter descriptor , cudnnGetFilter4dDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Get(cudnnDataType_t* dataType, cudnnTensorFormat_t* format, int* k, int* c, int* h, int* w)
        const;

    /**
     * Get an a N-D filter descriptor , cudnnGetFilterNdDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Get(int nbDimsRequested,
             cudnnDataType_t* dataType,
             cudnnTensorFormat_t* format,
             int* nbDims,
             int filterDimA[]) const;

    inline cudnnDataType_t GetDataType() const { return data_type_; }

    inline int GetDataTypeSizeInBytes() const {
        int data_type_size = 0;
        if (DNN_DATA_FLOAT == this->data_type_ || DNN_DATA_INT32 == this->data_type_ ||
            DNN_DATA_INT8x4 == this->GetDataType() || DNN_DATA_UINT8x4 == this->GetDataType()) {
            data_type_size = 4;
        } else if (DNN_DATA_HALF == this->GetDataType()) {
            data_type_size = 2;
        } else if (DNN_DATA_INT8 == this->GetDataType()) {
            data_type_size = 8;
        } else if (DNN_DATA_DOUBLE == this->GetDataType()) {
            data_type_size = 8;
        }
        return data_type_size;
    }

    inline cudnnTensorFormat_t GetDataType() { return format_; }

    inline int GetNbDim() const { return nbDims_; }

    inline std::vector<int> GetFilterDimA() const { return filter_dimA_; }

    inline int GetFilterDimA(int dim_id) const { return filter_dimA_[dim_id]; }

    inline int GetOutC() const { return filter_dimA_[0]; }

    inline int GetInC() const { return filter_dimA_[1]; }

    inline int GetFilterH() const { return filter_dimA_[2]; }

    inline int GetFilterW() const { return filter_dimA_[3]; }

    operator meta::CuMetaFilter() const;

 private:
    std::shared_ptr<logger> logger_           = GetLogger();
    std::shared_ptr<meta::CuMetaFilter> data_ = std::make_shared<meta::CuMetaFilter>(
        0, std::vector<int>(0).data(), DNN_DATA_FLOAT, DNN_TENSOR_NCHW);
    cudnnDataType_t data_type_;
    cudnnTensorFormat_t format_;
    int nbDims_                   = 0;
    std::vector<int> filter_dimA_ = std::vector<int>(DNN_DIM_MAX);
    bool packed_;
};

REGIST_CONCRETE_OBJECT(cudnnFilterStruct, CuFilterDescriptor);

}  // namespace impl
}  // namespace cudnn
