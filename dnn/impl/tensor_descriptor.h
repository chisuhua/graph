/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <cudnn/impl/cudnn_deref.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/impl/meta/cudnn_meta_tensor.h>

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

// give cudnnTensorStruct a definition
struct cudnnTensorStruct {
    virtual ~cudnnTensorStruct() = default;
};

namespace cudnn {
namespace impl {

/** data type, tensor format, number of dimensions, dimension size list, stride list */
using TensorProperties = std::tuple<cudnnDataType_t,
                                    cudnnTensorFormat_t,
                                    int,
                                    const std::vector<int>,
                                    const std::vector<int>>;

/**
 * @brief cudnnTensorStruct implementation class
 *
 * Below description which is directly copied from cudnn document:
 * The first dimension of the tensor defines the batch size n, and the second dimension defines the
 * number of features maps c.
 */
class CuTensorDescriptor : public cudnnTensorStruct {
 public:
    CuTensorDescriptor();
    ~CuTensorDescriptor() = default;

    /**
     * set a 4-dimension tensor, cudnnSetTensor4dDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(cudnnTensorFormat_t format, cudnnDataType_t data_type, int n, int c, int h, int w);

    /**
     * set a 4-dimension tensor, cudnnSetTensor4dDescriptorEx should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(cudnnDataType_t data_type,
             int n,
             int c,
             int h,
             int w,
             int n_stride,
             int c_stride,
             int h_stride,
             int w_stride);

    /**
     * set an variable dimension tensor (3-8), cudnnSetTensorNdDescriptor should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(cudnnDataType_t data_type, int nb_dims, const int dim_a[], const int stride_a[]);

    /**
     * set an variable dimension tensor (3-8), cudnnSetTensorNdDescriptorEx should call this.
     * NOTE: exception might be thrown out due to invalid parameters
     */
    void Set(cudnnTensorFormat_t format, cudnnDataType_t data_type, int nb_dims, const int dim_a[]);

    /**
     * get descriptor
     */
    void Get(cudnnDataType_t* data_type,
             int* n,
             int* c,
             int* h,
             int* w,
             int* n_stride,
             int* c_stride,
             int* h_stride,
             int* w_stride) const;

    void Get(int nb_dims_requested,
             cudnnDataType_t* data_type,
             int* nb_dims,
             int dim_a[],
             int stride_a[]) const;

    /* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h *
       h_stride + w * w_stride

       1)Example of all images in row major order one batch of features after the
       other (with an optional padding on row) input_stride :  c x h x h_stride
       feature_stride : h x h_stride h_stride  :  >= w  ( h_stride = w if no
       padding) w_stride  : 1


       2)Example of all images in row major with features maps interleaved
       input_stride :  c x h x h_stride
       feature_stride : 1
       h_stride  :  w x c
       w_stride  : c

       3)Example of all images in column major order one batch of features after the
       other (with optional padding on column) input_stride :  c x w x w_stride
       feature_stride : w x w_stride h_stride  :  1 w_stride  :  >= h
    */
    size_t GetSizeInBytes() const;

    /**
     * get all properties one calling
     *
     * @return a tuple has data type, tensor format, number of dimensions, list of size of each
     * dimension and list of stride of each dimension.
     * NOTE: recommend use this method over other get methods.
     */
    TensorProperties Get() const;

    operator meta::CuMetaTensor() const;

    size_t Unit() const;

    inline int GetNbDims() const { return data_->GetNbDims(); }

    inline std::vector<int> GetDim() const {
        std::vector<int> dim(data_->GetNbDims());
        for (int i = 0; i < dim.size(); ++i) {
            dim[i] = data_->GetDim(i + 1);
        }
        return dim;
    }

    inline std::vector<int> GetStride() const {
        std::vector<int> stride(data_->GetNbDims());
        for (int i = 0; i < stride.size(); ++i) {
            stride[i] = data_->GetStride(i + 1);
        }
        return stride;
    }

    inline cudnnDataType_t GetDataType() const { return data_->GetDataType(); }

    inline int GetDataTypeSizeInBytes() const {
        int data_type_size = 0;
        if (DNN_DATA_FLOAT == data_->GetDataType() || DNN_DATA_INT32 == data_->GetDataType() ||
            DNN_DATA_INT8x4 == data_->GetDataType() ||
            DNN_DATA_UINT8x4 == data_->GetDataType()) {
            data_type_size = 4;
        } else if (DNN_DATA_HALF == data_->GetDataType()) {
            data_type_size = 2;
        } else if (DNN_DATA_INT8 == data_->GetDataType()) {
            data_type_size = 8;
        } else if (DNN_DATA_DOUBLE == data_->GetDataType()) {
            data_type_size = 8;
        }
        return data_type_size;
    }

    inline cudnnTensorFormat_t GetTensorFormat() const { return data_->GetFormat(); }

    inline void Get(cudnnDataType_t* data_type, int* nb_dims, int dim_a[], int stride_a[]) const {
        *nb_dims   = data_->GetNbDims();
        *data_type = data_->GetDataType();

        for (int i = 0; i < *nb_dims; ++i) {
            dim_a[i]    = data_->GetDim(i + 1);
            stride_a[i] = data_->GetStride(i + 1);
        }
    }

    /** NOTE: inside DNN, we count DIM from 1 */
    inline int GetDim(int dim) const {
        assert(dim <= DNN_DIM_MAX);
        return data_->GetDim(dim);
    }

    /** NOTE: inside DNN, we count DIM from 1 */
    inline int GetStride(int dim) const {
        assert(dim <= DNN_DIM_MAX);
        return data_->GetStride(dim);
    }

    /** alias to getSize(1) */
    inline int GetN() const { return data_->GetDim(1); }

    /** alias to getSize(2) */
    inline int GetC() const { return data_->GetDim(2); }

    /** alias to getSize(3) */
    inline int GetH() const { return data_->GetDim(3); }

    /** alias to getSize(4) */
    inline int GetW() const { return data_->GetDim(4); }

    /** alias to getStride(1) */
    inline int GetStrideN() const { return data_->GetStride(1); }

    /** alias to getStride(2) */
    inline int GetStrideC() const { return data_->GetStride(2); }

    /** alias to getStride(3) */
    inline int GetStrideH() const { return data_->GetStride(3); }

    /** alias to getStride(4) */
    inline int GetStrideW() const { return data_->GetStride(4); }

    inline bool GetPackMode() const { return packed_; }

 private:
    bool packed_ = true;
    std::shared_ptr<meta::CuMetaTensor> data_ =
        std::make_shared<meta::CuMetaTensor>(0,
                                             std::vector<int>(0).data(),
                                             std::vector<int>(0).data(),
                                             DNN_DATA_FLOAT,
                                             DNN_TENSOR_NCHW);
};

REGIST_CONCRETE_OBJECT(cudnnTensorStruct, CuTensorDescriptor);

void TransformTensor(CuHandle& handle,
                     const void* alpha,
                     const CuTensorDescriptor& x_desc,
                     const void* x,
                     const void* beta,
                     const CuTensorDescriptor& y_desc,
                     void* y);

void AddTensor(CuHandle& handle,
               const void* alpha,
               const CuTensorDescriptor& a_desc,
               const void* a,
               const void* beta,
               const CuTensorDescriptor& c_desc,
               void* c);

template <typename T1, typename T2>
void LaunchAddTensorKernel(const T2* alpha,
                           const CuTensorDescriptor& a_desc,
                           const T1* a,
                           const T2* beta,
                           const CuTensorDescriptor& c_desc,
                           T1* c);

template <class T>
void LaunchTransformTensorKernel(const float* alpha,
                                 const CuTensorDescriptor& x_desc,
                                 const T* x,
                                 const float* beta,
                                 const CuTensorDescriptor& y_desc,
                                 T* y);

}  // namespace impl
}  // namespace cudnn
