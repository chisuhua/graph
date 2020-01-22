/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#include <cudnn.h>
#include <cudnn/api/cudnn_api_param_check.h>
#include <cudnn/impl/cudnn_handle.h>
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>

#include <map>

using cudnn::GetLogger;
using cudnn::Try;
using cudnn::impl::Deref;
using cudnn::impl::CuHandle;

namespace {
static std::map<cudnnStatus_t, const char*> kErrorString = {
    {DNN_STATUS_SUCCESS, "DNN_STATUS_SUCCESS"},
    {DNN_STATUS_NOT_INITIALIZED, "DNN_STATUS_NOT_INITIALIZED"},
    {DNN_STATUS_ALLOC_FAILED, "DNN_STATUS_ALLOC_FAILED"},
    {DNN_STATUS_BAD_PARAM, "DNN_STATUS_BAD_PARAM"},
    {DNN_STATUS_ARCH_MISMATCH, "DNN_STATUS_ARCH_MISMATCH"},
    {DNN_STATUS_MAPPING_ERROR, "DNN_STATUS_MAPPING_ERROR"},
    {DNN_STATUS_EXECUTION_FAILED, "DNN_STATUS_EXECUTION_FAILED"},
    {DNN_STATUS_INTERNAL_ERROR, "DNN_STATUS_INTERNAL_ERROR"},
    {DNN_STATUS_NOT_SUPPORTED, "DNN_STATUS_NOT_SUPPORTED"},
    {DNN_STATUS_LICENSE_ERROR, "DNN_STATUS_LICENSE_ERROR"},
    {DNN_STATUS_RUNTIME_PREREQUISITE_MISSING, "DNN_STATUS_RUNTIME_PREREQUISITE_MISSING"},
    {DNN_STATUS_RUNTIME_IN_PROGRESS, "DNN_STATUS_RUNTIME_IN_PROGRESS"},
    {DNN_STATUS_RUNTIME_FP_OVERFLOW, "DNN_STATUS_RUNTIME_FP_OVERFLOW"},
};
}  //  namespace

extern "C" {
cudnnStatus_t DNNWINAPI cudnnGetProperty(libraryPropertyType type, int* value) {
    cudnnStatus_t result = DNN_STATUS_SUCCESS;
    switch (type) {
    case MAJOR_VERSION: *value = DNN_MAJOR; break;
    case MINOR_VERSION: *value = DNN_MINOR; break;
    case PATCH_LEVEL: *value = DNN_PATCHLEVEL; break;
    default: result = DNN_STATUS_NOT_SUPPORTED;
    }
    return result;
}

cudnnStatus_t DNNWINAPI cudnnCreate(cudnnHandle_t* handle) {
    return Try([&] {
        cudnn::api::CheckNull(handle);

        *handle = new CuHandle();
    });
}

cudnnStatus_t DNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
    return Try([&] {
        Deref(handle);
        delete handle;
    });
}

cudnnStatus_t DNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    return Try([&] { Deref(handle).SetStream(streamId); });
}

cudnnStatus_t DNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t* streamId) {
    return Try([&] { *streamId = Deref(handle).GetStream(); });
}

const char* DNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
    try {
        return kErrorString.at(status);
    } catch (const std::out_of_range&) { return "DNN_UNKNOWN_STATUS"; }
}

size_t DNNWINAPI cudnnGetVersion(void) { return DNN_VERSION; }

size_t DNNWINAPI cudnnGetCudartVersion(void) {
    // TODO(Peter Han): need to align with runtime
    assert(("not implement", false));
    return 0;
}

cudnnStatus_t DNNWINAPI cudnnSetCallback(unsigned mask, void* udata, cudnnCallback_t fptr) {
    // TODO(Peter Han): beyond our target
    assert(("not supported", false));
    return DNN_STATUS_NOT_SUPPORTED;
}

cudnnStatus_t DNNWINAPI cudnnGetCallback(unsigned* mask, void** udata, cudnnCallback_t* fptr) {
    // TODO(Peter Han): beyond our target
    assert(("not supported", false));
    return DNN_STATUS_NOT_SUPPORTED;
}
}
