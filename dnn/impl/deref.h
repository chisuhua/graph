/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn/cudnn_exception.h>
#include <cudnn/cudnn_logger.h>
#include <typeinfo>

#define REGIST_CONCRETE_OBJECT(INTERFACE, CONCRETE)                   \
    inline CONCRETE& Deref(INTERFACE* obj) {                          \
        if (obj == nullptr) {                                         \
            auto log = cudnn::GetLogger();                            \
            log->debug("{}: {} is nullptr", __FUNCTION__, #CONCRETE); \
            throw cudnn::CuException(DNN_STATUS_BAD_PARAM);         \
        }                                                             \
        try {                                                         \
            return dynamic_cast<CONCRETE&>(*obj);                     \
        } catch (const std::bad_cast&) {                              \
            auto log = cudnn::GetLogger();                            \
            log->debug("{}: {} not valid", __FUNCTION__, #CONCRETE);  \
            throw cudnn::CuException(DNN_STATUS_BAD_PARAM);         \
        }                                                             \
    }                                                                 \
    inline const CONCRETE& Deref(const INTERFACE* obj) {              \
        if (obj == nullptr) {                                         \
            auto log = cudnn::GetLogger();                            \
            log->debug("{}: {} is nullptr", __FUNCTION__, #CONCRETE); \
            throw cudnn::CuException(DNN_STATUS_BAD_PARAM);         \
        }                                                             \
        try {                                                         \
            return dynamic_cast<const CONCRETE&>(*obj);               \
        } catch (const std::bad_cast&) {                              \
            auto log = cudnn::GetLogger();                            \
            log->debug("{}: {} not valid", __FUNCTION__, #CONCRETE);  \
            throw cudnn::CuException(DNN_STATUS_BAD_PARAM);         \
        }                                                             \
    }
