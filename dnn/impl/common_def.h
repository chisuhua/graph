/* 
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */
#pragma once
#include <cudnn.h>
#include <stdint.h>  // int8_t, uint8_t, int32_t

#include <map>
#include <vector>

namespace cudnn {
namespace impl {

#ifdef HW_ILUVATAR_COREX
// FIXME(Peter Han): need confirmation with HW team
static const int kMaxThreadNbPerBlock                  = 4096;
static const int kMaxBlockNbPerSM                      = 64;
static const int kWarpSize                             = 64;
static const dim3 kMaxBlockDimension                   = {1024, 1024, 64};
static const dim3 kMaxGridDimension                    = {2147483647, 65535, 65535};
static const int kNbThreadsPerBlockGainBestPerformance = 256;
static const int kMaxSharedMemSizePerBlock             = 48 * 1024;
#else
// collected from GTX1050
static const int kMaxThreadNbPerBlock                  = 1024;
static const int kMaxBlockNbPerSM                      = 8;  // used to achieve best occupancy
static const int kWarpSize                             = 32;
static const dim3 kMaxBlockDimension                   = {1024, 1024, 64};
static const dim3 kMaxGridDimension                    = {2147483647, 65535, 65535};
static const int kNbThreadsPerBlockGainBestPerformance = 256;
static const int kMaxSharedMemSizePerBlock             = 48 * 1024;
#endif

static const std::map<cudnnDataType_t, size_t> kUnit = {
    {DNN_DATA_FLOAT, sizeof(float)},
    {DNN_DATA_HALF, sizeof(short)},  // FIXME(Peter Han): can not compile using __half
    {DNN_DATA_INT8, sizeof(int8_t)},
    {DNN_DATA_UINT8, sizeof(uint8_t)},
    {DNN_DATA_INT32, sizeof(int32_t)},
};

/** used for matrixMul, Cut Matrix into Blocks to do matrix multiply */
#define TILE_WIDTH 16

}  // namespace impl
}  // namespace cudnn
