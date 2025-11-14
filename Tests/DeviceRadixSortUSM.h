#pragma once
#include "Backend/Resource.h"
#include <cstdint>

namespace ARBD {
// Constants
constexpr uint32_t DRS_RADIX = 256;
constexpr uint32_t DRS_RADIX_MASK = 255;
constexpr uint32_t DRS_RADIX_LOG = 8;
constexpr uint32_t DRS_PART_SIZE = 7680;
constexpr uint32_t DRS_VEC_PART_SIZE = 1920;
constexpr uint32_t DRS_BIN_PART_SIZE = 7680;
constexpr uint32_t DRS_BIN_HISTS_SIZE = 4096;
constexpr uint32_t DRS_BIN_SUB_PART_SIZE = 480;
constexpr uint32_t DRS_BIN_WARPS = 16;
constexpr uint32_t DRS_BIN_KEYS_PER_THREAD = 15;
constexpr uint32_t DRS_BIN_THREADS = DRS_BIN_WARPS * 32; // 512
constexpr uint32_t DRS_SCAN_THREADS = 128;
constexpr uint32_t DRS_UPSWEEP_THREADS = 512;
// DeviceRadixSort using SYCL USM - simpler and more direct implementation
void device_radix_sort_pairs_usm(const Resource &device, uint32_t *keys,
                                 uint32_t *payloads, uint32_t *alt_keys,
                                 uint32_t *alt_payloads,
                                 uint32_t *globalHistogram,
                                 uint32_t *passHistogram, uint32_t size);

} // namespace ARBD
