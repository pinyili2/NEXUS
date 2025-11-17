#pragma once
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Resource.h"
#include <cstdint>

namespace ARBD {

// DeviceRadixSort constants
constexpr uint32_t DRS_RADIX = 256;
constexpr uint32_t DRS_RADIX_MASK = 255;
constexpr uint32_t DRS_RADIX_LOG = 8;

// Upsweep kernel
constexpr uint32_t DRS_PART_SIZE = 7680;
constexpr uint32_t DRS_VEC_PART_SIZE = 1920;

// Downsweep kernel
constexpr uint32_t DRS_BIN_PART_SIZE = 7680;
constexpr uint32_t DRS_BIN_HISTS_SIZE = 4096;
constexpr uint32_t DRS_BIN_SUB_PART_SIZE = 480;
constexpr uint32_t DRS_BIN_WARPS = 16;
constexpr uint32_t DRS_BIN_KEYS_PER_THREAD = 15;
constexpr uint32_t DRS_BIN_THREADS = DRS_BIN_WARPS * 32;

// Scan kernel
constexpr uint32_t DRS_SCAN_THREADS = 128;

// Upsweep: Compute histograms per work-group
struct DeviceRadixSortUpsweepFunctor {
  uint32_t *sort;
  uint32_t *globalHist;
  uint32_t *passHist;
  uint32_t size;
  uint32_t radixShift;
  uint32_t gridDimX;

  template <typename WorkItem>
  DEVICE void operator()(idx_t gx, WorkItem &item) const;
};

// Scan: Compute prefix sums across work-groups
struct DeviceRadixSortScanFunctor {
  uint32_t *passHist;
  uint32_t threadBlocks;

  template <typename WorkItem>
  DEVICE void operator()(idx_t gx, WorkItem &item) const;
};

// Downsweep: Scatter keys and values using pre-computed offsets
struct DeviceRadixSortDownsweepPairsFunctor {
  uint32_t *sort;
  uint32_t *sortPayload;
  uint32_t *alt;
  uint32_t *altPayload;
  uint32_t *globalHist;
  uint32_t *passHist;
  uint32_t size;
  uint32_t radixShift;
  uint32_t gridDimX;

  template <typename WorkItem>
  DEVICE void operator()(idx_t gx, WorkItem &item) const;
};

// Main sorting function
void device_radix_sort_pairs(const Resource &device, uint32_t *keys,
                             uint32_t *payloads, uint32_t size);

} // namespace ARBD
