#ifdef USE_SYCL
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "DeviceRadixSortKernels.h"
#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

namespace ARBD {

// Helper functions for warp-level operations
template <typename SubGroup>
inline uint32_t drs_get_lane_id(const SubGroup &sg) {
  return sg.get_local_linear_id();
}

template <typename SubGroup>
inline uint32_t drs_get_lane_mask_lt(const SubGroup &sg) {
  const uint32_t lane_id = drs_get_lane_id(sg);
  return (1u << lane_id) - 1;
}

// Inclusive warp scan with circular shift
template <typename WorkItem>
inline uint32_t drs_inclusive_warp_scan_circular_shift(uint32_t val,
                                                       WorkItem &item,
                                                       uint32_t *shared_mem) {
  auto sg = item.get_sub_group();
  const uint32_t lane_id = drs_get_lane_id(sg);
  const uint32_t lane_mask = sg.get_max_local_range()[0] - 1;

  // Inclusive scan
  val = sycl::inclusive_scan_over_group(sg, val, sycl::plus<uint32_t>());

  // Circular shift using shared memory
  const uint32_t warp_id = item.local_id() / sg.get_max_local_range()[0];
  const uint32_t warp_offset = warp_id * sg.get_max_local_range()[0];
  shared_mem[warp_offset + lane_id] = val;
  item.barrier();

  uint32_t target_lane = (lane_id + lane_mask) & lane_mask;
  uint32_t result = shared_mem[warp_offset + target_lane];
  item.barrier();

  return result;
}

// Active exclusive warp scan (only for active lanes)
template <typename WorkItem>
inline uint32_t drs_active_exclusive_warp_scan(uint32_t val, WorkItem &item) {
  auto sg = item.get_sub_group();
  return sycl::exclusive_scan_over_group(sg, val, sycl::plus<uint32_t>());
}

// Inclusive warp scan
template <typename WorkItem>
inline uint32_t drs_inclusive_warp_scan(uint32_t val, WorkItem &item) {
  auto sg = item.get_sub_group();
  return sycl::inclusive_scan_over_group(sg, val, sycl::plus<uint32_t>());
}

// Active inclusive warp scan
template <typename WorkItem>
inline uint32_t drs_active_inclusive_warp_scan(uint32_t val, WorkItem &item) {
  auto sg = item.get_sub_group();
  return sycl::inclusive_scan_over_group(sg, val, sycl::plus<uint32_t>());
}

// ============================================================================
// Upsweep Kernel
// ============================================================================

template <typename WorkItem>
inline void DeviceRadixSortUpsweepFunctor::operator()(idx_t gx,
                                                      WorkItem &item) const {
  uint32_t *s_globalHist = item.template get_shared_mem<uint32_t>(0);

  const idx_t local_id = item.local_id();
  const idx_t group_id = item.group_id();
  const idx_t block_dim = 512; // Fixed block size for upsweep

  // Clear shared memory
  for (uint32_t i = local_id; i < DRS_RADIX * 2; i += block_dim)
    s_globalHist[i] = 0;
  item.barrier();

  // Histogram - 64 threads : 1 histogram in shared memory
  // With 512 threads and 2 histograms (512 elements), each group of 256 threads
  // uses one histogram
  uint32_t *s_wavesHist = &s_globalHist[(local_id / 256) * DRS_RADIX];

  if (group_id < gridDimX - 1) {
    const uint32_t partEnd = (group_id + 1) * DRS_VEC_PART_SIZE;
    for (uint32_t i = local_id + (group_id * DRS_VEC_PART_SIZE); i < partEnd;
         i += block_dim) {
      // Load 4 keys at once (vectorized)
      const uint32_t base_idx = i * 4;
      if (base_idx + 3 < size) {
        const uint32_t k0 = sort[base_idx + 0];
        const uint32_t k1 = sort[base_idx + 1];
        const uint32_t k2 = sort[base_idx + 2];
        const uint32_t k3 = sort[base_idx + 3];

        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group>
            a0(s_wavesHist[(k0 >> radixShift) & DRS_RADIX_MASK]);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group>
            a1(s_wavesHist[(k1 >> radixShift) & DRS_RADIX_MASK]);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group>
            a2(s_wavesHist[(k2 >> radixShift) & DRS_RADIX_MASK]);
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group>
            a3(s_wavesHist[(k3 >> radixShift) & DRS_RADIX_MASK]);

        a0.fetch_add(1);
        a1.fetch_add(1);
        a2.fetch_add(1);
        a3.fetch_add(1);
      }
    }
  }

  if (group_id == gridDimX - 1) {
    for (uint32_t i = local_id + (group_id * DRS_PART_SIZE); i < size;
         i += block_dim) {
      const uint32_t t = sort[i];
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::work_group>
          a(s_wavesHist[(t >> radixShift) & DRS_RADIX_MASK]);
      a.fetch_add(1);
    }
  }
  item.barrier();

  // Reduce to the first hist, pass out, begin prefix sum
  for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
    s_globalHist[i] += s_globalHist[i + DRS_RADIX];
    passHist[i * gridDimX + group_id] = s_globalHist[i];
    s_globalHist[i] = drs_inclusive_warp_scan_circular_shift(
        s_globalHist[i], item, s_globalHist + DRS_RADIX);
  }
  item.barrier();

  if (local_id < (DRS_RADIX / 32))
    s_globalHist[local_id * 32] =
        drs_active_exclusive_warp_scan(s_globalHist[local_id * 32], item);
  item.barrier();

  // Atomically add to device memory
  // This part adds the prefix sum values to the global histogram
  // The CUDA version uses __shfl_sync to get values from adjacent lanes,
  // but we'll simplify by just adding the scanned values directly
  for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
    uint32_t val = s_globalHist[i];
    // Add lane_id contribution if not lane 0
    auto sg = item.get_sub_group();
    const uint32_t lane_id = drs_get_lane_id(sg);
    if (lane_id > 0 && i > 0) {
      val += s_globalHist[i - 1];
    }
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        a(globalHist[i + (radixShift << 5)]);
    a.fetch_add(val);
  }
}

// ============================================================================
// Scan Kernel
// ============================================================================

template <typename WorkItem>
inline void DeviceRadixSortScanFunctor::operator()(idx_t gx,
                                                   WorkItem &item) const {
  uint32_t *s_scan = item.template get_shared_mem<uint32_t>(0);

  const idx_t local_id = item.local_id();
  const idx_t group_id = item.group_id();
  const idx_t block_dim = DRS_SCAN_THREADS; // Fixed block size for scan

  uint32_t reduction = 0;
  auto sg = item.get_sub_group();
  const uint32_t lane_id = drs_get_lane_id(sg);
  const uint32_t lane_mask = sg.get_max_local_range()[0] - 1;
  const uint32_t circularLaneShift = (lane_id + 1) & lane_mask;
  const uint32_t partitionsEnd = threadBlocks / block_dim * block_dim;
  const uint32_t digitOffset = group_id * threadBlocks;

  uint32_t i = local_id;
  for (; i < partitionsEnd; i += block_dim) {
    s_scan[local_id] = passHist[i + digitOffset];
    s_scan[local_id] = drs_inclusive_warp_scan(s_scan[local_id], item);
    item.barrier();

    if (local_id < (block_dim / 32)) {
      s_scan[(local_id + 1) * 32 - 1] =
          drs_active_inclusive_warp_scan(s_scan[(local_id + 1) * 32 - 1], item);
    }
    item.barrier();

    uint32_t val = (lane_id != lane_mask ? s_scan[local_id] : 0);
    if (local_id >= 32) {
      val += sycl::group_broadcast(sg, s_scan[local_id - 1], 0);
    }
    passHist[circularLaneShift + (i & ~lane_mask) + digitOffset] =
        val + reduction;

    reduction += s_scan[block_dim - 1];
    item.barrier();
  }

  // Handle remaining elements
  if (i < threadBlocks)
    s_scan[local_id] = passHist[i + digitOffset];
  else
    s_scan[local_id] = 0;

  s_scan[local_id] = drs_inclusive_warp_scan(s_scan[local_id], item);
  item.barrier();

  if (local_id < (block_dim / 32)) {
    s_scan[(local_id + 1) * 32 - 1] =
        drs_active_inclusive_warp_scan(s_scan[(local_id + 1) * 32 - 1], item);
  }
  item.barrier();

  const uint32_t index = circularLaneShift + (i & ~lane_mask);
  if (index < threadBlocks) {
    uint32_t val = (lane_id != lane_mask ? s_scan[local_id] : 0);
    if (local_id >= 32) {
      val += s_scan[(local_id & ~lane_mask) - 1];
    }
    passHist[index + digitOffset] = val + reduction;
  }
}

// ============================================================================
// Downsweep Pairs Kernel
// ============================================================================

template <typename WorkItem>
inline void
DeviceRadixSortDownsweepPairsFunctor::operator()(idx_t gx,
                                                 WorkItem &item) const {
  uint32_t *s_warpHistograms = item.template get_shared_mem<uint32_t>(0);
  uint32_t *s_localHistogram = item.template get_shared_mem<uint32_t>(
      DRS_BIN_PART_SIZE * sizeof(uint32_t));

  const idx_t local_id = item.local_id();
  const idx_t group_id = item.group_id();
  auto sg = item.get_sub_group();
  const uint32_t warp_index = local_id / 32;
  const uint32_t lane_id = drs_get_lane_id(sg);

  volatile uint32_t *s_warpHist = &s_warpHistograms[warp_index * DRS_RADIX];

  // Clear shared memory
  for (uint32_t i = local_id; i < DRS_BIN_HISTS_SIZE; i += DRS_BIN_THREADS)
    s_warpHistograms[i] = 0;
  item.barrier();

  // Load keys
  uint32_t keys[DRS_BIN_KEYS_PER_THREAD];
  const uint32_t bin_sub_part_start = warp_index * DRS_BIN_SUB_PART_SIZE;
  const uint32_t bin_part_start = group_id * DRS_BIN_PART_SIZE;

  if (group_id < gridDimX - 1) {
#pragma unroll
    for (uint32_t i = 0, t = lane_id + bin_sub_part_start + bin_part_start;
         i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32)
      keys[i] = sort[t];
  }

  if (group_id == gridDimX - 1) {
#pragma unroll
    for (uint32_t i = 0, t = lane_id + bin_sub_part_start + bin_part_start;
         i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32)
      keys[i] = t < size ? sort[t] : 0xffffffff;
  }
  item.barrier();

  // Simple histogram-based offset calculation (replaces complex WLMS)
  // Each thread atomically increments the histogram for its key's digit
  uint16_t offsets[DRS_BIN_KEYS_PER_THREAD];
#pragma unroll
  for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
    const uint32_t digit = (keys[i] >> radixShift) & DRS_RADIX_MASK;
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group>
        a(const_cast<uint32_t &>(s_warpHist[digit]));
    offsets[i] = a.fetch_add(1);
  }
  item.barrier();

  // Exclusive prefix sum up the warp histograms using SYCL group algorithms
  // First, reduce across warps for each digit using SYCL joint_reduce
  if (local_id < DRS_RADIX) {
    // Create a strided view of the histogram values for this digit
    // We need to reduce: s_warpHistograms[local_id], s_warpHistograms[local_id
    // + 256], s_warpHistograms[local_id + 512], ... up to DRS_BIN_HISTS_SIZE
    const uint32_t num_warps = DRS_BIN_HISTS_SIZE / DRS_RADIX; // 16 warps
    uint32_t reduction = 0;

    // Use joint_reduce on the strided memory locations
    // Create a pointer to the first element for this digit
    uint32_t *digit_start = &s_warpHistograms[local_id];
    auto group = item.get_group();

    // For each warp's contribution to this digit, accumulate
    for (uint32_t warp = 0; warp < num_warps; ++warp) {
      uint32_t warp_val = s_warpHistograms[local_id + warp * DRS_RADIX];
      reduction += warp_val;
      // Also compute exclusive prefix sum for this warp
      if (warp > 0) {
        s_warpHistograms[local_id + warp * DRS_RADIX] = reduction - warp_val;
      }
    }

    // Store reduction for scan
    s_localHistogram[local_id] = reduction;
  }
  item.barrier();

  // Compute exclusive prefix sums for the 256 digits.
  // Use a single thread to avoid complex synchronization requirements.
  if (local_id == 0) {
    uint32_t running_total = 0;
    for (uint32_t i = 0; i < DRS_RADIX; ++i) {
      const uint32_t value = s_localHistogram[i];
      s_warpHistograms[i] = running_total;
      running_total += value;
    }
  }
  item.barrier();

  // Update offsets
  if (warp_index) {
#pragma unroll
    for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
      const uint32_t t2 = (keys[i] >> radixShift) & DRS_RADIX_MASK;
      offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
      offsets[i] += s_warpHistograms[(keys[i] >> radixShift) & DRS_RADIX_MASK];
  }

  // Load in threadblock reductions (pre-computed offsets from scan!)
  if (local_id < DRS_RADIX) {
    s_localHistogram[local_id] = globalHist[local_id + (radixShift << 5)] +
                                 passHist[local_id * gridDimX + group_id] -
                                 s_warpHistograms[local_id];
  }
  item.barrier();

  // Scatter keys into shared memory
#pragma unroll
  for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
    s_warpHistograms[offsets[i]] = keys[i];
  item.barrier();

  // Scatter keys to device memory
  uint8_t digits[DRS_BIN_KEYS_PER_THREAD];
  if (group_id < gridDimX - 1) {
#pragma unroll
    for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
         ++i, t += DRS_BIN_THREADS) {
      digits[i] = (s_warpHistograms[t] >> radixShift) & DRS_RADIX_MASK;
      alt[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    }
    item.barrier();

    // Load payloads
#pragma unroll
    for (uint32_t i = 0, t = lane_id + bin_sub_part_start + bin_part_start;
         i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32) {
      keys[i] = sortPayload[t];
    }

    // Scatter payloads into shared memory
#pragma unroll
    for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
      s_warpHistograms[offsets[i]] = keys[i];
    item.barrier();

    // Scatter payloads to device
#pragma unroll
    for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
         ++i, t += DRS_BIN_THREADS) {
      altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    }
  }

  if (group_id == gridDimX - 1) {
    const uint32_t finalPartSize = size - bin_part_start;
#pragma unroll
    for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
         ++i, t += DRS_BIN_THREADS) {
      if (t < finalPartSize) {
        digits[i] = (s_warpHistograms[t] >> radixShift) & DRS_RADIX_MASK;
        alt[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
      }
    }
    item.barrier();

#pragma unroll
    for (uint32_t i = 0, t = lane_id + bin_sub_part_start + bin_part_start;
         i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32) {
      if (t < size)
        keys[i] = sortPayload[t];
    }

#pragma unroll
    for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
      s_warpHistograms[offsets[i]] = keys[i];
    item.barrier();

#pragma unroll
    for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
         ++i, t += DRS_BIN_THREADS) {
      if (t < finalPartSize)
        altPayload[s_localHistogram[digits[i]] + t] = s_warpHistograms[t];
    }
  }
}

// ============================================================================
// Main Sorting Function
// ============================================================================

void device_radix_sort_pairs(const Resource &device, uint32_t *keys,
                             uint32_t *payloads, uint32_t size) {
  const uint32_t maxThreadBlocks = 2048;
  const uint32_t threadBlocks =
      std::min((size + DRS_PART_SIZE - 1) / DRS_PART_SIZE, maxThreadBlocks);

  // Allocate buffers
  DeviceBuffer<uint32_t> alt_keys(size, device.id());
  DeviceBuffer<uint32_t> alt_payloads(size, device.id());
  DeviceBuffer<uint32_t> globalHistogram(DRS_RADIX * 4, device.id());
  DeviceBuffer<uint32_t> passHistogram(DRS_RADIX * threadBlocks, device.id());

  // Initialize to zero
  std::vector<uint32_t> zeros_hist(DRS_RADIX * 4, 0);
  std::vector<uint32_t> zeros_pass(DRS_RADIX * threadBlocks, 0);
  globalHistogram.copy_from_host(zeros_hist.data(), zeros_hist.size());
  passHistogram.copy_from_host(zeros_pass.data(), zeros_pass.size());

  std::cout << "DeviceRadixSort: size=" << size
            << ", threadBlocks=" << threadBlocks << std::endl;

  // Four radix passes
  for (uint32_t pass = 0; pass < 4; ++pass) {
    const uint32_t radixShift = pass * 8;

    // Reset histograms
    globalHistogram.copy_from_host(zeros_hist.data(), zeros_hist.size());
    passHistogram.copy_from_host(zeros_pass.data(), zeros_pass.size());

    // Upsweep
    KernelConfig upsweep_cfg;
    upsweep_cfg.grid_size = {threadBlocks, 1, 1};
    upsweep_cfg.block_size = {512, 1, 1};
    upsweep_cfg.shared_memory = DRS_RADIX * 2 * sizeof(uint32_t);

    DeviceRadixSortUpsweepFunctor upsweep_func{
        (pass % 2 == 0) ? keys : alt_keys.data(),
        globalHistogram.data(),
        passHistogram.data(),
        size,
        radixShift,
        threadBlocks};
    Event upsweep_event =
        launch_kernel_with_workitem(device, upsweep_cfg, upsweep_func);
    std::cout << "Upsweep launched" << std::endl;
    // Scan
    KernelConfig scan_cfg;
    scan_cfg.grid_size = {DRS_RADIX, 1, 1};
    scan_cfg.block_size = {DRS_SCAN_THREADS, 1, 1};
    scan_cfg.shared_memory = DRS_SCAN_THREADS * sizeof(uint32_t);

    DeviceRadixSortScanFunctor scan_func{passHistogram.data(), threadBlocks};
    Event scan_event = launch_kernel_with_workitem(device, scan_cfg, scan_func);
    std::cout << "Scan launched" << std::endl;
    // Downsweep
    KernelConfig downsweep_cfg;
    downsweep_cfg.grid_size = {threadBlocks, 1, 1};
    downsweep_cfg.block_size = {DRS_BIN_THREADS, 1, 1};
    downsweep_cfg.shared_memory =
        (DRS_BIN_PART_SIZE + DRS_RADIX) * sizeof(uint32_t);

    DeviceRadixSortDownsweepPairsFunctor downsweep_func{
        (pass % 2 == 0) ? keys : alt_keys.data(),
        (pass % 2 == 0) ? payloads : alt_payloads.data(),
        (pass % 2 == 0) ? alt_keys.data() : keys,
        (pass % 2 == 0) ? alt_payloads.data() : payloads,
        globalHistogram.data(),
        passHistogram.data(),
        size,
        radixShift,
        threadBlocks};
    Event downsweep_event =
        launch_kernel_with_workitem(device, downsweep_cfg, downsweep_func);
    downsweep_event.wait();

    std::cout << "Pass " << pass << " completed" << std::endl;
  }
}

} // namespace ARBD
#endif // USE_SYCL
