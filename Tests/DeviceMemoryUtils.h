// Tests/DeviceMemoryUtils.h
#pragma once

#include "Backend/Resource.h"
#include <cstddef>
#include <iomanip>
#include <iostream>

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace ARBD {
namespace Tests {

/**
 * @brief Memory information structure
 */
struct DeviceMemoryInfo {
  size_t total_bytes = 0;
  size_t free_bytes = 0;
  size_t used_bytes = 0;
  bool free_available = false; // Whether free memory info is available
};

/**
 * @brief Query device memory information
 * @param device The resource to query
 * @return DeviceMemoryInfo structure with memory stats
 */
inline DeviceMemoryInfo query_device_memory(const Resource &device) {
  DeviceMemoryInfo info;

#ifdef USE_SYCL
  try {
    auto &queue = device.get_sycl_queue();
    auto sycl_device = queue.get_device();
    info.total_bytes =
        sycl_device.get_info<sycl::info::device::global_mem_size>();
#ifdef USE_SYCL_ICPX
    info.free_bytes =
        sycl_device.get_info<sycl::ext::intel::info::device::free_memory>();
#else
    info.free_bytes = 0;
#endif
    info.used_bytes = info.total_bytes - info.free_bytes;

  } catch (const std::exception &e) {
    std::cerr << "Failed to query device memory: " << e.what() << std::endl;
  }
#elif defined(USE_CUDA)
  // Direct CUDA path
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err == cudaSuccess) {
    info.total_bytes = total_bytes;
    info.free_bytes = free_bytes;
    info.used_bytes = total_bytes - free_bytes;
    info.free_available = true;
  }
#else
  // No GPU backend available
  info.total_bytes = 0;
  info.free_bytes = 0;
  info.used_bytes = 0;
  info.free_available = false;
#endif

  return info;
}

/**
 * @brief Print device memory information
 * @param device The resource to query and print
 */
inline void print_device_memory_info(const Resource &device) {
  DeviceMemoryInfo info = query_device_memory(device);

  std::cout << "Device [" << device.id() << "] Memory:" << std::endl;
  std::cout << "  Total: " << std::fixed << std::setprecision(2)
            << (info.total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB ("
            << (info.total_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;

  if (info.free_available) {
    std::cout << "  Free:  " << std::fixed << std::setprecision(2)
              << (info.free_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB ("
              << (info.free_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "  Used:  " << std::fixed << std::setprecision(2)
              << (info.used_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB ("
              << (info.used_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
  } else {
    std::cout << "  (Free memory not available - use 'nvidia-smi' to check)"
              << std::endl;
  }
}

/**
 * @brief Check if allocation would exceed available memory
 * @param device The resource to check
 * @param required_bytes Bytes needed for allocation
 * @param safety_margin_percent Safety margin as percentage (default 10%)
 * @return true if allocation is safe, false otherwise
 */
inline bool can_allocate_safely(const Resource &device, size_t required_bytes,
                                double safety_margin_percent = 10.0) {
  DeviceMemoryInfo info = query_device_memory(device);

  if (info.free_available) {
    // We have free memory info, check against it
    size_t available_with_margin =
        info.free_bytes * (100.0 - safety_margin_percent) / 100.0;
    return required_bytes <= available_with_margin;
  } else {
    // Only have total memory, check against 90% of total as conservative
    // estimate
    size_t safe_total =
        info.total_bytes * (100.0 - safety_margin_percent) / 100.0;
    return required_bytes <= safe_total;
  }
}

/**
 * @brief Estimate memory required for radix sort operation
 * @param num_elements Number of elements to sort
 * @return Estimated bytes needed
 */
inline size_t estimate_radix_sort_memory(size_t num_elements) {
  constexpr uint32_t DRS_RADIX = 256;
  constexpr uint32_t DRS_PART_SIZE = 7680;

  // Main buffers (4 buffers: keys, payloads, alt_keys, alt_payloads)
  size_t main_mem = 4 * num_elements * sizeof(uint32_t);

  // Global histogram
  size_t global_hist = DRS_RADIX * 4 * sizeof(uint32_t);

  // Pass histogram
  uint32_t thread_blocks = (num_elements + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
  size_t pass_hist = DRS_RADIX * thread_blocks * sizeof(uint32_t);

  return main_mem + global_hist + pass_hist;
}

} // namespace Tests
} // namespace ARBD
