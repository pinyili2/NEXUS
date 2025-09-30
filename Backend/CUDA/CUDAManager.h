#pragma once

#ifdef USE_CUDA
// Prevent conflicts between CUDA headers and C++ standard library
#ifndef __CUDA_RUNTIME_H__
#define _GLIBCXX_USE_CXX11_ABI 1
#endif

#include "ARBDException.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <type_traits>
#include <utility>
#include <vector>

// Include CUDA headers after standard library headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace ARBD {
inline void check_cuda_error(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    ARBD_Exception(ExceptionType::CUDARuntimeError, "CUDA error at %s:%d: %s",
                   file, line, cudaGetErrorString(error));
  }
}
#define CUDA_CHECK(call) check_cuda_error(call, __FILE__, __LINE__)
namespace CUDA {
/**
 * @brief Simplified CUDA Manager - Device discovery and peer access only
 *
 * Stream management has been moved to Resource class.
 * Manager now only handles:
 * - Device enumeration
 * - Device properties
 * - Peer-to-peer access setup
 */
class Manager {
public:
  // Device discovery and initialization
  static void init();
  static void finalize();

  // Device information
  static int device_count();
  static cudaDeviceProp get_device_properties(int device_id);

  // Peer access management
  static void enable_peer_access();
  static bool can_access_peer(int device1, int device2);
  static void init_for_rank(int local_rank = 0, int ranks_per_node = 1,
                            int threads_per_rank = -1, bool verbose = false);

  /**
   * Initialize GPU for current OpenMP thread
   * Call this within an OpenMP parallel region
   */
  static void init_for_omp_thread();

  /**
   * Get GPU assigned to current OpenMP thread
   */
  static int get_thread_gpu();

  /**
   * Set GPU-OpenMP thread affinity mapping
   * @param strategy "block" or "cyclic" distribution
   */
  static void set_omp_gpu_affinity(const std::string &strategy = "block");

  /**
   * Get the GPUs assigned to this rank
   */
  static std::vector<int> get_rank_devices() {
    std::lock_guard<std::mutex> lock(mtx_);
    return rank_devices_;
  }

  /**
   * Check if running in multi-rank mode
   */
  static bool is_multi_rank() {
    std::lock_guard<std::mutex> lock(mtx_);
    return multi_rank_mode_;
  }

  /**
   * Check if OpenMP is enabled
   */
  static bool is_omp_enabled() { return omp_threads_ > 1; }

private:
  static std::vector<int> rank_devices_;   // GPU IDs assigned to this rank
  static bool multi_rank_mode_;            // Whether initialized for multi-rank
  static int rank_id_;                     // This process's rank ID
  static int omp_threads_;                 // Number of OpenMP threads
  static std::vector<int> thread_gpu_map_; // OpenMP thread -> GPU mapping
  static std::string gpu_affinity_strategy_; // "block" or "cyclic"

  // Helper to setup OpenMP thread-GPU mapping
  static void setup_omp_gpu_mapping();
  static std::vector<cudaDeviceProp> device_properties_;
  static std::vector<std::vector<bool>> peer_access_matrix_;
  static bool initialized_;
  static std::mutex mtx_;
  static void discover_devices();
  static void query_peer_access();
};
} // namespace CUDA

} // namespace ARBD
#endif
