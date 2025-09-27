#pragma once

#ifdef USE_CUDA
// Prevent conflicts between CUDA headers and C++ standard library
#ifndef __CUDA_RUNTIME_H__
#define _GLIBCXX_USE_CXX11_ABI 1
#endif

#include "ARBDException.h"
#include <algorithm>
#include <array>
#include <mutex>
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

private:
  static std::vector<cudaDeviceProp> device_properties_;
  static std::vector<std::vector<bool>> peer_access_matrix_;
  static bool initialized_;

  static void discover_devices();
  static void query_peer_access();
};

} // namespace ARBD
#endif
