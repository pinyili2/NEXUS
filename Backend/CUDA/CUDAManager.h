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
} // namespace ARBD
#endif
