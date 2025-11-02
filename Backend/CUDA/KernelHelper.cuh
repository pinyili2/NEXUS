#pragma once

#include "../ARBDException.h"
#include "../Buffer.h"
#include "../Events.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#ifdef __CUDACC__
// Only include CUDA headers when compiling with nvcc
#include "CUDAManager.h"
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#endif
using idx_t = size_t;
namespace ARBD {
namespace CUDA {

struct WorkItem {
  idx_t global_id_;
  idx_t local_id_;
  idx_t group_id_;
  void *shared_mem_;
  size_t shared_mem_size_;

#ifdef __CUDACC__
  __device__ WorkItem(idx_t global_id, idx_t local_id, idx_t group_id,
                      void *shared_mem, size_t shared_mem_size)
      : global_id_(global_id), local_id_(local_id), group_id_(group_id),
        shared_mem_(shared_mem), shared_mem_size_(shared_mem_size) {}

  __device__ idx_t global_id() const { return global_id_; }
  __device__ idx_t local_id() const { return local_id_; }
  __device__ idx_t group_id() const { return group_id_; }

  __device__ void barrier() { __syncthreads(); }

  template <typename T> __device__ T *get_shared_mem(size_t offset_bytes = 0) {
    return reinterpret_cast<T *>(static_cast<char *>(shared_mem_) +
                                 offset_bytes);
  }

  template <typename T>
  __device__ const T *get_shared_mem(size_t offset_bytes = 0) const {
    return reinterpret_cast<const T *>(static_cast<const char *>(shared_mem_) +
                                       offset_bytes);
  }
#else
  // Host-side stub methods (should not be called on host)
  WorkItem(idx_t, idx_t, idx_t, void *, size_t) {}
  idx_t global_id() const { return 0; }
  idx_t local_id() const { return 0; }
  idx_t group_id() const { return 0; }
  void barrier() {}
  template <typename T> T *get_shared_mem(size_t = 0) { return nullptr; }
  template <typename T> const T *get_shared_mem(size_t = 0) const {
    return nullptr;
  }
#endif
};
} // namespace CUDA
// Generic kernel wrapper that can call any functor
#ifdef __CUDACC__
template <typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(idx_t n, Functor kernel, Args... args) {
  idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    kernel(i, args...);
  }
}
template <typename Functor, typename... Args>
__global__ void
cuda_kernel_wrapper_with_workitem(idx_t n, size_t shared_mem_size,
                                  Functor kernel, Args... args) {
  extern __shared__ char shared_mem[];

  idx_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
  idx_t local_id = threadIdx.x;
  idx_t group_id = blockIdx.x;

  CUDA::WorkItem item(global_id, local_id, group_id, shared_mem,
                      shared_mem_size);

  idx_t i = global_id;
  // All threads must execute for workitem kernels (needed for barriers and
  // shared memory)
  kernel(i, item, args...);
}
/**
 * @brief Generic CUDA kernel implementation with full template support.
 *
 * This function handles all the CUDA setup, dependency management, and cleanup
 * while delegating the actual kernel launch to launch_cuda_wrapper_impl.
 *
 * By placing this in a header file, it can be instantiated for any user-defined
 * kernel types without requiring explicit instantiations.
 */

template <typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource &resource, const KernelConfig &config,
                         Functor kernel_func, Args... args) {
  // Get queue from config
  cudaStream_t stream = static_cast<cudaStream_t>(config.get_queue(resource));

  // Handle dependencies
  for (const auto &dep_event : config.dependencies.get_cuda_events()) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, dep_event, 0));
  }
  // Ensure standardized problem/grid/block are available
  KernelConfig local_config = config;
  if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0 ||
      local_config.problem_size.z == 0) {
    kerneldim3 new_problem{};
    new_problem.x = std::max<idx_t>(1, local_config.grid_size.x *
                                           local_config.block_size.x);
    new_problem.y = std::max<idx_t>(1, local_config.grid_size.y *
                                           local_config.block_size.y);
    new_problem.z = std::max<idx_t>(1, local_config.grid_size.z *
                                           local_config.block_size.z);
    local_config.problem_size = new_problem;
  }

  // Set __device__ context
  int old_device;
  CUDA_CHECK(cudaGetDevice(&old_device));
  CUDA_CHECK(cudaSetDevice(static_cast<int>(resource.id())));

  // Launch kernel using generic wrapper
  dim3 grid(local_config.grid_size.x, local_config.grid_size.y,
            local_config.grid_size.z);
  dim3 block(local_config.block_size.x, local_config.block_size.y,
             local_config.block_size.z);
  idx_t thread_count = local_config.problem_size.x *
                       local_config.problem_size.y *
                       local_config.problem_size.z;
  cuda_kernel_wrapper<<<grid, block, local_config.shared_memory, stream>>>(
      thread_count, kernel_func, args...);

  // Check for launch errors
  CUDA_CHECK(cudaGetLastError());

  // Create completion event
  cudaEvent_t completion_event;
  CUDA_CHECK(
      cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(completion_event, stream));

  // Restore __device__ context
  CUDA_CHECK(cudaSetDevice(old_device));

  return Event(completion_event, resource);
}

/**
 * @brief CUDA kernel launcher with WorkItem support
 *
 * Use this for kernels that need shared memory and barriers.
 * The functor must have signature: void operator()(size_t i, WorkItem& item,
 * Args...)
 */
template <typename Functor, typename... Args>
Event launch_cuda_kernel_with_workitem(const Resource &resource,
                                       const KernelConfig &config,
                                       Functor kernel_func, Args... args) {
  cudaStream_t stream = static_cast<cudaStream_t>(config.get_queue(resource));

  // Handle dependencies
  for (const auto &dep_event : config.dependencies.get_cuda_events()) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, dep_event, 0));
  }

  KernelConfig local_config = config;
  if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0 ||
      local_config.problem_size.z == 0) {
    kerneldim3 new_problem{};
    new_problem.x = std::max<idx_t>(1, local_config.grid_size.x *
                                           local_config.block_size.x);
    new_problem.y = std::max<idx_t>(1, local_config.grid_size.y *
                                           local_config.block_size.y);
    new_problem.z = std::max<idx_t>(1, local_config.grid_size.z *
                                           local_config.block_size.z);
    local_config.problem_size = new_problem;
  }

  // Set device context
  int old_device;
  CUDA_CHECK(cudaGetDevice(&old_device));
  CUDA_CHECK(cudaSetDevice(static_cast<int>(resource.id())));

  // Launch kernel using workitem wrapper
  dim3 grid(local_config.grid_size.x, local_config.grid_size.y,
            local_config.grid_size.z);
  dim3 block(local_config.block_size.x, local_config.block_size.y,
             local_config.block_size.z);
  idx_t thread_count = local_config.problem_size.x *
                       local_config.problem_size.y *
                       local_config.problem_size.z;

  cuda_kernel_wrapper_with_workitem<<<grid, block, local_config.shared_memory,
                                      stream>>>(
      thread_count, local_config.shared_memory, kernel_func, args...);

  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t completion_event;
  CUDA_CHECK(
      cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(completion_event, stream));

  CUDA_CHECK(cudaSetDevice(old_device));

  return Event(completion_event, resource);
}

#else // __CUDACC__

// Non-CUDA compilation - provide stub implementation
template <typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource &resource, const KernelConfig &config,
                         Functor kernel_func, Args... args) {
  // Non-CUDA compilation unit - provide stub implementation
  throw ARBD::Exception(
      ARBD::ExceptionType::NotImplementedError, ARBD::SourceLocation(),
      "launch_cuda_kernel can only be used in CUDA compilation units");
}

template <typename Functor, typename... Args>
Event launch_cuda_kernel_with_workitem(const Resource &resource,
                                       const KernelConfig &config,
                                       Functor kernel_func, Args... args) {
  throw ARBD::Exception(ARBD::ExceptionType::NotImplementedError,
                        ARBD::SourceLocation(),
                        "launch_cuda_kernel_with_workitem can only be used in "
                        "CUDA compilation units");
}

#endif // __CUDACC__

} // namespace ARBD
