#pragma once

#include "../Buffer.h"
#include "../Events.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "../ARBDException.h"

#ifdef __CUDACC__
// Only include CUDA headers when compiling with nvcc
#include "CUDAManager.h"
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#endif

namespace ARBD {

// Generic kernel wrapper that can call any functor
#ifdef __CUDACC__
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(idx_t n, Functor kernel, Args... args) {
	idx_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		kernel(i, args...);
	}
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

template<typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource& resource,
						 const KernelConfig& config,
						 Functor kernel_func,
						 Args... args) {
	// Get queue from config
	cudaStream_t stream = static_cast<cudaStream_t>(config.get_queue(resource));

	// Handle dependencies
	for (const auto& dep_event : config.dependencies.get_cuda_events()) {
		CUDA_CHECK(cudaStreamWaitEvent(stream, dep_event, 0));
	}
	// Ensure standardized problem/grid/block are available
	KernelConfig local_config = config;
	if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0 ||
		local_config.problem_size.z == 0) {
		kerneldim3 new_problem{};
		new_problem.x = std::max<idx_t>(1, local_config.grid_size.x * local_config.block_size.x);
		new_problem.y = std::max<idx_t>(1, local_config.grid_size.y * local_config.block_size.y);
		new_problem.z = std::max<idx_t>(1, local_config.grid_size.z * local_config.block_size.z);
		local_config.problem_size = new_problem;
	}

	// Set device context
	int old_device;
	CUDA_CHECK(cudaGetDevice(&old_device));
	CUDA_CHECK(cudaSetDevice(static_cast<int>(resource.id())));

	// Launch kernel using generic wrapper
	dim3 grid(local_config.grid_size.x, local_config.grid_size.y, local_config.grid_size.z);
	dim3 block(local_config.block_size.x, local_config.block_size.y, local_config.block_size.z);
	idx_t thread_count =
		local_config.problem_size.x * local_config.problem_size.y * local_config.problem_size.z;
	cuda_kernel_wrapper<<<grid, block, local_config.shared_memory, stream>>>(
		thread_count,
		kernel_func,
		get_buffer_pointer(args)...);

	// Check for launch errors
	CUDA_CHECK(cudaGetLastError());

	// Create completion event
	cudaEvent_t completion_event;
	CUDA_CHECK(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
	CUDA_CHECK(cudaEventRecord(completion_event, stream));

	// Restore device context
	CUDA_CHECK(cudaSetDevice(old_device));

	return Event(completion_event, resource);
}

#else // __CUDACC__

// Non-CUDA compilation - provide stub implementation
template<typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource& resource,
						 const KernelConfig& config,
						 Functor kernel_func,
						 Args... args) {
	// Non-CUDA compilation unit - provide stub implementation
	throw ARBD::Exception(ARBD::ExceptionType::NotImplementedError, ARBD::SourceLocation(), "launch_cuda_kernel can only be used in CUDA compilation units");
}

#endif // __CUDACC__

} // namespace ARBD
