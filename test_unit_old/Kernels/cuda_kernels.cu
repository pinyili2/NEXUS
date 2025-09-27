#include "Backend/Buffer.h"
#include "Backend/CUDA/KernelHelper.cuh"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "cuda_kernels.cuh"

// The kernel function objects are now defined in cuda_kernels.h
// This file ensures they are compiled with nvcc

#ifdef USE_CUDA

namespace ARBD {

// ============================================================================
// Explicit template instantiations for test kernels
// ============================================================================

// ScaleKernel instantiation
template Event launch_cuda_kernel<std::tuple<DeviceBuffer<float>&>,
								  std::tuple<DeviceBuffer<float>&>,
								  ScaleKernel&>(const Resource& resource,
												size_t thread_count,
												const std::tuple<DeviceBuffer<float>&>& inputs,
												const std::tuple<DeviceBuffer<float>&>& outputs,
												const KernelConfig& config,
												ScaleKernel& kernel_func);

// MultiplyKernel instantiation
template Event launch_cuda_kernel<std::tuple<DeviceBuffer<float>&>,
								  std::tuple<DeviceBuffer<float>&>,
								  MultiplyKernel&,
								  float&>(const Resource& resource,
										  size_t thread_count,
										  const std::tuple<DeviceBuffer<float>&>& inputs,
										  const std::tuple<DeviceBuffer<float>&>& outputs,
										  const KernelConfig& config,
										  MultiplyKernel& kernel_func,
										  float& args);

// SquareKernel instantiation
template Event launch_cuda_kernel<std::tuple<DeviceBuffer<float>&>,
								  std::tuple<DeviceBuffer<float>&>,
								  SquareKernel&>(const Resource& resource,
												 size_t thread_count,
												 const std::tuple<DeviceBuffer<float>&>& inputs,
												 const std::tuple<DeviceBuffer<float>&>& outputs,
												 const KernelConfig& config,
												 SquareKernel& kernel_func);

} // namespace ARBD
#endif
