#include "Backend/Buffer.h"
#include "Backend/CUDA/KernelHelper.cuh"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "cuda_omp_multi_gpu_kernels.h"

#ifdef USE_CUDA

namespace ARBD {

// ============================================================================
// Explicit template instantiations for multi-GPU Jacobi solver kernels
// ============================================================================

using BufferFloat = DeviceBuffer<float>;

// Template instantiations for CUDA kernel launches
// Based on the error message: launch_cuda_kernel<ARBD::initialize_boundaries_omp_kernel, float*, float*, float, int, int>

// Template instantiations matching the actual launch_cuda_kernel signature
template Event launch_cuda_kernel<initialize_boundaries_omp_kernel, float*, float*, float, int, int>(
    const Resource& resource,
    const KernelConfig& config,
    initialize_boundaries_omp_kernel kernel_func,
    float* arg1,
    float* arg2,
    float arg3,
    int arg4,
    int arg5);

template Event launch_cuda_kernel<jacobi_omp_kernel, float*, float*, float*, int, int, bool>(
    const Resource& resource,
    const KernelConfig& config,
    jacobi_omp_kernel kernel_func,
    float* arg1,
    float* arg2,
    float* arg3,
    int arg4,
    int arg5,
    bool arg6);

} // namespace ARBD

#endif
