
#include "Backend/Buffer.h"
#include "Backend/CUDA/KernelHelper.cuh"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Kernel_for_random.h"
#include "Random/RandomKernels.h"

// This file is compiled with nvcc to instantiate CUDA kernels.
// All wrapper function DEFINITIONS are here so they have access to
// launch_cuda_kernel.

namespace ARBD {

// Explicit instantiations for kernel functors (ensures they're compiled for
// device)
template struct UniformFunctor<float>;
template struct UniformFunctor<int>;
template struct GaussianFunctor<float>;

// Wrapper function DEFINITIONS (not inline, only in this .cu file)
template <typename T>
Event launch_uniform_kernel(const Resource &resource,
                            const KernelConfig &config,
                            const UniformFunctor<T> &func,
                            DeviceBuffer<T> &output) {
  return launch_kernel(resource, config, func, output);
}

template <typename T>
Event launch_gaussian_kernel(const Resource &resource,
                             const KernelConfig &config,
                             const GaussianFunctor<T> &func,
                             DeviceBuffer<T> &output) {
  return launch_kernel(resource, config, func, output);
}

template <typename T>
Event launch_transform_kernel(const Resource &resource,
                              const KernelConfig &config,
                              const TransformKernel &func,
                              DeviceBuffer<T> &input, DeviceBuffer<T> &output) {
  return launch_kernel(resource, config, func, input, output);
}

template <typename T>
Event launch_combine_kernel(const Resource &resource,
                            const KernelConfig &config,
                            const CombineKernel &func, DeviceBuffer<T> &uniform,
                            DeviceBuffer<T> &gaussian,
                            DeviceBuffer<T> &combined) {
  return launch_kernel(resource, config, func, uniform, gaussian, combined);
}

// Explicit instantiations
template Event launch_uniform_kernel<float>(const Resource &,
                                            const KernelConfig &,
                                            const UniformFunctor<float> &,
                                            DeviceBuffer<float> &);

template Event launch_uniform_kernel<int>(const Resource &,
                                          const KernelConfig &,
                                          const UniformFunctor<int> &,
                                          DeviceBuffer<int> &);

template Event launch_gaussian_kernel<float>(const Resource &,
                                             const KernelConfig &,
                                             const GaussianFunctor<float> &,
                                             DeviceBuffer<float> &);

template Event launch_transform_kernel<float>(const Resource &,
                                              const KernelConfig &,
                                              const TransformKernel &,
                                              DeviceBuffer<float> &,
                                              DeviceBuffer<float> &);

template Event
launch_combine_kernel<float>(const Resource &, const KernelConfig &,
                             const CombineKernel &, DeviceBuffer<float> &,
                             DeviceBuffer<float> &, DeviceBuffer<float> &);

} // namespace ARBD
