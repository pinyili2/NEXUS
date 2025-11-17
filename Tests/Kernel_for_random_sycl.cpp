#ifdef USE_SYCL

#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Kernel_for_random.h"
#include "Random/RandomKernels.h"

// This file provides SYCL implementations of the random kernel wrappers
// that are defined in Kernel_for_random.cu for CUDA

namespace ARBD {

// Explicit instantiations for kernel functors (ensures they're compiled for
// device)
template struct UniformFunctor<float>;
template struct UniformFunctor<int>;
template struct UniformFunctor<uint32_t>;
template struct GaussianFunctor<float>;

// Wrapper function DEFINITIONS for SYCL
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

template Event launch_uniform_kernel<uint32_t>(const Resource &,
                                               const KernelConfig &,
                                               const UniformFunctor<uint32_t> &,
                                               DeviceBuffer<uint32_t> &);

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

#endif // USE_SYCL
