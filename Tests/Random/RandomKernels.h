#pragma once
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"
#include "philox.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <thrust/tuple.h>
#elif defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
#elif defined(__SYCL_DEVICE_ONLY__)
#include <sycl/sycl.hpp>
#else
#include <cmath>
#endif

// --- Functor for Uniform Float Generation ---
namespace ARBD {
HOST DEVICE inline float int2float(uint32_t i) {

  constexpr float factor = 1.0f / (4294967295.0f + 1.0f); // 1.0f / 2^32
  constexpr float halffactor = 0.5f * factor;
  return static_cast<float>(i) * factor + halffactor;
}

template <typename T> struct UniformFunctor {
  T min_val;
  T max_val;
  uint64_t base_seed;
  uint32_t base_ctr;
  uint32_t global_seed;

  HOST DEVICE void operator()(size_t i, T *output) const {
    // Create a fresh Philox instance with deterministic parameters
    // Each thread gets a unique counter value based on its index
    openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i),
                         global_seed);

    uint32_t random_int = rng.draw();

    float random_float_01 = int2float(random_int);
    output[i] = min_val + random_float_01 * (max_val - min_val);
  }
};

template <typename T> struct GaussianFunctor {
  T mean;
  T stddev;
  size_t output_size;
  uint64_t base_seed;
  uint32_t base_ctr;
  uint32_t global_seed;

  HOST DEVICE void operator()(size_t i, T *output) const {
    if (i >= output_size)
      return;

    // Create a fresh Philox instance with deterministic parameters
    openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i),
                         global_seed);
    uint32_t i1 = rng.draw();
    uint32_t i2 = rng.draw();

    float u1 = (int2float(i1) < 1e-7f) ? 1e-7f : int2float(i1);
    float u2 = (int2float(i2) < 1e-7f) ? 1e-7f : int2float(i2);

    // Box-Muller transform - generate one value per thread
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.1415926535f * u2;
    float gaussian_val = r * cosf(theta);

    output[i] = mean + stddev * gaussian_val;
  }
};

// Forward declarations for wrapper functions (implementations in .cu file)
template <typename T>
Event launch_uniform_kernel(const Resource &resource,
                            const KernelConfig &config,
                            const UniformFunctor<T> &func,
                            DeviceBuffer<T> &output);

template <typename T>
Event launch_gaussian_kernel(const Resource &resource,
                             const KernelConfig &config,
                             const GaussianFunctor<T> &func,
                             DeviceBuffer<T> &output);

} // namespace ARBD
