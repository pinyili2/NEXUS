#pragma once
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "Header.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <omp.h>

constexpr int MAX_NUM_DEVICES = 32;

typedef float real;
constexpr real tol = 1.0e-8;

const real PI = 2.0 * std::asin(1.0);

struct kernel_func {
  DEVICE void operator()(size_t i, const float *input, float *output) const {
    output[i] = input[i] * 3;
  }
};
struct initialize_boundaries_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
                         float *__restrict__ const a, const float pi,
                         const int offset, const int nx, const int my_ny,
                         const int ny) const {
    // Convert linear thread index to local row index (matches original NVIDIA
    // logic)
    int iy = static_cast<int>(i);

    // Guard against extra threads launched by the backend
    if (iy >= my_ny) {
      return; // This thread has no work to do
    }

    // Calculate boundary value using global coordinate (offset + iy)
    // but store using local coordinate (iy) - matches original NVIDIA logic
    const float y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
    a[iy * nx + 0] = y0;
    a[iy * nx + (nx - 1)] = y0;
    a_new[iy * nx + 0] = y0;
    a_new[iy * nx + (nx - 1)] = y0;
  }
};

// Portable 2D Jacobi sweep functor for non-CUDA backends
struct jacobi_sweep_portable {
  DEVICE void operator()(size_t i, float *a_new, const float *a,
                         float *l2_accum, const int nx, const int ny) const {
    int total = (nx - 2) * (ny - 2);
    if (static_cast<int>(i) >= total)
      return;
    int local = static_cast<int>(i);
    int iy = 1 + local / (nx - 2);
    int ix = 1 + local % (nx - 2);

    float new_val = 0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                             a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
    a_new[iy * nx + ix] = new_val;
    float r = new_val - a[iy * nx + ix];
    ATOMIC_ADD(l2_accum, r * r);
  }
};

// Portable Jacobi kernel using linear index decomposition
struct jacobi_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
                         const float *__restrict__ const a,
                         float *__restrict__ const l2_norm, const int iy_start,
                         const int iy_end, const int nx,
                         float *__restrict__ const a_new_top, const int top_iy,
                         float *__restrict__ const a_new_bottom,
                         const int bottom_iy) const {
    // Decompose linear index i to 2D coordinates
    // Problem size is (nx-2) x (iy_end - iy_start) for the interior

    int width = nx - 2;
    int local = static_cast<int>(i);
    int local_iy = local / width;
    int local_ix = local % width;

    int iy = iy_start + local_iy;
    int ix = 1 + local_ix;

    // 1. Is the current row in our assigned chunk?
    if (iy >= iy_start && iy < iy_end && ix > 0 && ix < (nx - 1)) {
      // 2. Are we away from the left/right physical boundaries?

      // If the checks pass, it is now safe to perform the stencil read.
      const float new_val =
          0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                  a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);

      a_new[iy * nx + ix] = new_val;

      // This logic for updating neighbor halos remains the same.
      if (iy_start == iy) {
        a_new_top[top_iy * nx + ix] = new_val;
      }
      if ((iy_end - 1) == iy) {
        a_new_bottom[bottom_iy * nx + ix] = new_val;
      }

      // L2 norm calculation is safe here.
      float residue = new_val - a[iy * nx + ix];
      float local_l2_norm = residue * residue;
      ATOMIC_ADD(l2_norm, local_l2_norm);
    }
  }
};

double single_gpu(const int nx, const int ny, const int iter_max,
                  real *const a_ref_h, const int nccheck, const bool print);

template <typename T>
T get_argval(char **begin, char **end, const std::string &arg,
             const T default_val) {
  T argval = default_val;
  char **itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool inline get_arg(char **begin, char **end, const std::string &arg) {
  char **itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

struct l2_norm_buf {
  ARBD::Event copy_done;
  ARBD::DeviceBuffer<float> d_buf;
  std::vector<float> h_buf;
};

// ----------------------------------------------------------------------------
// Backend-agnostic kernel profiling test helper
// - Uses existing kernel_functor pattern and NEXUS launch API
// - Records ARBD::Event from launch and profiles with ARBD::Profiling
// - Works for CUDA or SYCL backends (single-backend assumption)
// ----------------------------------------------------------------------------
// Function declarations for CUDA kernel launches
void launch_initialize_boundaries(const ARBD::Resource &device,
                                  const ARBD::KernelConfig &cfg, float *a_new,
                                  float *a, float pi, int offset, int nx,
                                  int my_ny, int ny);

void launch_jacobi_kernel(const ARBD::Resource &device,
                          const ARBD::KernelConfig &cfg, float *a_new,
                          const float *a, float *l2_norm, int iy_start,
                          int iy_end, int nx, float *a_new_top, int top_iy,
                          float *a_new_bottom, int bottom_iy);

inline void run_backend_kernel_profile_test(size_t n = 1024) {
  using namespace ARBD;

  // Initialize profiling
  Profiling::ProfilingConfig profile_cfg;
  profile_cfg.enable_timing = true;
  profile_cfg.enable_backend_markers = true;
  Profiling::ProfileManager::init(profile_cfg);

  // Select resource for the active backend
#ifdef USE_CUDA
  Resource resource = Resource::create_cuda_device(0);
#elif defined(USE_SYCL)
  Resource resource(ResourceType::SYCL, 0);
#else
  Resource resource(ResourceType::CPU, 0);
#endif

  // Allocate device buffers on the selected device
  DeviceBuffer<float> input_buf(n, resource.id());
  DeviceBuffer<float> output_buf(n, resource.id());

  // Initialize input on host and copy to device
  std::vector<float> host_in(n, 5.0f);
  input_buf.copy_from_host(host_in);

  // Configure kernel launch for 1D problem
  KernelConfig config = KernelConfig::for_1d(static_cast<idx_t>(n), resource);
  config.sync = true;

  // Start profiling range
  Profiling::ProfileManager::start_range("triple_kernel_profile",
                                         resource.type());

  // Launch kernel functor and capture event
  auto event = launch_kernel(resource, config, kernel_func{},
                             get_buffer_pointer(input_buf),
                             get_buffer_pointer(output_buf));

  // Wait for completion via Event
  event.wait();

  // End profiling range
  Profiling::ProfileManager::end_range("triple_kernel_profile",
                                       resource.type());

  // Validate results
  std::vector<float> host_out(n, 0.0f);
  output_buf.copy_to_host(host_out);

#pragma omp parallel for if (n > 1024)
  for (long i = 0; i < static_cast<long>(n); ++i) {
    // Expect output = input * 3.0f
    if (std::fabs(host_out[static_cast<size_t>(i)] - 15.0f) > 1e-5f) {
      // Minimal guard; in unit tests this would be REQUIRE/ASSERT
      fprintf(stderr, "Validation failed at %ld: got %f expected %f\n", i,
              host_out[static_cast<size_t>(i)], 15.0f);
    }
  }

  // Finalize profiling (writes out data if configured)
  Profiling::ProfileManager::finalize();
}
