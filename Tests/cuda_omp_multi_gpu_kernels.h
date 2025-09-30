#pragma once

#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"

namespace ARBD {

// Multi-GPU Jacobi solver kernels - DEVICE only for CUDA atomic operations
struct initialize_boundaries_omp_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
                  float *__restrict__ const a, const float pi, const int nx,
                  const int ny) const {
    // Convert linear thread index to actual row index
    int iy = static_cast<int>(i);

    // Guard against extra threads launched by the backend
    if (iy >= ny) {
      return; // This thread has no work to do
    }

    const float y0 = std::sin(2.0f * pi * iy / (ny - 1));
    a[iy * nx + 0] = y0;
    a[iy * nx + (nx - 1)] = y0;
    a_new[iy * nx + 0] = y0;
    a_new[iy * nx + (nx - 1)] = y0;
  }
};

struct jacobi_omp_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
                  const float *__restrict__ const a,
                  float *__restrict__ const l2_norm, const int nx, const int ny,
                  const bool calculate_norm) const {
    // Convert linear thread index to 2D coordinates
    int total_width = nx - 2; // Interior points only
    int total_height = ny - 2;
    int thread_idx = static_cast<int>(i);

    // Guard against extra threads launched by the backend
    int total_interior_points = total_width * total_height;
    if (thread_idx >= total_interior_points) {
      return; // This thread has no work to do
    }

    int iy = thread_idx / total_width + 1; // Start from iy=1
    int ix = thread_idx % total_width + 1; // Start from ix=1

    // Additional bounds checking
    if (iy >= (ny - 1) || ix >= (nx - 1)) {
      return;
    }

    // Perform Jacobi iteration: new = 0.25 * (left + right + top + bottom)
    const float new_val =
        0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                 a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
    a_new[iy * nx + ix] = new_val;

    if (calculate_norm) {
      float residue = new_val - a[iy * nx + ix];
      float local_l2_norm = residue * residue;
      ATOMIC_ADD(l2_norm, local_l2_norm);
    }
  }
};

// ============================================================================
// Template declarations for CUDA kernel instantiations
// These tell the compiler that these templates are instantiated elsewhere
// ============================================================================

#ifdef USE_CUDA
// Forward declare the template instantiations that are defined in the .cu file
extern template Event launch_cuda_kernel<initialize_boundaries_omp_kernel, float*, float*, float, int, int>(
    const Resource& resource,
    const KernelConfig& config,
    initialize_boundaries_omp_kernel kernel_func,
    float* arg1,
    float* arg2,
    float arg3,
    int arg4,
    int arg5);

extern template Event launch_cuda_kernel<jacobi_omp_kernel, float*, float*, float*, int, int, bool>(
    const Resource& resource,
    const KernelConfig& config,
    jacobi_omp_kernel kernel_func,
    float* arg1,
    float* arg2,
    float* arg3,
    int arg4,
    int arg5,
    bool arg6);
#endif

} // namespace ARBD
