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

struct initialize_boundaries_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
                         const float pi, const int offset, const int nx,
                         const int my_ny, const int ny) const {
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
    a_new[iy * nx + 0] = y0;
    a_new[iy * nx + (nx - 1)] = y0;
  }
};

// Portable Jacobi kernel using linear index decomposition - IN-PLACE VERSION
struct jacobi_kernel {
  DEVICE void
  operator()(size_t i,
             float *__restrict__ const a, // Single buffer for in-place updates
             float *__restrict__ const l2_norm, const int iy_start,
             const int iy_end, const int nx, float *__restrict__ const a_top,
             const int top_iy, float *__restrict__ const a_bottom,
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

      // READ OLD VALUE BEFORE OVERWRITING (for residue calculation)
      const float old_val = a[iy * nx + ix];

      // Compute new value using neighbors from the same buffer
      const float new_val =
          0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                  a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);

      // Write new value back to same buffer (in-place update)
      a[iy * nx + ix] = new_val;

      // Update neighbor halos (still needed for MPI communication)
      if (iy_start == iy) {
        a_top[top_iy * nx + ix] = new_val;
      }
      if ((iy_end - 1) == iy) {
        a_bottom[bottom_iy * nx + ix] = new_val;
      }

      // L2 norm calculation using stored old_val
      float residue = new_val - old_val;
      float local_l2_norm = residue * residue;
      atomic_add(l2_norm, local_l2_norm);
    }
  }
};
