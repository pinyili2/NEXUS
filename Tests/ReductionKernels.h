#pragma once
#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"

/**
 * @brief Reduction kernel using WorkItem and shared memory
 *
 * Implements the reduction6 pattern from NVIDIA CUDA samples.
 * Uses shared memory for block-level reduction with warp-level optimization.
 *
 * Note: This kernel expects block_size elements per block, and processes
 * 2*block_size elements per block to maximize memory bandwidth.
 */
void cuda_reduction_kernel(const ARBD::Resource &device,
                           const ARBD::KernelConfig &cfg, const int *g_idata,
                           int *g_odata, unsigned int n,
                           unsigned int block_size, unsigned int grid_size);
void cuda_simple_reduction_kernel(const ARBD::Resource &device,
                                  const ARBD::KernelConfig &cfg,
                                  const int *input, int *output, unsigned int n,
                                  unsigned int block_size);
struct reduction_kernel {
  template <typename WorkItem>
  DEVICE void
  operator()(size_t i, WorkItem &item, const int *__restrict__ g_idata,
             int *__restrict__ g_odata, unsigned int n, unsigned int block_size,
             unsigned int grid_size) const {
    auto *sdata = item.template get_shared_mem<int>();
    unsigned int tid = static_cast<unsigned int>(item.local_id());

    // Load data into shared memory with multiple elements per thread
    sdata[tid] = 0;
    unsigned int idx =
        static_cast<unsigned int>(item.group_id()) * (block_size * 2) + tid;

    // Reduce multiple elements per thread (2 per thread for better bandwidth)
    while (idx < n) {
      sdata[tid] += g_idata[idx];
      if (idx + block_size < n) {
        sdata[tid] += g_idata[idx + block_size];
      }
      idx += grid_size;
    }

    item.barrier();

    unsigned int s = block_size;
    while (s > 1) {
      s >>= 1;
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      item.barrier();
    }

    // Write block sum to output
    if (tid == 0) {
      g_odata[item.group_id()] = sdata[0];
    }
  }
};

/**
 * @brief Simplified reduction kernel for testing
 * Single pass reduction using tree reduction pattern
 */
struct simple_reduction_kernel {
  template <typename WorkItem>
  DEVICE void operator()(size_t i, WorkItem &item,
                         const int *__restrict__ input,
                         int *__restrict__ output, unsigned int n,
                         unsigned int block_size) const {
    auto *sdata = item.template get_shared_mem<int>();
    unsigned int tid = static_cast<unsigned int>(item.local_id());

    // Load data (one element per thread)
    unsigned int idx = static_cast<unsigned int>(item.global_id());
    sdata[tid] = (idx < n) ? input[idx] : 0;

    item.barrier();

    // Tree reduction in shared memory
    // Use block_size parameter which should match blockDim.x
    unsigned int s = block_size;
    while (s > 1) {
      s >>= 1;
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      item.barrier();
    }

    // Write result
    if (tid == 0) {
      output[item.group_id()] = sdata[0];
    }
  }
};
