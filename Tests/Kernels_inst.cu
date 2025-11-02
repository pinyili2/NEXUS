#include "JacobiKernels.h"
#include "ReductionKernels.h"

// CUDA-specific kernel launch implementations
void cuda_initialize_boundaries(const ARBD::Resource &device,
                                const ARBD::KernelConfig &cfg, float *a_new,
                                float pi, int offset, int nx, int my_ny,
                                int ny) {
  ARBD::launch_kernel(device, cfg, initialize_boundaries_kernel{}, a_new, pi,
                      offset, nx, my_ny, ny);
}

void cuda_jacobi_kernel(const ARBD::Resource &device,
                        const ARBD::KernelConfig &cfg, float *a_new,
                        float *l2_norm, int iy_start, int iy_end, int nx,
                        float *a_new_top, int top_iy, float *a_new_bottom,
                        int bottom_iy) {
  ARBD::launch_kernel(device, cfg, jacobi_kernel{}, a_new, l2_norm, iy_start,
                      iy_end, nx, a_new_top, top_iy, a_new_bottom, bottom_iy);
}

void cuda_reduction_kernel(const ARBD::Resource &device,
                           const ARBD::KernelConfig &cfg, const int *g_idata,
                           int *g_odata, unsigned int n,
                           unsigned int block_size, unsigned int grid_size) {
  ARBD::launch_kernel_with_workitem(device, cfg, reduction_kernel{}, g_idata,
                                    g_odata, n, block_size, grid_size);
}
void cuda_simple_reduction_kernel(const ARBD::Resource &device,
                                  const ARBD::KernelConfig &cfg,
                                  const int *input, int *output, unsigned int n,
                                  unsigned int block_size) {
  ARBD::launch_kernel_with_workitem(device, cfg, simple_reduction_kernel{},
                                    input, output, n, block_size);
}
