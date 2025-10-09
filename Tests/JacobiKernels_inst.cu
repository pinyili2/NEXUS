#include "JacobiKernels.h"

// CUDA-specific kernel launch implementations
void launch_initialize_boundaries(const ARBD::Resource& device, const ARBD::KernelConfig& cfg,
                                   float* a_new, float* a, float pi, int offset, int nx, int my_ny, int ny) {
    ARBD::launch_kernel(device, cfg, initialize_boundaries_kernel{}, a_new, a, pi, offset, nx, my_ny, ny);
}

void launch_jacobi_kernel(const ARBD::Resource& device, const ARBD::KernelConfig& cfg,
                          float* a_new, const float* a, float* l2_norm,
                          int iy_start, int iy_end, int nx,
                          float* a_new_top, int top_iy, float* a_new_bottom, int bottom_iy) {
    ARBD::launch_kernel(device, cfg, jacobi_kernel{}, a_new, a, l2_norm, iy_start, iy_end, nx,
                        a_new_top, top_iy, a_new_bottom, bottom_iy);
}
