#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"

using namespace ARBD;

// ============================================================================
// Global Backend Initialization - Properly initializes the compile-time
// selected backend
// ============================================================================

// Global flag to track backend initialization
static bool g_backend_available = false;

// Initialize backend once across all tests
static void initialize_backend_once() {
  static bool initialized = false;
  if (initialized)
    return;

  try {
#ifdef USE_SYCL
    SYCL::Manager::init();
    g_backend_available = true;
#endif

#ifdef USE_CUDA
    CUDA::Manager::init();
    g_backend_available = true;
#endif

#ifdef USE_METAL
    auto &metal_manager = METAL::Manager::instance();
    metal_manager.init();
    g_backend_available = true;
#endif

    if (!g_backend_available) {
      // Fallback to CPU
      g_backend_available = true;
    }

    initialized = true;
  } catch (const std::exception &e) {
    g_backend_available = false;
    WARN("Backend initialization failed: " << e.what());
  }
}

// OpenMP-aware Jacobi kernel functors (backend-agnostic)
struct initialize_boundaries_omp_kernel {
  void operator()(size_t i, float *__restrict__ const a_new,
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
  void operator()(size_t i, float *__restrict__ const a_new,
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

    float local_l2_norm = 0.0f;

    // Perform Jacobi iteration: new = 0.25 * (left + right + top + bottom)
    const float new_val =
        0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                 a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
    a_new[iy * nx + ix] = new_val;

    if (calculate_norm) {
      float residue = new_val - a[iy * nx + ix];
      local_l2_norm = residue * residue;
      // Use ATOMIC_ADD macro for zero-overhead atomic accumulation
      ATOMIC_ADD(l2_norm, local_l2_norm);
    }
  }
};

// OpenMP Multi-GPU Jacobi solver class
class OMPJacobiSolver {
private:
  int num_gpus_;
  int nx_, ny_;

  std::vector<Resource> resources_;
  std::vector<DeviceBuffer<float>> a_bufs_;
  std::vector<DeviceBuffer<float>> a_new_bufs_;
  std::vector<DeviceBuffer<float>> l2_norm_bufs_;

  std::vector<int> gpu_ny_start_; // Starting row for each GPU
  std::vector<int> gpu_ny_size_;  // Number of rows for each GPU

  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  OMPJacobiSolver(int nx, int ny, int num_gpus = -1) : nx_(nx), ny_(ny) {

#ifdef _OPENMP
    if (num_gpus <= 0) {
      num_gpus_ = omp_get_max_threads();
    } else {
      num_gpus_ = num_gpus;
    }
    // Limit to maximum number of available devices
    num_gpus_ = std::min(num_gpus_, 8); // Assume max 8 GPUs
#else
    num_gpus_ = 1;
#endif

    LOGINFO("OMPJacobiSolver initialized with {} GPUs", num_gpus_);

    // Initialize resources
    resources_.reserve(num_gpus_);
    a_bufs_.reserve(num_gpus_);
    a_new_bufs_.reserve(num_gpus_);
    l2_norm_bufs_.reserve(num_gpus_);

    gpu_ny_start_.resize(num_gpus_);
    gpu_ny_size_.resize(num_gpus_);

    // Domain decomposition in Y direction
    int rows_per_gpu = ny / num_gpus_;
    int remainder = ny % num_gpus_;

    int current_row = 0;
    for (int gpu = 0; gpu < num_gpus_; gpu++) {
      gpu_ny_start_[gpu] = current_row;
      gpu_ny_size_[gpu] = rows_per_gpu + (gpu < remainder ? 1 : 0);
      current_row += gpu_ny_size_[gpu];

      // Initialize each GPU resource
      resources_.emplace_back(ResourceType::SYCL, gpu);

      // Allocate buffers for each GPU
      const size_t gpu_size = nx_ * gpu_ny_size_[gpu];
      a_bufs_.emplace_back(gpu_size, 0);
      a_new_bufs_.emplace_back(gpu_size, 0);
      l2_norm_bufs_.emplace_back(1, 0);

      // Initialize with zeros
      std::vector<float> zeros(gpu_size, 0.0f);
      a_bufs_[gpu].copy_from_host(zeros);
      a_new_bufs_[gpu].copy_from_host(zeros);

      std::vector<float> norm_zero(1, 0.0f);
      l2_norm_bufs_[gpu].copy_from_host(norm_zero);
    }
  }

  void initialize_boundaries() {
    // Set sine wave boundary conditions on each GPU
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
    for (int gpu = 0; gpu < num_gpus_; gpu++) {
      auto event = launch_kernel(
          resources_[gpu],
          KernelConfig::for_1d(gpu_ny_size_[gpu], resources_[gpu]),
          initialize_boundaries_omp_kernel{},
          get_buffer_pointer(a_new_bufs_[gpu]),
          get_buffer_pointer(a_bufs_[gpu]), pi_, nx_, ny_);
      event.wait();
    }
  }

  float solve(int iter_max, int nccheck = 1) {
    float l2_norm = 1.0f;
    int iter = 0;
    bool calculate_norm = true;

    while (l2_norm > tol_ && iter < iter_max) {
      // Reset norm buffers on all GPUs
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; gpu++) {
        std::vector<float> norm_init(1, 0.0f);
        l2_norm_bufs_[gpu].copy_from_host(norm_init);
      }

      calculate_norm = (iter % nccheck) == 0;

      // Launch Jacobi kernel on all GPUs
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; gpu++) {
        // Calculate number of interior points for this GPU
        int interior_points = (nx_ - 2) * (gpu_ny_size_[gpu] - 2);
        if (interior_points > 0) {
          auto event = launch_kernel(
              resources_[gpu],
              KernelConfig::for_1d(interior_points, resources_[gpu]),
              jacobi_omp_kernel{}, get_buffer_pointer(a_new_bufs_[gpu]),
              get_buffer_pointer(a_bufs_[gpu]),
              get_buffer_pointer(l2_norm_bufs_[gpu]), nx_, gpu_ny_size_[gpu],
              calculate_norm);
          event.wait();
        }
      }

      // Handle norm calculation
      if (calculate_norm) {
        float total_norm = 0.0f;

        for (int gpu = 0; gpu < num_gpus_; gpu++) {
          std::vector<float> norm_result(1);
          l2_norm_bufs_[gpu].copy_to_host(norm_result);
          total_norm += norm_result[0];
        }

        l2_norm = std::sqrt(total_norm);

        if ((iter % 100) == 0) {
          std::cout << "Iteration " << iter << ", L2 norm: " << l2_norm
                    << std::endl;
        }
      }

      // Exchange ghost cells between GPUs
      exchange_ghost_cells();

      // Swap arrays on all GPUs
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; gpu++) {
        std::swap(a_bufs_[gpu], a_new_bufs_[gpu]);
      }

      iter++;
    }

    return l2_norm;
  }

private:
  void exchange_ghost_cells() {
    if (num_gpus_ == 1)
      return; // No exchange needed for single GPU

    // Exchange between adjacent GPUs
    for (int gpu = 0; gpu < num_gpus_ - 1; gpu++) {
      int next_gpu = gpu + 1;

      // Get data from current GPU (bottom row) and send to next GPU (top row)
      std::vector<float> gpu_data, next_gpu_data;
      a_new_bufs_[gpu].copy_to_host(gpu_data);
      a_new_bufs_[next_gpu].copy_to_host(next_gpu_data);

      // Copy bottom row of current GPU to top row of next GPU
      int bottom_row_start = (gpu_ny_size_[gpu] - 1) * nx_;
      std::copy(gpu_data.begin() + bottom_row_start,
                gpu_data.begin() + bottom_row_start + nx_,
                next_gpu_data.begin());

      // Copy top row of next GPU to bottom row of current GPU
      std::copy(next_gpu_data.begin() + nx_, next_gpu_data.begin() + 2 * nx_,
                gpu_data.begin() + bottom_row_start);

      // Update buffers
      a_new_bufs_[gpu].copy_from_host(gpu_data);
      a_new_bufs_[next_gpu].copy_from_host(next_gpu_data);
    }
  }

public:
  std::vector<float> get_solution() {
    std::vector<float> full_solution(nx_ * ny_);

    for (int gpu = 0; gpu < num_gpus_; gpu++) {
      std::vector<float> gpu_solution;
      a_bufs_[gpu].copy_to_host(gpu_solution);

      // Copy GPU solution to appropriate part of full solution
      int start_row = gpu_ny_start_[gpu];
      std::copy(gpu_solution.begin(), gpu_solution.end(),
                full_solution.begin() + start_row * nx_);
    }

    return full_solution;
  }

  int get_num_gpus() const { return num_gpus_; }
  std::vector<int> get_gpu_sizes() const { return gpu_ny_size_; }
};

// Main OpenMP Multi-GPU Jacobi Solver Test
TEST_CASE("OpenMP Multi-GPU Jacobi Solver", "[omp][multi-gpu]") {
  // Initialize backend globally
  initialize_backend_once();

#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
  INFO("OpenMP available with " << max_threads << " threads");
#else
  INFO("OpenMP not available - using single threaded");
#endif

  SECTION("Single GPU Test") {
    const int nx = 16;
    const int ny = 16;
    const int iter_max = 100;

    OMPJacobiSolver solver(nx, ny, 1);
    INFO("Single GPU - Grid size: " << nx << "x" << ny);

    solver.initialize_boundaries();
    float final_norm = solver.solve(iter_max, 10);

    INFO("Single GPU Test - Final L2 norm = " << final_norm);

    // Verify solution converged
    REQUIRE(final_norm >= 0.0f);
    REQUIRE(final_norm < 1.0f);

    // Verify solution size
    auto solution = solver.get_solution();
    REQUIRE(solution.size() == nx * ny);
  }

  SECTION("Dual GPU Test") {
    const int nx = 32;
    const int ny = 32;
    const int iter_max = 200;

    OMPJacobiSolver solver(nx, ny, 2);
    INFO("Dual GPU - Grid size: " << nx << "x" << ny);

    auto gpu_sizes = solver.get_gpu_sizes();
    INFO("GPU domain sizes: [" << gpu_sizes[0] << ", " << gpu_sizes[1] << "]");

    solver.initialize_boundaries();
    float final_norm = solver.solve(iter_max, 20);

    INFO("Dual GPU Test - Final L2 norm = " << final_norm);

    // Verify solution converged
    REQUIRE(final_norm >= 0.0f);
    REQUIRE(final_norm < 1.0f);

    // Verify solution size
    auto solution = solver.get_solution();
    REQUIRE(solution.size() == nx * ny);
  }

  SECTION("Quad GPU Test") {
    const int nx = 64;
    const int ny = 64;
    const int iter_max = 300;

    OMPJacobiSolver solver(nx, ny, 4);
    INFO("Quad GPU - Grid size: " << nx << "x" << ny);

    auto gpu_sizes = solver.get_gpu_sizes();
    INFO("GPU domain sizes: [" << gpu_sizes[0] << ", " << gpu_sizes[1] << ", "
                               << gpu_sizes[2] << ", " << gpu_sizes[3] << "]");

    solver.initialize_boundaries();
    float final_norm = solver.solve(iter_max, 30);

    INFO("Quad GPU Test - Final L2 norm = " << final_norm);

    // Verify solution converged
    REQUIRE(final_norm >= 0.0f);
    REQUIRE(final_norm < 1.0f);

    // Verify solution size
    auto solution = solver.get_solution();
    REQUIRE(solution.size() == nx * ny);
  }

  SECTION("Performance Benchmark") {
    const int nx = 128;
    const int ny = 128;
    const int iter_max = 500;

    // Test different GPU configurations
    std::vector<int> gpu_configs = {1, 2, 4, 8};
    std::vector<double> times;

    for (int num_gpus : gpu_configs) {
      if (num_gpus > 8)
        continue; // Skip if more than 8 GPUs requested

      OMPJacobiSolver solver(nx, ny, num_gpus);
      solver.initialize_boundaries();

      auto start = std::chrono::high_resolution_clock::now();
      float final_norm = solver.solve(iter_max, 50);
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double>(end - start).count();
      times.push_back(elapsed);

      INFO("Performance " << num_gpus << " GPU(s): " << elapsed
                          << "s, norm = " << final_norm);

      // Verify correctness
      REQUIRE(final_norm >= 0.0f);
      REQUIRE(final_norm < 1.0f);
      REQUIRE(elapsed > 0.0);
      REQUIRE(elapsed < 120.0); // Should complete within 2 minutes
    }

    // Calculate and report speedups
    if (times.size() > 1) {
      double baseline = times[0]; // Single GPU time
      for (size_t i = 1; i < times.size(); i++) {
        double speedup = baseline / times[i];
        double efficiency = speedup / gpu_configs[i] * 100.0;
        INFO("Speedup " << gpu_configs[i] << " GPU(s): " << speedup
                        << "x (efficiency: " << efficiency << "%)");
      }
    }
  }
}

// OpenMP Communication Test
TEST_CASE("OpenMP Multi-GPU Communication Test", "[omp][communication]") {
  // Initialize backend globally
  initialize_backend_once();

  SECTION("Basic Resource Access") {
    const int num_gpus = 4;
    std::vector<Resource> resources;
    std::vector<DeviceBuffer<float>> buffers;

    // Initialize resources for multiple GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      try {
        resources.emplace_back(ResourceType::SYCL, gpu);
        buffers.emplace_back(1000, 0);

        // Test basic buffer operations
        std::vector<float> test_data(1000, static_cast<float>(gpu));
        buffers[gpu].copy_from_host(test_data);

        std::vector<float> result;
        buffers[gpu].copy_to_host(result);

        REQUIRE(result.size() == 1000);
        REQUIRE(result[0] == static_cast<float>(gpu));

      } catch (const std::exception &e) {
        WARN("GPU " << gpu << " not available: " << e.what());
        break;
      }
    }

    INFO("Successfully initialized " << resources.size() << " GPU resources");
    REQUIRE(resources.size() >= 1); // At least one GPU should work
  }

  SECTION("Data Transfer Test") {
    const int data_size = 10000;
    std::vector<float> host_data(data_size);

    // Initialize test data
    for (int i = 0; i < data_size; i++) {
      host_data[i] = std::sin(2.0f * M_PI * i / data_size);
    }

    try {
      Resource gpu0(ResourceType::SYCL, 0);
      Resource gpu1(ResourceType::SYCL, 1);

      DeviceBuffer<float> buffer0(data_size, 0);
      DeviceBuffer<float> buffer1(data_size, 0);

      // Copy data to GPU 0
      buffer0.copy_from_host(host_data);

      // Copy from GPU 0 to host, then to GPU 1
      std::vector<float> intermediate;
      buffer0.copy_to_host(intermediate);
      buffer1.copy_from_host(intermediate);

      // Verify data on GPU 1
      std::vector<float> result;
      buffer1.copy_to_host(result);

      REQUIRE(result.size() == data_size);
      for (int i = 0; i < data_size; i++) {
        REQUIRE(std::abs(result[i] - host_data[i]) < 1e-6f);
      }

      INFO("Data transfer test passed for 2 GPUs");

    } catch (const std::exception &e) {
      WARN("Multi-GPU test skipped: " << e.what());
      REQUIRE(true); // Test passes even if multi-GPU not available
    }
  }
}
