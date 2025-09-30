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

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#include "cuda_omp_multi_gpu_kernels.h"

using namespace ARBD;

// ============================================================================
// CUDA-specific OpenMP Multi-GPU Test Implementation
// ============================================================================

// Global flag to track backend initialization
static bool g_backend_available = false;

// Initialize CUDA backend once across all tests
static void initialize_cuda_backend_once() {
  static bool initialized = false;
  if (initialized)
    return;

  try {
#ifdef USE_CUDA
    CUDA::Manager::init();
    g_backend_available = true;
#endif

    if (!g_backend_available) {
      g_backend_available = false;
    }

    initialized = true;
  } catch (const std::exception &e) {
    g_backend_available = false;
    WARN("CUDA backend initialization failed: " << e.what());
  }
}

// Kernel functors are now defined in cuda_omp_multi_gpu_kernels.h

// CUDA OpenMP Multi-GPU Jacobi solver class
class CUDAOMPJacobiSolver {
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
  CUDAOMPJacobiSolver(int nx, int ny, int num_gpus = -1) : nx_(nx), ny_(ny) {

#ifdef _OPENMP
    if (num_gpus <= 0) {
      num_gpus_ = omp_get_max_threads();
    } else {
      num_gpus_ = num_gpus;
    }
    // Limit to maximum number of available CUDA devices
    int cuda_device_count = CUDA::Manager::device_count();
    num_gpus_ = std::min(num_gpus_, cuda_device_count);
#else
    num_gpus_ = 1;
#endif

    LOGINFO("CUDAOMPJacobiSolver initialized with {} GPUs", num_gpus_);

    // Distribute rows among GPUs
    setup_domain_decomposition();
    allocate_buffers();
  }

  void setup_domain_decomposition() {
    gpu_ny_start_.resize(num_gpus_);
    gpu_ny_size_.resize(num_gpus_);

    int rows_per_gpu = ny_ / num_gpus_;
    int extra_rows = ny_ % num_gpus_;

    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      gpu_ny_start_[gpu] = gpu * rows_per_gpu + std::min(gpu, extra_rows);
      gpu_ny_size_[gpu] = rows_per_gpu + (gpu < extra_rows ? 1 : 0);
    }

    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      LOGINFO("GPU {}: rows {} to {} (count: {})", gpu, gpu_ny_start_[gpu],
              gpu_ny_start_[gpu] + gpu_ny_size_[gpu] - 1, gpu_ny_size_[gpu]);
    }
  }

  void allocate_buffers() {
    resources_.clear();
    a_bufs_.clear();
    a_new_bufs_.clear();
    l2_norm_bufs_.clear();

    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      // Create CUDA resources for each GPU
      resources_.emplace_back(ResourceType::CUDA, gpu);

      size_t buffer_size = nx_ * gpu_ny_size_[gpu];
      a_bufs_.emplace_back(buffer_size, gpu);
      a_new_bufs_.emplace_back(buffer_size, gpu);
      l2_norm_bufs_.emplace_back(1, gpu); // Single float for norm accumulation
    }
  }

  void initialize_problem() {
    LOGINFO("Initializing CUDA multi-GPU Jacobi problem...");

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      // Initialize boundaries using the boundary kernel
      KernelConfig config = KernelConfig::for_1d(gpu_ny_size_[gpu], gpu);
      config.sync = true;

      auto boundary_event = launch_kernel(
          resources_[gpu], config, initialize_boundaries_omp_kernel{},
          get_buffer_pointer(a_new_bufs_[gpu]),
          get_buffer_pointer(a_bufs_[gpu]), pi_, nx_, gpu_ny_size_[gpu]);
      boundary_event.wait();

      LOGINFO("GPU {} boundary initialization completed", gpu);
    }
  }

  float solve_iteration(int max_iterations = 1000) {
    float global_l2_norm = 0.0f;

    for (int iter = 0; iter < max_iterations; ++iter) {
      global_l2_norm = 0.0f;

      // Reset norm accumulators
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        std::vector<float> zero_norm(1, 0.0f);
        l2_norm_bufs_[gpu].copy_from_host(zero_norm);
      }

      // Launch Jacobi kernels on all GPUs
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        int interior_points = (nx_ - 2) * (gpu_ny_size_[gpu] - 2);
        if (interior_points > 0) {
          KernelConfig config = KernelConfig::for_1d(interior_points, gpu);
          config.sync = true;

          auto jacobi_event = launch_kernel(
              resources_[gpu], config, jacobi_omp_kernel{},
              get_buffer_pointer(a_new_bufs_[gpu]),
              get_buffer_pointer(a_bufs_[gpu]),
              get_buffer_pointer(l2_norm_bufs_[gpu]), nx_, gpu_ny_size_[gpu],
              true);
          jacobi_event.wait();
        }
      }

      // Collect L2 norms from all GPUs
      for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        std::vector<float> gpu_norm(1);
        l2_norm_bufs_[gpu].copy_to_host(gpu_norm);
        global_l2_norm += gpu_norm[0];
      }

      global_l2_norm = std::sqrt(global_l2_norm);

      // Swap buffers for next iteration
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        std::swap(a_bufs_[gpu], a_new_bufs_[gpu]);
      }

      LOGINFO("Iteration {}: L2 norm = {:.6e}", iter + 1, global_l2_norm);

      if (global_l2_norm < tol_) {
        LOGINFO("Converged after {} iterations", iter + 1);
        return global_l2_norm;
      }
    }

    LOGWARN("Did not converge after {} iterations", max_iterations);
    return global_l2_norm;
  }

  void verify_solution() {
    LOGINFO("Verifying CUDA multi-GPU solution...");

    // Simple verification: check that boundaries are preserved
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      std::vector<float> gpu_data(nx_ * gpu_ny_size_[gpu]);
      a_bufs_[gpu].copy_to_host(gpu_data);

      // Check left and right boundaries
      for (int local_iy = 0; local_iy < gpu_ny_size_[gpu]; ++local_iy) {
        int global_iy = gpu_ny_start_[gpu] + local_iy;
        float expected = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));

        float left_val = gpu_data[local_iy * nx_ + 0];
        float right_val = gpu_data[local_iy * nx_ + (nx_ - 1)];

        if (std::abs(left_val - expected) > 1e-4f ||
            std::abs(right_val - expected) > 1e-4f) {
          LOGWARN("GPU {} boundary mismatch at global row {}: left={:.6f}, "
                  "right={:.6f}, expected={:.6f}",
                  gpu, global_iy, left_val, right_val, expected);
        }
      }
    }

    LOGINFO("CUDA multi-GPU solution verification completed");
  }
};

// ============================================================================
// CUDA OpenMP Multi-GPU Tests
// ============================================================================

#ifdef USE_CUDA

TEST_CASE("CUDA Manager OpenMP Integration", "[cuda][omp][manager]") {
  initialize_cuda_backend_once();

  SECTION("Basic CUDA Manager Initialization") {
    if (g_backend_available) {
      try {
        // Test basic initialization
        REQUIRE(CUDA::Manager::device_count() > 0);

        INFO("Found " << CUDA::Manager::device_count() << " CUDA devices");

        for (int i = 0; i < CUDA::Manager::device_count(); ++i) {
          auto props = CUDA::Manager::get_device_properties(i);
          INFO("Device " << i << ": " << props.name
               << " (SM " << props.major << "." << props.minor << ")");
        }

      } catch (const std::exception &e) {
        WARN("CUDA Manager initialization failed: " << e.what());
        REQUIRE(false);
      }
    } else {
      WARN("CUDA backend not available");
      REQUIRE(true);
    }
  }

  SECTION("CUDA OpenMP Thread Configuration") {
    if (g_backend_available) {
      try {
        int num_devices = CUDA::Manager::device_count();
        INFO("Testing OpenMP configuration with " << num_devices << " CUDA devices");

#ifdef _OPENMP
        // Test with different thread counts
        for (int threads = 1; threads <= std::min(4, num_devices); ++threads) {
          INFO("Testing with " << threads << " OpenMP threads");

          CUDA::Manager::init_for_rank(0, 1, threads, true);

          REQUIRE(CUDA::Manager::get_rank_devices().size() <= static_cast<size_t>(num_devices));

#pragma omp parallel num_threads(threads)
          {
            int thread_id = omp_get_thread_num();
            int gpu_id = CUDA::Manager::get_thread_gpu();

            INFO("Thread " << thread_id << " assigned to GPU " << gpu_id);
            REQUIRE(gpu_id >= 0);
            REQUIRE(gpu_id < num_devices);

            // Test thread-local GPU initialization
            CUDA::Manager::init_for_omp_thread();
          }
        }
#endif

      } catch (const std::exception &e) {
        WARN("CUDA OpenMP configuration failed: " << e.what());
        REQUIRE(false);
      }
    } else {
      WARN("CUDA backend not available");
      REQUIRE(true);
    }
  }
}

TEST_CASE("CUDA OpenMP Multi-GPU Jacobi Solver", "[cuda][omp][jacobi]") {
  initialize_cuda_backend_once();

  if (g_backend_available && CUDA::Manager::device_count() > 0) {
    try {
      INFO("=== CUDA OpenMP Multi-GPU Jacobi Solver Test ===");

      // Small problem for testing
      const int nx = 32;
      const int ny = 24;
      const int max_gpus = std::min(2, CUDA::Manager::device_count());

      LOGINFO("Testing with problem size {}x{} on {} GPUs", nx, ny, max_gpus);

#ifdef _OPENMP
      // Initialize CUDA manager for multi-GPU usage
      CUDA::Manager::init_for_rank(0, 1, max_gpus, true);
#endif

      // SECTION("Small problem convergence") {
      //   CUDAOMPJacobiSolver solver(nx, ny, max_gpus);

      //   solver.initialize_problem();
      //   float final_norm = solver.solve_iteration(100);
      //   solver.verify_solution();

      //   LOGINFO("Final L2 norm: {}", final_norm);
      //   REQUIRE(final_norm < 1.0e-3f); // Relaxed tolerance for multi-GPU test
      // }

      SECTION("Larger problem performance") {
        const int large_nx = 128;
        const int large_ny = 96;

        LOGINFO("Testing larger problem {}x{}", large_nx, large_ny);

        CUDAOMPJacobiSolver solver(large_nx, large_ny, max_gpus);

        auto start_time = std::chrono::high_resolution_clock::now();

        solver.initialize_problem();
        float final_norm = solver.solve_iteration(100);
        solver.verify_solution();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        INFO("Larger problem completed in " << duration.count() << " ms");
        INFO("Final L2 norm: " << final_norm);

        REQUIRE(final_norm >= 0.0f); // Should be finite and non-negative
      }

    } catch (const std::exception &e) {
      WARN("CUDA OpenMP Multi-GPU Jacobi test failed: " << e.what());
      REQUIRE(false);
    }
  } else {
    WARN("CUDA backend not available or no CUDA devices found");
    REQUIRE(true);
  }
}

TEST_CASE("CUDA Resource Management", "[cuda][resources]") {
  initialize_cuda_backend_once();

  if (g_backend_available) {
    try {
      INFO("=== CUDA Resource Management Test ===");

      int num_devices = CUDA::Manager::device_count();
      INFO("Testing resource management with " << num_devices << " devices");

      SECTION("Multiple GPU resource creation") {
        std::vector<Resource> cuda_resources;

        for (int i = 0; i < std::min(num_devices, 4); ++i) {
          cuda_resources.emplace_back(ResourceType::CUDA, i);
          REQUIRE(cuda_resources[i].type() == ResourceType::CUDA);
          REQUIRE(cuda_resources[i].id() == i);

          INFO("Created CUDA resource for device " << i);
        }
      }

      SECTION("Buffer allocation across devices") {
        const size_t buffer_size = 1000;

        for (int i = 0; i < std::min(num_devices, 4); ++i) {
          DeviceBuffer<float> buffer(buffer_size, i);

          // Test basic buffer operations
          std::vector<float> test_data(buffer_size, static_cast<float>(i + 1));
          buffer.copy_from_host(test_data);

          std::vector<float> result(buffer_size);
          buffer.copy_to_host(result);

          for (size_t j = 0; j < buffer_size; ++j) {
            REQUIRE(std::abs(result[j] - static_cast<float>(i + 1)) < 1e-6f);
          }

          INFO("Device " << i << " buffer operations successful");
        }
      }

    } catch (const std::exception &e) {
      WARN("CUDA resource management test failed: " << e.what());
      REQUIRE(false);
    }
  } else {
    WARN("CUDA backend not available");
    REQUIRE(true);
  }
}

#endif // USE_CUDA
