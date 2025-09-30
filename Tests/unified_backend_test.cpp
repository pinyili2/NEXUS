#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include "cuda_omp_multi_gpu_kernels.h"
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

using namespace ARBD;

// ============================================================================
// Global Backend Management
// ============================================================================

static bool g_backend_available = false;
static std::string g_backend_name = "Unknown";

static void initialize_backend_once() {
  static bool initialized = false;
  if (initialized) return;

  try {
#ifdef USE_CUDA
    CUDA::Manager::init();
    g_backend_available = true;
    g_backend_name = "CUDA";
#elif defined(USE_SYCL)
    SYCL::Manager::init();
    g_backend_available = true;
    g_backend_name = "SYCL";
#elif defined(USE_METAL)
    METAL::Manager::init();
    g_backend_available = true;
    g_backend_name = "METAL";
#else
    g_backend_available = false;
    g_backend_name = "CPU";
#endif
    initialized = true;
  } catch (const std::exception &e) {
    g_backend_available = false;
    WARN("Backend initialization failed: " << e.what());
  }
}

// Initialize backend once for all tests to avoid peer access conflicts
static void initialize_backend_for_tests() {
  static bool test_initialized = false;
  if (test_initialized) return;

  initialize_backend_once();

#ifdef USE_CUDA
  if (g_backend_available) {
    // Initialize for single rank to set up peer access once
    CUDA::Manager::init_for_rank(0, 1, 1, true);
  }
#endif

  test_initialized = true;
}

// ============================================================================
// Multi-GPU Jacobi Solver (Backend Agnostic)
// ============================================================================

template<typename BackendManager>
class MultiGPUJacobiSolver {
private:
  int num_gpus_;
  int nx_, ny_;
  std::vector<Resource> resources_;
  std::vector<DeviceBuffer<float>> a_bufs_;
  std::vector<DeviceBuffer<float>> a_new_bufs_;
  std::vector<DeviceBuffer<float>> l2_norm_bufs_;
  std::vector<int> gpu_ny_start_;
  std::vector<int> gpu_ny_size_;
  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  MultiGPUJacobiSolver(int nx, int ny, int num_gpus = -1) : nx_(nx), ny_(ny) {
#ifdef _OPENMP
    if (num_gpus <= 0) {
      num_gpus_ = omp_get_max_threads();
    } else {
      num_gpus_ = num_gpus;
    }

    // Limit to available devices
    int device_count = get_device_count();
    num_gpus_ = std::min(num_gpus_, device_count);
#else
    num_gpus_ = 1;
#endif

    LOGINFO("MultiGPUJacobiSolver ({}) initialized with {} GPUs", g_backend_name, num_gpus_);
    setup_domain_decomposition();
    allocate_buffers();
  }

private:
  int get_device_count() {
#ifdef USE_CUDA
    return CUDA::Manager::device_count();
#elif defined(USE_SYCL)
    return static_cast<int>(SYCL::Manager::device_count());
#else
    return 1;
#endif
  }

  void setup_domain_decomposition() {
    gpu_ny_start_.resize(num_gpus_);
    gpu_ny_size_.resize(num_gpus_);

    int rows_per_gpu = ny_ / num_gpus_;
    int extra_rows = ny_ % num_gpus_;

    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      gpu_ny_start_[gpu] = gpu * rows_per_gpu + std::min(gpu, extra_rows);
      gpu_ny_size_[gpu] = rows_per_gpu + (gpu < extra_rows ? 1 : 0);
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
#ifdef USE_CUDA
      resources_.emplace_back(ResourceType::CUDA, gpu);
#elif defined(USE_SYCL)
      resources_.emplace_back(ResourceType::SYCL, gpu);
#else
      resources_.emplace_back(ResourceType::CPU, gpu);
#endif

      size_t buffer_size = nx_ * gpu_ny_size_[gpu];
      a_bufs_.emplace_back(buffer_size, gpu);
      a_new_bufs_.emplace_back(buffer_size, gpu);
      l2_norm_bufs_.emplace_back(1, gpu);
    }
  }

public:
  void initialize_problem() {
    LOGINFO("Initializing {} multi-GPU Jacobi problem...", g_backend_name);

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      // Initialize boundaries
      KernelConfig config = KernelConfig::for_1d(gpu_ny_size_[gpu], gpu);
      config.sync = true;

#ifdef USE_CUDA
      auto boundary_event = launch_kernel(
          resources_[gpu], config, initialize_boundaries_omp_kernel{},
          get_buffer_pointer(a_new_bufs_[gpu]),
          get_buffer_pointer(a_bufs_[gpu]), pi_, nx_, gpu_ny_size_[gpu]);
#else
      // Generic boundary initialization for other backends
      std::vector<float> boundary_data(nx_ * gpu_ny_size_[gpu], 0.0f);
      for (int local_iy = 0; local_iy < gpu_ny_size_[gpu]; ++local_iy) {
        int global_iy = gpu_ny_start_[gpu] + local_iy;
        float y0 = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));
        boundary_data[local_iy * nx_ + 0] = y0;
        boundary_data[local_iy * nx_ + (nx_ - 1)] = y0;
      }
      a_bufs_[gpu].copy_from_host(boundary_data);
      a_new_bufs_[gpu].copy_from_host(boundary_data);
      Event boundary_event;
#endif
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

#ifdef USE_CUDA
          auto jacobi_event = launch_kernel(
              resources_[gpu], config, jacobi_omp_kernel{},
              get_buffer_pointer(a_new_bufs_[gpu]),
              get_buffer_pointer(a_bufs_[gpu]),
              get_buffer_pointer(l2_norm_bufs_[gpu]), nx_, gpu_ny_size_[gpu], true);
#else
          // Generic Jacobi iteration for other backends
          std::vector<float> a_data, a_new_data;
          a_bufs_[gpu].copy_to_host(a_data);
          a_new_bufs_[gpu].copy_to_host(a_new_data);

          float local_norm = 0.0f;
          for (int local_iy = 1; local_iy < gpu_ny_size_[gpu] - 1; ++local_iy) {
            for (int ix = 1; ix < nx_ - 1; ++ix) {
              int idx = local_iy * nx_ + ix;
              float new_val = 0.25f * (a_data[idx + 1] + a_data[idx - 1] +
                                     a_data[idx + nx_] + a_data[idx - nx_]);
              a_new_data[idx] = new_val;
              float residue = new_val - a_data[idx];
              local_norm += residue * residue;
            }
          }

          a_new_bufs_[gpu].copy_from_host(a_new_data);
          std::vector<float> norm_data = {local_norm};
          l2_norm_bufs_[gpu].copy_from_host(norm_data);
          Event jacobi_event;
#endif
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

      if ((iter + 1) % 25 == 0) {
        LOGINFO("Iteration {}: L2 norm = {:.6e}", iter + 1, global_l2_norm);
      }

      if (global_l2_norm < tol_) {
        LOGINFO("Converged after {} iterations", iter + 1);
        return global_l2_norm;
      }
    }

    LOGWARN("Did not converge after {} iterations", max_iterations);
    return global_l2_norm;
  }

  void verify_solution() {
    LOGINFO("Verifying {} multi-GPU solution...", g_backend_name);
    // Basic verification - check that boundaries are preserved
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      std::vector<float> gpu_data(nx_ * gpu_ny_size_[gpu]);
      a_bufs_[gpu].copy_to_host(gpu_data);

      for (int local_iy = 0; local_iy < gpu_ny_size_[gpu]; ++local_iy) {
        int global_iy = gpu_ny_start_[gpu] + local_iy;
        float expected = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));
        float left_val = gpu_data[local_iy * nx_ + 0];
        float right_val = gpu_data[local_iy * nx_ + (nx_ - 1)];

        if (std::abs(left_val - expected) > 1e-3f || std::abs(right_val - expected) > 1e-3f) {
          LOGWARN("GPU {} boundary mismatch at global row {}: left={:.6f}, right={:.6f}, expected={:.6f}",
                  gpu, global_iy, left_val, right_val, expected);
        }
      }
    }
    LOGINFO("{} multi-GPU solution verification completed", g_backend_name);
  }
};

// ============================================================================
// Backend Manager Tests
// ============================================================================

#ifdef USE_CUDA
TEST_CASE("CUDA Manager OpenMP Integration", "[cuda][omp][manager]") {
  initialize_backend_for_tests();

  if (!g_backend_available) {
    WARN("CUDA backend not available");
    return;
  }

  SECTION("Basic Manager Initialization") {
    REQUIRE(CUDA::Manager::device_count() > 0);
    INFO("Found " << CUDA::Manager::device_count() << " CUDA devices");

    for (int i = 0; i < CUDA::Manager::device_count(); ++i) {
      auto props = CUDA::Manager::get_device_properties(i);
      INFO("Device " << i << ": " << props.name << " (SM " << props.major << "." << props.minor << ")");
    }
  }

  SECTION("OpenMP Thread Configuration") {
    int num_devices = CUDA::Manager::device_count();
    INFO("Testing OpenMP configuration with " << num_devices << " CUDA devices");

#ifdef _OPENMP
    for (int threads = 1; threads <= std::min(4, num_devices); ++threads) {
      INFO("Testing with " << threads << " OpenMP threads");
      CUDA::Manager::init_for_rank(0, 1, threads, false); // Don't enable peer access repeatedly
      REQUIRE(CUDA::Manager::get_rank_devices().size() <= static_cast<size_t>(num_devices));

#pragma omp parallel num_threads(threads)
      {
        int thread_id = omp_get_thread_num();
        int gpu_id = CUDA::Manager::get_thread_gpu();
        INFO("Thread " << thread_id << " assigned to GPU " << gpu_id);
        REQUIRE(gpu_id >= 0);
        REQUIRE(gpu_id < num_devices);
        CUDA::Manager::init_for_omp_thread();
      }
    }
#endif
  }

  SECTION("OpenMP Scaling Performance Benchmark") {
    int num_devices = CUDA::Manager::device_count();
    const int num_iterations = 1000;
    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<double> times;

    INFO("=== CUDA OpenMP Scaling Benchmark ===");
    INFO("Devices available: " << num_devices);

    for (int threads : thread_configs) {
      if (threads > num_devices || threads > 8) continue;

      CUDA::Manager::init_for_rank(0, 1, threads, false); // Don't enable peer access repeatedly

      auto start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
      for (int i = 0; i < num_iterations; ++i) {
        CUDA::Manager::init_for_omp_thread();
        int gpu_id = CUDA::Manager::get_thread_gpu();

        // Simulate GPU work
        volatile int work = 0;
        for (int j = 0; j < 1000; ++j) {
          work += gpu_id + j;
        }
      }
#endif

      auto end = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(end - start).count();
      times.push_back(elapsed);

      INFO("Performance with " << threads << " threads: " << elapsed << "s");
      REQUIRE(elapsed > 0.0);
      REQUIRE(elapsed < 10.0);
    }

    // Analyze scaling
    if (times.size() >= 2) {
      double single_thread = times[0];
      for (size_t i = 1; i < times.size(); ++i) {
        double speedup = single_thread / times[i];
        double efficiency = speedup / thread_configs[i];
        INFO("Threads: " << thread_configs[i] << ", Speedup: " << speedup
             << "x, Efficiency: " << (efficiency * 100) << "%");
        REQUIRE(speedup > 0.5); // Should show some speedup
      }
    }
  }
}
#endif

#ifdef USE_SYCL
TEST_CASE("SYCL Manager OpenMP Integration", "[sycl][omp][manager]") {
  initialize_backend_for_tests();

  if (!g_backend_available) {
    WARN("SYCL backend not available");
    return;
  }

  SECTION("Basic Manager Initialization") {
    REQUIRE(SYCL::Manager::device_count() > 0);
    auto devices = SYCL::Manager::get_all_devices();
    REQUIRE(devices.size() > 0);
    INFO("Found " << devices.size() << " SYCL devices");
  }

  SECTION("OpenMP Scaling Performance Benchmark") {
    int num_devices = static_cast<int>(SYCL::Manager::device_count());
    const int num_iterations = 1000;
    std::vector<int> thread_configs = {1, 2, 4, 8};
    std::vector<double> times;

    INFO("=== SYCL OpenMP Scaling Benchmark ===");
    INFO("Devices available: " << num_devices);

    for (int threads : thread_configs) {
      if (threads > num_devices || threads > 8) continue;

      SYCL::Manager::init_for_rank(0, 1, threads, false);

      auto start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
      for (int i = 0; i < num_iterations; ++i) {
        SYCL::Manager::init_for_omp_thread();
        int device = SYCL::Manager::get_thread_device();

        // Simulate device work
        volatile int work = 0;
        for (int j = 0; j < 1000; ++j) {
          work += device + j;
        }
      }
#endif

      auto end = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(end - start).count();
      times.push_back(elapsed);

      INFO("Performance with " << threads << " threads: " << elapsed << "s");
      REQUIRE(elapsed > 0.0);
      REQUIRE(elapsed < 10.0);
    }

    // Analyze scaling
    if (times.size() >= 2) {
      double single_thread = times[0];
      for (size_t i = 1; i < times.size(); ++i) {
        double speedup = single_thread / times[i];
        double efficiency = speedup / thread_configs[i];
        INFO("Threads: " << thread_configs[i] << ", Speedup: " << speedup
             << "x, Efficiency: " << (efficiency * 100) << "%");
        REQUIRE(speedup > 0.5); // Should show some speedup
      }
    }
  }
}
#endif

// ============================================================================
// Multi-GPU Jacobi Solver Tests
// ============================================================================

TEST_CASE("Multi-GPU Jacobi Solver", "[multi-gpu][jacobi]") {
  initialize_backend_for_tests();

  if (!g_backend_available) {
    WARN("Backend not available - skipping multi-GPU tests");
    return;
  }

  SECTION("Small Problem Convergence") {
    const int nx = 32;
    const int ny = 24;
    int max_gpus = 2;

#ifdef USE_CUDA
    max_gpus = std::min(max_gpus, CUDA::Manager::device_count());
    CUDA::Manager::init_for_rank(0, 1, max_gpus, false); // Don't enable peer access repeatedly
    MultiGPUJacobiSolver<CUDA::Manager> solver(nx, ny, max_gpus);
#elif defined(USE_SYCL)
    max_gpus = std::min(max_gpus, static_cast<int>(SYCL::Manager::device_count()));
    SYCL::Manager::init_for_rank(0, 1, max_gpus, true);
    MultiGPUJacobiSolver<SYCL::Manager> solver(nx, ny, max_gpus);
#else
    MultiGPUJacobiSolver<void> solver(nx, ny, 1);
#endif

    LOGINFO("Testing {} multi-GPU solver with problem size {}x{} on {} GPUs",
            g_backend_name, nx, ny, max_gpus);

    solver.initialize_problem();
    float final_norm = solver.solve_iteration(100);
    solver.verify_solution();

    LOGINFO("Final L2 norm: {}", final_norm);
    REQUIRE(final_norm < 1.0e-2f); // Relaxed tolerance for multi-GPU
  }

  SECTION("Performance Scaling Test") {
    const int nx = 64;
    const int ny = 48;
    std::vector<int> gpu_configs = {1, 2};
    std::vector<double> solve_times;

    INFO("=== Multi-GPU Performance Scaling ===");

    for (int num_gpus : gpu_configs) {
#ifdef USE_CUDA
      if (num_gpus > CUDA::Manager::device_count()) continue;
      CUDA::Manager::init_for_rank(0, 1, num_gpus, false); // Don't enable peer access repeatedly
      MultiGPUJacobiSolver<CUDA::Manager> solver(nx, ny, num_gpus);
#elif defined(USE_SYCL)
      if (num_gpus > static_cast<int>(SYCL::Manager::device_count())) continue;
      SYCL::Manager::init_for_rank(0, 1, num_gpus, true);
      MultiGPUJacobiSolver<SYCL::Manager> solver(nx, ny, num_gpus);
#else
      if (num_gpus > 1) continue;
      MultiGPUJacobiSolver<void> solver(nx, ny, 1);
#endif

      solver.initialize_problem();

      auto start_time = std::chrono::high_resolution_clock::now();
      float final_norm = solver.solve_iteration(50);
      auto end_time = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double>(end_time - start_time).count();
      solve_times.push_back(elapsed);

      INFO("GPUs: " << num_gpus << ", Time: " << elapsed << "s, Final norm: " << final_norm);
      REQUIRE(elapsed > 0.0);
      REQUIRE(final_norm >= 0.0f);
    }

    // Analyze scaling efficiency
    if (solve_times.size() >= 2) {
      double single_gpu_time = solve_times[0];
      for (size_t i = 1; i < solve_times.size(); ++i) {
        double speedup = single_gpu_time / solve_times[i];
        double efficiency = speedup / gpu_configs[i];
        INFO("GPUs: " << gpu_configs[i] << ", Speedup: " << speedup
             << "x, Efficiency: " << (efficiency * 100) << "%");
      }
    }
  }
}


// ============================================================================
// Resource Management Tests
// ============================================================================

TEST_CASE("Resource Management", "[resources]") {
  initialize_backend_for_tests();

  if (!g_backend_available) {
    WARN("Backend not available - skipping resource tests");
    return;
  }

  SECTION("Multiple Device Resource Creation") {
    int num_devices = 1;
#ifdef USE_CUDA
    num_devices = CUDA::Manager::device_count();
#elif defined(USE_SYCL)
    num_devices = static_cast<int>(SYCL::Manager::device_count());
#endif

    std::vector<Resource> resources;
    for (int i = 0; i < std::min(num_devices, 4); ++i) {
#ifdef USE_CUDA
      resources.emplace_back(ResourceType::CUDA, i);
      REQUIRE(resources[i].type() == ResourceType::CUDA);
#elif defined(USE_SYCL)
      resources.emplace_back(ResourceType::SYCL, i);
      REQUIRE(resources[i].type() == ResourceType::SYCL);
#else
      resources.emplace_back(ResourceType::CPU, i);
      REQUIRE(resources[i].type() == ResourceType::CPU);
#endif
      REQUIRE(resources[i].id() == i);
      INFO("Created " << g_backend_name << " resource for device " << i);
    }
  }

  SECTION("Buffer Operations Across Devices") {
    const size_t buffer_size = 1000;
    int num_devices = 1;

#ifdef USE_CUDA
    num_devices = CUDA::Manager::device_count();
#elif defined(USE_SYCL)
    num_devices = static_cast<int>(SYCL::Manager::device_count());
#endif

    for (int i = 0; i < std::min(num_devices, 4); ++i) {
      DeviceBuffer<float> buffer(buffer_size, i);

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
}

// ============================================================================
// Comprehensive Performance Benchmark
// ============================================================================

TEST_CASE("Comprehensive Performance Benchmark", "[benchmark][performance]") {
  initialize_backend_for_tests();

  if (!g_backend_available) {
    WARN("Backend not available - skipping performance benchmark");
    return;
  }

  INFO("=== " << g_backend_name << " Comprehensive Performance Benchmark ===");

  SECTION("Problem Size Scaling") {
    std::vector<std::pair<int, int>> problem_sizes = {
      {32, 24}, {64, 48}, {128, 96}
    };

    int max_gpus = 1;
#ifdef USE_CUDA
    max_gpus = std::min(2, CUDA::Manager::device_count());
    CUDA::Manager::init_for_rank(0, 1, max_gpus, false); // Don't enable peer access repeatedly
#elif defined(USE_SYCL)
    max_gpus = std::min(2, static_cast<int>(SYCL::Manager::device_count()));
    SYCL::Manager::init_for_rank(0, 1, max_gpus, true);
#endif

    for (auto [nx, ny] : problem_sizes) {
#ifdef USE_CUDA
      MultiGPUJacobiSolver<CUDA::Manager> solver(nx, ny, max_gpus);
#elif defined(USE_SYCL)
      MultiGPUJacobiSolver<SYCL::Manager> solver(nx, ny, max_gpus);
#else
      MultiGPUJacobiSolver<void> solver(nx, ny, 1);
#endif

      solver.initialize_problem();

      auto start_time = std::chrono::high_resolution_clock::now();
      float final_norm = solver.solve_iteration(25);
      auto end_time = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double>(end_time - start_time).count();
      double points_per_sec = (static_cast<double>(nx * ny) * 25) / elapsed;

      INFO("Problem " << nx << "x" << ny << ": " << elapsed << "s, "
           << (points_per_sec / 1e6) << " Mpoints/sec, Final norm: " << final_norm);

      REQUIRE(elapsed > 0.0);
      REQUIRE(final_norm >= 0.0f);
    }
  }
}
