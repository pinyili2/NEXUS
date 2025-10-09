#include "../extern/Catch2/extras/catch_amalgamated.hpp"
#include "catch_boiler.h"
#include <algorithm>
#include <chrono>
#include <cmath>
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

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

// Include kernel definitions
#include "JacobiKernels.h"

using namespace ARBD;

// Initialize backend once for all tests to avoid peer access conflicts
static void initialize_backend_for_tests() {
  static bool test_initialized = false;
  if (test_initialized)
    return;

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

template <typename BackendManager> class MultiGPUJacobiSolver {
private:
  int num_gpus_;
  int nx_, ny_;
  std::vector<Resource> resources_;
  std::vector<DeviceBuffer<float>> a_bufs_;
  std::vector<DeviceBuffer<float>> a_new_bufs_;
  std::vector<DeviceBuffer<float>> l2_norm_bufs_;
  std::vector<int> gpu_ny_start_global_;  // Global row start indices
  std::vector<int> gpu_chunk_size_;       // Number of interior rows per GPU
  std::vector<int> neighbor_top_;         // Top neighbor GPU indices
  std::vector<int> neighbor_bottom_;      // Bottom neighbor GPU indices
  std::vector<float*> a_new_top_ptrs_;    // Pointers to top neighbor halo
  std::vector<float*> a_new_bottom_ptrs_; // Pointers to bottom neighbor halo
  bool p2p_enabled_ = false;
  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  MultiGPUJacobiSolver(int nx, int ny, int num_gpus = -1) : nx_(nx), ny_(ny) {
#ifdef _OPENMP
    INFO("Using OpenMP to determine number of GPUs");
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

    LOGINFO("MultiGPUJacobiSolver ({}) initialized with {} GPUs",
            g_backend_name.c_str(), num_gpus_);
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
    gpu_ny_start_global_.resize(num_gpus_);
    gpu_chunk_size_.resize(num_gpus_);
    neighbor_top_.resize(num_gpus_);
    neighbor_bottom_.resize(num_gpus_);
    a_new_top_ptrs_.resize(num_gpus_);
    a_new_bottom_ptrs_.resize(num_gpus_);

    // Distribute (ny-2) interior points among GPUs (like reference jacobi.cu)
    int interior_rows = ny_ - 2;
    int chunk_size_low = interior_rows / num_gpus_;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low = num_gpus_ * chunk_size_low + num_gpus_ - interior_rows;

    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      if (gpu < num_ranks_low) {
        gpu_chunk_size_[gpu] = chunk_size_low;
      } else {
        gpu_chunk_size_[gpu] = chunk_size_high;
      }

      // Calculate global starting row for this GPU's interior domain
      if (gpu < num_ranks_low) {
        gpu_ny_start_global_[gpu] = gpu * chunk_size_low + 1;
      } else {
        gpu_ny_start_global_[gpu] = num_ranks_low * chunk_size_low +
                                    (gpu - num_ranks_low) * chunk_size_high + 1;
      }

      // Set up neighbor topology (circular for now)
      neighbor_top_[gpu] = (gpu > 0) ? gpu - 1 : (num_gpus_ - 1);
      neighbor_bottom_[gpu] = (gpu + 1) % num_gpus_;

      LOGINFO("GPU {}: global rows {} to {} (chunk_size: {}), neighbors: top={}, bottom={}",
              gpu, gpu_ny_start_global_[gpu],
              gpu_ny_start_global_[gpu] + gpu_chunk_size_[gpu] - 1,
              gpu_chunk_size_[gpu], neighbor_top_[gpu], neighbor_bottom_[gpu]);
    }
  }

  void allocate_buffers() {
    resources_.clear();
    a_bufs_.clear();
    a_new_bufs_.clear();
    l2_norm_bufs_.clear();

    setup_p2p_access();

    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
#ifdef USE_CUDA
      resources_.emplace_back(ResourceType::CUDA, gpu);
#elif defined(USE_SYCL)
      resources_.emplace_back(ResourceType::SYCL, gpu);
#else
      resources_.emplace_back(ResourceType::CPU, gpu);
#endif

      // Allocate with halo regions: chunk_size + 2 (top halo + interior + bottom halo)
      size_t buffer_size = nx_ * (gpu_chunk_size_[gpu] + 2);
      a_bufs_.emplace_back(buffer_size, resources_[gpu]);
      a_new_bufs_.emplace_back(buffer_size, resources_[gpu]);
      l2_norm_bufs_.emplace_back(1, resources_[gpu]);

      LOGINFO("GPU {}: allocated {} x {} buffer with halos",
              gpu, nx_, gpu_chunk_size_[gpu] + 2);
    }

    setup_neighbor_pointers();
  }

  void setup_p2p_access() {
#ifdef USE_CUDA
    p2p_enabled_ = true;
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      cudaSetDevice(gpu);

      int top = neighbor_top_[gpu];
      int bottom = neighbor_bottom_[gpu];

      if (top != gpu) {
        int canAccessPeer = 0;
        cudaDeviceCanAccessPeer(&canAccessPeer, gpu, top);
        if (canAccessPeer) {
          cudaError_t err = cudaDeviceEnablePeerAccess(top, 0);
          if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            LOGWARN("Failed to enable P2P access from GPU {} to {}", gpu, top);
            p2p_enabled_ = false;
          }
        } else {
          LOGWARN("P2P access not supported from GPU {} to {}", gpu, top);
          p2p_enabled_ = false;
        }
      }

      if (bottom != gpu && bottom != top) {
        int canAccessPeer = 0;
        cudaDeviceCanAccessPeer(&canAccessPeer, gpu, bottom);
        if (canAccessPeer) {
          cudaError_t err = cudaDeviceEnablePeerAccess(bottom, 0);
          if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            LOGWARN("Failed to enable P2P access from GPU {} to {}", gpu, bottom);
            p2p_enabled_ = false;
          }
        } else {
          LOGWARN("P2P access not supported from GPU {} to {}", gpu, bottom);
          p2p_enabled_ = false;
        }
      }
    }

    if (p2p_enabled_) {
      LOGINFO("P2P access enabled for all GPU pairs");
    } else {
      LOGWARN("P2P access not fully available, falling back to explicit communication");
    }
#else
    p2p_enabled_ = false;
    LOGINFO("P2P access not supported on non-CUDA backend");
#endif
  }

  void setup_neighbor_pointers() {
    if (!p2p_enabled_) {
      LOGINFO("Skipping neighbor pointer setup (P2P not enabled)");
      return;
    }

#ifdef USE_CUDA
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      int top = neighbor_top_[gpu];
      int bottom = neighbor_bottom_[gpu];

      if (top != gpu) {
        // Point to the bottom halo row of the top neighbor
        a_new_top_ptrs_[gpu] = a_new_bufs_[top].device_data() +
                               nx_ * (gpu_chunk_size_[top] + 1); // Last row (bottom halo)
      } else {
        a_new_top_ptrs_[gpu] = nullptr;
      }

      if (bottom != gpu) {
        // Point to the top halo row of the bottom neighbor
        a_new_bottom_ptrs_[gpu] = a_new_bufs_[bottom].device_data(); // First row (top halo)
      } else {
        a_new_bottom_ptrs_[gpu] = nullptr;
      }

      LOGINFO("GPU {}: neighbor pointers set up (top={}, bottom={})",
              gpu, (void*)a_new_top_ptrs_[gpu], (void*)a_new_bottom_ptrs_[gpu]);
    }
#endif
  }

public:
  void initialize_problem() {
    LOGINFO("Initializing {} multi-GPU Jacobi problem with P2P={}...",
            g_backend_name.c_str(), p2p_enabled_);

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
      // Initialize boundaries with halo region layout
      int local_ny = gpu_chunk_size_[gpu] + 2; // Interior + 2 halo rows
      KernelConfig config = KernelConfig::for_1d(local_ny, gpu);
      config.sync = true;

#ifdef USE_CUDA
      launch_initialize_boundaries(
          resources_[gpu], config, a_new_bufs_[gpu].device_data(),
          a_bufs_[gpu].device_data(), pi_,
          gpu_ny_start_global_[gpu] - 1, // Offset for halo region
          nx_, local_ny, ny_);
      Event boundary_event;
#else
      // Generic boundary initialization for other backends with halo
      std::vector<float> boundary_data(nx_ * local_ny, 0.0f);
      for (int local_iy = 0; local_iy < local_ny; ++local_iy) {
        // Map local index to global, accounting for halo offset
        int global_iy = gpu_ny_start_global_[gpu] - 1 + local_iy;
        global_iy = std::max(0, std::min(global_iy, ny_ - 1)); // Clamp bounds

        float y0 = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));
        boundary_data[local_iy * nx_ + 0] = y0;
        boundary_data[local_iy * nx_ + (nx_ - 1)] = y0;
      }
      a_bufs_[gpu].copy_from_host(boundary_data);
      a_new_bufs_[gpu].copy_from_host(boundary_data);
      Event boundary_event;
#endif
      boundary_event.wait();
      LOGINFO("GPU {} boundary initialization completed (local_ny={})", gpu, local_ny);
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
        l2_norm_bufs_[gpu].copy_from_host(zero_norm.data(), 1);
      }

      // Launch Jacobi kernels on all GPUs with halo-aware indexing
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_gpus_)
#endif
      for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        // Interior points calculation: (nx-2) * chunk_size
        int interior_points = (nx_ - 2) * gpu_chunk_size_[gpu];
        if (interior_points > 0) {
          KernelConfig config = KernelConfig::for_1d(interior_points, gpu);
          config.sync = true;

          // Local indexing with halo regions:
          // Row 0 = top halo, Rows 1 to chunk_size = interior, Row chunk_size+1 = bottom halo
          int iy_start_local = 1;  // Start after top halo
          int iy_end_local = gpu_chunk_size_[gpu] + 1;  // End before bottom halo

#ifdef USE_CUDA
          // Get neighbor pointers for halo exchange
          float* top_neighbor_halo = (p2p_enabled_ && neighbor_top_[gpu] != gpu) ?
                                     a_new_top_ptrs_[gpu] : a_new_bufs_[gpu].device_data();
          float* bottom_neighbor_halo = (p2p_enabled_ && neighbor_bottom_[gpu] != gpu) ?
                                        a_new_bottom_ptrs_[gpu] :
                                        a_new_bufs_[gpu].device_data() + nx_ * (gpu_chunk_size_[gpu] + 1);

          launch_jacobi_kernel(
              resources_[gpu], config,
              a_new_bufs_[gpu].device_data(),  // Output buffer
              a_bufs_[gpu].device_data(),      // Input buffer
              l2_norm_bufs_[gpu].device_data(), // L2 norm accumulator
              iy_start_local, iy_end_local, nx_,
              top_neighbor_halo, gpu_chunk_size_[neighbor_top_[gpu]],  // Top neighbor halo location
              bottom_neighbor_halo, 0);        // Bottom neighbor halo location
          Event jacobi_event;
#else
          // Generic Jacobi iteration for other backends with halo regions
          int local_ny = gpu_chunk_size_[gpu] + 2;
          std::vector<float> a_data(nx_ * local_ny), a_new_data(nx_ * local_ny);
          a_bufs_[gpu].copy_to_host(a_data);
          a_new_bufs_[gpu].copy_to_host(a_new_data);

          float local_norm = 0.0f;
          // Process only interior points (skip halo rows)
          for (int local_iy = iy_start_local; local_iy < iy_end_local; ++local_iy) {
            for (int ix = 1; ix < nx_ - 1; ++ix) {
              int idx = local_iy * nx_ + ix;
              float new_val = 0.25f * (a_data[idx + 1] + a_data[idx - 1] +
                                       a_data[idx + nx_] + a_data[idx - nx_]);
              a_new_data[idx] = new_val;
              float residue = new_val - a_data[idx];
              local_norm += residue * residue;
            }
          }

          a_new_bufs_[gpu].copy_from_host(a_new_data.data(), a_new_data.size());
          std::vector<float> norm_data = {local_norm};
          l2_norm_bufs_[gpu].copy_from_host(norm_data.data(), 1);
          Event jacobi_event;
#endif
          jacobi_event.wait();
        }
      }

      // Collect L2 norms from all GPUs
      for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        std::vector<float> gpu_norm(1);
        l2_norm_bufs_[gpu].copy_to_host(gpu_norm.data(), 1);
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
      a_bufs_[gpu].copy_to_host(gpu_data.data(), gpu_data.size());

      for (int local_iy = 0; local_iy < gpu_ny_size_[gpu]; ++local_iy) {
        int global_iy = gpu_ny_start_[gpu] + local_iy;
        float expected = std::sin(2.0f * pi_ * global_iy / (ny_ - 1));
        float left_val = gpu_data[local_iy * nx_ + 0];
        float right_val = gpu_data[local_iy * nx_ + (nx_ - 1)];

        if (std::abs(left_val - expected) > 1e-3f ||
            std::abs(right_val - expected) > 1e-3f) {
          LOGWARN("GPU {} boundary mismatch at global row {}: left={:.6f}, "
                  "right={:.6f}, expected={:.6f}",
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
    num_devices = static_cast<int>(CUDA::Manager::device_count());
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
      Resource device_resource;
#ifdef USE_CUDA
      device_resource = Resource(ResourceType::CUDA, i);
#elif defined(USE_SYCL)
      device_resource = Resource(ResourceType::SYCL, i);
#else
      device_resource = Resource(ResourceType::CPU, i);
#endif
      DeviceBuffer<float> buffer(buffer_size, device_resource);

      std::vector<float> test_data(buffer_size, static_cast<float>(i + 1));
      buffer.copy_from_host(test_data.data(), test_data.size());

      std::vector<float> result(buffer_size);
      buffer.copy_to_host(result.data(), result.size());

      for (size_t j = 0; j < buffer_size; ++j) {
        REQUIRE(std::abs(result[j] - static_cast<float>(i + 1)) < 1e-6f);
      }

      INFO("Device " << i << " buffer operations successful");
    }
  }
}

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
      INFO("Device " << i << ": " << props.name << " (SM " << props.major << "."
                     << props.minor << ")");
    }
  }

  SECTION("OpenMP Thread Configuration") {
    int num_devices = CUDA::Manager::device_count();
    INFO("Testing OpenMP configuration with " << num_devices
                                              << " CUDA devices");

#ifdef _OPENMP
    // Test with a safe number of threads to avoid repeated Manager reinit
    int test_threads = std::min(2, num_devices);
    INFO("Testing with " << test_threads << " OpenMP threads");
    CUDA::Manager::init_for_rank(
        0, 1, test_threads, false); // Don't enable peer access repeatedly
    REQUIRE(CUDA::Manager::get_rank_devices().size() <=
            static_cast<size_t>(num_devices));

#pragma omp parallel num_threads(test_threads)
    {
      int thread_id = omp_get_thread_num();
      int gpu_id = CUDA::Manager::get_thread_gpu();
      INFO("Thread " << thread_id << " assigned to GPU " << gpu_id);
      REQUIRE(gpu_id >= 0);
      REQUIRE(gpu_id < num_devices);
      CUDA::Manager::init_for_omp_thread();
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
      if (threads > num_devices || threads > 8)
        continue;

      CUDA::Manager::init_for_rank(
          0, 1, threads, false); // Don't enable peer access repeatedly

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
      if (threads > num_devices || threads > 8)
        continue;

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
    std::vector<std::pair<int, int>> problem_sizes = {{163, 163}};

    int max_gpus = 1;
#ifdef USE_CUDA
    max_gpus = std::min(2, CUDA::Manager::device_count());
    CUDA::Manager::init_for_rank(0, 1, max_gpus,
                                 false); // Don't enable peer access repeatedly
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

      double elapsed =
          std::chrono::duration<double>(end_time - start_time).count();
      double points_per_sec = (static_cast<double>(nx * ny) * 25) / elapsed;

      INFO("Problem " << nx << "x" << ny << ": " << elapsed << "s, "
                      << (points_per_sec / 1e6)
                      << " Mpoints/sec, Final norm: " << final_norm);

      REQUIRE(elapsed > 0.0);
      REQUIRE(final_norm >= 0.0f);
    }
  }
}
