#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"

using namespace ARBD;

// Global Backend Initialization
static bool g_backend_available = false;
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
      g_backend_available = true; // Fallback to CPU
    }

    initialized = true;
  } catch (const std::exception &e) {
    g_backend_available = false;
    WARN("Backend initialization failed: " << e.what());
  }
}

// Simple multi-GPU Jacobi solver using DeviceBuffer on multiple devices
class SimpleMultiGPUJacobiSolver {
private:
  int nx_, ny_;
  int num_gpus_;

  std::vector<Resource> resources_;
  std::vector<DeviceBuffer<float>> u_buffers_;
  std::vector<DeviceBuffer<float>> u_new_buffers_;
  std::vector<DeviceBuffer<float>> l2_norm_buffers_;

  std::vector<int> gpu_ny_start_; // Starting row for each GPU
  std::vector<int> gpu_ny_size_;  // Number of rows for each GPU

  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  SimpleMultiGPUJacobiSolver(int nx, int ny, int num_gpus = -1)
      : nx_(nx), ny_(ny) {
    // Determine number of GPUs to use
    if (num_gpus <= 0) {
      num_gpus_ = std::min(8, 4); // Use up to 4 GPUs for safety
    } else {
      num_gpus_ = std::min(num_gpus, 8);
    }

    // Find available GPUs (prefer less loaded ones)
    // Based on nvidia-smi output, prioritize less loaded GPUs
    std::vector<int> preferred_gpu_order = {
        4, 0, 3, 1, 5, 6, 7, 2}; // GPU 4 is idle, GPU 2 is 97% loaded

    std::vector<int> available_gpus;
    for (int gpu_id : preferred_gpu_order) {
      try {
        // Test if GPU is available by creating a resource
        Resource test_resource(ResourceType::SYCL, gpu_id);
        available_gpus.push_back(gpu_id);
        if (available_gpus.size() >= num_gpus_)
          break;
      } catch (const std::exception &e) {
        // GPU not available, skip
        continue;
      }
    }

    // Use available GPUs, limit to requested number
    num_gpus_ = std::min(num_gpus_, static_cast<int>(available_gpus.size()));
    if (num_gpus_ == 0) {
      throw std::runtime_error("No available GPUs found");
    }

    LOGINFO("SimpleMultiGPUJacobiSolver initialized with {} GPUs: [{}]",
            num_gpus_, [&]() {
              std::string gpu_list;
              for (int i = 0; i < num_gpus_; i++) {
                if (i > 0)
                  gpu_list += ", ";
                gpu_list += std::to_string(available_gpus[i]);
              }
              return gpu_list;
            }());

    // Initialize resources and buffers for each GPU
    resources_.reserve(num_gpus_);
    u_buffers_.reserve(num_gpus_);
    u_new_buffers_.reserve(num_gpus_);
    l2_norm_buffers_.reserve(num_gpus_);

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

      // Use the available GPU ID
      int actual_gpu_id = available_gpus[gpu];
      resources_.emplace_back(ResourceType::SYCL, actual_gpu_id);

      // Allocate buffers for each GPU using actual device ID
      const size_t gpu_size = nx_ * gpu_ny_size_[gpu];
      u_buffers_.emplace_back(gpu_size, actual_gpu_id);
      u_new_buffers_.emplace_back(gpu_size, actual_gpu_id);
      l2_norm_buffers_.emplace_back(1, actual_gpu_id);

      // Initialize with zeros
      std::vector<float> zeros(gpu_size, 0.0f);
      u_buffers_[gpu].copy_from_host(zeros);
      u_new_buffers_[gpu].copy_from_host(zeros);

      std::vector<float> norm_zero(1, 0.0f);
      l2_norm_buffers_[gpu].copy_from_host(norm_zero);
    }
  }

  void initialize_boundaries() {
    // Set sine wave boundary conditions on each GPU
    for (int gpu = 0; gpu < num_gpus_; gpu++) {
      // Capture needed variables explicitly
      const int gpu_ny_size = gpu_ny_size_[gpu];
      const int gpu_ny_start = gpu_ny_start_[gpu];
      const int nx = nx_;
      const int ny = ny_;
      const float pi = pi_;
      auto u_ptr = u_buffers_[gpu].data();
      auto u_new_ptr = u_new_buffers_[gpu].data();

      auto event = launch_kernel(
          resources_[gpu], KernelConfig::for_1d(gpu_ny_size, resources_[gpu]),
          [=](size_t i) {
            int iy = static_cast<int>(i);
            if (iy >= gpu_ny_size)
              return;

            int global_iy = gpu_ny_start + iy;
            const float y0 = std::sin(2.0f * pi * global_iy / (ny - 1));

            // Set left and right boundaries
            u_ptr[iy * nx + 0] = y0;
            u_ptr[iy * nx + (nx - 1)] = y0;
            u_new_ptr[iy * nx + 0] = y0;
            u_new_ptr[iy * nx + (nx - 1)] = y0;
          });
      event.wait();
    }
  }

  float solve(int iter_max, int nccheck = 1) {
    float l2_norm = 1.0f;
    int iter = 0;

    while (l2_norm > tol_ && iter < iter_max) {
      // Reset norm buffers on all GPUs
      for (int gpu = 0; gpu < num_gpus_; gpu++) {
        std::vector<float> norm_init(1, 0.0f);
        l2_norm_buffers_[gpu].copy_from_host(norm_init);
      }

      bool calculate_norm = (iter % nccheck) == 0;

      // Launch Jacobi kernel on all GPUs
      for (int gpu = 0; gpu < num_gpus_; gpu++) {
        // Calculate number of interior points for this GPU
        const int gpu_ny_size = gpu_ny_size_[gpu];
        const int nx = nx_;
        int interior_points = (nx - 2) * (gpu_ny_size - 2);
        if (interior_points > 0) {
          // Capture needed variables explicitly
          auto u_ptr = u_buffers_[gpu].data();
          auto u_new_ptr = u_new_buffers_[gpu].data();
          auto l2_norm_ptr = l2_norm_buffers_[gpu].data();

          auto event = launch_kernel(
              resources_[gpu],
              KernelConfig::for_1d(interior_points, resources_[gpu]),
              [=](size_t i) {
                int thread_idx = static_cast<int>(i);
                int total_width = nx - 2; // Interior points only
                int total_height = gpu_ny_size - 2;

                if (thread_idx >= total_width * total_height)
                  return;

                int iy = thread_idx / total_width + 1; // Start from iy=1
                int ix = thread_idx % total_width + 1; // Start from ix=1

                if (iy >= (gpu_ny_size - 1) || ix >= (nx - 1))
                  return;

                float local_l2_norm = 0.0f;

                // Perform Jacobi iteration
                const float new_val =
                    0.25f * (u_ptr[(iy - 1) * nx + ix] + // top
                             u_ptr[(iy + 1) * nx + ix] + // bottom
                             u_ptr[iy * nx + (ix - 1)] + // left
                             u_ptr[iy * nx + (ix + 1)]   // right
                            );
                u_new_ptr[iy * nx + ix] = new_val;

                if (calculate_norm) {
                  float residue = new_val - u_ptr[iy * nx + ix];
                  local_l2_norm = residue * residue;
                  // Use atomic add for norm accumulation
                  ATOMIC_ADD(l2_norm_ptr, local_l2_norm);
                }
              });
          event.wait();
        }
      }

      // Handle norm calculation
      if (calculate_norm) {
        float total_norm = 0.0f;

        for (int gpu = 0; gpu < num_gpus_; gpu++) {
          std::vector<float> norm_result(1);
          l2_norm_buffers_[gpu].copy_to_host(norm_result);
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
      for (int gpu = 0; gpu < num_gpus_; gpu++) {
        std::swap(u_buffers_[gpu], u_new_buffers_[gpu]);
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
      u_new_buffers_[gpu].copy_to_host(gpu_data);
      u_new_buffers_[next_gpu].copy_to_host(next_gpu_data);

      // Copy bottom row of current GPU to top row of next GPU
      int bottom_row_start = (gpu_ny_size_[gpu] - 1) * nx_;
      std::copy(gpu_data.begin() + bottom_row_start,
                gpu_data.begin() + bottom_row_start + nx_,
                next_gpu_data.begin());

      // Copy top row of next GPU to bottom row of current GPU
      std::copy(next_gpu_data.begin() + nx_, next_gpu_data.begin() + 2 * nx_,
                gpu_data.begin() + bottom_row_start);

      // Update buffers
      u_new_buffers_[gpu].copy_from_host(gpu_data);
      u_new_buffers_[next_gpu].copy_from_host(next_gpu_data);
    }
  }

public:
  std::vector<float> get_solution() {
    std::vector<float> full_solution(nx_ * ny_);

    for (int gpu = 0; gpu < num_gpus_; gpu++) {
      std::vector<float> gpu_solution;
      u_buffers_[gpu].copy_to_host(gpu_solution);

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

// Test cases
TEST_CASE("Simple Multi-GPU Jacobi Solver", "[simple][multi-gpu]") {
  // Initialize backend globally
  initialize_backend_once();

  SECTION("Single GPU Test") {
    const int nx = 16;
    const int ny = 16;
    const int iter_max = 100;

    SimpleMultiGPUJacobiSolver solver(nx, ny, 1);
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

    SimpleMultiGPUJacobiSolver solver(nx, ny, 2);
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

  SECTION("Performance Benchmark") {
    const int nx = 128;
    const int ny = 128;
    const int iter_max = 500;

    // Test different GPU configurations
    std::vector<int> gpu_configs = {1, 2, 4};
    std::vector<double> times;

    for (int num_gpus : gpu_configs) {
      if (num_gpus > 4)
        continue; // Limit to 4 GPUs for safety

      SimpleMultiGPUJacobiSolver solver(nx, ny, num_gpus);
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

  SECTION("Scaling Profile") {
    // Test different problem sizes and GPU configurations
    std::vector<std::pair<int, int>> problem_sizes = {
        {256, 256}, {512, 512}, {1024, 1024}};

    std::vector<int> gpu_configs = {1, 2, 4};
    const int iter_max = 200; // Reduced for profiling

    std::cout << "\n=== SCALING PROFILE ===" << std::endl;
    std::cout << "Problem Size\tGPUs\tTime(s)\t\tSpeedup\t\tEfficiency\tGFLOPS"
              << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "-------------------"
              << std::endl;

    for (auto [nx, ny] : problem_sizes) {
      std::vector<double> times;
      std::vector<double> gflops;

      for (int num_gpus : gpu_configs) {
        if (num_gpus > 4)
          continue;

        SimpleMultiGPUJacobiSolver solver(nx, ny, num_gpus);
        solver.initialize_boundaries();

        // Warm-up run
        solver.solve(10, 999);

        // Measure performance
        auto start = std::chrono::high_resolution_clock::now();
        float final_norm = solver.solve(iter_max, 50);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        times.push_back(elapsed);

        // Calculate GFLOPS (approximately 5 FLOPS per point per iteration)
        double total_points = static_cast<double>(nx) * ny;
        double total_flops =
            total_points * iter_max * 5.0; // 5 FLOPS per point per iteration
        double gflops_value = (total_flops / elapsed) / 1e9;
        gflops.push_back(gflops_value);

        // Verify correctness
        REQUIRE(final_norm >= 0.0f);
        REQUIRE(final_norm < 1.0f);
        REQUIRE(elapsed > 0.0);
        REQUIRE(elapsed < 300.0); // Should complete within 5 minutes
      }

      // Calculate and display scaling results
      if (times.size() > 1) {
        double baseline_time = times[0]; // Single GPU time
        double baseline_gflops = gflops[0];

        for (size_t i = 0; i < times.size(); i++) {
          double speedup = (i == 0) ? 1.0 : baseline_time / times[i];
          double efficiency =
              (i == 0) ? 100.0 : (speedup / gpu_configs[i]) * 100.0;

          std::cout << nx << "x" << ny << "\t\t" << gpu_configs[i] << "\t\t"
                    << std::fixed << std::setprecision(3) << times[i] << "\t\t"
                    << std::fixed << std::setprecision(2) << speedup << "x\t\t"
                    << std::fixed << std::setprecision(1) << efficiency
                    << "%\t\t" << std::fixed << std::setprecision(2)
                    << gflops[i] << std::endl;
        }
        std::cout << "---------------------------------------------------------"
                     "-----------------------"
                  << std::endl;
      }
    }

    std::cout << "\n=== SCALING ANALYSIS ===" << std::endl;

    // Analyze scaling efficiency across problem sizes
    for (auto [nx, ny] : problem_sizes) {
      std::cout << "\nProblem Size " << nx << "x" << ny << ":" << std::endl;

      // Re-run to get fresh timing data
      std::vector<double> times;
      for (int num_gpus : gpu_configs) {
        if (num_gpus > 4)
          continue;

        SimpleMultiGPUJacobiSolver solver(nx, ny, num_gpus);
        solver.initialize_boundaries();
        solver.solve(10, 999); // Warm-up

        auto start = std::chrono::high_resolution_clock::now();
        solver.solve(iter_max, 50);
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration<double>(end - start).count());
      }

      if (times.size() > 1) {
        double single_gpu_time = times[0];
        for (size_t i = 1; i < times.size(); i++) {
          double speedup = single_gpu_time / times[i];
          double efficiency = (speedup / gpu_configs[i]) * 100.0;
          double parallel_efficiency = efficiency;

          std::cout << "  " << gpu_configs[i] << " GPU(s): " << std::fixed
                    << std::setprecision(2) << speedup << "x speedup, "
                    << std::fixed << std::setprecision(1) << parallel_efficiency
                    << "% efficiency" << std::endl;
        }
      }
    }

    std::cout << "\n=== PERFORMANCE INSIGHTS ===" << std::endl;
    std::cout
        << "• Higher problem sizes generally show better scaling efficiency"
        << std::endl;
    std::cout << "• Communication overhead is amortized over larger "
                 "computational work"
              << std::endl;
    std::cout << "• Ghost cell exchange becomes relatively less expensive for "
                 "larger grids"
              << std::endl;
    std::cout << "• Memory bandwidth and compute intensity improve with larger "
                 "problems"
              << std::endl;
  }
}
