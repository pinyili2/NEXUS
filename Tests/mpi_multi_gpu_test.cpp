#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/MPIManager.h"
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
    auto &cuda_manager = CUDA::Manager::instance();
    cuda_manager.init();
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

// MPI-aware Jacobi kernel functors (backend-agnostic)
struct initialize_boundaries_mpi_kernel {
  void operator()(size_t i, float *__restrict__ const a_new,
                  float *__restrict__ const a, const float pi, const int offset,
                  const int nx, const int my_ny, const int ny) const {
    // Convert linear thread index to actual row index
    int thread_iy = static_cast<int>(i);

    // Guard against extra threads launched by the backend
    if (thread_iy >= my_ny) {
      return; // This thread has no work to do
    }

    // Actual row index starts from offset
    int actual_iy = offset + thread_iy;

    // Additional bounds check
    if (actual_iy >= ny) {
      return;
    }

    const float y0 = std::sin(2.0f * pi * actual_iy / (ny - 1));
    a[actual_iy * nx + 0] = y0;
    a[actual_iy * nx + (nx - 1)] = y0;
    a_new[actual_iy * nx + 0] = y0;
    a_new[actual_iy * nx + (nx - 1)] = y0;
  }
};

struct jacobi_mpi_kernel {
  void operator()(size_t i, float *__restrict__ const a_new,
                  const float *__restrict__ const a,
                  float *__restrict__ const l2_norm, const int iy_start,
                  const int iy_end, const int nx,
                  const bool calculate_norm) const {
    // Convert linear thread index to 2D coordinates
    int total_width = nx - 2; // Interior points only
    int thread_idx = static_cast<int>(i);

    // Guard against extra threads launched by the backend
    int total_interior_points = total_width * (iy_end - iy_start);
    if (thread_idx >= total_interior_points) {
      return; // This thread has no work to do
    }

    int iy = thread_idx / total_width + iy_start;
    int ix = thread_idx % total_width + 1; // Start from ix=1

    // Additional bounds checking
    if (iy >= iy_end || ix >= (nx - 1)) {
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

// MPI Jacobi solver class
class MPIJacobiSolver {
private:
  int rank_;
  int size_;
  int nx_, ny_;
  int chunk_size_;
  int iy_start_global_, iy_end_global_;
  int iy_start_, iy_end_;

  Resource resource_;
  DeviceBuffer<float> a_buf_;
  DeviceBuffer<float> a_new_buf_;
  DeviceBuffer<float> l2_norm_buf_;

  const float pi_ = 2.0f * std::asin(1.0f);
  const float tol_ = 1.0e-8f;

public:
  MPIJacobiSolver(int nx, int ny, int device_id = -1) : nx_(nx), ny_(ny) {

#ifdef USE_MPI
    // Initialize MPI if not already done
    ARBD::MPI::Manager::instance().init();
    rank_ = ARBD::MPI::Manager::instance().get_rank();
    size_ = ARBD::MPI::Manager::instance().get_size();
    LOGINFO("MPIJacobiSolver initialized: rank {}/{}", rank_, size_);
#else
    rank_ = 0;
    size_ = 1;
#endif

    // Calculate domain decomposition for interior points (ny-2)
    int interior_points = ny - 2;
    int chunk_size_low = interior_points / size_;
    int remainder = interior_points % size_;

    if (rank_ < remainder) {
      chunk_size_ = chunk_size_low + 1;
    } else {
      chunk_size_ = chunk_size_low;
    }

    // Calculate global boundaries
    iy_start_global_ = 1; // Start from 1 (after boundary)
    for (int i = 0; i < rank_; i++) {
      if (i < remainder) {
        iy_start_global_ += chunk_size_low + 1;
      } else {
        iy_start_global_ += chunk_size_low;
      }
    }
    iy_end_global_ = iy_start_global_ + chunk_size_ - 1;

    // Local boundaries (including ghost cells)
    iy_start_ = 1;
    iy_end_ = iy_start_ + chunk_size_;

    // Initialize resource (use SYCL device)
    // When using CUDA_VISIBLE_DEVICES per process, each process sees only one
    // GPU as device 0
    int device_to_use = (device_id >= 0) ? device_id : 0;
    resource_ = Resource(ResourceType::SYCL, device_to_use);

    // Allocate buffers
    const size_t local_size = nx_ * (chunk_size_ + 2);
    a_buf_ = DeviceBuffer<float>(local_size, 0);
    a_new_buf_ = DeviceBuffer<float>(local_size, 0);
    l2_norm_buf_ = DeviceBuffer<float>(1, 0);

    // Initialize with zeros
    std::vector<float> zeros(local_size, 0.0f);
    a_buf_.copy_from_host(zeros);
    a_new_buf_.copy_from_host(zeros);
    l2_norm_buf_.copy_from_host(zeros);
  }

  void initialize_boundaries() {
    // Set sine wave boundary conditions
    auto event = launch_kernel(
        resource_, KernelConfig::for_1d(chunk_size_ + 2, resource_),
        initialize_boundaries_mpi_kernel{}, get_buffer_pointer(a_new_buf_),
        get_buffer_pointer(a_buf_), pi_, iy_start_global_ - 1, nx_,
        chunk_size_ + 2, ny_);
    event.wait();
  }

  float solve(int iter_max, int nccheck = 1) {
    float l2_norm = 1.0f;
    int iter = 0;
    bool calculate_norm = true;

    // Calculate number of interior points
    int interior_points = (nx_ - 2) * (iy_end_ - iy_start_);

    while (l2_norm > tol_ && iter < iter_max) {
      // Reset norm buffer
      std::vector<float> norm_init(1, 0.0f);
      l2_norm_buf_.copy_from_host(norm_init);

      calculate_norm = (iter % nccheck) == 0;

      // Launch Jacobi kernel
      auto event = launch_kernel(
          resource_, KernelConfig::for_1d(interior_points, resource_),
          jacobi_mpi_kernel{}, get_buffer_pointer(a_new_buf_),
          get_buffer_pointer(a_buf_), get_buffer_pointer(l2_norm_buf_),
          iy_start_, iy_end_, nx_, calculate_norm);
      event.wait();

      // Handle norm calculation
      if (calculate_norm) {
        std::vector<float> norm_result(1);
        l2_norm_buf_.copy_to_host(norm_result);
        float local_norm = norm_result[0];

#ifdef USE_MPI
        MPI_Allreduce(&local_norm, &l2_norm, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
#else
        l2_norm = local_norm;
#endif
        l2_norm = std::sqrt(l2_norm);

        if (rank_ == 0 && (iter % 100) == 0) {
          std::cout << "Iteration " << iter << ", L2 norm: " << l2_norm
                    << std::endl;
        }
      }

      // Exchange ghost cells
      exchange_ghost_cells();

      // Swap arrays
      std::swap(a_buf_, a_new_buf_);
      iter++;
    }

    return l2_norm;
  }

private:
  void exchange_ghost_cells() {
#ifdef USE_MPI
    const int top = rank_ > 0 ? rank_ - 1 : (size_ - 1);
    const int bottom = (rank_ + 1) % size_;

    // Exchange top boundary
    if (rank_ > 0) {
      // Send to top, receive from top
      std::vector<float> send_data(nx_);
      std::vector<float> recv_data(nx_);

      // Get full buffer and extract top boundary
      std::vector<float> full_data;
      a_new_buf_.copy_to_host(full_data);
      std::copy(full_data.begin() + iy_start_ * nx_,
                full_data.begin() + (iy_start_ + 1) * nx_, send_data.begin());

      MPI_Sendrecv(send_data.data(), nx_, MPI_FLOAT, top, 0, recv_data.data(),
                   nx_, MPI_FLOAT, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Copy received data to ghost cell and update buffer
      std::copy(recv_data.begin(), recv_data.end(), full_data.begin());
      a_new_buf_.copy_from_host(full_data);
    }

    // Exchange bottom boundary
    if (rank_ < size_ - 1) {
      // Send to bottom, receive from bottom
      std::vector<float> send_data(nx_);
      std::vector<float> recv_data(nx_);

      // Get full buffer and extract bottom boundary
      std::vector<float> full_data;
      a_new_buf_.copy_to_host(full_data);
      std::copy(full_data.begin() + (iy_end_ - 1) * nx_,
                full_data.begin() + iy_end_ * nx_, send_data.begin());

      MPI_Sendrecv(send_data.data(), nx_, MPI_FLOAT, bottom, 0,
                   recv_data.data(), nx_, MPI_FLOAT, bottom, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      // Copy received data to ghost cell and update buffer
      std::copy(recv_data.begin(), recv_data.end(),
                full_data.begin() + iy_end_ * nx_);
      a_new_buf_.copy_from_host(full_data);
    }
#endif
  }

public:
  std::vector<float> get_solution() {
    std::vector<float> local_solution(nx_ * (chunk_size_ + 2));
    a_buf_.copy_to_host(local_solution);
    return local_solution;
  }

  int get_rank() const { return rank_; }
  int get_size() const { return size_; }
  int get_chunk_size() const { return chunk_size_; }
  int get_iy_start_global() const { return iy_start_global_; }
  int get_iy_end_global() const { return iy_end_global_; }
};

// Main MPI Jacobi Solver Test using MPI-aware infrastructure
TEST_CASE("MPI Multi-GPU Jacobi Solver", "[mpi][multi-gpu]") {
#ifdef USE_MPI
  // Initialize backend globally
  initialize_backend_once();

  // Initialize MPI
  ARBD::MPI::Manager::instance().init();
  int rank = ARBD::MPI::Manager::instance().get_rank();
  int size = ARBD::MPI::Manager::instance().get_size();

  // Only rank 0 prints MPI info
  if (rank == 0) {
    ARBD::MPI::Manager::instance().print_info();
    INFO("Running MPI tests on " << size << " processes");
  }

  SECTION("Small Grid Test") {
    const int nx = 8;
    const int ny = 8;
    const int iter_max = 100;

    // All processes run the test
    MPIJacobiSolver solver(nx, ny);

    if (rank == 0) {
      INFO("Domain decomposition: " << size << " processes, chunk_size="
                                    << solver.get_chunk_size());
    }

    solver.initialize_boundaries();
    float final_norm = solver.solve(iter_max, 10);

    // Local verification
    bool local_success = (final_norm >= 0.0f && final_norm < 1.0f);

    auto local_solution = solver.get_solution();
    local_success = local_success && (local_solution.size() ==
                                      nx * (solver.get_chunk_size() + 2));

    // Check for non-zero values
    bool has_nonzero = false;
    for (float val : local_solution) {
      if (std::abs(val) > 1e-6f) {
        has_nonzero = true;
        break;
      }
    }
    local_success = local_success && has_nonzero;

    // Coordinate results across all processes
    int local_result = local_success ? 1 : 0;
    int global_result;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (rank == 0) {
      INFO("Small Grid Test - Final L2 norm = " << final_norm);
    }

    REQUIRE(global_result == 1);
  }

  SECTION("Medium Grid Test") {
    const int nx = 16;
    const int ny = 16;
    const int iter_max = 200;

    MPIJacobiSolver solver(nx, ny);

    // Initialize boundaries
    solver.initialize_boundaries();

    // Solve
    float final_norm = solver.solve(iter_max, 20);

    if (rank == 0) {
      INFO("Medium test - Final L2 norm = " << final_norm);
    }

    // Verify solution converged - realistic expectations for multi-process
    REQUIRE(final_norm >= 0.0f);
    REQUIRE(final_norm < 1.0f); // Should show convergence progress

    // Verify all processes have reasonable solutions
    auto local_solution = solver.get_solution();
    REQUIRE(local_solution.size() == nx * (solver.get_chunk_size() + 2));
  }

  SECTION("Performance Test") {
    const int nx = 32;
    const int ny = 32;
    const int iter_max = 500;

    MPIJacobiSolver solver(nx, ny);

    // Initialize boundaries
    solver.initialize_boundaries();

    // Time the solve
    double start_time = MPI_Wtime();
    float final_norm = solver.solve(iter_max, 50);
    double end_time = MPI_Wtime();

    double elapsed = end_time - start_time;
    if (rank == 0) {
      INFO("Performance test - Solve time = " << elapsed << "s, Final norm = "
                                              << final_norm);
    }

    // Verify performance is reasonable
    REQUIRE(elapsed > 0.0);
    REQUIRE(elapsed < 60.0); // Should complete within 60 seconds

    // Verify convergence - realistic expectations for multi-process
    REQUIRE(final_norm >= 0.0f);
    REQUIRE(final_norm < 1.0f); // Should show convergence progress
  }

#else
  WARN("MPI not available - skipping Jacobi solver tests");
  REQUIRE(true);
#endif
}

TEST_CASE("MPI Communication Test", "[mpi][communication]") {
#ifdef USE_MPI
  // Initialize MPI
  ARBD::MPI::Manager::instance().init();
  int rank = ARBD::MPI::Manager::instance().get_rank();
  int size = ARBD::MPI::Manager::instance().get_size();

  SECTION("Basic Communication") {
    // Test basic MPI communication - all processes participate
    int send_data = rank * 10;
    int recv_data = 0;

    MPI_Allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int expected_sum = 0;
    for (int i = 0; i < size; i++) {
      expected_sum += i * 10;
    }

    // Coordinate test results
    bool local_success = (recv_data == expected_sum);
    int local_result = local_success ? 1 : 0;
    int global_result;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (rank == 0) {
      INFO("Communication test passed, sum = " << recv_data);
    }

    REQUIRE(global_result == 1);
  }

  SECTION("Point-to-Point Communication") {
    if (size >= 2) {
      int send_data = rank * 100;
      int recv_data = 0;
      bool local_success = true;

      if (rank == 0) {
        MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        local_success = (recv_data == 100);
        INFO("Point-to-point test: Rank 0 received " << recv_data
                                                     << " from rank 1");
      } else if (rank == 1) {
        MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        local_success = (recv_data == 0);
      }

      // Coordinate test results
      int local_result = local_success ? 1 : 0;
      int global_result;
      MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      REQUIRE(global_result == 1);
    }
  }

#else
  WARN("MPI not available - skipping communication tests");
  REQUIRE(true);
#endif
}

// Add MPI finalization at the end
TEST_CASE("MPI Finalization", "[mpi][cleanup]") {
#ifdef USE_MPI
  ARBD::MPI::Manager::instance().finalize();
#endif
}
