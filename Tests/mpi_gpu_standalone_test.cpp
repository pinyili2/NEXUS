/**
 * @file mpi_gpu_standalone_test.cpp
 * @brief Standalone MPI GPU test that doesn't use Catch2 framework
 * @details This test is designed to run with mpirun where only rank 0 handles
 * output
 */

#include "catch_boiler.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef USE_MPI
#include "Backend/MPIManager.h"
#include <mpi.h>
#else
#error "This test requires MPI support"
#endif

using namespace ARBD;

// Global backend initialization flag
static bool backend_initialized = false;

void initialize_backend_once() {
  if (!backend_initialized) {
    Tests::TestBackendManager::getInstance();
    backend_initialized = true;
  }
}

// Simple MPI Jacobi solver for testing
class MPIJacobiSolver {
private:
  int nx_, ny_;
  int rank_, size_;
  int chunk_size_;
  int iy_start_global_, iy_end_global_;
  Resource resource_;
  DeviceBuffer<float> *u_, *u_new_;

public:
  MPIJacobiSolver(int nx, int ny, int device_id = -1) : nx_(nx), ny_(ny) {
    rank_ = ARBD::MPI::Manager::instance().get_rank();
    size_ = ARBD::MPI::Manager::instance().get_size();

    // Device selection - when using CUDA_VISIBLE_DEVICES per process,
    // each process sees only one GPU as device 0
    int device_to_use = (device_id >= 0) ? device_id : 0;
    resource_ = Resource(ResourceType::SYCL, device_to_use);

    // Domain decomposition
    int interior_points = ny - 2;
    int chunk_size_low = interior_points / size_;
    int remainder = interior_points % size_;

    if (rank_ < remainder) {
      chunk_size_ = chunk_size_low + 1;
    } else {
      chunk_size_ = chunk_size_low;
    }

    iy_start_global_ = 1; // Start from 1 (after boundary)
    for (int i = 0; i < rank_; i++) {
      if (i < remainder) {
        iy_start_global_ += chunk_size_low + 1;
      } else {
        iy_start_global_ += chunk_size_low;
      }
    }
    iy_end_global_ = iy_start_global_ + chunk_size_ - 1;

    // Allocate local arrays with ghost cells
    int local_height = chunk_size_ + 2; // +2 for ghost cells
    size_t total_size = nx_ * local_height;

    u_ = new DeviceBuffer<float>(total_size, resource_);
    u_new_ = new DeviceBuffer<float>(total_size, resource_);
  }

  ~MPIJacobiSolver() {
    delete u_;
    delete u_new_;
  }

  void initialize_boundaries() {
    std::vector<float> host_data(nx_ * (chunk_size_ + 2), 0.0f);

    // Set boundary conditions only if this rank owns the actual boundaries
    if (rank_ == 0) {
      // Top boundary (y=0)
      for (int ix = 0; ix < nx_; ix++) {
        host_data[ix] = 1.0f;
      }
    }
    if (rank_ == size_ - 1) {
      // Bottom boundary (y=ny-1)
      int local_height = chunk_size_ + 2;
      for (int ix = 0; ix < nx_; ix++) {
        host_data[ix + (local_height - 1) * nx_] = 1.0f;
      }
    }

    u_->copy_from_host(host_data);
    u_new_->copy_from_host(host_data);
  }

  float solve(int iter_max, int print_freq = 0) {
    float l2_norm = 0.0f;

    for (int iter = 1; iter <= iter_max; iter++) {
      // Exchange ghost cells
      exchange_ghost_cells();

      // Jacobi update kernel
      update_jacobi();

      // Compute L2 norm every print_freq iterations
      if (print_freq > 0 && iter % print_freq == 0) {
        l2_norm = compute_l2_norm();
        if (rank_ == 0 && l2_norm < 1e-6f) {
          break; // Converged
        }
      }

      // Swap arrays
      std::swap(u_, u_new_);
    }

    return compute_l2_norm();
  }

private:
  void exchange_ghost_cells() {
    if (size_ == 1)
      return;

    int local_height = chunk_size_ + 2;
    std::vector<float> send_top(nx_), send_bottom(nx_);
    std::vector<float> recv_top(nx_), recv_bottom(nx_);

    // Extract boundary rows - get full buffer and extract portions
    std::vector<float> full_data;
    u_->copy_to_host(full_data);

    // Extract first interior row
    std::copy(full_data.begin() + 1 * nx_, full_data.begin() + 2 * nx_,
              send_top.begin());

    // Extract last interior row
    std::copy(full_data.begin() + (local_height - 2) * nx_,
              full_data.begin() + (local_height - 1) * nx_,
              send_bottom.begin());

    MPI_Request requests[4];
    int req_count = 0;

    // Send/receive with neighbors
    if (rank_ > 0) {
      MPI_Isend(send_top.data(), nx_, MPI_FLOAT, rank_ - 1, 0, MPI_COMM_WORLD,
                &requests[req_count++]);
      MPI_Irecv(recv_top.data(), nx_, MPI_FLOAT, rank_ - 1, 1, MPI_COMM_WORLD,
                &requests[req_count++]);
    }
    if (rank_ < size_ - 1) {
      MPI_Isend(send_bottom.data(), nx_, MPI_FLOAT, rank_ + 1, 1,
                MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(recv_bottom.data(), nx_, MPI_FLOAT, rank_ + 1, 0,
                MPI_COMM_WORLD, &requests[req_count++]);
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // Update ghost cells - modify the full buffer
    if (rank_ > 0) {
      std::copy(recv_top.begin(), recv_top.end(),
                full_data.begin()); // Top ghost
    }
    if (rank_ < size_ - 1) {
      std::copy(recv_bottom.begin(), recv_bottom.end(),
                full_data.begin() + (local_height - 1) * nx_); // Bottom ghost
    }

    // Copy modified data back to device
    u_->copy_from_host(full_data);
  }

  void update_jacobi() {
    void *stream_ptr = resource_.get_stream();
    auto &queue = *static_cast<sycl::queue *>(stream_ptr);

    const int nx = nx_;
    const int local_height = chunk_size_ + 2;

    queue
        .submit([&](sycl::handler &h) {
          auto u_ptr = u_->data();
          auto u_new_ptr = u_new_->data();

          h.parallel_for(
              sycl::range<2>(chunk_size_, nx - 2), [=](sycl::id<2> idx) {
                int iy = idx[0] + 1; // Local index (skip ghost cells)
                int ix = idx[1] + 1; // Skip boundary

                int center = iy * nx + ix;
                int top = (iy - 1) * nx + ix;
                int bottom = (iy + 1) * nx + ix;
                int left = iy * nx + (ix - 1);
                int right = iy * nx + (ix + 1);

                u_new_ptr[center] = 0.25f * (u_ptr[top] + u_ptr[bottom] +
                                             u_ptr[left] + u_ptr[right]);
              });
        })
        .wait();
  }

  float compute_l2_norm() {
    void *stream_ptr = resource_.get_stream();
    auto &queue = *static_cast<sycl::queue *>(stream_ptr);

    const int nx = nx_;
    DeviceBuffer<float> local_sum(1, resource_);
    std::vector<float> zero_val = {0.0f};
    local_sum.copy_from_host(zero_val);

    queue
        .submit([&](sycl::handler &h) {
          auto u_ptr = u_->data();
          auto u_new_ptr = u_new_->data();
          auto sum_ptr = local_sum.data();

          h.parallel_for(sycl::range<2>(chunk_size_, nx - 2),
                         [=](sycl::id<2> idx) {
                           int iy = idx[0] + 1;
                           int ix = idx[1] + 1;
                           int center = iy * nx + ix;
                           float diff = u_new_ptr[center] - u_ptr[center];
                           sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device>
                               atomic_sum(*sum_ptr);
                           atomic_sum += diff * diff;
                         });
        })
        .wait();

    std::vector<float> sum_vec;
    local_sum.copy_to_host(sum_vec);
    float local_norm_sq = sum_vec[0];

    float global_norm_sq;
    MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);

    return std::sqrt(global_norm_sq);
  }

public:
  int get_chunk_size() const { return chunk_size_; }
  int get_iy_start_global() const { return iy_start_global_; }
  int get_iy_end_global() const { return iy_end_global_; }
};

// Main test runner
int main(int argc, char *argv[]) {
  try {
    // Initialize backend
    initialize_backend_once();

    // Initialize our MPI manager (this will call MPI_Init internally)
    ARBD::MPI::Manager::instance().init();

    int rank = ARBD::MPI::Manager::instance().get_rank();
    int size = ARBD::MPI::Manager::instance().get_size();

    if (rank == 0) {
      std::cout << "\n=== Standalone MPI GPU Test ===" << std::endl;
      ARBD::MPI::Manager::instance().print_info();
    }

    bool all_tests_passed = true;

    // Test 1: Small Grid Test
    {
      const int nx = 8;
      const int ny = 8;
      const int iter_max = 100;

      if (rank == 0) {
        std::cout << "\nTest 1: Small Grid Test (" << nx << "x" << ny
                  << " grid)" << std::endl;
      }

      MPIJacobiSolver solver(nx, ny);
      solver.initialize_boundaries();
      float final_norm = solver.solve(iter_max, 10);

      bool test1_passed = (final_norm >= 0.0f && final_norm < 1.0f);

      int local_result = test1_passed ? 1 : 0;
      int global_result;
      MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      if (rank == 0) {
        std::cout << "  Final L2 norm = " << final_norm << std::endl;
        std::cout << "  Result: " << (global_result ? "PASSED" : "FAILED")
                  << std::endl;
        all_tests_passed &= (global_result == 1);
      }
    }

    // Test 2: Communication Test
    {
      if (rank == 0) {
        std::cout << "\nTest 2: Basic MPI Communication" << std::endl;
      }

      int send_data = rank * 10;
      int recv_data = 0;
      MPI_Allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

      int expected_sum = 0;
      for (int i = 0; i < size; i++) {
        expected_sum += i * 10;
      }

      bool test2_passed = (recv_data == expected_sum);

      int local_result = test2_passed ? 1 : 0;
      int global_result;
      MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      if (rank == 0) {
        std::cout << "  Sum result = " << recv_data << " (expected "
                  << expected_sum << ")" << std::endl;
        std::cout << "  Result: " << (global_result ? "PASSED" : "FAILED")
                  << std::endl;
        all_tests_passed &= (global_result == 1);
      }
    }

    // Final result
    if (rank == 0) {
      std::cout << "\n=== Final Result ===" << std::endl;
      std::cout << "All tests: " << (all_tests_passed ? "PASSED" : "FAILED")
                << std::endl;
      std::cout << "===================" << std::endl;
    }

    // Cleanup
    Tests::cleanup();
    ARBD::MPI::Manager::instance().finalize();

    return all_tests_passed ? 0 : 1;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    ARBD::MPI::Manager::instance().finalize();
    return 1;
  }
}
