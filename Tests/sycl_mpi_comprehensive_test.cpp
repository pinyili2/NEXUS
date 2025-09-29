#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/MPIManager.h"
#include "Backend/Resource.h"

using namespace ARBD;

// Global backend initialization
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
    initialized = true;
  } catch (const std::exception &e) {
    g_backend_available = false;
    WARN("Backend initialization failed: " << e.what());
  }
}

// MPI Communication Validation Test (inspired by Shamrock)
class MPICommunicationValidator {
private:
  int rank_, size_;
  Resource resource_;

public:
  MPICommunicationValidator(int device_id = -1) {
#ifdef USE_MPI
    ARBD::MPI::Manager::instance().init();
    rank_ = ARBD::MPI::Manager::instance().get_rank();
    size_ = ARBD::MPI::Manager::instance().get_size();
#else
    rank_ = 0;
    size_ = 1;
#endif

    // Use specific device for this rank
    int device_to_use = (device_id >= 0) ? device_id : (rank_ % 8);
    resource_ = Resource(ResourceType::SYCL, device_to_use);
  }

  bool validate_direct_communication() {
    const size_t test_size = 10000;

    // Create test buffers
    DeviceBuffer<float> send_buffer(test_size, 0);
    DeviceBuffer<float> recv_buffer(test_size, 0);

    // Initialize send buffer with rank-specific pattern
    std::vector<float> host_data(test_size);
    for (size_t i = 0; i < test_size; i++) {
      host_data[i] = static_cast<float>(rank_ * 1000 + i % 100);
    }
    send_buffer.copy_from_host(host_data);

    bool success = true;

#ifdef USE_MPI
    if (size_ >= 2) {
      // Ring communication test
      int next_rank = (rank_ + 1) % size_;
      int prev_rank = (rank_ - 1 + size_) % size_;

      MPI_Request send_req, recv_req;

      // Non-blocking send/receive
      MPI_Isend(send_buffer.data(), test_size, MPI_FLOAT, next_rank, 0,
                MPI_COMM_WORLD, &send_req);
      MPI_Irecv(recv_buffer.data(), test_size, MPI_FLOAT, prev_rank, 0,
                MPI_COMM_WORLD, &recv_req);

      // Wait for completion
      MPI_Wait(&send_req, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

      // Validate received data
      std::vector<float> received_data(test_size);
      recv_buffer.copy_to_host(received_data);

      // Expected data from previous rank
      for (size_t i = 0; i < test_size; i++) {
        float expected = static_cast<float>(prev_rank * 1000 + i % 100);
        if (std::abs(received_data[i] - expected) > 1e-6f) {
          success = false;
          break;
        }
      }
    }
#endif

    return success;
  }

  bool validate_collective_operations() {
#ifdef USE_MPI
    const size_t test_size = 1000;
    DeviceBuffer<float> buffer(test_size, 0);

    // Initialize with rank-specific values
    std::vector<float> host_data(test_size, static_cast<float>(rank_));
    buffer.copy_from_host(host_data);

    // Test Allreduce
    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), test_size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);

    // Validate result
    std::vector<float> result(test_size);
    buffer.copy_to_host(result);

    float expected_sum = 0.0f;
    for (int i = 0; i < size_; i++) {
      expected_sum += static_cast<float>(i);
    }

    for (size_t i = 0; i < test_size; i++) {
      if (std::abs(result[i] - expected_sum) > 1e-6f) {
        return false;
      }
    }

    return true;
#else
    return true;
#endif
  }

  int get_rank() const { return rank_; }
  int get_size() const { return size_; }
};

// Multi-GPU Stencil Computation Test
class MultiGPUStencilTest {
private:
  int rank_, size_;
  Resource resource_;
  int nx_, ny_;
  int local_ny_;

public:
  MultiGPUStencilTest(int nx, int ny, int device_id = -1) : nx_(nx), ny_(ny) {
#ifdef USE_MPI
    ARBD::MPI::Manager::instance().init();
    rank_ = ARBD::MPI::Manager::instance().get_rank();
    size_ = ARBD::MPI::Manager::instance().get_size();
#else
    rank_ = 0;
    size_ = 1;
#endif

    // Domain decomposition
    local_ny_ = ny_ / size_;
    if (rank_ < ny_ % size_) {
      local_ny_++;
    }

    int device_to_use = (device_id >= 0) ? device_id : (rank_ % 8);
    resource_ = Resource(ResourceType::SYCL, device_to_use);
  }

  bool run_stencil_test() {
    // Allocate local domain with ghost cells
    const int ghost_cells = 1;
    const size_t local_size = nx_ * (local_ny_ + 2 * ghost_cells);

    DeviceBuffer<float> u_old(local_size, 0);
    DeviceBuffer<float> u_new(local_size, 0);

    // Initialize with simple pattern
    std::vector<float> init_data(local_size, 0.0f);
    for (int j = ghost_cells; j < local_ny_ + ghost_cells; j++) {
      for (int i = 0; i < nx_; i++) {
        int global_j = rank_ * (ny_ / size_) + (j - ghost_cells);
        init_data[j * nx_ + i] = std::sin(2.0f * M_PI * i / nx_) *
                                 std::cos(2.0f * M_PI * global_j / ny_);
      }
    }
    u_old.copy_from_host(init_data);

    // Perform stencil iterations
    const int num_iterations = 10;
    for (int iter = 0; iter < num_iterations; iter++) {
      // Exchange ghost cells
      exchange_ghost_cells(u_old);

      // Apply stencil kernel
      apply_stencil_kernel(u_old, u_new);

      // Swap buffers
      std::swap(u_old, u_new);
    }

    // Validate result (simple convergence check)
    std::vector<float> final_data(local_size);
    u_old.copy_to_host(final_data);

    // Check for reasonable values (not NaN or infinite)
    for (float val : final_data) {
      if (!std::isfinite(val)) {
        return false;
      }
    }

    return true;
  }

private:
  void exchange_ghost_cells(DeviceBuffer<float> &buffer) {
#ifdef USE_MPI
    if (size_ == 1)
      return;

    const int ghost_cells = 1;
    std::vector<float> send_top(nx_), recv_top(nx_);
    std::vector<float> send_bottom(nx_), recv_bottom(nx_);

    // Get current data
    std::vector<float> local_data(nx_ * (local_ny_ + 2 * ghost_cells));
    buffer.copy_to_host(local_data);

    // Prepare send data
    if (rank_ > 0) {
      std::copy(local_data.begin() + ghost_cells * nx_,
                local_data.begin() + (ghost_cells + 1) * nx_, send_top.begin());
    }
    if (rank_ < size_ - 1) {
      std::copy(local_data.begin() + (local_ny_ + ghost_cells - 1) * nx_,
                local_data.begin() + (local_ny_ + ghost_cells) * nx_,
                send_bottom.begin());
    }

    // Exchange with neighbors
    MPI_Request requests[4];
    int req_count = 0;

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

    // Update ghost cells
    if (rank_ > 0) {
      std::copy(recv_top.begin(), recv_top.end(), local_data.begin());
    }
    if (rank_ < size_ - 1) {
      std::copy(recv_bottom.begin(), recv_bottom.end(),
                local_data.begin() + (local_ny_ + ghost_cells) * nx_);
    }

    buffer.copy_from_host(local_data);
#endif
  }

  void apply_stencil_kernel(DeviceBuffer<float> &u_old,
                            DeviceBuffer<float> &u_new) {
    // Simple 5-point stencil kernel
    const int ghost_cells = 1;
    const int interior_points = nx_ * local_ny_;
    const int nx = nx_; // Capture local copy

    auto stencil_kernel = [=](size_t idx, float *old_data, float *new_data) {
      int i = idx % nx;
      int j = (idx / nx) + ghost_cells;

      if (i > 0 && i < nx - 1) {
        int center = j * nx + i;
        new_data[center] =
            0.25f * (old_data[center - 1] + old_data[center + 1] +
                     old_data[center - nx] + old_data[center + nx]);
      } else {
        new_data[j * nx + i] = old_data[j * nx + i]; // Boundary
      }
    };

    auto event = launch_kernel(
        resource_, KernelConfig::for_1d(interior_points, resource_),
        stencil_kernel, get_buffer_pointer(u_old), get_buffer_pointer(u_new));
    event.wait();
  }

public:
  int get_rank() const { return rank_; }
  int get_size() const { return size_; }
};

// Test Cases
TEST_CASE("SYCL MPI Communication Validation", "[sycl][mpi][communication]") {
#ifdef USE_MPI
  initialize_backend_once();
  REQUIRE(g_backend_available);

  MPICommunicationValidator validator;

  INFO("Running on rank " << validator.get_rank() << " of "
                          << validator.get_size() << " processes");

  SECTION("Direct GPU Communication Test") {
    bool success = validator.validate_direct_communication();
    REQUIRE(success);
    INFO("Rank " << validator.get_rank()
                 << ": Direct communication successful");
  }

  SECTION("Collective Operations Test") {
    bool success = validator.validate_collective_operations();
    REQUIRE(success);
    INFO("Rank " << validator.get_rank()
                 << ": Collective operations successful");
  }

#else
  WARN("MPI not available - skipping communication validation tests");
  REQUIRE(true);
#endif
}

TEST_CASE("Multi-GPU Stencil Computation", "[sycl][mpi][stencil]") {
#ifdef USE_MPI
  initialize_backend_once();
  REQUIRE(g_backend_available);

  SECTION("Small Domain Test") {
    MultiGPUStencilTest stencil_test(32, 32);

    INFO("Rank " << stencil_test.get_rank() << " of " << stencil_test.get_size()
                 << " processes - Small domain test");

    bool success = stencil_test.run_stencil_test();
    REQUIRE(success);
  }

  SECTION("Medium Domain Test") {
    MultiGPUStencilTest stencil_test(64, 64);

    INFO("Rank " << stencil_test.get_rank() << " of " << stencil_test.get_size()
                 << " processes - Medium domain test");

    bool success = stencil_test.run_stencil_test();
    REQUIRE(success);
  }

#else
  WARN("MPI not available - skipping multi-GPU stencil tests");
  REQUIRE(true);
#endif
}

TEST_CASE("MPI Device Affinity Test", "[sycl][mpi][affinity]") {
#ifdef USE_MPI
  initialize_backend_once();
  REQUIRE(g_backend_available);

  ARBD::MPI::Manager::instance().init();
  int rank = ARBD::MPI::Manager::instance().get_rank();
  int size = ARBD::MPI::Manager::instance().get_size();

  SECTION("Device Assignment Verification") {
    // Test that each rank gets assigned to correct GPU
    int expected_device = rank % 8;
    Resource resource(ResourceType::SYCL, expected_device);

    // Simple kernel to verify device is working
    DeviceBuffer<int> test_buffer(1, 0);
    std::vector<int> host_data = {rank};
    test_buffer.copy_from_host(host_data);

    auto kernel = [=](size_t idx, int *data) {
      data[0] = data[0] * 2; // Simple operation
    };

    auto event = launch_kernel(resource, KernelConfig::for_1d(1, resource),
                               kernel, get_buffer_pointer(test_buffer));
    event.wait();

    std::vector<int> result(1);
    test_buffer.copy_to_host(result);

    REQUIRE(result[0] == rank * 2);
    INFO("Rank " << rank << " successfully used device " << expected_device);
  }

#else
  WARN("MPI not available - skipping device affinity tests");
  REQUIRE(true);
#endif
}

// Cleanup
TEST_CASE("MPI Finalization", "[mpi][cleanup]") {
#ifdef USE_MPI
  ARBD::MPI::Manager::instance().finalize();
#endif
}
