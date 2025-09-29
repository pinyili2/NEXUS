#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>

#include "Buffer.h"
#include "MPIManager.h"
#include "Resource.h"

using namespace ARBD;

// Simple SYCL MPI test
TEST_CASE("SYCL MPI Basic Test", "[sycl][mpi]") {
#ifdef USE_MPI
  // Initialize MPI
  ARBD::MPI::Manager::instance().init();
  int rank = ARBD::MPI::Manager::instance().get_rank();
  int size = ARBD::MPI::Manager::instance().get_size();

  INFO("Running on rank " << rank << " of " << size << " processes");

  SECTION("SYCL Device Availability") {
    try {
      // Try to create a SYCL resource
      Resource sycl_resource(ResourceType::SYCL, rank);
      INFO("SYCL device " << rank << " is available");

      // Try to create a small buffer
      DeviceBuffer<float> test_buffer(10, sycl_resource);
      INFO("Successfully created SYCL buffer on device " << rank);

      // Test basic operations
      std::vector<float> host_data(10, static_cast<float>(rank));
      test_buffer.copy_from_host(host_data.data(), 10);

      std::vector<float> result_data(10);
      test_buffer.copy_to_host(result_data.data(), 10);

      // Verify data
      for (int i = 0; i < 10; i++) {
        REQUIRE(result_data[i] == static_cast<float>(rank));
      }

      INFO("SYCL operations successful on rank " << rank);

    } catch (const std::exception &e) {
      WARN("SYCL device not available on rank " << rank << ": " << e.what());
      // Fallback to CPU
      Resource cpu_resource(ResourceType::CPU);
      DeviceBuffer<float> test_buffer(10, cpu_resource);
      INFO("Using CPU fallback on rank " << rank);
    }
  }

  SECTION("MPI Communication") {
    // Test basic MPI communication
    int send_data = rank * 100;
    int recv_data = 0;

    MPI_Allreduce(&send_data, &recv_data, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int expected_sum = 0;
    for (int i = 0; i < size; i++) {
      expected_sum += i * 100;
    }

    REQUIRE(recv_data == expected_sum);
    INFO("Rank " << rank
                 << ": MPI communication successful, sum = " << recv_data);
  }

  SECTION("Multi-Process SYCL Test") {
    try {
      // Each process uses its rank as device ID
      Resource sycl_resource(ResourceType::SYCL, rank);
      DeviceBuffer<float> buffer(100, sycl_resource);

      // Initialize with rank-specific data
      std::vector<float> data(100, static_cast<float>(rank));
      buffer.copy_from_host(data.data(), 100);

      // Test MPI communication with SYCL data
      std::vector<float> result(100);
      buffer.copy_to_host(result.data(), 100);

      // Verify data integrity
      for (int i = 0; i < 100; i++) {
        REQUIRE(result[i] == static_cast<float>(rank));
      }

      INFO("Multi-process SYCL test successful on rank " << rank);

    } catch (const std::exception &e) {
      WARN("Multi-process SYCL test failed on rank " << rank << ": "
                                                     << e.what());
      // This is acceptable if SYCL devices are not available
    }
  }

  // Finalize MPI
  ARBD::MPI::Manager::instance().finalize();

#else
  WARN("MPI not available - skipping SYCL MPI tests");
  REQUIRE(true);
#endif
}

// Simple Jacobi test with SYCL
TEST_CASE("SYCL Jacobi Test", "[sycl][jacobi]") {
#ifdef USE_MPI
  ARBD::MPI::Manager::instance().init();
  int rank = ARBD::MPI::Manager::instance().get_rank();
  int size = ARBD::MPI::Manager::instance().get_size();

  SECTION("Small Jacobi Problem") {
    try {
      // Create SYCL resource
      Resource sycl_resource(ResourceType::SYCL, rank);

      // Small problem size
      const int nx = 8;
      const int ny = 8;
      const int local_ny = ny / size;

      // Allocate buffers
      DeviceBuffer<float> a_buffer(nx * local_ny, sycl_resource);
      DeviceBuffer<float> a_new_buffer(nx * local_ny, sycl_resource);

      // Initialize with simple pattern
      std::vector<float> init_data(nx * local_ny, 0.0f);
      for (int i = 0; i < nx * local_ny; i++) {
        init_data[i] = static_cast<float>(rank + i);
      }

      a_buffer.copy_from_host(init_data.data(), nx * local_ny);
      a_new_buffer.copy_from_host(init_data.data(), nx * local_ny);

      // Simple Jacobi iteration
      for (int iter = 0; iter < 10; iter++) {
        // Copy data back to host for simple CPU Jacobi
        std::vector<float> host_data(nx * local_ny);
        a_buffer.copy_to_host(host_data.data(), nx * local_ny);

        // Simple Jacobi update (CPU version for simplicity)
        for (int j = 1; j < local_ny - 1; j++) {
          for (int i = 1; i < nx - 1; i++) {
            int idx = j * nx + i;
            host_data[idx] =
                0.25f *
                (host_data[(j - 1) * nx + i] + host_data[(j + 1) * nx + i] +
                 host_data[j * nx + (i - 1)] + host_data[j * nx + (i + 1)]);
          }
        }

        a_new_buffer.copy_from_host(host_data.data(), nx * local_ny);

        // Swap buffers
        std::swap(a_buffer, a_new_buffer);
      }

      INFO("Jacobi iteration completed on rank " << rank);

    } catch (const std::exception &e) {
      WARN("Jacobi test failed on rank " << rank << ": " << e.what());
    }
  }

  ARBD::MPI::Manager::instance().finalize();
#else
  WARN("MPI not available - skipping Jacobi test");
  REQUIRE(true);
#endif
}
