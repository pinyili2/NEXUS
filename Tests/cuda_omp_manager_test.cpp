#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

using namespace ARBD;

// ============================================================================
// CUDA Manager OpenMP Tests
// ============================================================================

#ifdef USE_CUDA

TEST_CASE("CUDA Manager OpenMP Initialization", "[cuda][omp][manager]") {

  SECTION("Basic Manager Initialization") {
    // Test basic initialization
    CUDA::Manager::init();

    REQUIRE(CUDA::Manager::device_count() > 0);

    INFO("Found " << CUDA::Manager::device_count() << " CUDA devices");

    for (int i = 0; i < CUDA::Manager::device_count(); ++i) {
      auto props = CUDA::Manager::get_device_properties(i);
      INFO("Device " << i << ": " << props.name
           << " (SM " << props.major << "." << props.minor
           << ", " << props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB)");
    }
  }

  SECTION("Single Rank OpenMP Configuration") {
    CUDA::Manager::init();

    int num_devices = CUDA::Manager::device_count();
    INFO("Testing single rank configuration with " << num_devices << " devices");

#ifdef _OPENMP
    // Test with different numbers of OpenMP threads
    for (int threads : {1, 2, 4}) {
      if (threads > omp_get_max_threads()) continue;

      INFO("Testing with " << threads << " OpenMP threads");

      // Initialize for single rank with specified thread count
      CUDA::Manager::init_for_rank(0, 1, threads, true);

      REQUIRE(CUDA::Manager::get_rank_devices().size() > 0);
      REQUIRE(CUDA::Manager::get_rank_devices().size() <= static_cast<size_t>(num_devices));

      // Test OpenMP parallel region
#pragma omp parallel num_threads(threads)
      {
        int thread_id = omp_get_thread_num();
        int gpu_id = CUDA::Manager::get_thread_gpu();

        INFO("Thread " << thread_id << " assigned to GPU " << gpu_id);

        // Verify GPU assignment is valid
        REQUIRE(gpu_id >= 0);
        REQUIRE(gpu_id < num_devices);

        // Test thread-local initialization
        CUDA::Manager::init_for_omp_thread();
      }
    }
#else
    INFO("OpenMP not available, testing single-threaded mode");
    CUDA::Manager::init_for_rank(0, 1, 1, true);
    REQUIRE(CUDA::Manager::get_rank_devices().size() > 0);
#endif
  }

  SECTION("Multi-Rank Configuration Simulation") {
    CUDA::Manager::init();

    int num_devices = CUDA::Manager::device_count();
    if (num_devices < 2) {
      WARN("Need at least 2 CUDA devices for multi-rank testing");
      return;
    }

    INFO("Testing multi-rank configuration with " << num_devices << " devices");

    // Simulate different ranks
    for (int rank = 0; rank < std::min(num_devices, 4); ++rank) {
      INFO("Testing rank " << rank);

      CUDA::Manager::init_for_rank(rank, std::min(num_devices, 4), 2, true);

      auto rank_devices = CUDA::Manager::get_rank_devices();
      REQUIRE(rank_devices.size() > 0);

      INFO("Rank " << rank << " assigned to " << rank_devices.size() << " devices");

      // Verify assigned devices are valid
      for (int gpu_id : rank_devices) {
        REQUIRE(gpu_id >= 0);
        REQUIRE(gpu_id < num_devices);
        INFO("  - GPU " << gpu_id);
      }

      REQUIRE(CUDA::Manager::is_multi_rank());
    }
  }

  SECTION("GPU Affinity Strategies") {
    CUDA::Manager::init();

    int num_devices = CUDA::Manager::device_count();
    INFO("Testing GPU affinity strategies with " << num_devices << " devices");

#ifdef _OPENMP
    int num_threads = std::min(4, omp_get_max_threads());

    for (const std::string& strategy : {"block", "cyclic"}) {
      INFO("Testing " << strategy << " affinity strategy");

      CUDA::Manager::init_for_rank(0, 1, num_threads, false);
      CUDA::Manager::set_omp_gpu_affinity(strategy);

      // Test that all threads get valid GPU assignments
#pragma omp parallel num_threads(num_threads)
      {
        int thread_id = omp_get_thread_num();
        int gpu_id = CUDA::Manager::get_thread_gpu();

        INFO("Strategy " << strategy << ": Thread " << thread_id
             << " -> GPU " << gpu_id);

        REQUIRE(gpu_id >= 0);
        REQUIRE(gpu_id < num_devices);

        CUDA::Manager::init_for_omp_thread();
      }
    }
#endif
  }

  SECTION("Peer-to-Peer Access") {
    CUDA::Manager::init();

    int num_devices = CUDA::Manager::device_count();
    if (num_devices < 2) {
      WARN("Need at least 2 CUDA devices for P2P testing");
      return;
    }

    INFO("Testing peer-to-peer access between " << num_devices << " devices");

    // Test P2P capabilities
    for (int i = 0; i < num_devices; ++i) {
      for (int j = 0; j < num_devices; ++j) {
        if (i != j) {
          bool can_access = CUDA::Manager::can_access_peer(i, j);
          INFO("P2P access from device " << i << " to device " << j
               << ": " << (can_access ? "YES" : "NO"));
        }
      }
    }

    // Enable P2P access
    try {
      CUDA::Manager::enable_peer_access();
      INFO("P2P access enabled successfully");
    } catch (const std::exception& e) {
      WARN("P2P access enablement failed: " << e.what());
    }
  }

  SECTION("Performance and Scalability") {
    CUDA::Manager::init();

    int num_devices = CUDA::Manager::device_count();
    INFO("Testing performance with " << num_devices << " devices");

#ifdef _OPENMP
    // Test different thread counts and measure setup time
    for (int threads : {1, 2, 4, 8}) {
      if (threads > omp_get_max_threads()) continue;

      auto start_time = std::chrono::high_resolution_clock::now();

      CUDA::Manager::init_for_rank(0, 1, threads, false);

#pragma omp parallel num_threads(threads)
      {
        CUDA::Manager::init_for_omp_thread();
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);

      INFO("Setup time for " << threads << " threads: "
           << duration.count() << " μs");

      REQUIRE(duration.count() < 100000); // Should be less than 100ms
    }
#endif
  }

  SECTION("Error Handling and Edge Cases") {
    CUDA::Manager::init();

    // Test invalid device access
    REQUIRE_FALSE(CUDA::Manager::can_access_peer(-1, 0));
    REQUIRE_FALSE(CUDA::Manager::can_access_peer(0, 999));

    // Test with invalid rank parameters
    try {
      CUDA::Manager::init_for_rank(-1, 1, 1, false);
      // Should not throw but handle gracefully
    } catch (const std::exception& e) {
      INFO("Handled invalid rank gracefully: " << e.what());
    }

    // Test with more ranks than devices
    int num_devices = CUDA::Manager::device_count();
    if (num_devices > 0) {
      CUDA::Manager::init_for_rank(0, num_devices + 2, 1, true);
      auto rank_devices = CUDA::Manager::get_rank_devices();
      REQUIRE(rank_devices.size() > 0); // Should still assign some device
    }
  }
}

TEST_CASE("CUDA Manager Memory and Resource Management", "[cuda][manager][memory]") {
  CUDA::Manager::init();

  int num_devices = CUDA::Manager::device_count();
  REQUIRE(num_devices > 0);

  SECTION("Device Properties Verification") {
    for (int i = 0; i < num_devices; ++i) {
      auto props = CUDA::Manager::get_device_properties(i);

      // Verify basic properties
      REQUIRE(props.major >= 1); // At least SM 1.x
      REQUIRE(props.totalGlobalMem > 0);
      REQUIRE(props.multiProcessorCount > 0);
      REQUIRE(props.maxThreadsPerBlock > 0);

      INFO("Device " << i << " detailed properties:");
      INFO("  Name: " << props.name);
      INFO("  Compute Capability: SM " << props.major << "." << props.minor);
      INFO("  Global Memory: " << props.totalGlobalMem / (1024*1024) << " MB");
      INFO("  Multiprocessors: " << props.multiProcessorCount);
      INFO("  Max Threads per Block: " << props.maxThreadsPerBlock);
      INFO("  Max Block Dimensions: "
           << props.maxThreadsDim[0] << "x"
           << props.maxThreadsDim[1] << "x"
           << props.maxThreadsDim[2]);
      INFO("  Max Grid Dimensions: "
           << props.maxGridSize[0] << "x"
           << props.maxGridSize[1] << "x"
           << props.maxGridSize[2]);
    }
  }

  SECTION("Concurrent Multi-Device Operations") {
    if (num_devices < 2) {
      WARN("Need at least 2 devices for concurrent operations test");
      return;
    }

#ifdef _OPENMP
    const int test_devices = std::min(num_devices, 4);
    CUDA::Manager::init_for_rank(0, 1, test_devices, false);

    std::vector<bool> success(test_devices, false);

#pragma omp parallel for num_threads(test_devices)
    for (int i = 0; i < test_devices; ++i) {
      try {
        CUDA::Manager::init_for_omp_thread();

        int assigned_gpu = CUDA::Manager::get_thread_gpu();
        auto props = CUDA::Manager::get_device_properties(assigned_gpu);

        // Verify we can access device properties concurrently
        success[i] = (props.totalGlobalMem > 0);

      } catch (const std::exception& e) {
        WARN("Concurrent operation failed on device " << i << ": " << e.what());
        success[i] = false;
      }
    }

    // Verify all operations succeeded
    for (int i = 0; i < test_devices; ++i) {
      REQUIRE(success[i]);
    }
#endif
  }

  SECTION("Stress Test - Rapid Initialization/Finalization") {
    const int iterations = 10;

    for (int iter = 0; iter < iterations; ++iter) {
      try {
        CUDA::Manager::init_for_rank(0, 1, 2, false);

#ifdef _OPENMP
#pragma omp parallel num_threads(2)
        {
          CUDA::Manager::init_for_omp_thread();
          int gpu_id = CUDA::Manager::get_thread_gpu();
          REQUIRE(gpu_id >= 0);
        }
#endif

        // Brief operation to ensure everything is working
        auto props = CUDA::Manager::get_device_properties(0);
        REQUIRE(props.totalGlobalMem > 0);

      } catch (const std::exception& e) {
        FAIL("Stress test iteration " << iter << " failed: " << e.what());
      }
    }

    INFO("Completed " << iterations << " stress test iterations successfully");
  }
}

#endif // USE_CUDA
