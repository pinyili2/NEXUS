#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

using namespace ARBD;

// ============================================================================
// SYCL Manager OpenMP Tests
// ============================================================================

#ifdef USE_SYCL

TEST_CASE("SYCL Manager OpenMP Initialization", "[sycl][omp][manager]") {

  SECTION("Basic Manager Initialization") {
    // Test basic initialization
    SYCL::Manager::init();

    REQUIRE(SYCL::Manager::device_count() > 0);

    auto devices = SYCL::Manager::get_all_devices();
    REQUIRE(devices.size() > 0);

    INFO("Found " << devices.size() << " SYCL devices");

    SYCL::Manager::finalize();
  }

  SECTION("Single Rank OpenMP Configuration") {
    SYCL::Manager::init();

    int num_devices = static_cast<int>(SYCL::Manager::device_count());
    INFO("Testing with " << num_devices << " SYCL devices");

    // Test single rank initialization with different thread counts
    std::vector<int> thread_configs = {1, 2, 4};

    for (int threads : thread_configs) {
      if (threads > 8) continue; // Limit test threads

      INFO("Testing with " << threads << " OpenMP threads");

      SYCL::Manager::init_for_rank(0, 1, threads, true);

      // Verify OpenMP configuration
      REQUIRE(SYCL::Manager::is_omp_enabled() == (threads > 1));

      // Verify rank devices assignment
      auto rank_devices = SYCL::Manager::get_rank_devices();
      REQUIRE(rank_devices.size() == static_cast<size_t>(num_devices));

      // Test thread device mapping
      for (int t = 0; t < threads; ++t) {
#ifdef _OPENMP
        int original_thread = omp_get_thread_num();
        omp_set_num_threads(threads);

        #pragma omp parallel num_threads(threads)
        {
          if (omp_get_thread_num() == t) {
            SYCL::Manager::init_for_omp_thread();
            int assigned_device = SYCL::Manager::get_thread_device();

            // Verify device assignment is valid
            REQUIRE(assigned_device >= 0);
            REQUIRE(assigned_device < num_devices);

            INFO("Thread " << t << " assigned to device " << assigned_device);
          }
        }
#endif
      }
    }

    SYCL::Manager::finalize();
  }

  SECTION("Multi-Rank Configuration") {
    SYCL::Manager::init();

    int num_devices = static_cast<int>(SYCL::Manager::device_count());

    if (num_devices >= 2) {
      // Test multi-rank scenario with 2 ranks
      int ranks_per_node = 2;
      int threads_per_rank = 2;

      for (int rank = 0; rank < ranks_per_node; ++rank) {
        INFO("Testing rank " << rank << " of " << ranks_per_node);

        SYCL::Manager::init_for_rank(rank, ranks_per_node, threads_per_rank, true);

        auto rank_devices = SYCL::Manager::get_rank_devices();
        REQUIRE(rank_devices.size() > 0);

        // Verify each rank gets different devices (or shares if oversubscribed)
        for (int device_id : rank_devices) {
          REQUIRE(device_id >= 0);
          REQUIRE(device_id < num_devices);
        }

        INFO("Rank " << rank << " assigned " << rank_devices.size() << " devices");
      }
    } else {
      INFO("Skipping multi-rank test - only " << num_devices << " devices available");
    }

    SYCL::Manager::finalize();
  }

  SECTION("Device Affinity Strategies") {
    SYCL::Manager::init();

    int num_devices = static_cast<int>(SYCL::Manager::device_count());
    int threads = 4;

    SYCL::Manager::init_for_rank(0, 1, threads, false);

    // Test block affinity strategy
    SYCL::Manager::set_omp_device_affinity("block");

#ifdef _OPENMP
    std::vector<int> block_assignments(threads);
    #pragma omp parallel num_threads(threads)
    {
      int thread_id = omp_get_thread_num();
      SYCL::Manager::init_for_omp_thread();
      block_assignments[thread_id] = SYCL::Manager::get_thread_device();
    }

    // Test cyclic affinity strategy
    SYCL::Manager::set_omp_device_affinity("cyclic");

    std::vector<int> cyclic_assignments(threads);
    #pragma omp parallel num_threads(threads)
    {
      int thread_id = omp_get_thread_num();
      SYCL::Manager::init_for_omp_thread();
      cyclic_assignments[thread_id] = SYCL::Manager::get_thread_device();
    }

    // Verify assignments are valid
    for (int t = 0; t < threads; ++t) {
      REQUIRE(block_assignments[t] >= 0);
      REQUIRE(block_assignments[t] < num_devices);
      REQUIRE(cyclic_assignments[t] >= 0);
      REQUIRE(cyclic_assignments[t] < num_devices);

      INFO("Thread " << t << " - Block: " << block_assignments[t]
           << ", Cyclic: " << cyclic_assignments[t]);
    }

    // If we have multiple devices, block and cyclic should potentially differ
    if (num_devices > 1 && threads > num_devices) {
      bool strategies_differ = false;
      for (int t = 0; t < threads; ++t) {
        if (block_assignments[t] != cyclic_assignments[t]) {
          strategies_differ = true;
          break;
        }
      }
      INFO("Block and cyclic strategies " << (strategies_differ ? "differ" : "are same"));
    }
#else
    INFO("OpenMP not available - skipping thread affinity tests");
#endif

    SYCL::Manager::finalize();
  }

  SECTION("Error Handling") {
    SYCL::Manager::init();

    // Test invalid affinity strategy
    SYCL::Manager::set_omp_device_affinity("invalid");

    // Test thread device access without initialization
    int device = SYCL::Manager::get_thread_device();
    REQUIRE(device >= 0); // Should return 0 as fallback

    SYCL::Manager::finalize();
  }
}

TEST_CASE("SYCL Manager OpenMP Performance Test", "[sycl][omp][performance]") {
#ifdef USE_SYCL

  SECTION("Thread Scaling Performance") {
    SYCL::Manager::init();

    const int num_iterations = 1000;
    std::vector<int> thread_configs = {1, 2, 4};
    std::vector<double> times;

    for (int threads : thread_configs) {
      if (threads > 8) continue;

      SYCL::Manager::init_for_rank(0, 1, threads, false);

      auto start = std::chrono::high_resolution_clock::now();

      // Simple parallel work simulation
#ifdef _OPENMP
      #pragma omp parallel for num_threads(threads)
      for (int i = 0; i < num_iterations; ++i) {
        SYCL::Manager::init_for_omp_thread();
        int device = SYCL::Manager::get_thread_device();

        // Simulate some work with device
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
      REQUIRE(elapsed < 10.0); // Should complete reasonably quickly
    }

    // Check that we're not getting dramatically worse performance with more threads
    if (times.size() >= 2) {
      double single_thread = times[0];
      for (size_t i = 1; i < times.size(); ++i) {
        double ratio = times[i] / single_thread;
        INFO("Thread scaling ratio " << thread_configs[i] << " threads: " << ratio);
        REQUIRE(ratio < 10.0); // Shouldn't be more than 10x slower
      }
    }

    SYCL::Manager::finalize();
  }

#else
  INFO("SYCL not available - skipping OpenMP performance tests");
#endif
}

#else // !USE_SYCL

TEST_CASE("SYCL Manager OpenMP - Not Available", "[sycl][omp]") {
  INFO("SYCL not enabled in build - skipping SYCL Manager OpenMP tests");
  REQUIRE(true); // Test passes when SYCL not available
}

#endif // USE_SYCL
