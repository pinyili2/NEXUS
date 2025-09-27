#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "catch_boiler.h"
#include <chrono>
#include <future>
#include <memory>
#include <numeric>
#include <vector>

using namespace ARBD;

// Include the Catch2 test runner
DEF_RUN_TRIAL

// Test constants
constexpr size_t PINNED_BUFFER_SIZE = 1024;
constexpr size_t UNIFIED_BUFFER_SIZE = 2048;
constexpr size_t LARGE_BUFFER_SIZE = 10000;
constexpr size_t STRESS_BUFFER_SIZE = 1000;

// Helper function to skip buffer transfer tests on macOS with SYCL
inline void skip_if_sycl_on_macos() {
#ifdef USE_SYCL
#ifdef __APPLE__
  SKIP("Skipping buffer transfer tests on macOS - unified memory architecture");
#endif
#endif
}

// ============================================================================
// Backend Initialization Fixture
// ============================================================================

struct BackendInitFixture {
  static std::atomic<bool> initialized_;

  BackendInitFixture() {
    // Use atomic flag to ensure initialization only happens once
    bool expected = false;
    if (initialized_.compare_exchange_strong(expected, true)) {
      // Only initialize once across all test fixtures
      try {
#ifdef USE_CUDA
        CUDA::Manager::init();
        CUDA::Manager::load_info();
        if (CUDA::Manager::device_count() > 0) {
          CUDA::Manager::use(0);
          std::cout << "Initialized CUDA with " << CUDA::Manager::device_count()
                    << " device(s)" << std::endl;
        }
#endif

#ifdef USE_SYCL
        SYCL::Manager::init();
        SYCL::Manager::load_info();
        if (SYCL::Manager::device_count() > 0) {
          std::cout << "Initialized SYCL with " << SYCL::Manager::device_count()
                    << " device(s)" << std::endl;
        }
#endif

#ifdef USE_METAL
        METAL::Manager::init();
        METAL::Manager::load_info();
        if (METAL::Manager::get_device_count() > 0) {
          METAL::Manager::use(0);
          LOGDEBUG("Initialized Metal with {} device(s)",
                   METAL::Manager::get_device_count());
        }
#endif
      } catch (const std::exception &e) {
        LOGWARN("Backend initialization failed: {}", e.what());
      }
    }
  }

  ~BackendInitFixture() {
    // No individual finalization - let the program cleanup handle it
    // This avoids double-finalization and mutex issues
  }
};

// Static member definition
std::atomic<bool> BackendInitFixture::initialized_{false};

// ============================================================================
// Pinned Buffer Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Pinned Buffer Creation",
                 "[pinned][creation]") {

  SECTION("Basic pinned buffer creation") {
    Resource resource;

    PinnedBuffer<float> pinned_buffer(PINNED_BUFFER_SIZE, resource);

    REQUIRE(static_cast<const Buffer<float, PinnedPolicy> &>(pinned_buffer)
                .size() == PINNED_BUFFER_SIZE);
    REQUIRE(static_cast<const Buffer<float, PinnedPolicy> &>(pinned_buffer)
                .empty() == false);
    REQUIRE(static_cast<const Buffer<float, PinnedPolicy> &>(pinned_buffer)
                .data() != nullptr);
    REQUIRE(static_cast<const Buffer<float, PinnedPolicy> &>(pinned_buffer)
                .bytes() == PINNED_BUFFER_SIZE * sizeof(float));
  }

  SECTION("Pinned buffer with different data types") {
    Resource resource;

    // Test int buffer
    PinnedBuffer<int> int_buffer(512, resource);
    REQUIRE(static_cast<const Buffer<int, PinnedPolicy> &>(int_buffer).size() ==
            512);
    REQUIRE(
        static_cast<const Buffer<int, PinnedPolicy> &>(int_buffer).bytes() ==
        512 * sizeof(int));

    // Test double buffer
    PinnedBuffer<double> double_buffer(256, resource);
    REQUIRE(static_cast<const Buffer<double, PinnedPolicy> &>(double_buffer)
                .size() == 256);
    REQUIRE(static_cast<const Buffer<double, PinnedPolicy> &>(double_buffer)
                .bytes() == 256 * sizeof(double));
  }

  SECTION("Empty pinned buffer") {
    Resource resource;

    PinnedBuffer<float> empty_buffer(0, resource);
    REQUIRE(empty_buffer.size() == 0);
    REQUIRE(empty_buffer.empty());
    REQUIRE(empty_buffer.bytes() == 0);
  }

  SECTION("Large pinned buffer allocation") {
    Resource resource;

    const size_t large_size = 1000000; // 1M elements
    PinnedBuffer<float> large_buffer(large_size, resource);

    REQUIRE(large_buffer.size() == large_size);
    REQUIRE(large_buffer.bytes() == large_size * sizeof(float));
    REQUIRE(large_buffer.data() != nullptr);
  }
}

// Sometimes this failed on macs with SYCL for performance and memory.
TEST_CASE_METHOD(BackendInitFixture, "Pinned Buffer Memory Operations",
                 "[pinned][memory]") {

  SECTION("Host to pinned buffer transfer") {
    Resource resource;

    // skip_if_sycl_on_macos();

    PinnedBuffer<float> pinned_buffer(PINNED_BUFFER_SIZE, resource);

    // Create test data
    std::vector<float> host_data(PINNED_BUFFER_SIZE);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Copy to pinned buffer using PINBuffer specific method
    REQUIRE_NOTHROW(
        pinned_buffer.upload_to_device(host_data.data(), PINNED_BUFFER_SIZE));

    // Verify data integrity using PINBuffer specific method
    std::vector<float> result_data(PINNED_BUFFER_SIZE);
    REQUIRE_NOTHROW(pinned_buffer.download_from_device(result_data.data(),
                                                       PINNED_BUFFER_SIZE));
    REQUIRE(result_data == host_data);
  }

  SECTION("Pinned buffer to device transfer") {
    Resource resource;

    PinnedBuffer<float> pinned_buffer(PINNED_BUFFER_SIZE, resource);

    // Initialize pinned buffer with data using PINBuffer specific method
    std::vector<float> test_data(PINNED_BUFFER_SIZE, 42.0f);
    pinned_buffer.upload_to_device(test_data.data(), PINNED_BUFFER_SIZE);

    // Create device buffer
    DeviceBuffer<float> device_buffer(PINNED_BUFFER_SIZE);

    // Copy from pinned to device
    REQUIRE_NOTHROW(
        device_buffer.copy_from_host(pinned_buffer.data(), PINNED_BUFFER_SIZE));

    // Verify device buffer has correct data
    std::vector<float> device_result;
    device_buffer.copy_to_host(device_result);
    REQUIRE(device_result == test_data);
  }

  SECTION("Device to pinned buffer transfer") {
    Resource resource;

    // skip_if_sycl_on_macos();

    PinnedBuffer<float> pinned_buffer(PINNED_BUFFER_SIZE, resource);
    DeviceBuffer<float> device_buffer(PINNED_BUFFER_SIZE);

    // Initialize device buffer
    std::vector<float> test_data(PINNED_BUFFER_SIZE);
    std::iota(test_data.begin(), test_data.end(), 100.0f);
    device_buffer.copy_from_host(test_data);

    // Get data from device buffer to host, then upload to pinned buffer using
    // PINBuffer method
    std::vector<float> host_data(PINNED_BUFFER_SIZE);
    device_buffer.copy_to_host(host_data.data(), PINNED_BUFFER_SIZE);
    REQUIRE_NOTHROW(
        pinned_buffer.upload_to_device(host_data.data(), PINNED_BUFFER_SIZE));

    // Verify pinned buffer has correct data using PINBuffer specific method
    std::vector<float> pinned_result(PINNED_BUFFER_SIZE);
    REQUIRE_NOTHROW(pinned_buffer.download_from_device(pinned_result.data(),
                                                       PINNED_BUFFER_SIZE));
    REQUIRE(pinned_result == test_data);
  }

  SECTION("Partial transfers") {
    Resource resource;

    // skip_if_sycl_on_macos();

    PinnedBuffer<int> pinned_buffer(PINNED_BUFFER_SIZE, resource);

    const size_t partial_size = PINNED_BUFFER_SIZE / 2;
    std::vector<int> partial_data(partial_size, 123);

    // Partial copy to pinned buffer using PINBuffer specific method
    REQUIRE_NOTHROW(
        pinned_buffer.upload_to_device(partial_data.data(), partial_size));

    // Partial copy from pinned buffer using PINBuffer specific method
    std::vector<int> result_data(partial_size);
    REQUIRE_NOTHROW(
        pinned_buffer.download_from_device(result_data.data(), partial_size));

    REQUIRE(result_data == partial_data);
  }
}

// Sometimes this failed on macs with SYCL for performance and memory
TEST_CASE_METHOD(BackendInitFixture, "Pinned Buffer Performance",
                 "[pinned][performance]") {

  SECTION("Pinned vs regular host memory performance") {
    Resource resource;

    const size_t test_size = 100000; // 100K elements

    // Create pinned buffer
    PinnedBuffer<float> pinned_buffer(test_size, resource);

    // Create regular host vector
    std::vector<float> host_vector(test_size);

    // Initialize both with same data
    std::vector<float> test_data(test_size);
    std::iota(test_data.begin(), test_data.end(), 1.0f);

    // Measure pinned buffer copy using PINBuffer specific method
    auto pinned_start = std::chrono::high_resolution_clock::now();
    pinned_buffer.upload_to_device(test_data.data(), test_size);
    auto pinned_end = std::chrono::high_resolution_clock::now();

    // Measure regular vector copy
    auto host_start = std::chrono::high_resolution_clock::now();
    host_vector = test_data;
    auto host_end = std::chrono::high_resolution_clock::now();

    auto pinned_time = std::chrono::duration_cast<std::chrono::microseconds>(
        pinned_end - pinned_start);
    auto host_time = std::chrono::duration_cast<std::chrono::microseconds>(
        host_end - host_start);

    INFO("Pinned buffer copy took: " << pinned_time.count() << " microseconds");
    INFO("Host vector copy took: " << host_time.count() << " microseconds");

    // Both should complete successfully
    REQUIRE(static_cast<const Buffer<float, PinnedPolicy> &>(pinned_buffer)
                .size() == test_size);
    REQUIRE(host_vector.size() == test_size);
  }

  SECTION("Concurrent pinned buffer access") {
    // SKIP_IF_SYCL_UNSTABLE();

    Resource resource;

    const size_t num_buffers = 4;
    const size_t buffer_size = 1000;

    std::vector<std::unique_ptr<PinnedBuffer<int>>> buffers;
    std::vector<std::future<bool>> futures;

    // Create multiple pinned buffers
    for (size_t i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(
          std::make_unique<PinnedBuffer<int>>(buffer_size, resource));
    }

    // Concurrent operations on different buffers
    for (size_t i = 0; i < num_buffers; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [&buffers, i, buffer_size]() {
            try {
              auto &buffer = *buffers[i];

              // Initialize with unique data using PINBuffer specific methods
              std::vector<int> test_data(buffer_size, static_cast<int>(i));
              buffer.upload_to_device(test_data.data(), buffer_size);

              // Verify data using PINBuffer specific methods
              std::vector<int> result(buffer_size);
              buffer.download_from_device(result.data(), buffer_size);

              return result == test_data;
            } catch (...) {
              return false;
            }
          }));
    }

    // Wait for all operations to complete
    for (auto &future : futures) {
      REQUIRE(future.get() == true);
    }
  }
}

// ============================================================================
// Unified Buffer Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Unified Buffer Creation",
                 "[unified][creation]") {

  SECTION("Basic unified buffer creation") {
    Resource resource;

    UnifiedBuffer<float> unified_buffer(UNIFIED_BUFFER_SIZE, resource);

    REQUIRE(unified_buffer.size() == UNIFIED_BUFFER_SIZE);
    REQUIRE_FALSE(unified_buffer.empty());
    REQUIRE(unified_buffer.data() != nullptr);
    REQUIRE(unified_buffer.bytes() == UNIFIED_BUFFER_SIZE * sizeof(float));
  }

  SECTION("Unified buffer with different data types") {
    Resource resource;

    // Test int buffer
    UnifiedBuffer<int> int_buffer(512, resource);
    REQUIRE(int_buffer.size() == 512);
    REQUIRE(int_buffer.bytes() == 512 * sizeof(int));

    // Test double buffer
    UnifiedBuffer<double> double_buffer(256, resource);
    REQUIRE(double_buffer.size() == 256);
    REQUIRE(double_buffer.bytes() == 256 * sizeof(double));

    // Test Vector3 buffer
    UnifiedBuffer<std::vector<float>> vec_buffer(128, resource);
    REQUIRE(vec_buffer.size() == 128);
    REQUIRE(vec_buffer.bytes() == 128 * sizeof(std::vector<float>));
  }

  SECTION("Empty unified buffer") {
    Resource resource;

    UnifiedBuffer<float> empty_buffer(0, resource);
    REQUIRE(empty_buffer.size() == 0);
    REQUIRE(empty_buffer.empty());
    REQUIRE(empty_buffer.bytes() == 0);
  }

  SECTION("Large unified buffer allocation") {
    Resource resource;

    const size_t large_size = 500000; // 500K elements
    UnifiedBuffer<float> large_buffer(large_size, resource);

    REQUIRE(large_buffer.size() == large_size);
    REQUIRE(large_buffer.bytes() == large_size * sizeof(float));
    REQUIRE(large_buffer.data() != nullptr);
  }
}

TEST_CASE_METHOD(BackendInitFixture, "Unified Buffer Memory Operations",
                 "[unified][memory]") {

  SECTION("Host to unified buffer transfer") {
    Resource resource;

    // skip_if_sycl_on_macos();

    UnifiedBuffer<float> unified_buffer(UNIFIED_BUFFER_SIZE, resource);

    // Create test data
    std::vector<float> host_data(UNIFIED_BUFFER_SIZE);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Copy to unified buffer
    REQUIRE_NOTHROW(unified_buffer.copy_from_host(host_data));

    // Verify data integrity
    std::vector<float> result_data;
    REQUIRE_NOTHROW(unified_buffer.copy_to_host(result_data));
    REQUIRE(result_data == host_data);
  }

  SECTION("Unified buffer to device transfer") {
    Resource resource;

    // skip_if_sycl_on_macos();

    UnifiedBuffer<float> unified_buffer(UNIFIED_BUFFER_SIZE, resource);

    // Initialize unified buffer with data
    std::vector<float> test_data(UNIFIED_BUFFER_SIZE, 42.0f);
    unified_buffer.copy_from_host(test_data);

    // Create device buffer
    DeviceBuffer<float> device_buffer(UNIFIED_BUFFER_SIZE);

    // Copy from unified to device
    REQUIRE_NOTHROW(device_buffer.copy_from_host(unified_buffer.data(),
                                                 UNIFIED_BUFFER_SIZE));

    // Verify device buffer has correct data
    std::vector<float> device_result;
    device_buffer.copy_to_host(device_result);
    REQUIRE(device_result == test_data);
  }

  SECTION("Device to unified buffer transfer") {
    Resource resource;

    // skip_if_sycl_on_macos();

    UnifiedBuffer<float> unified_buffer(UNIFIED_BUFFER_SIZE, resource);
    DeviceBuffer<float> device_buffer(UNIFIED_BUFFER_SIZE);

    // Initialize device buffer
    std::vector<float> test_data(UNIFIED_BUFFER_SIZE);
    std::iota(test_data.begin(), test_data.end(), 100.0f);
    device_buffer.copy_from_host(test_data);

    // Copy from device to unified using raw memory copy
    // Since unified buffer can be accessed from device, we can copy the data
    // directly First, copy device data to host, then to unified buffer
    std::vector<float> temp_data;
    device_buffer.copy_to_host(temp_data);
    unified_buffer.copy_from_host(temp_data);

    // Verify unified buffer has correct data
    std::vector<float> unified_result;
    unified_buffer.copy_to_host(unified_result);
    REQUIRE(unified_result == test_data);
  }

  SECTION("Cross-device unified buffer operations") {
    // This test requires at least one device
    std::vector<Resource> devices;
#ifdef USE_CUDA
    for (size_t i = 0; i < CUDA::Manager::device_count(); ++i) {
      devices.emplace_back(ResourceType::CUDA, i);
    }
#endif
#ifdef USE_SYCL
    for (size_t i = 0; i < SYCL::Manager::device_count(); ++i) {
      devices.emplace_back(ResourceType::SYCL, i);
    }
#endif
#ifdef USE_METAL
    for (size_t i = 0; i < METAL::Manager::get_device_count(); ++i) {
      devices.emplace_back(ResourceType::METAL, i);
    }
#endif
    if (devices.empty()) {
      SKIP("No devices available for unified buffer test");
    }

    const size_t test_size = 1000;

    // Use a single device to avoid CUDA context issues
    Resource single_device = devices[0];

    // Create unified buffer on the device
    UnifiedBuffer<float> unified_buffer(test_size, single_device);

    // Create device buffer on the same device
    DeviceBuffer<float> device_buffer(test_size, single_device);

    // Initialize unified buffer
    std::vector<float> test_data(test_size, 3.14159f);
    unified_buffer.copy_from_host(test_data);

    // Copy from unified to device using host as intermediary
    // Since they have different policies, we need to go through host memory
    std::vector<float> temp_data;
    unified_buffer.copy_to_host(temp_data);
    device_buffer.copy_from_host(temp_data);

    // Verify data integrity
    std::vector<float> result;
    device_buffer.copy_to_host(result);
    REQUIRE(result == test_data);
  }
}

TEST_CASE_METHOD(BackendInitFixture, "Unified Buffer Performance",
                 "[unified][performance]") {
  // Skip buffer transfer tests on macOS when using SYCL due to unified memory
  // architecture skip_if_sycl_on_macos();

  SECTION("Unified vs device buffer performance") {
    Resource resource;

    const size_t test_size = 100000; // 100K elements

    // Create unified buffer
    UnifiedBuffer<float> unified_buffer(test_size, resource);

    // Create device buffer
    DeviceBuffer<float> device_buffer(test_size);

    // Initialize both with same data
    std::vector<float> test_data(test_size);
    std::iota(test_data.begin(), test_data.end(), 1.0f);

    // Measure unified buffer operations
    auto unified_start = std::chrono::high_resolution_clock::now();
    unified_buffer.copy_from_host(test_data);
    auto unified_end = std::chrono::high_resolution_clock::now();

    // Measure device buffer operations
    auto device_start = std::chrono::high_resolution_clock::now();
    device_buffer.copy_from_host(test_data);
    auto device_end = std::chrono::high_resolution_clock::now();

    auto unified_time = std::chrono::duration_cast<std::chrono::microseconds>(
        unified_end - unified_start);
    auto device_time = std::chrono::duration_cast<std::chrono::microseconds>(
        device_end - device_start);

    INFO("Unified buffer copy took: " << unified_time.count()
                                      << " microseconds");
    INFO("Device buffer copy took: " << device_time.count() << " microseconds");

    // Both should complete successfully
    REQUIRE(unified_buffer.size() == test_size);
    REQUIRE(device_buffer.size() == test_size);
  }

  SECTION("Concurrent unified buffer access") {
    Resource resource;

    const size_t num_buffers = 4;
    const size_t buffer_size = 1000;

    std::vector<std::unique_ptr<UnifiedBuffer<int>>> buffers;
    std::vector<std::future<bool>> futures;

    // Create multiple unified buffers
    for (size_t i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(
          std::make_unique<UnifiedBuffer<int>>(buffer_size, resource));
    }

    // Concurrent operations on different buffers
    for (size_t i = 0; i < num_buffers; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [&buffers, i, buffer_size]() {
            try {
              auto &buffer = *buffers[i];

              // Initialize with unique data
              std::vector<int> test_data(buffer_size, static_cast<int>(i));
              buffer.copy_from_host(test_data);

              // Verify data
              std::vector<int> result;
              buffer.copy_to_host(result);

              return result == test_data;
            } catch (...) {
              return false;
            }
          }));
    }

    // Wait for all operations to complete
    for (auto &future : futures) {
      REQUIRE(future.get() == true);
    }
  }
}

// ============================================================================
// Mixed Buffer Type Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Mixed Buffer Type Operations",
                 "[mixed][operations]") {
  // Skip buffer transfer tests on macOS when using SYCL due to unified memory
  // architecture skip_if_sycl_on_macos();

  SECTION("Pinned to unified to device transfer chain") {
    Resource resource;

    const size_t test_size = 2000;

    // Create buffers of different types
    PinnedBuffer<float> pinned_buffer(test_size, resource);
    UnifiedBuffer<float> unified_buffer(test_size, resource);
    DeviceBuffer<float> device_buffer(test_size, resource);

    // Initialize pinned buffer using PINBuffer specific method
    std::vector<float> test_data(test_size);
    std::iota(test_data.begin(), test_data.end(), 1.0f);
    pinned_buffer.upload_to_device(test_data.data(), test_size);

    // Transfer: Pinned -> Unified (using unified buffer's copy_from_host)
    REQUIRE_NOTHROW(
        unified_buffer.copy_from_host(pinned_buffer.data(), test_size));

    // Transfer: Unified -> Device (using device buffer's copy_from_host)
    REQUIRE_NOTHROW(
        device_buffer.copy_from_host(unified_buffer.data(), test_size));

    // Verify final result
    std::vector<float> final_result;
    device_buffer.copy_to_host(final_result);
    REQUIRE(final_result == test_data);
  }

  SECTION("Cross-resource buffer operations") {
    std::vector<Resource> devices;
#ifdef USE_CUDA
    for (size_t i = 0; i < CUDA::Manager::device_count(); ++i) {
      devices.emplace_back(ResourceType::CUDA, i);
    }
#endif
#ifdef USE_SYCL
    for (size_t i = 0; i < SYCL::Manager::device_count(); ++i) {
      devices.emplace_back(ResourceType::SYCL, i);
    }
#endif
#ifdef USE_METAL
    for (size_t i = 0; i < METAL::Manager::get_device_count(); ++i) {
      devices.emplace_back(ResourceType::METAL, i);
    }
#endif
    if (devices.size() < 2) {
      SKIP("Need at least 2 devices for cross-resource test");
    }

    const size_t test_size = 1500;

    // Create buffers on different resources
    PinnedBuffer<float> pinned_buffer(test_size, devices[0]);
    UnifiedBuffer<float> unified_buffer(test_size, devices[1]);
    DeviceBuffer<float> device_buffer(test_size, devices[0]);

    // Initialize with test data using PINBuffer specific method
    std::vector<float> test_data(test_size, 2.718f);
    pinned_buffer.upload_to_device(test_data.data(), test_size);

    // Cross-resource transfer
    REQUIRE_NOTHROW(
        unified_buffer.copy_from_host(pinned_buffer.data(), test_size));
    REQUIRE_NOTHROW(
        device_buffer.copy_from_host(unified_buffer.data(), test_size));

    // Verify data integrity
    std::vector<float> result;
    device_buffer.copy_to_host(result);
    REQUIRE(result == test_data);
  }
}

// ============================================================================
// Stress Tests-- this one passes on macos
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Pinned and Unified Buffer Stress Tests",
                 "[stress][pinned][unified]") {

  SECTION("Rapid allocation/deallocation") {
    Resource resource;

    const size_t iterations = 100;
    const size_t buffer_size = 1000;

    for (size_t i = 0; i < iterations; ++i) {
      // Create and immediately destroy buffers
      {
        PinnedBuffer<int> temp_pinned(buffer_size, resource);
        UnifiedBuffer<int> temp_unified(buffer_size, resource);
        DeviceBuffer<int> temp_device(buffer_size, resource);

        // Quick operation to ensure allocation worked
        std::vector<int> test_data(buffer_size, static_cast<int>(i));
        temp_pinned.upload_to_device(test_data.data(), buffer_size);
        temp_unified.copy_from_host(test_data);
        temp_device.copy_from_host(test_data);
      }
    }

    // If we get here without crashes, the test passed
    REQUIRE(true);
  }

  SECTION("Many small buffers") {
    Resource resource;

    const size_t num_buffers = 100;
    const size_t buffer_size = 100;

    std::vector<std::unique_ptr<PinnedBuffer<float>>> pinned_buffers;
    std::vector<std::unique_ptr<UnifiedBuffer<float>>> unified_buffers;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> device_buffers;

    // Create many small buffers
    for (size_t i = 0; i < num_buffers; ++i) {
      pinned_buffers.emplace_back(
          std::make_unique<PinnedBuffer<float>>(buffer_size, resource));
      unified_buffers.emplace_back(
          std::make_unique<UnifiedBuffer<float>>(buffer_size, resource));
      device_buffers.emplace_back(
          std::make_unique<DeviceBuffer<float>>(buffer_size, resource));
    }

    // Initialize all buffers with unique data
    for (size_t i = 0; i < num_buffers; ++i) {
      std::vector<float> data(buffer_size, static_cast<float>(i));

      pinned_buffers[i]->upload_to_device(data.data(), buffer_size);
      unified_buffers[i]->copy_from_host(data);
      device_buffers[i]->copy_from_host(data);
    }

    // Verify all buffers have correct data
    for (size_t i = 0; i < num_buffers; ++i) {
      std::vector<float> expected(buffer_size, static_cast<float>(i));

      std::vector<float> pinned_result(buffer_size),
          unified_result(buffer_size), device_result(buffer_size);

      pinned_buffers[i]->download_from_device(pinned_result.data(),
                                              buffer_size);
      unified_buffers[i]->copy_to_host(unified_result);
      device_buffers[i]->copy_to_host(device_result);

      REQUIRE(pinned_result == expected);
      REQUIRE(unified_result == expected);
      REQUIRE(device_result == expected);
    }
  }

  SECTION("Mixed size operations") {
    Resource resource;

    std::vector<size_t> sizes = {1, 10, 100, 1000, 10000};

    for (size_t size : sizes) {
      // Test all buffer types with different sizes
      PinnedBuffer<int> pinned_buffer(size, resource);
      UnifiedBuffer<int> unified_buffer(size, resource);
      DeviceBuffer<int> device_buffer(size, resource);

      REQUIRE(pinned_buffer.size() == size);
      REQUIRE(unified_buffer.size() == size);
      REQUIRE(device_buffer.size() == size);

      if (size > 0) {
        std::vector<int> test_data(size, 42);

        pinned_buffer.upload_to_device(test_data.data(), size);
        unified_buffer.copy_from_host(test_data);
        device_buffer.copy_from_host(test_data);

        std::vector<int> pinned_result(size), unified_result(size),
            device_result(size);

        pinned_buffer.download_from_device(pinned_result.data(), size);
        unified_buffer.copy_to_host(unified_result);
        device_buffer.copy_to_host(device_result);

        REQUIRE(pinned_result == test_data);
        REQUIRE(unified_result == test_data);
        REQUIRE(device_result == test_data);
      }
    }
  }
}

// ============================================================================
// Edge Cases and Error Handling-- failed
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Edge Cases", "[edge_cases]") {
  // Skip buffer transfer tests on macOS when using SYCL due to unified memory
  // architecture
  skip_if_sycl_on_macos();

  SECTION("Zero-size buffer operations") {
    Resource resource;

    PinnedBuffer<float> empty_pinned(0, resource);
    UnifiedBuffer<float> empty_unified(0, resource);
    DeviceBuffer<float> empty_device(0, resource);

    REQUIRE(empty_pinned.empty());
    REQUIRE(empty_unified.empty());
    REQUIRE(empty_device.empty());

    REQUIRE(empty_pinned.size() == 0);
    REQUIRE(empty_unified.size() == 0);
    REQUIRE(empty_device.size() == 0);

    // Test operations on empty buffers
    std::vector<float> empty_vec;
    REQUIRE_NOTHROW(empty_pinned.upload_to_device(empty_vec.data(), 0));
  }

  SECTION("Buffer resize operations") {
    Resource resource;

    PinnedBuffer<float> pinned_buffer(100, resource);
    UnifiedBuffer<float> unified_buffer(100, resource);
    DeviceBuffer<float> device_buffer(100, resource);

    // Resize to larger size
    pinned_buffer.resize(200);
    // For UnifiedBuffer, use the base class resize method to get proper data
    // preservation
    static_cast<Buffer<float, UnifiedPolicy> &>(unified_buffer).resize(200);
    device_buffer.resize(200);

    REQUIRE(pinned_buffer.size() == 200);
    REQUIRE(unified_buffer.size() == 200);
    REQUIRE(device_buffer.size() == 200);

    // Resize to smaller size
    pinned_buffer.resize(50);
    // For UnifiedBuffer, use the base class resize method to get proper data
    // preservation
    static_cast<Buffer<float, UnifiedPolicy> &>(unified_buffer).resize(50);
    device_buffer.resize(50);

    REQUIRE(pinned_buffer.size() == 50);
    REQUIRE(unified_buffer.size() == 50);
    REQUIRE(device_buffer.size() == 50);
  }
}

// ============================================================================
// Performance Benchmark Tests: Trace trap
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Performance Benchmarks",
                 "[benchmark][performance]") {
  // Skip buffer transfer tests on macOS when using SYCL due to unified memory
  // architecture skip_if_sycl_on_macos();

  SECTION("Memory bandwidth test") {
    Resource resource;

    const size_t test_size = 1000000; // 1M elements
    const size_t num_iterations = 10;

    // Create buffers
    PinnedBuffer<float> pinned_buffer(test_size, resource);
    UnifiedBuffer<float> unified_buffer(test_size, resource);
    DeviceBuffer<float> device_buffer(test_size, resource);

    // Test data
    std::vector<float> test_data(test_size);
    std::iota(test_data.begin(), test_data.end(), 1.0f);

    // Measure pinned buffer performance using PINBuffer specific methods
    auto pinned_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations; ++i) {
      pinned_buffer.upload_to_device(test_data.data(), test_size);
      std::vector<float> result(test_size);
      pinned_buffer.download_from_device(result.data(), test_size);
    }
    auto pinned_end = std::chrono::high_resolution_clock::now();

    // Measure unified buffer performance
    auto unified_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations; ++i) {
      unified_buffer.copy_from_host(test_data);
      std::vector<float> result;
      unified_buffer.copy_to_host(result);
    }
    auto unified_end = std::chrono::high_resolution_clock::now();

    // Measure device buffer performance
    auto device_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_iterations; ++i) {
      device_buffer.copy_from_host(test_data);
      std::vector<float> result;
      device_buffer.copy_to_host(result);
    }
    auto device_end = std::chrono::high_resolution_clock::now();

    auto pinned_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        pinned_end - pinned_start);
    auto unified_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        unified_end - unified_start);
    auto device_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        device_end - device_start);

    INFO("Pinned buffer total time: " << pinned_time.count() << " ms");
    INFO("Unified buffer total time: " << unified_time.count() << " ms");
    INFO("Device buffer total time: " << device_time.count() << " ms");

    // All operations should complete successfully
    REQUIRE(pinned_buffer.size() == test_size);
    REQUIRE(unified_buffer.size() == test_size);
    REQUIRE(device_buffer.size() == test_size);
  }

  SECTION("Concurrent access performance") {
    Resource resource;

    const size_t num_threads = 4;
    const size_t buffer_size = 50000;
    const size_t operations_per_thread = 100;

    // Create buffers for each thread
    std::vector<std::unique_ptr<PinnedBuffer<int>>> pinned_buffers;
    std::vector<std::unique_ptr<UnifiedBuffer<int>>> unified_buffers;
    std::vector<std::unique_ptr<DeviceBuffer<int>>> device_buffers;

    for (size_t i = 0; i < num_threads; ++i) {
      pinned_buffers.emplace_back(
          std::make_unique<PinnedBuffer<int>>(buffer_size, resource));
      unified_buffers.emplace_back(
          std::make_unique<UnifiedBuffer<int>>(buffer_size, resource));
      device_buffers.emplace_back(
          std::make_unique<DeviceBuffer<int>>(buffer_size, resource));
    }

    // Measure concurrent pinned buffer performance
    auto pinned_start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> pinned_futures;
    for (size_t i = 0; i < num_threads; ++i) {
      pinned_futures.emplace_back(
          std::async(std::launch::async, [&pinned_buffers, i, buffer_size,
                                          operations_per_thread]() {
            for (size_t j = 0; j < operations_per_thread; ++j) {
              std::vector<int> data(buffer_size,
                                    static_cast<int>(i * 1000 + j));
              pinned_buffers[i]->upload_to_device(data.data(), buffer_size);
              std::vector<int> result(buffer_size);
              pinned_buffers[i]->download_from_device(result.data(),
                                                      buffer_size);
            }
          }));
    }
    for (auto &future : pinned_futures) {
      future.wait();
    }
    auto pinned_end = std::chrono::high_resolution_clock::now();

    auto pinned_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        pinned_end - pinned_start);
    INFO("Concurrent pinned buffer operations took: " << pinned_time.count()
                                                      << " ms");

    // All operations should complete successfully
    REQUIRE(pinned_buffers.size() == num_threads);
    REQUIRE(unified_buffers.size() == num_threads);
    REQUIRE(device_buffers.size() == num_threads);
  }
}
