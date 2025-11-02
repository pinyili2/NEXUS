#include "catch_boiler.h"

// Test asynchronous buffer operations
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#endif
#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif
#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

#include <chrono>
#include <future>
#include <numeric>
#include <vector>

using namespace ARBD;

// ============================================================================
// Test Fixture for Asynchronous Buffer Tests
// ============================================================================

class AsyncBufferTestFixture {
public:
  std::vector<Resource> available_resources;

  AsyncBufferTestFixture() {
    // Initialize backends and populate resources
    try {
#ifdef USE_CUDA
      CUDA::Manager::init();
      if (CUDA::Manager::device_count() > 0) {
        for (size_t i = 0; i < CUDA::Manager::device_count(); ++i) {
          available_resources.emplace_back(ResourceType::CUDA, i);
        }
      }
#endif
#ifdef USE_SYCL
      SYCL::Manager::init();
      SYCL::Manager::load_info();
      if (SYCL::Manager::device_count() > 0) {
        for (size_t i = 0; i < SYCL::Manager::device_count(); ++i) {
          available_resources.emplace_back(ResourceType::SYCL, i);
        }
        std::cout << "Added " << SYCL::Manager::device_count()
                  << " SYCL device(s) to available resources" << std::endl;
      }
#endif
#ifdef USE_METAL
      METAL::Manager::init();
      METAL::Manager::load_info();
      if (METAL::Manager::get_device_count() > 0) {
        for (size_t i = 0; i < METAL::Manager::get_device_count(); ++i) {
          available_resources.emplace_back(ResourceType::METAL, i);
        }
      }
#endif
      available_resources.emplace_back(ResourceType::CPU, 0);

    } catch (const std::exception &e) {
      // Don't FAIL here, just add CPU as fallback
      std::cerr << "Backend initialization failed: " << e.what() << std::endl;
      available_resources.clear();
      available_resources.emplace_back(ResourceType::CPU, 0);
    }
  }

  ~AsyncBufferTestFixture() {
    // No explicit finalization - let the singleton handle cleanup
  }

  std::vector<Resource> get_device_resources() const {
    std::vector<Resource> devices;
    for (const auto &resource : available_resources) {
      if (resource.is_device()) {
        devices.push_back(resource);
      }
    }
    return devices;
  }

  Resource get_any_device_resource() const {
    auto devices = get_device_resources();
    if (devices.empty()) {
      SKIP("No device resources available for testing");
    }
    return devices[0];
  }

  // Helper function to synchronize with a stream handle
  void synchronize_stream(void *stream_handle, const Resource &resource) {
    if (!stream_handle)
      return;

#ifdef USE_CUDA
    if (resource.type() == ResourceType::CUDA) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(stream_handle));
    }
#endif
#ifdef USE_SYCL
    if (resource.type() == ResourceType::SYCL) {
      // For SYCL, we need to wait on the queue
      sycl::queue *queue = static_cast<sycl::queue *>(stream_handle);
      queue->wait();
    }
#endif
#ifdef USE_METAL
    if (resource.type() == ResourceType::METAL) {
      // Metal synchronization would be handled differently
      // For now, we'll use a brief pause
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
  }
};

// ============================================================================
// DeviceBuffer Async Tests
// ============================================================================

TEST_CASE_METHOD(AsyncBufferTestFixture,
                 "DeviceBuffer Async Host-Device Transfers",
                 "[AsyncBuffer][DeviceBuffer][host-device]") {

  auto device = get_any_device_resource();
  const size_t buffer_size = 1000;

  SECTION("Async copy_from_host") {
    DeviceBuffer<float> buffer(buffer_size, device);

    // Create test data
    std::vector<float> host_data(buffer_size);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Perform async copy from host
    void *stream = buffer.copy_from_host_async(host_data.data(), buffer_size);
    REQUIRE(stream != nullptr);

    // Synchronize and verify
    synchronize_stream(stream, device);

    std::vector<float> result_data(buffer_size);
    buffer.copy_to_host_sync(result_data.data(), buffer_size);
    REQUIRE(result_data == host_data);
  }

  SECTION("Async copy_to_host") {
    DeviceBuffer<int> buffer(buffer_size, device);

    // Initialize buffer with known data using async copy and proper
    // synchronization
    std::vector<int> host_data(buffer_size, 42);
    void *init_stream =
        buffer.copy_from_host_async(host_data.data(), buffer_size);
    REQUIRE(init_stream != nullptr);
    synchronize_stream(init_stream, device);

    // Perform async copy to host using the same stream as the buffer
    std::vector<int> result_data(buffer_size);
    void *stream = buffer.copy_to_host_async(result_data.data(), buffer_size);
    REQUIRE(stream != nullptr);

    // Synchronize and verify
    synchronize_stream(stream, device);
    REQUIRE(result_data == host_data);
  }

  SECTION("Event-based async operations") {
    DeviceBuffer<double> buffer(buffer_size, device);

    // Test copy_from_host_event
    std::vector<double> host_data(buffer_size, 3.14159);
    Event from_host_event =
        buffer.copy_from_host_event(host_data.data(), buffer_size);

    // Event should be valid
    REQUIRE((from_host_event.is_valid() || !device.supports_async()));

    // Test copy_to_host_event
    std::vector<double> result_data(buffer_size);
    Event to_host_event =
        buffer.copy_to_host_event(result_data.data(), buffer_size);

    // Event should be valid
    REQUIRE((to_host_event.is_valid() || !device.supports_async()));

    // Wait for completion and verify
    if (to_host_event.is_valid()) {
      to_host_event.wait();
    }
    REQUIRE(result_data == host_data);
  }

  SECTION("Multiple concurrent async transfers") {
    const size_t num_buffers = 3;
    const size_t small_size = 100;

    std::vector<DeviceBuffer<float>> buffers;
    std::vector<std::vector<float>> host_data_sets;
    std::vector<void *> streams;

    // Create buffers and data
    for (size_t i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(small_size, device);
      host_data_sets.emplace_back(small_size, static_cast<float>(i + 1));
    }

    // Launch all async transfers
    for (size_t i = 0; i < num_buffers; ++i) {
      void *stream =
          buffers[i].copy_from_host_async(host_data_sets[i].data(), small_size);
      streams.push_back(stream);
    }

    // Synchronize all streams
    for (size_t i = 0; i < num_buffers; ++i) {
      synchronize_stream(streams[i], device);
    }

    // Verify all transfers completed correctly
    for (size_t i = 0; i < num_buffers; ++i) {
      std::vector<float> result(small_size);
      buffers[i].copy_to_host_sync(result.data(), small_size);
      REQUIRE(result == host_data_sets[i]);
    }
  }
}

// ============================================================================
// PinnedBuffer Async Tests
// ============================================================================

TEST_CASE_METHOD(AsyncBufferTestFixture,
                 "PinnedBuffer Async Host-Device Transfers",
                 "[AsyncBuffer][PinnedBuffer][host-device]") {

  auto device = get_any_device_resource();
  const size_t buffer_size = 500;

  SECTION("PinnedBuffer async copy operations") {
    PinnedBuffer<float> buffer(buffer_size, device);

    // Create test data
    std::vector<float> host_data(buffer_size);
    std::iota(host_data.begin(), host_data.end(), 10.0f);

    // Test upload_to_device (pinned buffer specific method)
    REQUIRE_NOTHROW(buffer.upload_to_device(host_data.data(), buffer_size));

    // Test download_from_device (pinned buffer specific method)
    std::vector<float> result_data(buffer_size);
    REQUIRE_NOTHROW(
        buffer.download_from_device(result_data.data(), buffer_size));

    REQUIRE(result_data == host_data);
  }

  SECTION("PinnedBuffer performance characteristics") {
    // Test that pinned buffers provide faster transfers than regular buffers
    PinnedBuffer<int> pinned_buffer(buffer_size, device);
    DeviceBuffer<int> device_buffer(buffer_size, device);

    std::vector<int> test_data(buffer_size, 123);

    // Time pinned buffer transfer
    auto start = std::chrono::high_resolution_clock::now();
    pinned_buffer.copy_from_host(test_data.data(), buffer_size);
    auto pinned_time = std::chrono::high_resolution_clock::now() - start;

    // Time regular device buffer transfer
    start = std::chrono::high_resolution_clock::now();
    device_buffer.copy_from_host(test_data.data(), buffer_size, true);
    auto device_time = std::chrono::high_resolution_clock::now() - start;

    // Both should complete successfully (performance comparison is
    // informational)
    std::vector<int> pinned_result(buffer_size);
    std::vector<int> device_result(buffer_size);

    pinned_buffer.copy_to_host(pinned_result.data(), buffer_size);
    device_buffer.copy_to_host(device_result.data(), buffer_size, true);

    REQUIRE(pinned_result == test_data);
    REQUIRE(device_result == test_data);

    INFO("Pinned buffer transfer time: " << pinned_time.count() << " ns");
    INFO("Device buffer transfer time: " << device_time.count() << " ns");
  }

  SECTION("PinnedBuffer with async streams") {
    PinnedBuffer<double> buffer(buffer_size, device);

    // Get the stream from the buffer
    void *stream = buffer.get_queue();

    // Create test data
    std::vector<double> host_data(buffer_size, 2.71828);

    // Use the buffer's stream for async operations
    buffer.copy_from_host(host_data.data(), buffer_size);

    // Verify data integrity
    std::vector<double> result_data(buffer_size);
    buffer.copy_to_host(result_data.data(), buffer_size);
    REQUIRE(result_data == host_data);
  }
}

// ============================================================================
// UnifiedBuffer Async Tests
// ============================================================================

TEST_CASE_METHOD(AsyncBufferTestFixture, "UnifiedBuffer Async Operations",
                 "[AsyncBuffer][UnifiedBuffer][unified-memory]") {

  auto device = get_any_device_resource();
  const size_t buffer_size = 800;

  SECTION("UnifiedBuffer prefetch operations") {
    UnifiedBuffer<float> buffer(buffer_size, device);

    // Initialize with host data
    std::vector<float> host_data(buffer_size);
    std::iota(host_data.begin(), host_data.end(), 100.0f);
    buffer.copy_from_host(host_data);

    // Test prefetch to device
    REQUIRE_NOTHROW(buffer.prefetch(static_cast<int>(device.id())));

    // Test prefetch with explicit queue
    void *queue = buffer.get_queue();
    REQUIRE_NOTHROW(buffer.prefetch(static_cast<int>(device.id()), queue));

    // Verify data integrity after prefetch
    std::vector<float> result_data;
    buffer.copy_to_host(result_data);
    REQUIRE(result_data == host_data);
  }

  SECTION("UnifiedBuffer memory advice") {
    UnifiedBuffer<int> buffer(buffer_size, device);

    // Test memory advice operations
#ifdef USE_CUDA
    if (device.type() == ResourceType::CUDA) {
      REQUIRE_NOTHROW(buffer.mem_advise(cudaMemAdviseSetPreferredLocation,
                                        static_cast<int>(device.id())));
      REQUIRE_NOTHROW(buffer.mem_advise(cudaMemAdviseSetAccessedBy,
                                        static_cast<int>(device.id())));
    }
#endif

    // Generic memory advice (should work for all backends)
    REQUIRE_NOTHROW(buffer.mem_advise(0, static_cast<int>(device.id())));

    // Verify buffer still functions correctly
    std::vector<int> test_data(buffer_size, 456);
    buffer.copy_from_host(test_data);

    std::vector<int> result_data;
    buffer.copy_to_host(result_data);
    REQUIRE(result_data == test_data);
  }

  SECTION("UnifiedBuffer multi-device operations") {
    // Create buffer with multiple device resources
    std::vector<Resource> devices = get_device_resources();
    if (devices.empty()) {
      SKIP("No device resources for multi-device test");
    }

    UnifiedBuffer<double> buffer(buffer_size, buffer_size, devices);

    // Test multi-device prefetch
    REQUIRE_NOTHROW(buffer.prefetch_devices());

    // Test setting devices
    buffer.set_devices(devices);

#ifdef USE_CUDA
    if (!devices.empty() && devices[0].type() == ResourceType::CUDA) {
      REQUIRE_NOTHROW(buffer.set_preferred_location_all());
      REQUIRE_NOTHROW(buffer.set_accessed_by_all());
    }
#endif

    // Verify functionality
    std::vector<double> test_data(buffer_size, 789.012);
    buffer.copy_from_host(test_data);

    std::vector<double> result_data;
    buffer.copy_to_host(result_data);
    REQUIRE(result_data == test_data);
  }
}

// ============================================================================
// Cross-Buffer Async Operations
// ============================================================================

TEST_CASE_METHOD(AsyncBufferTestFixture, "Cross-Buffer Async Operations",
                 "[AsyncBuffer][cross-buffer][device-to-device]") {

  auto device = get_any_device_resource();
  const size_t buffer_size = 600;

  // Note: On macOS with single device, device-to-device copies are still valid
  // within the same device, testing the async API functionality

  SECTION("DeviceBuffer async device-to-device copy") {
    DeviceBuffer<float> src_buffer(buffer_size, device);
    DeviceBuffer<float> dst_buffer(buffer_size, device);

    // Initialize source buffer
    std::vector<float> test_data(buffer_size);
    std::iota(test_data.begin(), test_data.end(), 200.0f);
    src_buffer.copy_from_host_sync(test_data.data(), buffer_size);

    // Perform async device-to-device copy
    void *stream =
        dst_buffer.copy_device_to_device_async(src_buffer, buffer_size);
    REQUIRE(stream != nullptr);

    // Synchronize and verify
    synchronize_stream(stream, device);

    std::vector<float> result_data(buffer_size);
    dst_buffer.copy_to_host_sync(result_data.data(), buffer_size);
    REQUIRE(result_data == test_data);
  }

  SECTION("Event-based device-to-device copy") {
    DeviceBuffer<int> src_buffer(buffer_size, device);
    DeviceBuffer<int> dst_buffer(buffer_size, device);

    // Initialize source buffer
    std::vector<int> test_data(buffer_size, 999);
    src_buffer.copy_from_host_sync(test_data.data(), buffer_size);

    // Perform event-based device-to-device copy
    Event copy_event =
        dst_buffer.copy_device_to_device_event(src_buffer, buffer_size);

    if (copy_event.is_valid()) {
      copy_event.wait();
    }

    // Verify copy completed
    std::vector<int> result_data(buffer_size);
    dst_buffer.copy_to_host_sync(result_data.data(), buffer_size);
    REQUIRE(result_data == test_data);
  }
}

// ============================================================================
// Async Performance and Stress Tests
// ============================================================================

TEST_CASE_METHOD(AsyncBufferTestFixture, "Async Operations Performance Tests",
                 "[AsyncBuffer][performance][stress]") {

  auto device = get_any_device_resource();

  SECTION("Async vs Sync transfer comparison") {
    const size_t large_size = 10000;
    DeviceBuffer<float> buffer(large_size, device);

    std::vector<float> large_data(large_size);
    std::iota(large_data.begin(), large_data.end(), 1.0f);

    // Time synchronous transfer
    auto start = std::chrono::high_resolution_clock::now();
    buffer.copy_from_host_sync(large_data.data(), large_size);
    auto sync_time = std::chrono::high_resolution_clock::now() - start;

    // Time asynchronous transfer
    start = std::chrono::high_resolution_clock::now();
    void *stream = buffer.copy_from_host_async(large_data.data(), large_size);
    synchronize_stream(stream, device);
    auto async_time = std::chrono::high_resolution_clock::now() - start;

    INFO("Synchronous transfer time: " << sync_time.count() << " ns");
    INFO("Asynchronous transfer time: " << async_time.count() << " ns");

    // Both should complete successfully
    std::vector<float> result_sync(large_size), result_async(large_size);
    buffer.copy_to_host_sync(result_sync.data(), large_size);
    buffer.copy_to_host_sync(result_async.data(), large_size);

    REQUIRE(result_sync == large_data);
    REQUIRE(result_async == large_data);
  }

  SECTION("Multiple concurrent async operations stress test") {
    const size_t num_operations = 5;
    const size_t operation_size = 1000;

    std::vector<DeviceBuffer<int>> buffers;
    std::vector<std::vector<int>> data_sets;
    std::vector<void *> streams;

    // Create buffers and data
    for (size_t i = 0; i < num_operations; ++i) {
      buffers.emplace_back(operation_size, device);
      data_sets.emplace_back(operation_size, static_cast<int>(i * 100));
    }

    // Launch all async operations simultaneously
    for (size_t i = 0; i < num_operations; ++i) {
      void *stream =
          buffers[i].copy_from_host_async(data_sets[i].data(), operation_size);
      streams.push_back(stream);
    }

    // Synchronize all operations
    for (size_t i = 0; i < num_operations; ++i) {
      synchronize_stream(streams[i], device);
    }

    // Verify all operations completed correctly
    for (size_t i = 0; i < num_operations; ++i) {
      std::vector<int> result(operation_size);
      buffers[i].copy_to_host_sync(result.data(), operation_size);
      REQUIRE(result == data_sets[i]);
    }
  }

  SECTION("Async operation pipeline test") {
    const size_t pipeline_stages = 3;
    const size_t stage_size = 500;

    std::vector<DeviceBuffer<float>> stage_buffers;
    for (size_t i = 0; i < pipeline_stages; ++i) {
      stage_buffers.emplace_back(stage_size, device);
    }

    // Create initial data
    std::vector<float> initial_data(stage_size);
    std::iota(initial_data.begin(), initial_data.end(), 1.0f);

    // Pipeline: Host -> Buffer1 -> Buffer2 -> Buffer3 -> Host
    void *stream1 =
        stage_buffers[0].copy_from_host_async(initial_data.data(), stage_size);
    synchronize_stream(stream1, device);

    void *stream2 = stage_buffers[1].copy_device_to_device_async(
        stage_buffers[0], stage_size);
    synchronize_stream(stream2, device);

    void *stream3 = stage_buffers[2].copy_device_to_device_async(
        stage_buffers[1], stage_size);
    synchronize_stream(stream3, device);

    // Verify final result
    std::vector<float> final_data(stage_size);
    void *stream4 =
        stage_buffers[2].copy_to_host_async(final_data.data(), stage_size);
    synchronize_stream(stream4, device);

    REQUIRE(final_data == initial_data);
  }
}

// ============================================================================
// Error Handling and Edge Cases for Async Operations
// ============================================================================

TEST_CASE_METHOD(AsyncBufferTestFixture, "Async Operations Error Handling",
                 "[AsyncBuffer][error_handling][edge_cases]") {

  auto device = get_any_device_resource();

  SECTION("Async operations with null pointers") {
    DeviceBuffer<int> buffer(100, device);

    // Test with null host pointer (should throw or handle gracefully)
    REQUIRE_THROWS_AS(buffer.copy_from_host_async(nullptr, 100), Exception);

    // Test with zero size (should not crash)
    std::vector<int> data(100, 42);
    void *stream = buffer.copy_from_host_async(data.data(), 0);
    // Zero-size operations may return null stream, which is acceptable
  }

  SECTION("Async operations with oversized transfers") {
    const size_t buffer_size = 100;
    DeviceBuffer<float> buffer(buffer_size, device);

    std::vector<float> oversized_data(buffer_size * 2, 1.0f);

    // Oversized transfer should throw exception
    REQUIRE_THROWS_AS(buffer.copy_from_host_async(oversized_data.data(),
                                                  oversized_data.size()),
                      Exception);
  }

  SECTION("Stream synchronization with invalid handles") {
    // Test synchronization behavior with null stream
    synchronize_stream(nullptr, device);
    // Should not crash - null stream sync is a no-op
  }

  SECTION("Event operations on unsupported devices") {
    // Test event creation on devices that might not support it
    DeviceBuffer<double> buffer(100, device);

    std::vector<double> data(100, 3.14);
    Event event = buffer.copy_from_host_event(data.data(), 100);

    // Event might be invalid on some backends - that's acceptable
    if (event.is_valid()) {
      REQUIRE_NOTHROW(event.wait());
    }

    // Data should still be transferred correctly
    std::vector<double> result(100);
    buffer.copy_to_host_sync(result.data(), 100);
    REQUIRE(result == data);
  }

  SECTION("Concurrent async operations safety") {
    const size_t num_threads = 3;
    const size_t operations_per_thread = 5;
    const size_t data_size = 200;

    std::vector<std::future<bool>> futures;

    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
      futures.emplace_back(std::async(std::launch::async, [=, this]() -> bool {
        try {
          DeviceBuffer<int> buffer(data_size, device);

          for (size_t op = 0; op < operations_per_thread; ++op) {
            std::vector<int> data(data_size,
                                  static_cast<int>(thread_id * 1000 + op));

            // Async copy to device
            void *stream = buffer.copy_from_host_async(data.data(), data_size);
            synchronize_stream(stream, device);

            // Async copy back to host
            std::vector<int> result(data_size);
            stream = buffer.copy_to_host_async(result.data(), data_size);
            synchronize_stream(stream, device);

            // Verify data integrity
            if (result != data) {
              return false;
            }
          }
          return true;
        } catch (const std::exception &) {
          return false;
        }
      }));
    }

    // All threads should complete successfully
    for (auto &future : futures) {
      REQUIRE(future.get() == true);
    }
  }
}
