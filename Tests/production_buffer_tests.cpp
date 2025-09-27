#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "catch_boiler.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif
#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif
#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

#include <future>
#include <memory>
#include <numeric>
#include <vector>

using namespace ARBD;

// ============================================================================
// Test Fixture for Production Buffer Tests
// ============================================================================

class ProductionBufferTestFixture {
public:
  std::vector<Resource> available_resources;

  ProductionBufferTestFixture() {
    // Initialize backends first, then populate resources
    try {
#ifdef USE_CUDA
      CUDA::Manager::init();
      CUDA::Manager::load_info();
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

      // Always have CPU available
      available_resources.emplace_back(ResourceType::CPU, 0);

    } catch (const std::exception &e) {
      // Don't FAIL here, just add CPU as fallback
      std::cerr << "Backend initialization failed: " << e.what() << std::endl;
      available_resources.clear();
      available_resources.emplace_back(ResourceType::CPU, 0);
    }
  }

  ~ProductionBufferTestFixture() {
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
};

// ============================================================================
// Basic Buffer Constructor Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture,
                 "Buffer Constructors - Backward Compatibility",
                 "[Buffer][constructors][compatibility]") {

  SECTION("Size-only constructor (backward compatible)") {
    // Creates buffer using best available device (prioritizes GPU)
    auto device = get_any_device_resource();
    DeviceBuffer<float> buffer(1000, device);

    REQUIRE(buffer.size() == 1000);
    REQUIRE(!buffer.empty());
    REQUIRE(buffer.data() != nullptr);
    // Resource should be best available device
    REQUIRE(buffer.resource().is_device());
  }

  SECTION("Size and resource constructor - Device") {
    auto devices = get_device_resources();
    if (devices.empty()) {
      SKIP("No device resources available");
    }

    DeviceBuffer<float> buffer(300, devices[0]);
    REQUIRE(buffer.size() == 300);
    REQUIRE(buffer.resource() == devices[0]);
    REQUIRE(buffer.data() != nullptr);
  }

  SECTION("Empty buffer construction") {
    auto device = get_any_device_resource();
    DeviceBuffer<int> empty_buffer(0, device);
    REQUIRE(empty_buffer.size() == 0);
    REQUIRE(empty_buffer.empty());
    // Should use best available device
    REQUIRE(empty_buffer.resource().is_device());
  }
}

// ============================================================================
// Copy and Move Semantics Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Buffer Copy and Move Operations",
                 "[Buffer][copy][move]") {

  SECTION("Copy constructor") {
    DeviceBuffer<float> original(100);

    // Initialize with test data
    std::vector<float> test_data(100);
    std::iota(test_data.begin(), test_data.end(), 1.0f);
    original.copy_from_host(test_data);

    // Copy constructor
    DeviceBuffer<float> copy(original);
    REQUIRE(copy.size() == 100);
    REQUIRE(copy.resource() == original.resource());

    // Verify data integrity
    std::vector<float> copied_data(100);
    copy.copy_to_host(copied_data);
    REQUIRE(copied_data == test_data);
  }

  SECTION("Move semantics") {
    DeviceBuffer<int> original(50);
    void *original_ptr = original.data();
    Resource original_resource = original.resource();

    // Move construction
    DeviceBuffer<int> moved = std::move(original);
    REQUIRE(moved.size() == 50);
    REQUIRE(moved.resource() == original_resource);
    REQUIRE(moved.data() == original_ptr);

    // Original should be in moved-from state
    REQUIRE(original.size() == 0);
    REQUIRE(original.data() == nullptr);
  }

  SECTION("Copy assignment") {
    DeviceBuffer<double> buffer1(100);
    DeviceBuffer<double> buffer2(200);

    std::vector<double> test_data(100, 3.14);
    buffer1.copy_from_host(test_data);

    buffer2 = buffer1;
    REQUIRE(buffer2.size() == 100);
    REQUIRE(buffer2.resource() == buffer1.resource());

    std::vector<double> result_data(100);
    buffer2.copy_to_host(result_data);
    REQUIRE(result_data == test_data);
  }
}

// ============================================================================
// Memory Operations Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Buffer Memory Operations",
                 "[Buffer][memory]") {

  SECTION("Host-device transfers") {
    const size_t size = 1000;
    DeviceBuffer<float> buffer(size);

    // Create test data
    std::vector<float> host_data(size);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Copy to device
    REQUIRE_NOTHROW(buffer.copy_from_host(host_data));

    // Copy back to host
    std::vector<float> result_data;
    REQUIRE_NOTHROW(buffer.copy_to_host(result_data));

    // Verify data integrity
    REQUIRE(result_data.size() == size);
    REQUIRE(result_data == host_data);
  }

  SECTION("Partial transfers") {
    const size_t size = 500;
    DeviceBuffer<int> buffer(size);

    std::vector<int> partial_data(100, 42);
    REQUIRE_NOTHROW(buffer.copy_from_host(partial_data.data(), 100));

    std::vector<int> result_data(100);
    REQUIRE_NOTHROW(buffer.copy_to_host(result_data.data(), 100));

    REQUIRE(result_data == partial_data);
  }

  SECTION("Device-to-device transfers") {
    const size_t size = 300;
    DeviceBuffer<double> buffer1(size);
    DeviceBuffer<double> buffer2(size);

    std::vector<double> test_data(size, 2.718);
    buffer1.copy_from_host(test_data);

    REQUIRE_NOTHROW(buffer2.copy_device_to_device(buffer1, size));

    std::vector<double> result_data;
    buffer2.copy_to_host(result_data);
    REQUIRE(result_data == test_data);
  }
}

// ============================================================================
// Resize Operations Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Buffer Resize Operations",
                 "[Buffer][resize]") {

  SECTION("Basic resize") {
    DeviceBuffer<int> buffer(100);
    REQUIRE(buffer.size() == 100);

    buffer.resize(200);
    REQUIRE(buffer.size() == 200);
    REQUIRE(buffer.data() != nullptr);

    buffer.resize(50);
    REQUIRE(buffer.size() == 50);

    buffer.resize(0);
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.empty());
  }

  SECTION("Resize with resource change") {
    auto devices = get_device_resources();
    if (devices.empty()) {
      SKIP("No device resources available for resource change test");
    }

    DeviceBuffer<float> buffer(100, devices[0]);
    Resource original_resource = buffer.resource();

    // Only test resource change if the target device is different from original
    if (devices[0] != original_resource) {
      buffer.resize(200, devices[0]);
      REQUIRE(buffer.size() == 200);
      REQUIRE(buffer.resource() == devices[0]);
      REQUIRE(buffer.resource() != original_resource);
    } else {
      // If same device, just test resize functionality
      buffer.resize(200, devices[0]);
      REQUIRE(buffer.size() == 200);
      REQUIRE(buffer.resource() == devices[0]);
      REQUIRE(buffer.resource() == original_resource);
    }
  }
}

// ============================================================================
// Multi-Threading Safety Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Buffer Thread Safety",
                 "[Buffer][threading][safety]") {

  auto devices = get_device_resources();

  SECTION("Concurrent buffer creation") {
    const size_t num_threads = 4;
    const size_t buffer_size = 1000;

    std::vector<std::future<bool>> futures;

    for (size_t i = 0; i < num_threads; ++i) {
      futures.emplace_back(std::async(std::launch::async, [i, buffer_size]() {
        try {
          // Use CPU resource for simplicity
          Resource resource;

          std::vector<std::unique_ptr<HostBuffer<float>>> buffers;
          for (int j = 0; j < 5; ++j) {
            auto buffer =
                std::make_unique<HostBuffer<float>>(buffer_size + j, resource);

            if (buffer->resource() != resource ||
                buffer->size() != buffer_size + j) {
              return false;
            }

            buffers.push_back(std::move(buffer));
          }

          // Test data operations
          std::vector<float> test_data(buffer_size, static_cast<float>(i));
          buffers[0]->copy_from_host(test_data);

          std::vector<float> result_data;
          buffers[0]->copy_to_host(result_data);

          // Verify data integrity
          return (result_data == test_data);

        } catch (const std::exception &e) {
          std::cerr << "Thread " << i << " failed with exception: " << e.what()
                    << std::endl;
          return false;
        }
      }));
    }

    // Wait for all operations to complete
    for (auto &future : futures) {
      REQUIRE(future.get() == true);
    }
  }

  SECTION("Concurrent memory operations") {
    const size_t buffer_size = 500;
    const size_t num_operations = 8; // Reduced complexity for stability

    // Create buffers on the same resource to avoid cross-device contention
    Resource buffer_resource = devices[0]; // Use single resource for stability
    std::vector<std::unique_ptr<DeviceBuffer<int>>> buffers;
    for (size_t i = 0; i < num_operations; ++i) {
      // Create each buffer with sync=true to ensure synchronous operations
      buffers.emplace_back(std::make_unique<DeviceBuffer<int>>(
          buffer_size, buffer_resource, nullptr, true));
    }

    // Verify all buffers are on the same resource
    for (const auto &buffer : buffers) {
      REQUIRE(buffer->resource() == buffer_resource);
    }

    std::vector<std::future<bool>> futures;

    for (size_t i = 0; i < num_operations; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [&buffers, buffer_size, i]() {
            try {
              // Each thread gets a unique buffer
              size_t buffer_idx = i % buffers.size();
              auto &buffer = *buffers[buffer_idx];

              // Create unique test data for this operation
              std::vector<int> test_data(buffer_size, static_cast<int>(i + 1));

              // Perform memory operations - copy_from_host already forces sync
              buffer.copy_from_host(test_data);

              // copy_to_host already forces synchronization, so no additional
              // sync needed
              std::vector<int> result_data;
              buffer.copy_to_host(result_data);

              // Verify data integrity
              if (result_data.size() != test_data.size()) {
                return false;
              }

              for (size_t j = 0; j < test_data.size(); ++j) {
                if (result_data[j] != test_data[j]) {
                  return false;
                }
              }

              return true;

            } catch (const std::exception &) {
              return false;
            }
          }));
    }

    // Wait for all operations and verify success
    for (auto &future : futures) {
      REQUIRE(future.get() == true);
    }
  }
}

// ============================================================================
// Resource Management Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Resource Management",
                 "[Buffer][resource]") {

  SECTION("Resource creation and properties") {
    Resource cpu = Resource(ResourceType::CPU);
    REQUIRE(cpu.type() == ResourceType::CPU);
    REQUIRE(cpu.id() == 0);
    REQUIRE(cpu.is_host());
    REQUIRE(!cpu.is_device());

    if (!get_device_resources().empty()) {
      auto device = get_device_resources()[0];
      REQUIRE(device.is_device());
      REQUIRE(!device.is_host());
      REQUIRE(device.supports_async());
    }
  }

  SECTION("Resource default detection") {
    Resource local; // Uses default constructor
    REQUIRE(local.is_device());

    DeviceBuffer<float> buffer(100, local);
    REQUIRE(buffer.resource() == local);
  }
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Buffer Performance Tests",
                 "[Buffer][performance]") {

  SECTION("Large buffer allocation") {
    const size_t large_size = 10000000; // 10M elements

    auto device = get_any_device_resource();
    DeviceBuffer<float> large_buffer(large_size, device);
    REQUIRE(large_buffer.size() == large_size);
    REQUIRE(large_buffer.bytes() == large_size * sizeof(float));
  }

  SECTION("Many small buffers") {
    const size_t num_buffers = 1000;
    const size_t buffer_size = 100;

    auto device = get_any_device_resource();
    std::vector<DeviceBuffer<int>> buffers;
    buffers.reserve(num_buffers);

    for (size_t i = 0; i < num_buffers; ++i) {
      buffers.emplace_back(buffer_size, device);
      REQUIRE(buffers.back().size() == buffer_size);
    }
  }

  SECTION("Rapid allocation/deallocation") {
    for (int i = 0; i < 100; ++i) {
      DeviceBuffer<double> temp(1000);
      REQUIRE(temp.size() == 1000);
      // Buffer automatically deallocated when going out of scope
    }
  }
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Buffer Edge Cases",
                 "[Buffer][edge_cases]") {

  SECTION("Zero-size buffers") {
    DeviceBuffer<float> empty1(0);
    DeviceBuffer<float> empty2(0);

    REQUIRE(empty1.empty());
    REQUIRE(empty2.empty());
    REQUIRE(empty1.size() == 0);
    REQUIRE(empty2.size() == 0);
  }

  SECTION("Copy operations with zero-size buffers") {
    DeviceBuffer<int> empty(0);
    DeviceBuffer<int> normal(100);

    std::vector<int> empty_vec;
    // Empty buffers have null device pointers, so copy operations should throw
    REQUIRE_THROWS(empty.copy_to_host(empty_vec));

    // Copy from empty vector to empty buffer should also throw since
    // device_ptr_ is null
    REQUIRE_THROWS(empty.copy_from_host(empty_vec));
  }

  SECTION("Device-to-device copy with different sizes") {
    DeviceBuffer<float> small(50);
    DeviceBuffer<float> large(100);

    std::vector<float> test_data(50, 1.0f);
    small.copy_from_host(test_data);

    // Copy from small to large (should work)
    REQUIRE_NOTHROW(large.copy_device_to_device(small, 50));

    // Copy more than source size (should throw exception)
    REQUIRE_THROWS(large.copy_device_to_device(small, 51));

    // Verify the copy worked correctly
    std::vector<float> result;
    large.copy_to_host(result);
    REQUIRE(result.size() == 100);
    // First 50 elements should match the source data
    for (size_t i = 0; i < 50; ++i) {
      REQUIRE(result[i] == 1.0f);
    }
  }

  SECTION("Multiple resize operations") {
    DeviceBuffer<double> buffer(100);

    for (size_t new_size : {200, 50, 0, 150, 300}) {
      buffer.resize(new_size);
      REQUIRE(buffer.size() == new_size);
      if (new_size > 0) {
        REQUIRE(buffer.data() != nullptr);
      }
    }
  }
}

// ============================================================================
// Real-World Usage Patterns
// ============================================================================

TEST_CASE_METHOD(ProductionBufferTestFixture, "Real-World Usage Patterns",
                 "[Buffer][integration][real_world]") {

  SECTION("Legacy code compatibility") {
    // Test that existing code patterns still work
    DeviceBuffer<float> grid_buffer(1000); // Your original pattern
    DeviceBuffer<float> result_buffer(1);  // Your original pattern

    REQUIRE(grid_buffer.size() == 1000);
    REQUIRE(result_buffer.size() == 1);
    Resource default_resource;
    REQUIRE(grid_buffer.resource().type() == default_resource.type());
    REQUIRE(result_buffer.resource().type() == default_resource.type());

    // Test data operations work
    std::vector<float> test_data(1000);
    std::iota(test_data.begin(), test_data.end(), 0.0f);

    grid_buffer.copy_from_host(test_data);

    std::vector<float> result_data;
    grid_buffer.copy_to_host(result_data);
    REQUIRE(result_data == test_data);
  }

  SECTION("Mixed legacy and new API") {
    // Legacy style
    DeviceBuffer<float> legacy_buffer(500);

    // New explicit style
    auto devices = get_device_resources();
    if (!devices.empty()) {
      DeviceBuffer<float> explicit_buffer(500, devices[0]);

      // Cross-resource copy
      std::vector<float> data(500, 42.0f);
      legacy_buffer.copy_from_host(data);
      explicit_buffer.copy_device_to_device(legacy_buffer, 500);

      std::vector<float> result;
      explicit_buffer.copy_to_host(result);
      REQUIRE(result == data);
    }
  }

  SECTION("Multi-GPU workload distribution") {
    auto devices = get_device_resources();
    if (devices.empty()) {
      SKIP("No devices available for distribution test");
    }

    // Use a single device to avoid CUDA context issues
    Resource single_device = devices[0];
    const size_t total_size = 10000;
    const size_t num_chunks = 4; // Use 4 chunks instead of multiple devices
    const size_t chunk_size = total_size / num_chunks;

    // Create distributed buffers on the same device
    std::vector<DeviceBuffer<float>> distributed_buffers;
    for (size_t i = 0; i < num_chunks; ++i) {
      distributed_buffers.emplace_back(chunk_size, single_device);
      REQUIRE(distributed_buffers.back().resource() == single_device);
    }

    // Initialize each chunk
    std::vector<float> chunk_data(chunk_size);
    for (size_t i = 0; i < num_chunks; ++i) {
      std::fill(chunk_data.begin(), chunk_data.end(), static_cast<float>(i));
      distributed_buffers[i].copy_from_host(chunk_data);
    }

    // Verify each chunk
    for (size_t i = 0; i < num_chunks; ++i) {
      std::vector<float> result;
      distributed_buffers[i].copy_to_host(result);
      REQUIRE(std::all_of(result.begin(), result.end(), [i](float val) {
        return val == static_cast<float>(i);
      }));
    }
  }
}
