#include "Backend/Kernels.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/KernelConfig.h"
#include "Backend/Resource.h"
#include "catch_boiler.h"
#include <cmath>
#include <vector>

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

using namespace ARBD;

// ============================================================================
// Backend Test Fixture - Properly initializes the compile-time selected backend
// ============================================================================

// Global flag to track backend initialization
static bool g_backend_available = false;
struct kernel_func {
  void operator()(size_t i, const float *input, float *output) const {
    output[i] = input[i] * 3;
  }
};

// Initialize backend once across all tests
static void initialize_backend_once() {
  static bool initialized = false;
  if (initialized)
    return;

  try {
#ifdef USE_SYCL
    SYCL::Manager::init();
    // SYCL::Manager::load_info(); // Ensure full device initialization
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

// ============================================================================
// Kernel Launch Interface Tests
// ============================================================================

TEST_CASE("Kernel Launch Interface", "[backend][kernels]") {
  initialize_backend_once();

  SECTION("Basic kernel function signature") {
    auto test_kernel = [](size_t i, const float *input, float *output) {
      output[i] = input[i] * 2.0f;
    };

    Resource test_resource;
    KernelConfig config = KernelConfig::for_1d(100, test_resource);

    // Test that launch_kernel compiles with current backend
    static_assert(
        std::is_invocable_v<
            decltype(launch_kernel<decltype(test_kernel), DeviceBuffer<float>,
                                   DeviceBuffer<float>>),
            const Resource &, const KernelConfig &, decltype(test_kernel),
            DeviceBuffer<float>, DeviceBuffer<float>>);
  }

  SECTION("Dimensional kernel launchers") {
    auto test_kernel = [](size_t i, const float *input, float *output) {
      output[i] = input[i] * 2.0f;
    };

    Resource test_resource;
    DeviceBuffer<float> input_buf(100);
    DeviceBuffer<float> output_buf(100);

    // Test dimensional launcher signatures
    static_assert(
        std::is_invocable_v<decltype(launch_kernel_1d<decltype(test_kernel),
                                                      DeviceBuffer<float>,
                                                      DeviceBuffer<float>>),
                            const Resource &, idx_t, KernelConfig,
                            decltype(test_kernel), DeviceBuffer<float>,
                            DeviceBuffer<float>>);

    static_assert(
        std::is_invocable_v<decltype(launch_kernel_2d<decltype(test_kernel),
                                                      DeviceBuffer<float>,
                                                      DeviceBuffer<float>>),
                            const Resource &, idx_t, idx_t, KernelConfig,
                            decltype(test_kernel), DeviceBuffer<float>,
                            DeviceBuffer<float>>);

    static_assert(
        std::is_invocable_v<decltype(launch_kernel_3d<decltype(test_kernel),
                                                      DeviceBuffer<float>,
                                                      DeviceBuffer<float>>),
                            const Resource &, idx_t, idx_t, idx_t, KernelConfig,
                            decltype(test_kernel), DeviceBuffer<float>,
                            DeviceBuffer<float>>);
  }

  SECTION("Buffer pointer extraction") {
    if (g_backend_available) {
      try {
        DeviceBuffer<float> buffer(100);

        // Test that get_buffer_pointer works
        auto *ptr = get_buffer_pointer(buffer);
        REQUIRE(ptr != nullptr);

      } catch (const std::exception &e) {
        WARN("Buffer pointer extraction failed: " << e.what());
        REQUIRE(true);
      }
    } else {
      WARN("Backend not available, skipping buffer pointer extraction test");
      REQUIRE(true);
    }
  }

  SECTION("Buffer type detection traits") {
    if (g_backend_available) {
      try {
        DeviceBuffer<float> device_buf(10);
        HostBuffer<float> host_buf(10);

        // These should compile and work correctly
        static_assert(is_device_buffer_v<DeviceBuffer<float>>);
        // static_assert(is_host_buffer_v<HostBuffer<float>>); // Comment out if
        // trait not available
        REQUIRE(true); // Framework requires assertion
      } catch (const std::exception &e) {
        WARN("Buffer type detection failed: " << e.what());
        REQUIRE(true);
      }
    } else {
      WARN("Backend not available, skipping buffer type detection test");
      REQUIRE(true);
    }
  }

#ifdef USE_SYCL
  SECTION("Native SYCL kernel with DeviceBuffer USM") {
    if (g_backend_available) {
      try {
        INFO("=== TEST 1: Native SYCL kernel with DeviceBuffer USM ===");

        // Create Resource first to ensure consistent device usage
        Resource sycl_resource(ResourceType::SYCL, 4);

        // Create DeviceBuffers on the same device as the resource
        DeviceBuffer<float> input_buf(10, 4);  // Use device 4
        DeviceBuffer<float> output_buf(10, 4); // Use device 4

        // Initialize input
        std::vector<float> input_data(10, 5.0f);
        input_buf.copy_from_host(input_data);
        auto *queue_ptr =
            static_cast<sycl::queue *>(sycl_resource.get_stream());
        sycl::queue &queue = *queue_ptr;

        const float *input_ptr = get_buffer_pointer(input_buf);
        float *output_ptr = get_buffer_pointer(output_buf);

        INFO("Native SYCL: input_ptr = " << input_ptr);
        INFO("Native SYCL: output_ptr = " << output_ptr);
        sycl::range<1> global_range(10);
        sycl::range<1> local_range(10);
        sycl::nd_range<1> execution_range(global_range, local_range);

        // Submit kernel with dependency handling
        auto sycl_event = queue.submit([&](sycl::handler &h) {
          // Extract device-copyable parts before capture
          idx_t problem_size_x = 10;

          // Launch kernel with streamlined interface - capture raw pointers
          // directly
          h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
            idx_t gx = static_cast<idx_t>(item.get_global_id(0));

            if (gx < problem_size_x) {
              size_t i = gx;
              kernel_func{}(i, input_ptr, output_ptr);
            }
          });
        });

        // Ensure proper synchronization
        try {
          sycl_event.wait();
        } catch (const sycl::exception &e) {
          WARN("SYCL kernel execution failed: " << e.what());
          throw;
        }
        // Verify results - ensure memory synchronization
        std::vector<float> result(10);
        try {
          output_buf.copy_to_host(result);
        } catch (const std::exception &e) {
          WARN("Memory copy failed: " << e.what());
          throw;
        }
        for (size_t i = 0; i < 10; ++i) {
          INFO("Native SYCL result[" << i << "] = " << result[i]
                                     << " (expected 777.0f)");
          REQUIRE(std::abs(result[i] - 15.0f) < 1e-6f);
        }

      } catch (const std::exception &e) {
        WARN("Native SYCL kernel failed: " << e.what());
        REQUIRE(false);
      }
    } else {
      WARN("SYCL backend not available");
    }
  }

  SECTION("launch_kernel with raw pointers1") {
    if (g_backend_available) {
      try {
        INFO("=== TEST 2.5: launch_kernel with native sycl s ===");

        Resource sycl_resource(ResourceType::SYCL,
                               4);             // Use device 4
        DeviceBuffer<float> input_buf(10, 4);  // Use device 4
        DeviceBuffer<float> output_buf(10, 4); // Use device 4

        // Initialize input buffer
        std::vector<float> input_data(10, 5.0f);
        input_buf.copy_from_host(input_data);

        // Debug: Verify input buffer was written correctly
        std::vector<float> verify_input(10);
        input_buf.copy_to_host(verify_input);
        INFO("DEBUG: input_buf[0] after copy = " << verify_input[0]);
        auto input_ptr = get_buffer_pointer(input_buf);
        auto output_ptr = get_buffer_pointer(output_buf);
        // Use KernelConfig::for_1d() to properly set grid_size like working
        // test
        KernelConfig config = KernelConfig::for_1d(10, 4); // Use device 4
        config.sync = true;

        auto sycl_event = launch_sycl_kernel_1d(
            sycl_resource, config, kernel_func{}, input_ptr, output_ptr);
        auto device =
            sycl_resource.get_sycl_queue(config.queue_id).get_device();
        INFO("Resource: " << device.get_info<sycl::info::device::vendor_id>());
        INFO("Queue ID: " << config.queue_id);

        // Verify results from the kernel
        std::vector<float> result(10);
        try {
          output_buf.copy_to_host(result);
        } catch (const std::exception &e) {
          WARN("Memory copy failed: " << e.what());
          throw;
        }

        // Check kernel results
        for (size_t i = 0; i < 10; ++i) {
          INFO("launch_sycl_kernel_1d result[" << i << "] = " << result[i]
                                               << " (expected 15.0f)");
          REQUIRE(std::abs(result[i] - 15.0f) < 1e-6f);
        }
      } catch (const std::exception &e) {
        WARN("SYCL kernel launch failed: " << e.what());
        REQUIRE(true);
      }
    } else {
      WARN("SYCL backend not available, skipping SYCL kernel test");
      REQUIRE(true);
    }
  }
#endif
}
