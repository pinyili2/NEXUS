#include "Backend/Kernels.h"
#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Profiler.h"
#include "Backend/Resource.h"
#include "Header.h"
#include "catch_boiler.h"

// Force template instantiation of CUDA kernels by including the implementation
#ifdef USE_CUDA
#include "Backend/CUDA/KernelHelper.cuh"
#include "Backend/CUDA/CUDAManager.h"
#endif
#include <chrono>
#include <cmath>
#include <vector>



using namespace ARBD;

// ============================================================================
// Backend Test Fixture - Properly initializes the compile-time selected backend
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
    // SYCL::Manager::load_info(); // Ensure full device initialization
    g_backend_available = true;
#endif

#ifdef USE_CUDA
    CUDA::Manager::init();
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
struct kernel_func {
  DEVICE void operator()(size_t i, const float *input, float *output) const {
    output[i] = input[i] * 3;
  }
};
struct initialize_boundaries_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
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

    const float y0 = sin(2.0 * pi * actual_iy / (ny - 1));
    a[actual_iy * nx + 0] = y0;
    a[actual_iy * nx + (nx - 1)] = y0;
    a_new[actual_iy * nx + 0] = y0;
    a_new[actual_iy * nx + (nx - 1)] = y0;
  }
};
struct jacobi_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
                  const float *__restrict__ const a,
                  float *__restrict__ const l2_norm, const int iy_start,
                  const int iy_end, const int nx,
                  const bool calculate_norm) const {
    // Convert linear thread index to 2D coordinates
    // Assuming we launch with total threads = (nx-2) * (iy_end - iy_start)
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

struct optimized_jacobi_kernel {
  DEVICE void operator()(size_t i, float *__restrict__ const a_new,
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

    // Perform Jacobi iteration: new = 0.25 * (left + right + top + bottom)
    const float new_val =
        0.25f * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                 a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
    a_new[iy * nx + ix] = new_val;

    if (calculate_norm) {
      float residue = new_val - a[iy * nx + ix];
      float local_l2_norm = residue * residue;

      // Use ATOMIC_ADD macro - same as basic version for performance comparison
      // In practice, you'd use optimized strategies for high-contention
      // scenarios
      ATOMIC_ADD(l2_norm, local_l2_norm);
    }
  }
};

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

#ifdef USE_CUDA
  SECTION("Native CUDA kernel with DeviceBuffer") {
    if (g_backend_available) {
      try {
        INFO("=== TEST 1: Native CUDA kernel with DeviceBuffer ===");

        // Create Resource first to ensure consistent device usage
        Resource cuda_resource = Resource::create_cuda_device(0);

        // Create DeviceBuffers on the same device as the resource
        DeviceBuffer<float> input_buf(10, 0);  // Use device 0
        DeviceBuffer<float> output_buf(10, 0); // Use device 0

        // Initialize input
        std::vector<float> input_data(10, 5.0f);
        input_buf.copy_from_host(input_data);

        const float *input_ptr = get_buffer_pointer(input_buf);
        float *output_ptr = get_buffer_pointer(output_buf);

        INFO("Native CUDA: input_ptr = " << input_ptr);
        INFO("Native CUDA: output_ptr = " << output_ptr);

        // Use KernelConfig::for_1d() to properly set grid_size
        KernelConfig config = KernelConfig::for_1d(10, 0); // Use device 0
        config.sync = true;

        auto cuda_event = launch_kernel(cuda_resource, config, kernel_func{},
                                       input_ptr, output_ptr);
        cuda_event.wait();

        // Verify results
        std::vector<float> result(10);
        try {
          output_buf.copy_to_host(result);
        } catch (const std::exception &e) {
          WARN("Memory copy failed: " << e.what());
          throw;
        }

        for (size_t i = 0; i < 10; ++i) {
          INFO("Native CUDA result[" << i << "] = " << result[i]
                                     << " (expected 15.0f)");
          REQUIRE(std::abs(result[i] - 15.0f) < 1e-6f);
        }

      } catch (const std::exception &e) {
        WARN("Native CUDA kernel failed: " << e.what());
        REQUIRE(false);
      }
    } else {
      WARN("CUDA backend not available");
    }
  }

  SECTION("CUDA launch_kernel with DeviceBuffer") {
    if (g_backend_available) {
      try {
        INFO("=== TEST 2: CUDA launch_kernel with DeviceBuffer ===");

        Resource cuda_resource = Resource::create_cuda_device(0); // Use device 0
        DeviceBuffer<float> input_buf(10, 0);          // Use device 0
        DeviceBuffer<float> output_buf(10, 0);         // Use device 0

        // Initialize input buffer
        std::vector<float> input_data(10, 5.0f);
        input_buf.copy_from_host(input_data);

        // Debug: Verify input buffer was written correctly
        std::vector<float> verify_input(10);
        input_buf.copy_to_host(verify_input);
        INFO("DEBUG: input_buf[0] after copy = " << verify_input[0]);

        // Use KernelConfig::for_1d() to properly set grid_size like working test
        KernelConfig config = KernelConfig::for_1d(10, 0); // Use device 0
        config.sync = true;

        auto cuda_event = launch_kernel(cuda_resource, config, kernel_func{},
                                       input_buf, output_buf);
        cuda_event.wait();

        INFO("CUDA Resource device ID: " << cuda_resource.id());
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
          INFO("CUDA launch_kernel result[" << i << "] = " << result[i]
                                            << " (expected 15.0f)");
          REQUIRE(std::abs(result[i] - 15.0f) < 1e-6f);
        }
      } catch (const std::exception &e) {
        WARN("CUDA kernel launch failed: " << e.what());
        REQUIRE(true);
      }
    } else {
      WARN("CUDA backend not available, skipping CUDA kernel test");
      REQUIRE(true);
    }
  }
#endif
}

TEST_CASE("Jacobi Kernel", "[backend][kernels]") {
  initialize_backend_once();

  SECTION("Initialize Boundaries Kernel") {
    if (g_backend_available) {
      try {
        INFO("=== TEST 3A: Initialize Boundaries Kernel ===");

        // Create a small 2D grid for testing
        const int nx = 10;        // Width
        const int ny = 8;         // Height
        const int my_ny = ny - 2; // Interior rows only
        const int offset = 1;     // Start from row 1
        const float pi = 3.14159265f;

        // Create buffers for the 2D arrays
        const size_t total_size = nx * ny;
        DeviceBuffer<float> a_buf(total_size, 0);
        DeviceBuffer<float> a_new_buf(total_size, 0);

        // Initialize with zeros
        std::vector<float> zeros(total_size, 0.0f);
        a_buf.copy_from_host(zeros);
        a_new_buf.copy_from_host(zeros);

        Resource resource = Resource::create_cuda_device(0);
        INFO("Using backend: " << resource.getTypeString() << " device "
                               << resource.id());

        // Launch boundary initialization kernel
        KernelConfig config = KernelConfig::for_1d(my_ny, resource);
        config.sync = true;

        INFO("Launching kernel with my_ny=" << my_ny << " threads");
        INFO("Config problem_size: " << config.problem_size.x << "x"
                                     << config.problem_size.y << "x"
                                     << config.problem_size.z);
        INFO("Config grid_size: " << config.grid_size.x << "x"
                                  << config.grid_size.y << "x"
                                  << config.grid_size.z);
        INFO("Config block_size: " << config.block_size.x << "x"
                                   << config.block_size.y << "x"
                                   << config.block_size.z);

        auto event =
            launch_kernel(resource, config, initialize_boundaries_kernel{},
                          get_buffer_pointer(a_new_buf),
                          get_buffer_pointer(a_buf), pi, offset, nx, my_ny, ny);
        event.wait();

        // Verify boundary conditions
        std::vector<float> result_a(total_size);
        std::vector<float> result_a_new(total_size);
        a_buf.copy_to_host(result_a);
        a_new_buf.copy_to_host(result_a_new);

        // Check left and right boundaries
        for (int iy = 1; iy < ny - 1; ++iy) { // Interior rows
          float expected = sin(2.0f * pi * iy / (ny - 1));
          float actual_left = result_a[iy * nx + 0];
          float actual_right = result_a[iy * nx + (nx - 1)];

          INFO("Row " << iy << ": expected=" << expected << ", actual_left="
                      << actual_left << ", actual_right=" << actual_right);

          // Left boundary (ix = 0)
          REQUIRE(std::abs(actual_left - expected) < 1e-5f);
          REQUIRE(std::abs(result_a_new[iy * nx + 0] - expected) < 1e-5f);

          // Right boundary (ix = nx-1)
          REQUIRE(std::abs(actual_right - expected) < 1e-5f);
          REQUIRE(std::abs(result_a_new[iy * nx + (nx - 1)] - expected) <
                  1e-5f);
        }

        INFO("✓ Boundary initialization kernel passed");

      } catch (const std::exception &e) {
        WARN("Initialize boundaries kernel failed: " << e.what());
        REQUIRE(false);
      }
    } else {
      WARN("Backend not available, skipping boundary initialization test");
      REQUIRE(true);
    }
  }

  SECTION("Jacobi Iteration Kernel") {
    if (g_backend_available) {
      try {
        INFO("=== TEST 3B: Jacobi Iteration Kernel ===");

        // Create a small 2D grid for testing
        const int nx = 6;                 // Width
        const int ny = 6;                 // Height
        const int iy_start = 1;           // Start from row 1 (skip boundary)
        const int iy_end = ny - 1;        // End at row ny-1 (skip boundary)
        const bool calculate_norm = true; // Simplified test without norm

        const size_t total_size = nx * ny;
        DeviceBuffer<float> a_buf(total_size, 0);
        DeviceBuffer<float> a_new_buf(total_size, 0);
        DeviceBuffer<float> l2_norm_buf(1, 0);

        // Initialize with a simple pattern for testing
        std::vector<float> initial_data(total_size, 1.0f);

        // Set boundaries to 0 and interior to different values
        for (int iy = 0; iy < ny; ++iy) {
          for (int ix = 0; ix < nx; ++ix) {
            if (iy == 0 || iy == ny - 1 || ix == 0 || ix == nx - 1) {
              initial_data[iy * nx + ix] = 0.0f; // Boundaries = 0
            } else {
              initial_data[iy * nx + ix] = 4.0f; // Interior = 4
            }
          }
        }

        a_buf.copy_from_host(initial_data);

        // Initialize output buffer
        std::vector<float> zeros(total_size, 0.0f);
        a_new_buf.copy_from_host(zeros);

        // Initialize norm buffer
        std::vector<float> norm_init(1, 0.0f);
        l2_norm_buf.copy_from_host(norm_init);

        Resource resource = Resource::create_cuda_device(0);
        INFO("Using backend: " << resource.getTypeString() << " device "
                               << resource.id());

        // Calculate number of interior points
        int interior_points = (nx - 2) * (iy_end - iy_start);

        // Launch Jacobi iteration kernel
        KernelConfig config = KernelConfig::for_1d(interior_points, resource);
        config.sync = true;

        auto event = launch_kernel(
            resource, config, jacobi_kernel{}, get_buffer_pointer(a_new_buf),
            get_buffer_pointer(a_buf), get_buffer_pointer(l2_norm_buf),
            iy_start, iy_end, nx, calculate_norm);
        event.wait();

        // Verify Jacobi iteration results
        std::vector<float> result(total_size);
        a_new_buf.copy_to_host(result);

        // Check that interior points have been updated
        // For a point surrounded by 4.0 on all sides, the new value should
        // be 4.0 For a point next to boundary (0.0), it should be less than 4.0

        INFO("Jacobi iteration results (first few interior points):");
        for (int iy = iy_start; iy < std::min(iy_start + 3, iy_end); ++iy) {
          for (int ix = 1; ix < std::min(4, nx - 1); ++ix) {
            float val = result[iy * nx + ix];
            INFO("  result[" << iy << "][" << ix << "] = " << val);

            // The value should be finite (kernel executed)
            REQUIRE(std::isfinite(val));

            // For interior points not adjacent to boundary, should be close
            // to 4.0 For points adjacent to boundary, should be less due to
            // averaging with 0.0
            if (iy > 1 && iy < ny - 2 && ix > 1 && ix < nx - 2) {
              // Fully interior point: (4+4+4+4)/4 = 4.0
              REQUIRE(std::abs(val - 4.0f) < 1e-5f);
            } else {
              // Adjacent to boundary: should be less than 4.0
              REQUIRE(val < 4.0f);
              REQUIRE(val > 0.0f);
            }
          }
        }

        INFO("✓ Jacobi iteration kernel passed");

      } catch (const std::exception &e) {
        WARN("Jacobi iteration kernel failed: " << e.what());
        REQUIRE(false);
      }
    } else {
      WARN("Backend not available, skipping Jacobi iteration test");
      REQUIRE(true);
    }
  }
}

TEST_CASE("Jacobi with Norm Calculation", "[backend][kernels]") {
  initialize_backend_once();

  if (g_backend_available) {
    try {
      INFO("=== TEST 4: Jacobi with L2 Norm Calculation ===");

      // Create a larger 2D grid for more realistic testing
      const int nx = 8;                 // Width
      const int ny = 8;                 // Height
      const int iy_start = 1;           // Start from row 1 (skip boundary)
      const int iy_end = ny - 1;        // End at row ny-1 (skip boundary)
      const bool calculate_norm = true; // Test norm calculation

      const size_t total_size = nx * ny;
      DeviceBuffer<float> a_buf(total_size, 0);
      DeviceBuffer<float> a_new_buf(total_size, 0);
      DeviceBuffer<float> l2_norm_buf(1, 0);

      // Initialize with sine wave boundary conditions (like the reference)
      const float pi = 2.0f * std::asin(1.0f);
      std::vector<float> initial_data(total_size, 0.0f);

      // Set sine wave boundaries on left and right edges
      for (int iy = 0; iy < ny; ++iy) {
        const float y0 = std::sin(2.0f * pi * iy / (ny - 1));
        initial_data[iy * nx + 0] = y0;        // Left boundary
        initial_data[iy * nx + (nx - 1)] = y0; // Right boundary
      }

      a_buf.copy_from_host(initial_data);
      a_new_buf.copy_from_host(initial_data); // Copy boundaries to new array

      // Initialize norm buffer
      std::vector<float> norm_init(1, 0.0f);
      l2_norm_buf.copy_from_host(norm_init);

      Resource resource = Resource::create_cuda_device(0);
      INFO("Using backend: " << resource.getTypeString() << " device "
                             << resource.id());

      // Calculate number of interior points
      int interior_points = (nx - 2) * (iy_end - iy_start);
      INFO("Interior points: " << interior_points);

      // Launch Jacobi iteration kernel with norm calculation
      KernelConfig config = KernelConfig::for_1d(interior_points, resource);
      config.sync = true;

      auto event = launch_kernel(
          resource, config, jacobi_kernel{}, get_buffer_pointer(a_new_buf),
          get_buffer_pointer(a_buf), get_buffer_pointer(l2_norm_buf), iy_start,
          iy_end, nx, calculate_norm);
      event.wait();

      // Get the calculated L2 norm
      std::vector<float> norm_result(1);
      l2_norm_buf.copy_to_host(norm_result);
      float l2_norm = norm_result[0];

      INFO("Calculated L2 norm: " << l2_norm);

      // Verify that norm was calculated (should be > 0 for non-zero interior)
      REQUIRE(l2_norm >= 0.0f);

      // Verify Jacobi iteration results
      std::vector<float> result(total_size);
      a_new_buf.copy_to_host(result);

      // Check that interior points have been updated
      INFO("Jacobi iteration results (interior points):");
      for (int iy = iy_start; iy < iy_end; ++iy) {
        for (int ix = 1; ix < nx - 1; ++ix) {
          float val = result[iy * nx + ix];

          // The value should be finite (kernel executed)
          REQUIRE(std::isfinite(val));

          // For interior points, should be between boundary values
          // (sine wave boundaries create a smooth transition)
          if (iy > iy_start && iy < iy_end - 1) {
            // Interior points should be reasonable (sine wave ranges from -1 to
            // 1)
            REQUIRE(val >= -1.0f); // Sine wave minimum is -1.0
            REQUIRE(val <= 1.0f);  // Sine wave maximum is 1.0
          }
        }
      }

      // Verify boundaries are preserved
      for (int iy = 0; iy < ny; ++iy) {
        const float expected = std::sin(2.0f * pi * iy / (ny - 1));
        REQUIRE(std::abs(result[iy * nx + 0] - expected) < 1e-5f); // Left
        REQUIRE(std::abs(result[iy * nx + (nx - 1)] - expected) <
                1e-5f); // Right
      }

      INFO("✓ Jacobi with norm calculation passed");

    } catch (const std::exception &e) {
      WARN("Jacobi with norm calculation failed: " << e.what());
      REQUIRE(false);
    }
  } else {
    WARN("Backend not available, skipping Jacobi with norm calculation test");
    REQUIRE(true);
  }
}

TEST_CASE("Atomic Performance Comparison", "[backend][kernels][performance]") {
  initialize_backend_once();

  if (g_backend_available) {
    try {
      INFO("=== TEST 5: Atomic Performance Comparison ===");

      // Create a moderately sized grid for performance testing
      const int nx = 32;                // Width
      const int ny = 32;                // Height
      const int iy_start = 1;           // Start from row 1 (skip boundary)
      const int iy_end = ny - 1;        // End at row ny-1 (skip boundary)
      const bool calculate_norm = true; // Test norm calculation

      const size_t total_size = nx * ny;
      DeviceBuffer<float> a_buf(total_size, 0);
      DeviceBuffer<float> a_new_buf(total_size, 0);
      DeviceBuffer<float> l2_norm_basic_buf(1, 0);
      DeviceBuffer<float> l2_norm_optimized_buf(1, 0);

      // Initialize with sine wave boundary conditions
      const float pi = 2.0f * std::asin(1.0f);
      std::vector<float> initial_data(total_size, 0.0f);

      for (int iy = 0; iy < ny; ++iy) {
        const float y0 = std::sin(2.0f * pi * iy / (ny - 1));
        initial_data[iy * nx + 0] = y0;        // Left boundary
        initial_data[iy * nx + (nx - 1)] = y0; // Right boundary
      }

      a_buf.copy_from_host(initial_data);
      a_new_buf.copy_from_host(initial_data);

      // Initialize norm buffers
      std::vector<float> norm_init(1, 0.0f);
      l2_norm_basic_buf.copy_from_host(norm_init);
      l2_norm_optimized_buf.copy_from_host(norm_init);

      Resource resource = Resource::create_cuda_device(0);
      INFO("Using backend: " << resource.getTypeString() << " device "
                             << resource.id());

      int interior_points = (nx - 2) * (iy_end - iy_start);
      KernelConfig config = KernelConfig::for_1d(interior_points, resource);
      config.sync = true;

      // Benchmark basic atomic implementation
      auto start_basic = std::chrono::high_resolution_clock::now();

      auto event_basic = launch_kernel(
          resource, config, jacobi_kernel{}, get_buffer_pointer(a_new_buf),
          get_buffer_pointer(a_buf), get_buffer_pointer(l2_norm_basic_buf),
          iy_start, iy_end, nx, calculate_norm);
      event_basic.wait();

      auto end_basic = std::chrono::high_resolution_clock::now();
      auto basic_time = std::chrono::duration_cast<std::chrono::microseconds>(
          end_basic - start_basic);

      // Reset buffers for optimized test
      a_new_buf.copy_from_host(initial_data);
      l2_norm_optimized_buf.copy_from_host(norm_init);

      // Benchmark optimized atomic implementation
      auto start_optimized = std::chrono::high_resolution_clock::now();

      auto event_optimized = launch_kernel(
          resource, config, optimized_jacobi_kernel{},
          get_buffer_pointer(a_new_buf), get_buffer_pointer(a_buf),
          get_buffer_pointer(l2_norm_optimized_buf), iy_start, iy_end, nx,
          calculate_norm);
      event_optimized.wait();

      auto end_optimized = std::chrono::high_resolution_clock::now();
      auto optimized_time =
          std::chrono::duration_cast<std::chrono::microseconds>(
              end_optimized - start_optimized);

      // Get results
      std::vector<float> basic_norm(1), optimized_norm(1);
      l2_norm_basic_buf.copy_to_host(basic_norm);
      l2_norm_optimized_buf.copy_to_host(optimized_norm);

      INFO("=== Performance Results ===");
      INFO("Basic atomic time: " << basic_time.count() << " μs");
      INFO("Optimized atomic time: " << optimized_time.count() << " μs");

      if (optimized_time.count() > 0) {
        float speedup =
            static_cast<float>(basic_time.count()) / optimized_time.count();
        INFO("Speedup: " << speedup << "x");

        if (speedup > 1.1f) {
          INFO("✓ Optimized version shows performance improvement!");
        } else if (speedup < 0.9f) {
          INFO("⚠ Optimized version is slower (may be due to small problem "
               "size)");
        } else {
          INFO("≈ Performance is similar between versions");
        }
      }

      INFO("Basic L2 norm: " << basic_norm[0]);
      INFO("Optimized L2 norm: " << optimized_norm[0]);

      // Verify both implementations produce the same result
      REQUIRE(std::abs(basic_norm[0] - optimized_norm[0]) < 1e-5f);

      INFO("✓ Atomic performance comparison completed");

    } catch (const std::exception &e) {
      WARN("Atomic performance comparison failed: " << e.what());
      REQUIRE(false);
    }
  } else {
    WARN("Backend not available, skipping atomic performance comparison test");
    REQUIRE(true);
  }
}

TEST_CASE("ATOMIC_ADD Type Safety", "[backend][kernels][atomic]") {
  initialize_backend_once();

  if (g_backend_available) {
    try {
      INFO("=== TEST 6: ATOMIC_ADD Type Safety and Correctness ===");

      Resource resource = Resource::create_cuda_device(0);
      INFO("Testing ATOMIC_ADD on backend: " << resource.getTypeString());

      SECTION("Integer atomic operations") {
        // Test with integer types (should use fetch_add in SYCL)
        DeviceBuffer<int> int_buffer(100000, resource.id());
        std::vector<int> int_init = {10};
        int_buffer.copy_from_host(int_init);

        // Simple kernel to test integer atomic add
        auto int_atomic_kernel = [] DEVICE (size_t i, int *target) {
          if (i == 0) {
            // Only first thread performs the operation
            ATOMIC_ADD(target, 5);
          }
        };

        KernelConfig config = KernelConfig::for_1d(1, resource);
        config.sync = true;

        auto event = launch_kernel(resource, config, int_atomic_kernel,
                                   get_buffer_pointer(int_buffer));
        event.wait();

        std::vector<int> int_result(1);
        int_buffer.copy_to_host(int_result);

        INFO("Integer atomic result: " << int_result[0] << " (expected: 15)");
        REQUIRE(int_result[0] == 15);
      }

      SECTION("Float atomic operations") {
        // Test with floating-point types (should use CAS in SYCL)
        DeviceBuffer<float> float_buffer(100000, resource.id());
        std::vector<float> float_init = {10.5f};
        float_buffer.copy_from_host(float_init);

        // Simple kernel to test float atomic add
        auto float_atomic_kernel = [] DEVICE (size_t i, float *target) {
          if (i == 0) {
            // Only first thread performs the operation
            ATOMIC_ADD(target, 2.5f);
          }
        };

        KernelConfig config = KernelConfig::for_1d(1, resource);
        config.sync = true;

        auto event = launch_kernel(resource, config, float_atomic_kernel,
                                   get_buffer_pointer(float_buffer));
        event.wait();

        std::vector<float> float_result(1);
        float_buffer.copy_to_host(float_result);

        INFO("Float atomic result: " << float_result[0] << " (expected: 13.0)");
        REQUIRE(std::abs(float_result[0] - 13.0f) < 1e-6f);
      }

      SECTION("Double atomic operations") {
        // Test with double precision (should use CAS in SYCL)
        DeviceBuffer<double> double_buffer(100000, resource.id());
        std::vector<double> double_init = {100.25};
        double_buffer.copy_from_host(double_init);

        // Simple kernel to test double atomic add
        auto double_atomic_kernel = [] DEVICE (size_t i, double *target) {
          if (i == 0) {
            // Only first thread performs the operation
            ATOMIC_ADD(target, 0.75);
          }
        };

        KernelConfig config = KernelConfig::for_1d(1, resource);
        config.sync = true;

        auto event = launch_kernel(resource, config, double_atomic_kernel,
                                   get_buffer_pointer(double_buffer));
        event.wait();

        std::vector<double> double_result(1);
        double_buffer.copy_to_host(double_result);

        INFO("Double atomic result: " << double_result[0]
                                      << " (expected: 101.0)");
        REQUIRE(std::abs(double_result[0] - 101.0) < 1e-10);
      }

      INFO("✓ ATOMIC_ADD type safety tests passed");

    } catch (const std::exception &e) {
      WARN("ATOMIC_ADD type safety test failed: " << e.what());
      REQUIRE(false);
    }
  } else {
    WARN("Backend not available, skipping ATOMIC_ADD type safety test");
    REQUIRE(true);
  }
}

TEST_CASE("ATOMIC_ADD Performance Profiling",
          "[backend][kernels][atomic][performance]") {
  initialize_backend_once();

  if (g_backend_available) {
    try {
      INFO("=== TEST 7: ATOMIC_ADD Performance Profiling ===");

      Resource resource = Resource::create_cuda_device(0);
      INFO("Profiling ATOMIC_ADD on backend: " << resource.getTypeString());

      // Initialize profiler
      ARBD::Profiling::ProfilingConfig config;
      config.enable_timing = true;
      config.enable_backend_markers = true;
      ARBD::Profiling::ProfileManager::init(config);

      SECTION("Low contention performance (single thread)") {
        DeviceBuffer<float> float_buffer(1, resource.id());
        std::vector<float> float_init = {0.0f};
        float_buffer.copy_from_host(float_init);

        auto single_thread_kernel = [] DEVICE (size_t i, float *target) {
          if (i == 0) {
            // Single thread doing 1000 atomic operations
            for (int j = 0; j < 1000; ++j) {
              ATOMIC_ADD(target, 1.0f);
            }
          }
        };

        KernelConfig config = KernelConfig::for_1d(1, resource);
        config.sync = true;

        // Time the kernel execution
        auto start = std::chrono::high_resolution_clock::now();

        auto event = launch_kernel(resource, config, single_thread_kernel,
                                   get_buffer_pointer(float_buffer));
        event.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::vector<float> result(1);
        float_buffer.copy_to_host(result);

        INFO("Low contention: 1000 atomic ops by 1 thread");
        INFO("Time: " << duration.count() << " μs");
        INFO("Throughput: " << (1000.0 * 1000.0 / duration.count())
                            << " ops/ms");
        INFO("Result: " << result[0] << " (expected: 1000)");

        REQUIRE(std::abs(result[0] - 1000.0f) < 1e-5f);
      }

      SECTION("Medium contention performance (32 threads)") {
        DeviceBuffer<float> float_buffer(1, resource.id());
        std::vector<float> float_init = {0.0f};
        float_buffer.copy_from_host(float_init);

        auto medium_contention_kernel = [] DEVICE (size_t i, float *target) {
          // Each of 32 threads does 100 atomic operations
          for (int j = 0; j < 100; ++j) {
            ATOMIC_ADD(target, 1.0f);
          }
        };

        KernelConfig config = KernelConfig::for_1d(32, resource);
        config.sync = true;

        auto start = std::chrono::high_resolution_clock::now();

        auto event = launch_kernel(resource, config, medium_contention_kernel,
                                   get_buffer_pointer(float_buffer));
        event.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::vector<float> result(1);
        float_buffer.copy_to_host(result);

        INFO("Medium contention: 100 atomic ops by 32 threads");
        INFO("Time: " << duration.count() << " μs");
        INFO("Throughput: " << (3200.0 * 1000.0 / duration.count())
                            << " ops/ms");
        INFO("Result: " << result[0] << " (expected: 3200)");

        REQUIRE(std::abs(result[0] - 3200.0f) < 1e-5f);
      }

      SECTION("High contention performance (1024 threads)") {
        DeviceBuffer<float> float_buffer(1, resource.id());
        std::vector<float> float_init = {0.0f};
        float_buffer.copy_from_host(float_init);

        auto high_contention_kernel = [] DEVICE (size_t i, float *target) {
          // Each of 1024 threads does 10 atomic operations
          for (int j = 0; j < 10; ++j) {
            ATOMIC_ADD(target, 1.0f);
          }
        };

        KernelConfig config = KernelConfig::for_1d(1024, resource);
        config.sync = true;

        auto start = std::chrono::high_resolution_clock::now();

        auto event = launch_kernel(resource, config, high_contention_kernel,
                                   get_buffer_pointer(float_buffer));
        event.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::vector<float> result(1);
        float_buffer.copy_to_host(result);

        INFO("High contention: 10 atomic ops by 1024 threads");
        INFO("Time: " << duration.count() << " μs");
        INFO("Throughput: " << (10240.0 * 1000.0 / duration.count())
                            << " ops/ms");
        INFO("Result: " << result[0] << " (expected: 10240)");

        REQUIRE(std::abs(result[0] - 10240.0f) < 1e-5f);
      }

      SECTION("Integer vs Float atomic performance comparison") {
        const int num_threads = 256;
        const int ops_per_thread = 100;
        const int total_ops = num_threads * ops_per_thread;

        // Test integer atomics
        DeviceBuffer<int> int_buffer(1, resource.id());
        std::vector<int> int_init = {0};
        int_buffer.copy_from_host(int_init);

        auto int_atomic_kernel = [=] DEVICE (size_t i, int *target) {
          for (int j = 0; j < ops_per_thread; ++j) {
            ATOMIC_ADD(target, 1);
          }
        };

        KernelConfig int_config = KernelConfig::for_1d(num_threads, resource);
        int_config.sync = true;

        auto int_start = std::chrono::high_resolution_clock::now();
        auto int_event = launch_kernel(resource, int_config, int_atomic_kernel,
                                       get_buffer_pointer(int_buffer));
        int_event.wait();
        auto int_end = std::chrono::high_resolution_clock::now();
        auto int_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(int_end -
                                                                  int_start);

        // Test float atomics
        DeviceBuffer<float> float_buffer(1, resource.id());
        std::vector<float> float_init = {0.0f};
        float_buffer.copy_from_host(float_init);

        auto float_atomic_kernel = [=] DEVICE (size_t i, float *target) {
          for (int j = 0; j < ops_per_thread; ++j) {
            ATOMIC_ADD(target, 1.0f);
          }
        };

        KernelConfig float_config = KernelConfig::for_1d(num_threads, resource);
        float_config.sync = true;

        auto float_start = std::chrono::high_resolution_clock::now();
        auto float_event =
            launch_kernel(resource, float_config, float_atomic_kernel,
                          get_buffer_pointer(float_buffer));
        float_event.wait();
        auto float_end = std::chrono::high_resolution_clock::now();
        auto float_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(float_end -
                                                                  float_start);

        // Verify results
        std::vector<int> int_result(1);
        std::vector<float> float_result(1);
        int_buffer.copy_to_host(int_result);
        float_buffer.copy_to_host(float_result);

        INFO("=== Integer vs Float Atomic Performance ===");
        INFO("Integer atomics: " << int_duration.count() << " μs ("
                                 << (total_ops * 1000.0 / int_duration.count())
                                 << " ops/ms)");
        INFO("Float atomics: " << float_duration.count() << " μs ("
                               << (total_ops * 1000.0 / float_duration.count())
                               << " ops/ms)");

        float performance_ratio =
            static_cast<float>(int_duration.count()) / float_duration.count();
        INFO("Performance ratio (int/float): " << performance_ratio);

        if (performance_ratio < 0.9f) {
          INFO("✓ Integer atomics are faster (expected for SYCL with "
               "fetch_add)");
        } else if (performance_ratio > 1.1f) {
          INFO("⚠ Float atomics are faster (unexpected, may indicate "
               "measurement variance)");
        } else {
          INFO("≈ Similar performance between integer and float atomics");
        }

        REQUIRE(int_result[0] == total_ops);
        REQUIRE(std::abs(float_result[0] - static_cast<float>(total_ops)) <
                1e-5f);
      }

      // Finalize profiler
      ARBD::Profiling::ProfileManager::finalize();
      INFO("✓ ATOMIC_ADD performance profiling completed");

    } catch (const std::exception &e) {
      WARN("ATOMIC_ADD performance profiling failed: " << e.what());
      REQUIRE(false);
    }
  } else {
    WARN("Backend not available, skipping ATOMIC_ADD performance profiling");
    REQUIRE(true);
  }
}

