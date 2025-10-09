#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"
#include "JacobiKernels.h"
#include "catch_boiler.h"
#include <chrono>
#include <iomanip>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace Catch;

namespace {

static ::std::vector<ARBD::Resource>
get_devices_for_backend(size_t max_devices) {
  ::std::vector<ARBD::Resource> devices;
#ifdef USE_CUDA
  size_t count = ARBD::CUDA::Manager::device_count();
  for (size_t i = 0; i < std::min(count, max_devices); ++i) {
    devices.emplace_back(ARBD::ResourceType::CUDA, static_cast<short>(i));
  }
#elif defined(USE_SYCL)
  size_t count = ARBD::SYCL::Manager::device_count();
  for (size_t i = 0; i < std::min(count, max_devices); ++i) {
    devices.emplace_back(ARBD::ResourceType::SYCL, static_cast<short>(i));
  }
#else
  devices.emplace_back(ARBD::ResourceType::CPU, 0);
#endif
  return devices;
}

/**
 * @brief Run Jacobi iterations with P2P (peer-to-peer) multi-GPU support
 * @details Uses direct GPU-to-GPU memory copies for halo exchange without
 * staging through host
 */
static double
run_jacobi_iterations_p2p(const ::std::vector<ARBD::Resource> &devices, int nx,
                          int ny, int iters) {
  using clock_t = ::std::chrono::high_resolution_clock;
  const int num_gpus = static_cast<int>(devices.size());

  printf("DEBUG: run_jacobi_iterations_p2p started with %d GPUs\n", num_gpus);
  fflush(stdout);

  if (num_gpus <= 0)
    return 0.0;

  // Distribute domain across GPUs (exclude boundary rows)
  ::std::vector<int> ny_local(num_gpus, (ny - 2) / num_gpus);
  int remainder = (ny - 2) % num_gpus;
  for (int g = 0; g < remainder; ++g) {
    ny_local[g]++;
  }

  // Create buffers for each GPU (+2 for ghost cells: one top, one bottom)
  ::std::vector<ARBD::DeviceBuffer<float>> a_bufs;
  ::std::vector<ARBD::DeviceBuffer<float>> a_new_bufs;
  a_bufs.reserve(num_gpus);
  a_new_bufs.reserve(num_gpus);

  printf("DEBUG: Starting buffer allocation for %d GPUs\n", num_gpus);
  fflush(stdout);

  // Allocate buffers sequentially on each GPU
  for (int g = 0; g < num_gpus; ++g) {
    printf("DEBUG: Allocating buffers for GPU %d (device id=%d)\n", g,
           devices[g].id());
    fflush(stdout);

    // Each GPU has its local domain plus 2 ghost rows
    size_t size_local = static_cast<size_t>(nx) * (ny_local[g] + 2);
    printf("DEBUG: Buffer size = %zu floats for GPU %d\n", size_local, g);
    fflush(stdout);

#ifdef USE_SYCL
    // Set device context before allocation for SYCL
    ARBD::SYCL::Manager::get_device_by_id(devices[g].id());
#endif

    // Create buffers on the specific device
    ARBD::DeviceBuffer<float> a_buf(size_local, devices[g].id());
    ARBD::DeviceBuffer<float> a_new_buf(size_local, devices[g].id());

    // Initialize to zero
    a_buf.copy_from_host(std::vector<float>(size_local, 0.0f));
    a_new_buf.copy_from_host(std::vector<float>(size_local, 0.0f));

    // Move buffers into storage
    a_bufs.push_back(std::move(a_buf));
    a_new_bufs.push_back(std::move(a_new_buf));

    printf("DEBUG: GPU %d buffer allocation completed\n", g);
    fflush(stdout);
  }
  printf("DEBUG: Buffer allocation completed\n");
  fflush(stdout);

  // Initialize boundaries on each GPU
  printf("DEBUG: Starting boundary initialization\n");
  for (int g = 0; g < num_gpus; ++g) {
    printf("DEBUG: Initializing boundaries for GPU %d\n", g);

    // Calculate global y-offset for this GPU
    int iy_start_global = 1; // Global grid starts at y=1
    for (int gg = 0; gg < g; ++gg) {
      iy_start_global += ny_local[gg];
    }
    int my_offset = iy_start_global - 1;

    ARBD::KernelConfig cfg = ARBD::KernelConfig::for_1d(
        static_cast<idx_t>(ny_local[g] + 2), devices[g]);

#ifdef USE_CUDA
    launch_initialize_boundaries(g, cfg, a_new_bufs[g], a_bufs[g], M_PI,
                                 my_offset, nx, ny_local[g] + 2, ny);
#else
    launch_kernel(g, cfg, initialize_boundaries_kernel{}, a_new_bufs[g],
                  a_bufs[g], M_PI, my_offset, nx, ny_local[g] + 2, ny);
#endif
    printf("DEBUG: GPU %d boundary initialization completed\n", g);
  }

  // Synchronize all devices after initialization
  for (int g = 0; g < num_gpus; ++g) {
    devices[g].synchronize_streams();
  }

  printf("DEBUG: Starting timed iterations\n");
  fflush(stdout);

  auto t0 = clock_t::now();

  for (int it = 0; it < iters; ++it) {
    // --- P2P HALO EXCHANGE ---
    if (num_gpus > 1) {
      // Exchange ghost cells between neighboring GPUs using P2P copies
      for (int g = 0; g < num_gpus; ++g) {
        // Send bottom interior row to next GPU's top ghost row
        if (g < num_gpus - 1) {
          // Bottom interior row is at index ny_local[g] (0-indexed)
          size_t src_offset = ny_local[g] * nx;   // Bottom interior row
          size_t dst_offset = 0;                  // Top ghost row of next GPU
          size_t copy_bytes = nx * sizeof(float); // One row

          // Get raw pointers with offsets
          float *src_ptr = a_bufs[g].data() + src_offset;
          float *dst_ptr = a_bufs[g + 1].data() + dst_offset;

          // Perform P2P device-to-device copy
#ifdef USE_CUDA
          // Use CUDA P2P memcpy
          cudaMemcpyPeerAsync(
              dst_ptr, devices[g + 1].id(), // dst and dst device
              src_ptr, devices[g].id(),     // src and src device
              copy_bytes, static_cast<cudaStream_t>(devices[g].get_stream()));
#elif defined(USE_SYCL)
          // SYCL device-to-device copy
          auto src_queue = static_cast<sycl::queue *>(devices[g].get_stream());
          src_queue->memcpy(dst_ptr, src_ptr, copy_bytes).wait();
#else
          // CPU fallback
          std::memcpy(dst_ptr, src_ptr, copy_bytes);
#endif
        }

        // Send top interior row to previous GPU's bottom ghost row
        if (g > 0) {
          // Top interior row is at index 1 (0 is top ghost)
          size_t src_offset = 1 * nx; // Top interior row
          size_t dst_offset =
              (ny_local[g - 1] + 1) * nx; // Bottom ghost row of prev GPU
          size_t copy_bytes = nx * sizeof(float);

          // Get raw pointers with offsets
          float *src_ptr = a_bufs[g].data() + src_offset;
          float *dst_ptr = a_bufs[g - 1].data() + dst_offset;

          // Perform P2P device-to-device copy
#ifdef USE_CUDA
          // Use CUDA P2P memcpy
          cudaMemcpyPeerAsync(
              dst_ptr, devices[g - 1].id(), // dst and dst device
              src_ptr, devices[g].id(),     // src and src device
              copy_bytes, static_cast<cudaStream_t>(devices[g].get_stream()));
#elif defined(USE_SYCL)
          // SYCL device-to-device copy
          auto src_queue = static_cast<sycl::queue *>(devices[g].get_stream());
          src_queue->memcpy(dst_ptr, src_ptr, copy_bytes).wait();
#else
          // CPU fallback
          std::memcpy(dst_ptr, src_ptr, copy_bytes);
#endif
        }
      }

      // Synchronize all devices after halo exchange
      for (int g = 0; g < num_gpus; ++g) {
        devices[g].synchronize_streams();
      }
    }

    // --- LAUNCH JACOBI KERNEL ON EACH GPU ---
    // Process sequentially to avoid race conditions
    // (can be parallelized with proper stream management)
    for (int g = 0; g < num_gpus; ++g) {
      ARBD::KernelConfig cfg = ARBD::KernelConfig::for_2d(
          static_cast<idx_t>(nx - 2), static_cast<idx_t>(ny_local[g]),
          devices[g]);

#ifdef USE_CUDA
      launch_jacobi_kernel(g, cfg, a_new_bufs[g], a_bufs[g],
                           a_new_bufs[g],   // L2 norm buffer (reused)
                           1,               // iy_start (local coordinates)
                           ny_local[g] + 1, // iy_end (local coordinates)
                           nx, a_new_bufs[g], 0, a_new_bufs[g], 0);
#else
      launch_kernel(g, cfg, jacobi_kernel{}, a_new_bufs[g], a_bufs[g],
                    a_new_bufs[g],   // L2 norm buffer (reused)
                    1,               // iy_start (local coordinates)
                    ny_local[g] + 1, // iy_end (local coordinates)
                    nx, a_new_bufs[g], 0, a_new_bufs[g], 0);
#endif
    }

    // Swap buffers on each GPU
    for (int g = 0; g < num_gpus; ++g) {
      ::std::swap(a_bufs[g], a_new_bufs[g]);
    }

    // Synchronize all devices after iteration
    for (int g = 0; g < num_gpus; ++g) {
      devices[g].synchronize_streams();
    }
  }

  auto t1 = clock_t::now();

  printf("DEBUG: Timed iterations completed\n");
  fflush(stdout);

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace

TEST_CASE("Jacobi Multi-GPU with P2P", "[jacobi][perf][p2p]") {
  printf("DEBUG: Test started\n");

  Tests::BackendInit::initialize_backend_once();
  printf("DEBUG: Backend initialized\n");

  if (!Tests::Global::backend_available) {
    SKIP("Backend not available");
  }

#ifdef USE_CUDA
  printf("DEBUG: Initializing CUDA\n");
  fflush(stdout);
  ARBD::CUDA::Manager::init();

  // Enable P2P access between all GPU pairs
  int device_count = ARBD::CUDA::Manager::device_count();
  printf("DEBUG: Enabling P2P access for %d GPUs\n", device_count);
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i != j) {
        cudaSetDevice(i);
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, i, j);
        if (can_access) {
          cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          if (err == cudaSuccess) {
            printf("DEBUG: P2P enabled from GPU %d to GPU %d\n", i, j);
          } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
            // Already enabled, ignore
            cudaGetLastError(); // Clear error
          } else {
            printf("WARNING: Failed to enable P2P from GPU %d to GPU %d: %s\n",
                   i, j, cudaGetErrorString(err));
          }
        } else {
          printf("WARNING: P2P not supported from GPU %d to GPU %d\n", i, j);
        }
      }
    }
  }
  printf("DEBUG: CUDA initialization completed\n");
  fflush(stdout);
#elif defined(USE_SYCL)
  printf("DEBUG: Initializing SYCL\n");
  fflush(stdout);
  ARBD::SYCL::Manager::init();
  printf("DEBUG: SYCL initialization completed\n");
  fflush(stdout);
#endif

  // Problem size
  const int nx = 16384;
  const int ny = 16384;
  const int iters = 100;

  printf("DEBUG: Creating device list\n");
  fflush(stdout);

  // Create devices
  ::std::vector<ARBD::Resource> devices;
#ifdef USE_CUDA
  for (int i = 0; i < ARBD::CUDA::Manager::device_count(); ++i) {
    devices.emplace_back(ARBD::ResourceType::CUDA, static_cast<short>(i));
  }
  printf("DEBUG: Created %zu CUDA devices\n", devices.size());
  fflush(stdout);
#elif defined(USE_SYCL)
  for (int i = 0; i < ARBD::SYCL::Manager::device_count(); ++i) {
    devices.emplace_back(ARBD::ResourceType::SYCL, static_cast<short>(i));
  }
  printf("DEBUG: Created %zu SYCL devices\n", devices.size());
  fflush(stdout);
#else
  devices.emplace_back(ARBD::ResourceType::CPU, 0);
  printf("DEBUG: Created 1 CPU device\n");
  fflush(stdout);
#endif

  // Test single GPU first
  ::std::vector<ARBD::Resource> single_gpu = {devices[0]};
  printf("DEBUG: Starting single GPU test\n");
  fflush(stdout);
  double t1_ms = run_jacobi_iterations_p2p(single_gpu, nx, ny, iters);
  printf("Single GPU: %f ms\n", t1_ms);
  fflush(stdout);

  // Test multi-GPU with P2P
  if (devices.size() > 1) {
    printf("DEBUG: Starting multi-GPU P2P test with %zu GPUs\n",
           devices.size());
    fflush(stdout);
    double tN_ms = run_jacobi_iterations_p2p(devices, nx, ny, iters);
    printf("Multi-GPU (%zu GPUs): %f ms\n", devices.size(), tN_ms);

    double speedup = t1_ms / tN_ms;
    double efficiency = speedup / devices.size() * 100.0;

    INFO("Num GPUs: " << devices.size());
    INFO("Problem size: " << nx << "x" << ny);
    INFO("Single GPU time: " << std::fixed << std::setprecision(4)
                             << (t1_ms / 1000.0) << " s");
    INFO("Multi-GPU time: " << std::fixed << std::setprecision(4)
                            << (tN_ms / 1000.0) << " s");
    INFO("Speedup: " << std::fixed << std::setprecision(2) << speedup << "x");
    INFO("Efficiency: " << std::fixed << std::setprecision(1) << efficiency
                        << "%");

    REQUIRE(tN_ms > 0.0);
    REQUIRE(tN_ms < t1_ms); // Multi-GPU should be faster
  } else {
    printf("DEBUG: Skipping multi-GPU test (only 1 GPU available)\n");
    INFO("Only 1 GPU available, skipping multi-GPU test");
  }
}
