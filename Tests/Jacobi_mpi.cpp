#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"
#include "JacobiKernels.h"
#include "catch_boiler.h"
#include <chrono>
#include <iomanip>
#include <tuple>
#include <vector>

#ifdef USE_MPI
#include "Backend/MPIManager.h"
#include <mpi.h>
#endif

using namespace Catch;

namespace {

/**
 * @brief Run Jacobi iterations with MPI multi-process support
 * @details Each MPI rank manages one GPU and exchanges halo data via MPI
 */
static std::tuple<double, int, real>
run_jacobi_iterations_mpi(int rank, int size, ARBD::Resource &device, int nx,
                          int ny, int iters) {
  using clock_t = ::std::chrono::high_resolution_clock;

  if (rank == 0) {
    printf("DEBUG: run_jacobi_iterations_mpi started with %d MPI ranks\n",
           size);
    fflush(stdout);
  }

  // Domain decomposition following reference implementation
  int chunk_size_low = (ny - 2) / size;
  int chunk_size_high = chunk_size_low + 1;
  int num_ranks_low = size * chunk_size_low + size - (ny - 2);

  int ny_local;
  int iy_start_global;

  if (rank < num_ranks_low) {
    ny_local = chunk_size_low;
    iy_start_global = rank * chunk_size_low + 1;
  } else {
    ny_local = chunk_size_high;
    iy_start_global = num_ranks_low * chunk_size_low +
                      (rank - num_ranks_low) * chunk_size_high + 1;
  }

  int my_offset = iy_start_global - 1;

  printf("Rank %d: Managing %d rows (global offset %d) on device %d\n", rank,
         ny_local, my_offset, device.id());
  fflush(stdout);

  // Allocate buffers on this rank's GPU (+2 for ghost cells)
  size_t size_local = static_cast<size_t>(nx) * (ny_local + 2);
  ARBD::DeviceBuffer<float> a_buf(size_local, device.id());
  ARBD::DeviceBuffer<float> a_new_buf(size_local, device.id());

  // Allocate L2 norm buffer for convergence checking
  ARBD::DeviceBuffer<float> l2_norm_buf(1, device.id());
  std::vector<float> l2_norm_host(1, 0.0f);

  // Initialize to zero
  a_buf.copy_from_host(std::vector<float>(size_local, 0.0f));
  a_new_buf.copy_from_host(std::vector<float>(size_local, 0.0f));

  printf("Rank %d: Buffers allocated (%zu floats)\n", rank, size_local);
  fflush(stdout);

  // Initialize boundaries
  ARBD::KernelConfig cfg =
      ARBD::KernelConfig::for_1d(static_cast<idx_t>(ny_local + 2), device);

#ifdef USE_CUDA
  launch_initialize_boundaries(device, cfg, a_new_buf.data(), a_buf.data(),
                               M_PI, my_offset, nx, ny_local + 2, ny);
#else
  launch_kernel(device, cfg, initialize_boundaries_kernel{}, a_new_buf.data(),
                a_buf.data(), M_PI, my_offset, nx, ny_local + 2, ny);
#endif

  device.synchronize_streams();
  printf("Rank %d: Boundaries initialized\n", rank);
  fflush(stdout);

#ifdef USE_MPI
  printf("Rank %d: Using CUDA-aware MPI for direct device-to-device "
         "communication\n",
         rank);
  fflush(stdout);
#endif

  // Barrier before timing
#ifdef USE_MPI
  if (rank == 0) {
    printf("Rank %d: About to call MPI_Barrier with %d processes\n", rank,
           size);
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Rank %d: MPI_Barrier completed\n", rank);
    fflush(stdout);
  }
#endif

  auto t0 = clock_t::now();

  // Initialize L2 norm tracking
  real l2_norm = 1.0;
  int it = 0;

  // Main iteration loop - run all iterations for MD simulation
  for (it = 0; it < iters; ++it) {
#ifdef USE_MPI
    // --- MPI HALO EXCHANGE (CUDA-aware MPI) ---
    if (size > 1) {
      // Apply periodic boundary conditions (like reference implementation)
      const int top = rank > 0 ? rank - 1 : (size - 1);
      const int bottom = (rank + 1) % size;

      // Time MPI communication for performance debugging
      auto mpi_start = clock_t::now();

      // Match reference implementation's communication pattern exactly
      // Send interior row to top neighbor, receive into bottom ghost
      MPI_Sendrecv(a_buf.data() + 1 * nx, nx, MPI_FLOAT, top, 0,
                   a_buf.data() + (ny_local + 1) * nx, nx, MPI_FLOAT, bottom, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Send interior row to bottom neighbor, receive into top ghost
      MPI_Sendrecv(a_buf.data() + ny_local * nx, nx, MPI_FLOAT, bottom, 0,
                   a_buf.data() + 0, nx, MPI_FLOAT, top, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      auto mpi_end = clock_t::now();
      double mpi_time =
          std::chrono::duration<double, std::milli>(mpi_end - mpi_start)
              .count();

      if (it % 100 == 0) {
        printf("Rank %d: MPI halo exchange took %.6f ms (iteration %d)\n", rank,
               mpi_time, it);
        fflush(stdout);
      }
    }
#endif

    // Calculate L2 norm only every 100 iterations (like reference
    // implementation)
    bool calculate_norm = ((it + 1) % 100) == 0;

    if (calculate_norm) {
      // --- RESET L2 NORM ---
      l2_norm_host[0] = 0.0f;
      l2_norm_buf.copy_from_host(l2_norm_host);
    }

    // --- LAUNCH JACOBI KERNEL ---
    ARBD::KernelConfig kernel_cfg = ARBD::KernelConfig::for_2d(
        static_cast<idx_t>(nx - 2), static_cast<idx_t>(ny_local), device);

#ifdef USE_CUDA
    launch_jacobi_kernel(device, kernel_cfg, a_new_buf.data(), a_buf.data(),
                         l2_norm_buf.data(), // Always provide buffer, but only
                                             // use result if calculate_norm
                         1,                  // iy_start (local coordinates)
                         ny_local + 1,       // iy_end (local coordinates)
                         nx, a_new_buf.data(), 0, a_new_buf.data(), 0);
#else
    launch_kernel(device, kernel_cfg, jacobi_kernel{}, a_new_buf.data(),
                  a_buf.data(),
                  l2_norm_buf.data(), // Always provide buffer, but only use
                                      // result if calculate_norm
                  1,                  // iy_start (local coordinates)
                  ny_local + 1,       // iy_end (local coordinates)
                  nx, a_new_buf.data(), 0, a_new_buf.data(), 0);
#endif

    // Swap buffers
    ::std::swap(a_buf, a_new_buf);

    // Synchronize device
    device.synchronize_streams();

    if (calculate_norm) {
      // --- CONVERGENCE CHECK ---
      // Copy L2 norm from device and reduce across all MPI ranks
      l2_norm_buf.copy_to_host(l2_norm_host);
      float local_l2_norm = l2_norm_host[0];

#ifdef USE_MPI
      float global_l2_norm;
      MPI_Allreduce(&local_l2_norm, &global_l2_norm, 1, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      l2_norm = static_cast<real>(std::sqrt(global_l2_norm));
#else
      l2_norm = static_cast<real>(std::sqrt(local_l2_norm));
#endif

      // Print progress every 100 iterations
      if (rank == 0) {
        printf("Iteration %d, L2 norm: %.6e\n", it + 1, l2_norm);
        fflush(stdout);
      }
    }
  }

  auto t1 = clock_t::now();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  double local_time =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (rank == 0) {
    printf("DEBUG: Timed iterations completed\n");
    fflush(stdout);
  }

  return std::make_tuple(local_time, iters, l2_norm);
}

} // namespace

#ifdef USE_MPI

TEST_CASE("Jacobi Multi-GPU with MPI", "[jacobi][perf][mpi]") {
  auto &mpi_mgr = Tests::MPITestManager::getInstance();

  if (!mpi_mgr.isInitialized()) {
    SKIP("MPI not initialized");
  }

  int rank = mpi_mgr.getRank();
  int size = mpi_mgr.getSize();

  if (rank == 0) {
    printf("DEBUG: MPI Jacobi test started with %d processes\n", size);
    fflush(stdout);
  }

  // Initialize backend
  Tests::BackendInit::initialize_backend_once();

  if (!Tests::Global::backend_available) {
    SKIP("Backend not available");
  }

#ifdef USE_CUDA
  if (rank == 0) {
    printf("DEBUG: Initializing CUDA\n");
    fflush(stdout);
  }
  ARBD::CUDA::Manager::init();

  // Get local rank within node for proper GPU assignment
  int local_rank = -1;
  {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
  }

  int device_count = ARBD::CUDA::Manager::device_count();
  int my_device_id = local_rank % device_count;

  if (rank == 0) {
    printf("DEBUG: %d CUDA devices available\n", device_count);
  }
  printf("Rank %d (local_rank %d) assigned to GPU %d\n", rank, local_rank,
         my_device_id);
  fflush(stdout);

  // Validate device ID and set CUDA device for this rank
  if (my_device_id >= device_count) {
    printf("ERROR: Rank %d assigned invalid device %d (only %d devices "
           "available)\n",
           rank, my_device_id, device_count);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  cudaSetDevice(my_device_id);
  cudaFree(0); // Initialize device context

#elif defined(USE_SYCL)
  if (rank == 0) {
    printf("DEBUG: Initializing SYCL\n");
    fflush(stdout);
  }
  ARBD::SYCL::Manager::init();

  // Get local rank within node for proper GPU assignment
  int local_rank = -1;
  {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
  }

  int device_count = ARBD::SYCL::Manager::device_count();
  int my_device_id = local_rank % device_count;

  if (rank == 0) {
    printf("DEBUG: %d SYCL devices available\n", device_count);
  }
  printf("Rank %d (local_rank %d) assigned to device %d\n", rank, local_rank,
         my_device_id);
  fflush(stdout);
#else
  int local_rank = 0;
  int my_device_id = 0;
#endif

  // Create resource for this rank's device
#ifdef USE_CUDA
  ARBD::Resource device(ARBD::ResourceType::CUDA,
                        static_cast<short>(my_device_id));
#elif defined(USE_SYCL)
  ARBD::Resource device(ARBD::ResourceType::SYCL,
                        static_cast<short>(my_device_id));
#else
  ARBD::Resource device(ARBD::ResourceType::CPU, 0);
#endif

  // Problem size
  const int nx = 16384;
  const int ny = 16384;
  const int iters = 1000;

  if (rank == 0) {
    printf("Problem size: %dx%d, iterations: %d\n", nx, ny, iters);
    fflush(stdout);
  }

  // Run MPI-parallel Jacobi
  auto [time_ms, iterations_completed, final_l2_norm] =
      run_jacobi_iterations_mpi(rank, size, device, nx, ny, iters);

  // Gather timing from all ranks
  double max_time_ms;
  MPI_Reduce(&time_ms, &max_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\n=== MPI Jacobi Results ===\n");
    printf("Number of MPI ranks: %d\n", size);
    printf("Problem size: %dx%d\n", nx, ny);
    printf("Iterations completed: %d\n", iterations_completed);
    printf("Final L2 norm: %.6e\n", final_l2_norm);
    printf("Max time across ranks: %.4f ms (%.4f s)\n", max_time_ms,
           max_time_ms / 1000.0);
    printf("=========================\n");
    fflush(stdout);

    INFO("Num MPI ranks: " << size);
    INFO("Problem size: " << nx << "x" << ny);
    INFO("Time: " << std::fixed << std::setprecision(4)
                  << (max_time_ms / 1000.0) << " s");

    REQUIRE(max_time_ms > 0.0);
  } else {
    // Non-root ranks just need to pass
    REQUIRE(true);
  }
}

#else

TEST_CASE("Jacobi Multi-GPU with MPI", "[jacobi][perf][mpi]") {
  SKIP("MPI not available (USE_MPI not defined)");
}

#endif
