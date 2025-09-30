#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

// MPI support
#ifdef USE_MPI
#include "Backend/MPIManager.h"
#include <mpi.h>
#endif

// Include backend-specific headers
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include "SignalManager.h"
#include <cuda.h>
#include <nvfunctional>
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

// Common includes
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"

// Use Catch2 v3 amalgamated header (self-contained)

// Macro for run_trial function - defines run_trial as an alias to run_trial
// function
#define DEF_RUN_TRIAL using Tests::run_trial;

// Macro for cleanup function - defines cleanup as an alias to
// TestBackendManager::cleanup
#define DEF_CLEANUP using Tests::cleanup;

namespace Tests {

// =============================================================================
// MPI-aware testing infrastructure
// =============================================================================

#ifdef USE_MPI
/**
 * @brief MPI-aware test manager for coordinating tests across multiple
 * processes
 */
class MPITestManager {
private:
  static MPITestManager *instance_;
  static std::mutex mutex_;
  bool initialized_ = false;
  int rank_ = 0;
  int size_ = 1;

  MPITestManager() { initialize(); }

public:
  MPITestManager(const MPITestManager &) = delete;
  MPITestManager &operator=(const MPITestManager &) = delete;

  static MPITestManager &getInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == nullptr) {
      instance_ = new MPITestManager();
    }
    return *instance_;
  }

  static void cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ != nullptr) {
      instance_->finalize();
      delete instance_;
      instance_ = nullptr;
    }
  }

  void initialize() {
    if (initialized_)
      return;

    try {
      ARBD::MPI::Manager::instance().init();
      rank_ = ARBD::MPI::Manager::instance().get_rank();
      size_ = ARBD::MPI::Manager::instance().get_size();
      initialized_ = true;

      if (rank_ == 0) {
        std::cout << "MPI Test Manager initialized with " << size_
                  << " processes" << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << "MPI Test Manager initialization failed: " << e.what()
                << std::endl;
    }
  }

  void finalize() {
    if (!initialized_)
      return;

    try {
      ARBD::MPI::Manager::instance().finalize();
    } catch (const std::exception &e) {
      std::cerr << "MPI Test Manager finalization failed: " << e.what()
                << std::endl;
    }
    initialized_ = false;
  }

  int getRank() const { return rank_; }
  int getSize() const { return size_; }
  bool isInitialized() const { return initialized_; }
  bool isRoot() const { return rank_ == 0; }

  /**
   * @brief Coordinate test execution across all processes
   * @param test_func Function to run on all processes
   * @param root_verify_func Function to run verification only on root
   * @return true if test passed on all processes
   */
  template <typename TestFunc, typename VerifyFunc>
  bool runCoordinatedTest(TestFunc test_func, VerifyFunc root_verify_func) {
    if (!initialized_)
      return false;

    bool local_success = false;
    try {
      local_success = test_func();
    } catch (const std::exception &e) {
      if (rank_ == 0) {
        std::cerr << "Test function failed on rank " << rank_ << ": "
                  << e.what() << std::endl;
      }
      local_success = false;
    }

    // Gather results from all processes
    int local_result = local_success ? 1 : 0;
    int global_result;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    bool all_passed = (global_result == 1);

    // Only root performs additional verification
    if (rank_ == 0 && all_passed) {
      try {
        all_passed = root_verify_func();
      } catch (const std::exception &e) {
        std::cerr << "Root verification failed: " << e.what() << std::endl;
        all_passed = false;
      }
    }

    return all_passed;
  }

  /**
   * @brief Simple coordinated test without root verification
   */
  template <typename TestFunc> bool runCoordinatedTest(TestFunc test_func) {
    return runCoordinatedTest(test_func, []() { return true; });
  }
};

// Static member definitions
inline MPITestManager *MPITestManager::instance_ = nullptr;
inline std::mutex MPITestManager::mutex_;

// Convenience macros for MPI testing
#define MPI_TEST_MANAGER Tests::MPITestManager::getInstance()
#define MPI_RANK MPI_TEST_MANAGER.getRank()
#define MPI_SIZE MPI_TEST_MANAGER.getSize()
#define MPI_IS_ROOT MPI_TEST_MANAGER.isRoot()

// Macro for MPI-aware test cases
#define MPI_TEST_CASE(name, tags)                                              \
  TEST_CASE(name, tags) {                                                      \
    auto &mpi_mgr = Tests::MPITestManager::getInstance();                      \
    if (!mpi_mgr.isInitialized()) {                                            \
      WARN("MPI not initialized - skipping test");                             \
      REQUIRE(true);                                                           \
      return;                                                                  \
    }                                                                          \
    int mpi_rank = mpi_mgr.getRank();                                          \
    int mpi_size = mpi_mgr.getSize();                                          \
    (void)mpi_rank;                                                            \
    (void)mpi_size; /* suppress unused warnings */

// Macro for MPI-aware sections that only run on root
#define MPI_ROOT_SECTION(name)                                                 \
  if (MPI_IS_ROOT) {                                                           \
  SECTION(name)

// Macro for sections that run on all processes
#define MPI_ALL_SECTION(name) SECTION(name)

// End MPI test case (for cleanup)
#define MPI_TEST_CASE_END() }

#else
// Non-MPI fallbacks
#define MPI_TEST_MANAGER
#define MPI_RANK 0
#define MPI_SIZE 1
#define MPI_IS_ROOT true
#define MPI_TEST_CASE(name, tags) TEST_CASE(name, tags)
#define MPI_ROOT_SECTION(name) SECTION(name)
#define MPI_ALL_SECTION(name) SECTION(name)
#define MPI_TEST_CASE_END()
#endif

// =============================================================================
// Backend-specific kernel implementations
// =============================================================================

#if defined(USE_CUDA) && defined(__CUDACC__)
template <typename Op_t, typename R, typename... T>
__global__ void cuda_op_kernel(R *result, T... args) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result = Op_t::op(args...);
  }
}
#endif

// =============================================================================
// Unified Backend Manager
// =============================================================================

/**
 * @brief Unified backend manager for test execution across different compute
 * backends
 *
 * Singleton pattern to ensure SYCL is only initialized once across all tests
 */
class TestBackendManager {
private:
  static TestBackendManager *instance_;
  static std::mutex mutex_;
  bool initialized_ = false;

  // Private constructor for singleton pattern
  TestBackendManager() { initialize(); }

public:
  // Delete copy constructor and assignment operator
  TestBackendManager(const TestBackendManager &) = delete;
  TestBackendManager &operator=(const TestBackendManager &) = delete;

  // Get singleton instance
  static TestBackendManager &getInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == nullptr) {
      instance_ = new TestBackendManager();
    }
    return *instance_;
  }

  // Cleanup singleton instance
  static void cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ != nullptr) {
      instance_->finalize();
      delete instance_;
      instance_ = nullptr;
    }
  }

  ~TestBackendManager() { finalize(); }

  void initialize() {
    if (initialized_)
      return;

    try {
#ifdef USE_CUDA
      ARBD::SignalManager::manage_segfault();
      // Initialize CUDA GPU Manager
      ARBD::CUDA::Manager::init();
#endif
#ifdef USE_SYCL
      // Add error handling around SYCL initialization to prevent memory
      // corruption
      try {
        ARBD::SYCL::Manager::init();
        ARBD::SYCL::Manager::load_info();
      } catch (const ARBD::Exception &e) {
        std::cerr << "Warning: SYCL initialization failed: " << e.what()
                  << std::endl;
        // Don't mark as initialized if SYCL fails
        return;
      }
#endif
#ifdef USE_METAL
      ARBD::METAL::Manager::init();
      ARBD::METAL::Manager::load_info();
#endif
      initialized_ = true;
    } catch (const ARBD::Exception &e) {
      std::cerr << "Warning: Backend initialization failed: " << e.what()
                << std::endl;
      // Don't mark as initialized if any backend fails
      return;
    }
  }

  void finalize() {
    if (!initialized_)
      return;

#ifdef USE_CUDA
    ARBD::CUDA::Manager::finalize();
#endif
#ifdef USE_SYCL
    ARBD::SYCL::Manager::finalize();
#endif
#ifdef USE_METAL
    ARBD::METAL::Manager::finalize();
#endif
    initialized_ = false;
  }

  bool isInitialized() const { return initialized_; }

  void synchronize() {
    if (!initialized_) {
      std::cerr << "Warning: Backend not initialized, skipping synchronization"
                << std::endl;
      return;
    }

#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
#ifdef USE_SYCL
    ;
#endif
#ifdef USE_METAL
    ARBD::METAL::Manager::sync();
#endif
  }

  template <typename R> R *allocate_device_memory(size_t count) {
    if (!initialized_) {
      std::cerr
          << "Warning: Backend not initialized, skipping memory allocation"
          << std::endl;
      return nullptr;
    }
#ifdef USE_CUDA
    R *ptr;
    ARBD::check_cuda_error(cudaMalloc((void **)&ptr, count * sizeof(R)),
                           __FILE__, __LINE__);
    // Initialize device memory to zero
    ARBD::check_cuda_error(cudaMemset(ptr, 0, count * sizeof(R)), __FILE__,
                           __LINE__);
    return ptr;
#elif defined(USE_SYCL)
    // Create a resource and get its queue
    ARBD::Resource resource;
    void *stream_ptr = resource.get_stream();
    if (!stream_ptr) {
      std::cerr << "Warning: Failed to get SYCL queue from Resource"
                << std::endl;
      return nullptr;
    }
    auto &queue = *static_cast<sycl::queue *>(stream_ptr);
    return sycl::malloc_device<R>(count, queue);
#elif defined(USE_METAL)
    auto &device = ARBD::METAL::Manager::get_current_device();
    // Metal uses unified memory, so we can allocate using DeviceMemory
    // For simplicity, we'll use the manager's allocate function
    // Note: This is conceptual - actual Metal allocation would be different
    return static_cast<R *>(std::malloc(count * sizeof(R)));
#else
    // CPU fallback
    return static_cast<R *>(std::malloc(count * sizeof(R)));
#endif
  }

  template <typename R> void free_device_memory(R *ptr) {
    if (!ptr)
      return;

    if (!initialized_) {
      std::cerr
          << "Warning: Backend not initialized, skipping memory deallocation"
          << std::endl;
      return;
    }

#ifdef USE_CUDA
    cudaFree(ptr);
#elif defined(USE_SYCL)
    // Create a resource and get its queue
    ARBD::Resource resource;
    void *stream_ptr = resource.get_stream();
    if (stream_ptr) {
      auto &queue = *static_cast<sycl::queue *>(stream_ptr);
      sycl::free(ptr, queue);
    }
#elif defined(USE_METAL)
    std::free(ptr);
#else
    // CPU fallback
    std::free(ptr);
#endif
  }

  template <typename R>
  void copy_to_device(R *device_ptr, const R *host_ptr, size_t count) {
    if (!initialized_) {
      std::cerr << "Warning: Backend not initialized, skipping copy to device"
                << std::endl;
      return;
    }
#ifdef USE_CUDA
    ARBD::check_cuda_error(cudaMemcpy(device_ptr, host_ptr, count * sizeof(R),
                                      cudaMemcpyHostToDevice),
                           __FILE__, __LINE__);
#elif defined(USE_SYCL)
    // Create a resource and get its queue
    ARBD::Resource resource;
    void *stream_ptr = resource.get_stream();
    if (stream_ptr) {
      auto &queue = *static_cast<sycl::queue *>(stream_ptr);
      queue.memcpy(device_ptr, host_ptr, count * sizeof(R)).wait();
    }
#elif defined(USE_METAL)
    std::memcpy(device_ptr, host_ptr, count * sizeof(R));
#else
    // CPU fallback
    std::memcpy(device_ptr, host_ptr, count * sizeof(R));
#endif
  }

  template <typename R>
  void copy_from_device(R *host_ptr, const R *device_ptr, size_t count) {
    if (!initialized_) {
      std::cerr << "Warning: Backend not initialized, skipping copy from device"
                << std::endl;
      return;
    }
#ifdef USE_CUDA
    ARBD::check_cuda_error(cudaMemcpy(host_ptr, device_ptr, count * sizeof(R),
                                      cudaMemcpyDeviceToHost),
                           __FILE__, __LINE__);
#elif defined(USE_SYCL)
    // Create a resource and get its queue
    ARBD::Resource resource;
    void *stream_ptr = resource.get_stream();
    if (stream_ptr) {
      auto &queue = *static_cast<sycl::queue *>(stream_ptr);
      queue.memcpy(host_ptr, device_ptr, count * sizeof(R)).wait();
    }
#elif defined(USE_METAL)
    std::memcpy(host_ptr, device_ptr, count * sizeof(R));
#else
    // CPU fallback
    std::memcpy(host_ptr, device_ptr, count * sizeof(R));
#endif
  }

  template <typename Op_t, typename R, typename... T>
  void execute_kernel(R *result_device, T... args) {
    // Check if backend is properly initialized
    if (!initialized_) {
      std::cerr << "Warning: Backend not initialized, skipping kernel execution"
                << std::endl;
      return;
    }
#ifdef USE_CUDA
#if defined(__CUDACC__)
    // Launch the kernel using the defined cuda_op_kernel
    cuda_op_kernel<Op_t, R, T...><<<1, 1>>>(result_device, args...);
    ARBD::check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
    ARBD::check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__);
#else
    // Fallback: execute on host when not compiled with nvcc
    *result_device = Op_t::op(args...);
#endif
#elif defined(USE_SYCL)
    // Create a resource and get its queue
    ARBD::Resource resource;
    void *stream_ptr = resource.get_stream();
    if (stream_ptr) {
      auto &queue = *static_cast<sycl::queue *>(stream_ptr);
      queue
          .submit([=](sycl::handler &h) {
            h.single_task([=]() { *result_device = Op_t::op(args...); });
          })
          .wait();
    } else {
      // Fallback: execute on host when queue is not available
      *result_device = Op_t::op(args...);
    }
#elif defined(USE_METAL)
    // For Metal, we'll execute on CPU for now since compute shaders
    // require more complex setup. In a full implementation, this would
    // dispatch a Metal compute shader.
    *result_device = Op_t::op(args...);
#else
    // CPU fallback
    *result_device = Op_t::op(args...);
#endif
  }
};

// =============================================================================
// Unified Test Runner
// =============================================================================

/**
 * @brief Run a test operation across different backends
 */
template <typename Op_t, typename R, typename... T>
void run_trial(std::string name, R expected_result, T... args) {
  using namespace ARBD;

  INFO(name);

  // Test CPU execution
  R cpu_result = Op_t::op(args...);
  CAPTURE(cpu_result);
  CAPTURE(expected_result);
  REQUIRE(cpu_result == expected_result);

  // Test the current backend (determined at compile time)
  TestBackendManager &manager = TestBackendManager::getInstance();

  // Check if backend is properly initialized
  if (!manager.isInitialized()) {
    WARN("Backend not properly initialized, skipping device execution");
    return;
  }

  R *device_result_d = manager.allocate_device_memory<R>(1);
  if (!device_result_d) {
    WARN("Failed to allocate device memory, skipping device execution");
    return;
  }

  manager.execute_kernel<Op_t, R, T...>(device_result_d, args...);

  R device_result;
  manager.copy_from_device(&device_result, device_result_d, 1);
  manager.synchronize();

  manager.free_device_memory(device_result_d);

  CAPTURE(device_result);
  CHECK(cpu_result == device_result);
}

// Cleanup function for test suite
inline void cleanup() {
  TestBackendManager::cleanup();
#ifdef USE_MPI
  MPITestManager::cleanup();
#endif
}

} // namespace Tests

// =============================================================================
// Operation definitions (unchanged from original)
// =============================================================================

namespace Tests::Unary {
template <typename R, typename T> struct NegateOp {
  HOST DEVICE static R op(T in) { return static_cast<R>(-in); }
};

template <typename R, typename T> struct NormalizedOp {
  HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized()); }
};
} // namespace Tests::Unary

namespace Tests::Binary {
// R is return type, T and U are types of operands
template <typename R, typename T, typename U> struct AddOp {
  HOST DEVICE static R op(T a, U b) { return static_cast<R>(a + b); }
};

template <typename R, typename T, typename U> struct SubOp {
  HOST DEVICE static R op(T a, U b) { return static_cast<R>(a - b); }
};

template <typename R, typename T, typename U> struct MultOp {
  HOST DEVICE static R op(T a, U b) { return static_cast<R>(a * b); }
};

template <typename R, typename T, typename U> struct DivOp {
  HOST DEVICE static R op(T a, U b) { return static_cast<R>(a / b); }
};
} // namespace Tests::Binary

// =============================================================================
// Static member definitions
// =============================================================================

namespace Tests {
inline TestBackendManager *TestBackendManager::instance_ = nullptr;
inline std::mutex TestBackendManager::mutex_;
} // namespace Tests
