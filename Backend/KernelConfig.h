#pragma once
#include "Events.h"
#include "Header.h"
#include "Resource.h"

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif

#ifdef HOST_GUARD
#include <thread>
#endif

namespace ARBD {
struct kerneldim3 {
  idx_t x = 1, y = 1, z = 1;
};

/**
 * @brief Configuration for kernel launches.
 *
 * This class encapsulates the configuration parameters for kernel launches,
 * including grid and block sizes, shared memory requirements, and event
 * dependencies. It also provides methods for auto-configuring the kernel based
 * on the resource type.
 * @param grid_size The grid size for the kernel launch.
 * @param block_size The block size for the kernel launch.
 * @param problem_size The problem size for the kernel launch.
 * @note problem size is not necessarily the same as the total_thread_size!
 * @param shared_memory The shared memory size for the kernel launch.
 * @param sync Whether to sync the kernel launch.
 * @param async deprecated. use sync instead.
 * @param dependencies The dependencies for the kernel launch.
 * @param queue_id The queue id for the kernel launch.
 * @param explicit_queue The explicit queue for the kernel launch. Will override
 * queue_id.
 * @param validate_block_size Whether to validate the block size for the kernel
 * launch.
 * @param auto_configure Whether to auto-configure the kernel launch.
 * @param get_queue The queue for the kernel launch.
 * @note Using KernelConfig::for_1d, for_2d, for_3d is recommended instead of
 * manually setting the grid_size, and block_size.
 */
struct KernelConfig {
public:
  int dim{1};
  kerneldim3 grid_size{0, 0, 0};
  kerneldim3 block_size{256, 1, 1};
  kerneldim3 problem_size{0, 0, 0};
  idx_t shared_memory{0};
  bool sync{false}; // use this only. async is deprecated.
  EventList dependencies;
  int queue_id{0};
  void *explicit_queue{nullptr};

  static KernelConfig for_1d(idx_t size_x, const Resource &resource) {
    KernelConfig config;
    config.dim = 1;
    config.problem_size = {size_x, 1, 1};
    config.auto_configure_1d(size_x, resource);
    return config;
  }

  static KernelConfig for_2d(idx_t size_x, idx_t size_y,
                             const Resource &resource) {
    KernelConfig config;
    config.dim = 2;
    config.problem_size = {size_x, size_y, 1};
    config.auto_configure_2d(size_x, size_y, resource);
    return config;
  }

  static KernelConfig for_3d(idx_t size_x, idx_t size_y, idx_t size_z,
                             const Resource &resource) {
    KernelConfig config;
    config.dim = 3;
    config.problem_size = {size_x, size_y, size_z};
    config.auto_configure_3d(size_x, size_y, size_z, resource);
    return config;
  }
  inline void validate_block_size(const Resource &resource) {
#ifdef USE_SYCL
    if (resource.type() == ResourceType::SYCL) {
      try {
        auto sycl_device = SYCL::Manager::get_device_by_id(resource.id());
        idx_t max_work_group_size =
            sycl_device.get_info<sycl::info::device::max_work_group_size>();

        auto max_work_item_sizes =
            sycl_device.get_info<sycl::info::device::max_work_item_sizes<3>>();

        // Clamp each dimension to device limits
        block_size.x =
            std::min(block_size.x, static_cast<idx_t>(max_work_item_sizes[0]));
        block_size.y =
            std::min(block_size.y, static_cast<idx_t>(max_work_item_sizes[1]));
        block_size.z =
            std::min(block_size.z, static_cast<idx_t>(max_work_item_sizes[2]));

        // Ensure total work-group size doesn't exceed device limit
        idx_t total_work_items = block_size.x * block_size.y * block_size.z;
        if (total_work_items > max_work_group_size) {
          // Scale down proportionally
          double scale_factor = sycl::sqrt(
              static_cast<double>(max_work_group_size) / total_work_items);
          block_size.x =
              std::max(1UL, static_cast<idx_t>(block_size.x * scale_factor));
          block_size.y =
              std::max(1UL, static_cast<idx_t>(block_size.y * scale_factor));
          block_size.z =
              std::max(1UL, static_cast<idx_t>(block_size.z * scale_factor));
        }

        LOGDEBUG("SYCL block size clamped to ({}, {}, {}) for device with max "
                 "work-group "
                 "size {}",
                 block_size.x, block_size.y, block_size.z, max_work_group_size);

      } catch (const sycl::exception &e) {
        LOGWARN(
            "Failed to query SYCL device limits, using default block size: {}",
            e.what());
        block_size = {256, 1, 1};
      }
    }
#endif

#ifdef USE_CUDA
    if (resource.type == ResourceType::CUDA) {
      try {
        auto &device = CUDA::Manager::devices()[resource.id];
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device.id()));

        // Clamp each dimension to CUDA limits
        block_size.x =
            std::min(block_size.x, static_cast<idx_t>(prop.maxThreadsDim[0]));
        block_size.y =
            std::min(block_size.y, static_cast<idx_t>(prop.maxThreadsDim[1]));
        block_size.z =
            std::min(block_size.z, static_cast<idx_t>(prop.maxThreadsDim[2]));

        // Ensure total threads per block doesn't exceed limit
        idx_t total_threads = block_size.x * block_size.y * block_size.z;
        if (total_threads > static_cast<idx_t>(prop.maxThreadsPerBlock)) {
          double scale_factor = std::sqrt(
              static_cast<double>(prop.maxThreadsPerBlock) / total_threads);
          block_size.x =
              std::max(1UL, static_cast<idx_t>(block_size.x * scale_factor));
          block_size.y =
              std::max(1UL, static_cast<idx_t>(block_size.y * scale_factor));
          block_size.z =
              std::max(1UL, static_cast<idx_t>(block_size.z * scale_factor));
        }

        LOGDEBUG("CUDA block size clamped to ({}, {}, {}) for device with max "
                 "threads per "
                 "block {}",
                 block_size.x, block_size.y, block_size.z,
                 prop.maxThreadsPerBlock);

      } catch (...) {
        LOGWARN("Failed to query CUDA device limits, using default block size");
        block_size = {256, 1, 1};
      }
    }
#endif

#ifdef USE_METAL
    if (resource.type == ResourceType::METAL) {
      try {
        auto &device = METAL::Manager::devices()[resource.id];

        // Get the already-queried maximum threads per threadgroup from the
        // device
        idx_t max_threads_per_threadgroup = device.max_threads_per_group();

        // Metal also has per-dimension limits (typically 1024x1024x64)
        // For simplicity, we'll use conservative limits that work on all
        // devices
        idx_t max_threads_x = 1024;
        idx_t max_threads_y = 1024;
        idx_t max_threads_z = 64;

        // Clamp each dimension to Metal limits
        block_size.x = std::min(block_size.x, max_threads_x);
        block_size.y = std::min(block_size.y, max_threads_y);
        block_size.z = std::min(block_size.z, max_threads_z);

        // Ensure total threads per threadgroup doesn't exceed device limit
        idx_t total_threads = block_size.x * block_size.y * block_size.z;
        if (total_threads > max_threads_per_threadgroup) {
          double scale_factor = std::sqrt(
              static_cast<double>(max_threads_per_threadgroup) / total_threads);
          block_size.x =
              std::max(1UL, static_cast<idx_t>(block_size.x * scale_factor));
          block_size.y =
              std::max(1UL, static_cast<idx_t>(block_size.y * scale_factor));
          block_size.z =
              std::max(1UL, static_cast<idx_t>(block_size.z * scale_factor));
        }

        LOGDEBUG("Metal block size clamped to ({}, {}, {}) for device with max "
                 "threads per "
                 "threadgroup {}",
                 block_size.x, block_size.y, block_size.z,
                 max_threads_per_threadgroup);

      } catch (...) {
        LOGWARN(
            "Failed to query Metal device limits, using default block size");
        block_size = {32, 1, 1}; // Metal-optimized default
      }
    }
#endif

    // For CPU, any block size is technically fine since we use std::thread
    if (resource.type() == ResourceType::CPU) {
      const idx_t max_cpu_threads = std::thread::hardware_concurrency() * 4;
      idx_t total_threads = block_size.x * block_size.y * block_size.z;
      if (total_threads > max_cpu_threads) {
        block_size.x = std::min(block_size.x, max_cpu_threads);
        block_size.y = 1;
        block_size.z = 1;
      }
    }
  }

  /**
   * @brief Make sure problem size is not 0.
   */
  inline void set_problem_size() {
    if (problem_size.x == 0 && grid_size.x > 0 && block_size.x > 0) {
      problem_size.x = grid_size.x * block_size.x;
    }
    if (problem_size.y == 0 && grid_size.y > 0 && block_size.y > 0) {
      problem_size.y = grid_size.y * block_size.y;
    }
    if (problem_size.z == 0 && grid_size.z > 0 && block_size.z > 0) {
      problem_size.z = grid_size.z * block_size.z;
    }
  }

  void *get_queue(const Resource &resource) const {
    if (explicit_queue) {
      return explicit_queue;
    } else {
      return (queue_id == 0) ? resource.get_stream()
                             : resource.get_stream(queue_id);
    }
  }

private:
  void auto_configure_1d(idx_t thread_count, const Resource &resource) {
    // Backend-specific 1D auto-configuration
    switch (resource.type()) {
    case ResourceType::CUDA:
      // Use CUDA-specific configuration
      block_size.x = 256; // Optimal for most CUDA kernels
      grid_size.x =
          std::max(1UL, (thread_count + block_size.x - 1) / block_size.x);
      grid_size.y = 1;
      grid_size.z = 1;
      break;
    case ResourceType::SYCL:
      // SYCL work-group configuration
      block_size.x = 64; // Typical SYCL work-group size
      grid_size.x =
          std::max(1UL, (thread_count + block_size.x - 1) / block_size.x);
      grid_size.y = 1;
      grid_size.z = 1;
      break;
    case ResourceType::METAL:
      // Metal threadgroup configuration - use optimal threadgroup size
      block_size.x = 32; // Metal SIMD width (optimal for most kernels)
      block_size.y = 1;  // 1D processing
      block_size.z = 1;  // 1D processing

      // Calculate number of threadgroups needed
      grid_size.x =
          std::max(1UL, (thread_count + block_size.x - 1) / block_size.x);
      grid_size.y = 1; // 1D processing
      grid_size.z = 1; // 1D processing
      break;
    default:
      // CPU fallback
      block_size.x = 64; // CPU SIMD width (optimal for most kernels)
      grid_size.x =
          std::max(1UL, (thread_count + block_size.x - 1) / block_size.x);
      grid_size.y = 1;
      grid_size.z = 1;
      break;
    }
  }

  void auto_configure_2d(idx_t width, idx_t height, const Resource &resource) {
    // Backend-specific 2D auto-configuration
    switch (resource.type()) {
    case ResourceType::CUDA:
      // CUDA 2D configuration - use 16x16 blocks for good occupancy
      block_size.x = 16;
      block_size.y = 16;
      block_size.z = 1;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = 1;
      break;
    case ResourceType::SYCL:
      // SYCL 2D work-group configuration
      block_size.x = 8;
      block_size.y = 8;
      block_size.z = 1;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = 1;
      break;
    case ResourceType::METAL:
      // Metal 2D threadgroup configuration
      // Use 8x8 threadgroups for good memory access patterns and SIMD
      // efficiency
      block_size.x = 8;
      block_size.y = 8;
      block_size.z = 1;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = 1;
      break;
    default:
      // CPU 2D configuration - use 8x8 blocks for good occupancy
      block_size.x = 8;
      block_size.y = 8;
      block_size.z = 1;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = 1;
      break;
    }
  }

  void auto_configure_3d(idx_t width, idx_t height, idx_t depth,
                         const Resource &resource) {
    // Backend-specific 3D auto-configuration
    switch (resource.type()) {
    case ResourceType::CUDA:
      // CUDA 3D configuration - use 8x8x4 blocks
      block_size.x = 8;
      block_size.y = 8;
      block_size.z = 4;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = (depth + block_size.z - 1) / block_size.z;
      break;
    case ResourceType::SYCL:
      // SYCL 3D work-group configuration
      block_size.x = 4;
      block_size.y = 4;
      block_size.z = 4;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = (depth + block_size.z - 1) / block_size.z;
      break;
    case ResourceType::METAL:
      // Metal 3D threadgroup configuration
      block_size.x = 4;
      block_size.y = 4;
      block_size.z = 4;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = (depth + block_size.z - 1) / block_size.z;
      break;

    default:
      // CPU 3D configuration - use 4x4x4 blocks
      block_size.x = 4;
      block_size.y = 4;
      block_size.z = 4;

      grid_size.x = (width + block_size.x - 1) / block_size.x;
      grid_size.y = (height + block_size.y - 1) / block_size.y;
      grid_size.z = (depth + block_size.z - 1) / block_size.z;
    }
  }
};
} // namespace ARBD
