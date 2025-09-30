#ifdef USE_SYCL
#include "SYCLManager.h"
#include "ARBDLogger.h"
#include <sstream>
#include <mutex>

namespace ARBD {
namespace SYCL {

// Static member initialization
std::vector<sycl::device> Manager::all_devices_;
bool Manager::initialized_ = false;
std::vector<int> Manager::rank_devices_;
bool Manager::multi_rank_mode_ = false;
int Manager::rank_id_ = 0;
int Manager::omp_threads_ = 1;
std::vector<int> Manager::thread_device_map_;
std::string Manager::device_affinity_strategy_ = "block";
std::mutex Manager::mtx_;

void Manager::init() {
  if (initialized_) {
    LOGWARN("SYCL Manager already initialized");
    return;
  }

  LOGDEBUG("Initializing SYCL Manager...");
  discover_devices();

  if (all_devices_.empty()) {
    ARBD_Exception(ExceptionType::ValueError, "No SYCL devices found");
  }

  initialized_ = true;
  LOGINFO("SYCL Manager initialized with {} device(s)", all_devices_.size());
}

void Manager::discover_devices() {
  all_devices_.clear();

  try {
    // Get all available devices
    auto platforms = sycl::platform::get_platforms();

    for (const auto &platform : platforms) {
      auto devices = platform.get_devices();

      for (const auto &device : devices) {
        // Log device info
        try {
          std::string name = device.get_info<sycl::info::device::name>();
          std::string vendor = device.get_info<sycl::info::device::vendor>();
          auto type = device.get_info<sycl::info::device::device_type>();

          std::string type_str;
          switch (type) {
          case sycl::info::device_type::cpu:
            type_str = "CPU";
            break;
          case sycl::info::device_type::gpu:
            type_str = "GPU";
            break;
          case sycl::info::device_type::accelerator:
            type_str = "Accelerator";
            break;
          default:
            type_str = "Unknown";
            break;
          }

          LOGINFO("Found SYCL device [{}]: {} {} ({})", all_devices_.size(),
                  vendor, name, type_str);

          all_devices_.push_back(device);

        } catch (const sycl::exception &e) {
          LOGWARN("Could not query device info: {}", e.what());
          // Still add the device even if we can't query info
          all_devices_.push_back(device);
        }
      }
    }

  } catch (const sycl::exception &e) {
    LOGERROR("SYCL device discovery failed: {}", e.what());
    throw;
  }
}

void Manager::load_info() {
  init();

  // Additional device information logging
  for (size_t i = 0; i < all_devices_.size(); ++i) {
    try {
      const auto &device = all_devices_[i];

      auto max_compute_units =
          device.get_info<sycl::info::device::max_compute_units>();
      auto max_work_group =
          device.get_info<sycl::info::device::max_work_group_size>();
      auto global_mem = device.get_info<sycl::info::device::global_mem_size>();
      auto local_mem = device.get_info<sycl::info::device::local_mem_size>();

      LOGDEBUG("Device [{}] specs:", i);
      LOGDEBUG("  Max compute units: {}", max_compute_units);
      LOGDEBUG("  Max work group size: {}", max_work_group);
      LOGDEBUG("  Global memory: {:.2f} GB",
               global_mem / (1024.0 * 1024.0 * 1024.0));
      LOGDEBUG("  Local memory: {:.2f} KB", local_mem / 1024.0);

    } catch (const sycl::exception &e) {
      LOGWARN("Could not query detailed device info for device {}: {}", i,
              e.what());
    }
  }
}

void Manager::finalize() {
  if (!initialized_) {
    return;
  }

  LOGDEBUG("Finalizing SYCL Manager...");

  // Clear device list
  all_devices_.clear();
  initialized_ = false;

  LOGINFO("SYCL Manager finalized");
}

sycl::device Manager::get_device_by_id(size_t device_id) {
  if (!initialized_) {
    ARBD_Exception(ExceptionType::ValueError,
                   "SYCL Manager not initialized. Call Manager::init() first");
  }

  if (device_id >= all_devices_.size()) {
    ARBD_Exception(ExceptionType::ValueError,
                   "SYCL device {} not found (available: 0-{})", device_id,
                   all_devices_.size() - 1);
  }

  return all_devices_[device_id];
}

std::vector<sycl::device> Manager::get_all_devices() {
  if (!initialized_) {
    ARBD_Exception(ExceptionType::ValueError,
                   "SYCL Manager not initialized. Call Manager::init() first");
  }

  return all_devices_;
}

size_t Manager::device_count() {
  if (!initialized_) {
    return 0;
  }

  return all_devices_.size();
}

void Manager::init_for_rank(int local_rank, int ranks_per_node,
                            int threads_per_rank, bool verbose) {
  std::lock_guard<std::mutex> lock(mtx_);

  // First, do standard initialization if not already done
  if (!initialized_) {
    init();
  }

  // Store rank information
  rank_id_ = local_rank;
  multi_rank_mode_ = (ranks_per_node > 1);

  // Determine number of OpenMP threads
  if (threads_per_rank <= 0) {
// Use OMP_NUM_THREADS if set, otherwise use max available
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
      omp_threads_ = omp_get_num_threads();
    }
#else
    omp_threads_ = 1;
#endif
  } else {
    omp_threads_ = threads_per_rank;
#ifdef _OPENMP
    omp_set_num_threads(omp_threads_);
#endif
  }

  // Get total number of devices
  int num_devices = static_cast<int>(all_devices_.size());
  if (num_devices == 0) {
    ARBD_Exception(ExceptionType::ValueError,
                   "No SYCL devices available for rank {}", local_rank);
  }

  // Determine device assignment for this rank
  rank_devices_.clear();

  if (ranks_per_node <= 0) {
    ranks_per_node = 1; // Default to single rank
  }

  if (ranks_per_node == 1) {
    // Single rank gets all devices
    for (int i = 0; i < num_devices; ++i) {
      rank_devices_.push_back(i);
    }
    LOGINFO("Single rank mode: assigned all {} SYCL device(s)", num_devices);

  } else if (ranks_per_node <= num_devices) {
    // One or more devices per rank
    int devices_per_rank = num_devices / ranks_per_node;
    int remainder = num_devices % ranks_per_node;

    int start_device = local_rank * devices_per_rank;
    if (local_rank < remainder) {
      start_device += local_rank;
      devices_per_rank += 1;
    } else {
      start_device += remainder;
    }

    for (int i = 0; i < devices_per_rank; ++i) {
      rank_devices_.push_back(start_device + i);
    }

    LOGINFO("Rank {} assigned to {} SYCL device(s): [{}]", local_rank,
            rank_devices_.size(), [&]() {
              std::stringstream ss;
              for (size_t i = 0; i < rank_devices_.size(); ++i) {
                if (i > 0)
                  ss << ", ";
                ss << rank_devices_[i];
              }
              return ss.str();
            }());

  } else {
    // More ranks than devices: round-robin assignment
    int device_id = local_rank % num_devices;
    rank_devices_.push_back(device_id);
    LOGWARN("Rank {} sharing SYCL device {} (oversubscription: {} ranks, {} "
            "devices)",
            local_rank, device_id, ranks_per_node, num_devices);
  }

  // Setup OpenMP thread to device mapping
  setup_omp_device_mapping();

  if (verbose) {
    LOGINFO("Rank {} configuration:", local_rank);
    LOGINFO("  OpenMP threads: {}", omp_threads_);
    LOGINFO("  SYCL devices: {}", rank_devices_.size());
    for (size_t i = 0; i < rank_devices_.size(); ++i) {
      try {
        const auto &device = all_devices_[rank_devices_[i]];
        std::string name = device.get_info<sycl::info::device::name>();
        LOGINFO("    Device [{}]: {}", rank_devices_[i], name);
      } catch (const sycl::exception &e) {
        LOGINFO("    Device [{}]: <info unavailable>", rank_devices_[i]);
      }
    }

    LOGINFO("  Thread-device mapping:");
    for (int t = 0; t < omp_threads_; ++t) {
      LOGINFO("    Thread {} -> Device {}", t, thread_device_map_[t]);
    }
  }

  // Configure for OpenMP usage
  if (omp_threads_ > 1) {
#ifdef _OPENMP
// Set thread affinity for NUMA awareness
#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      if (thread_id < static_cast<int>(thread_device_map_.size())) {
        int assigned_device = thread_device_map_[thread_id];
        // Note: SYCL doesn't have explicit device setting like CUDA
        // Each thread will need to create contexts with their assigned device
        LOGTRACE("OpenMP thread {} assigned to SYCL device {}", thread_id,
                 assigned_device);
      }
    }
#endif

    LOGINFO(
        "OpenMP configuration complete: {} threads across {} SYCL device(s)",
        omp_threads_, rank_devices_.size());
  }

  LOGINFO(
      "SYCL Manager initialized for rank {} with {} device(s) and {} OpenMP "
      "thread(s)",
      local_rank, rank_devices_.size(), omp_threads_);
}

void Manager::setup_omp_device_mapping() {
  thread_device_map_.resize(omp_threads_);

  int num_rank_devices = static_cast<int>(rank_devices_.size());

  if (device_affinity_strategy_ == "block") {
    // Block distribution: consecutive threads use same device
    int threads_per_device =
        (omp_threads_ + num_rank_devices - 1) / num_rank_devices;

    for (int t = 0; t < omp_threads_; ++t) {
      int device_idx = t / threads_per_device;
      if (device_idx >= num_rank_devices)
        device_idx = num_rank_devices - 1;
      thread_device_map_[t] = rank_devices_[device_idx];
    }

  } else if (device_affinity_strategy_ == "cyclic") {
    // Cyclic distribution: round-robin threads across devices
    for (int t = 0; t < omp_threads_; ++t) {
      int device_idx = t % num_rank_devices;
      thread_device_map_[t] = rank_devices_[device_idx];
    }

  } else {
    // Default to block
    LOGWARN("Unknown device affinity strategy '{}', using 'block'",
            device_affinity_strategy_);
    device_affinity_strategy_ = "block";
    setup_omp_device_mapping();
    return;
  }
}

void Manager::set_omp_device_affinity(const std::string &strategy) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (strategy != "block" && strategy != "cyclic") {
    LOGWARN("Invalid device affinity strategy '{}'. Use 'block' or 'cyclic'",
            strategy);
    return;
  }

  device_affinity_strategy_ = strategy;
  setup_omp_device_mapping();

  LOGINFO("SYCL device affinity strategy set to '{}'", strategy);
}

void Manager::init_for_omp_thread() {
#ifdef _OPENMP
  int thread_id = omp_get_thread_num();

  if (thread_id >= static_cast<int>(thread_device_map_.size())) {
    LOGWARN("Thread {} has no device mapping. Using device 0", thread_id);
    return;
  }

  int assigned_device = thread_device_map_[thread_id];
  LOGTRACE("OpenMP thread {} using SYCL device {}", thread_id, assigned_device);
#endif
}

int Manager::get_thread_device() {
#ifdef _OPENMP
  int thread_id = omp_get_thread_num();

  if (thread_id >= static_cast<int>(thread_device_map_.size())) {
    LOGWARN("Thread {} has no device mapping. Returning device 0", thread_id);
    return 0;
  }

  return thread_device_map_[thread_id];
#else
  return 0;
#endif
}

} // namespace SYCL
} // namespace ARBD
#endif
