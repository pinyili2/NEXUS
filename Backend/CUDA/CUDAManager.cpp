// src/Backend/CUDA/CUDAManager.cpp
#ifdef USE_CUDA
#include "CUDAManager.h"
#include "ARBDLogger.h"

namespace ARBD {
namespace CUDA {

// Static member initialization
std::vector<cudaDeviceProp> Manager::device_properties_;
std::vector<std::vector<bool>> Manager::peer_access_matrix_;
bool Manager::initialized_ = false;
std::vector<int> Manager::rank_devices_;
bool Manager::multi_rank_mode_ = false;
int Manager::rank_id_ = -1;
int Manager::omp_threads_ = 1;
std::vector<int> Manager::thread_gpu_map_;
std::string Manager::gpu_affinity_strategy_ = "block";
std::mutex Manager::mtx_;

void Manager::init() {
  if (initialized_) {
    LOGWARN("CUDA Manager already initialized");
    return;
  }

  LOGDEBUG("Initializing CUDA Manager...");

  discover_devices();
  query_peer_access();

  initialized_ = true;
  LOGINFO("CUDA Manager initialized with {} device(s)",
          device_properties_.size());
}

void Manager::discover_devices() {
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));

  if (num_devices == 0) {
    ARBD_Exception(ExceptionType::ValueError, "No CUDA devices found");
  }

  device_properties_.clear();
  device_properties_.reserve(num_devices);

  for (int dev = 0; dev < num_devices; ++dev) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
    device_properties_.push_back(props);

    LOGINFO("Found CUDA device [{}]: {} (SM {}.{}, {:.2f} GB)", dev, props.name,
            props.major, props.minor,
            props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }
}

void Manager::query_peer_access() {
  size_t num_devices = device_properties_.size();
  peer_access_matrix_.resize(num_devices,
                             std::vector<bool>(num_devices, false));

  for (size_t i = 0; i < num_devices; ++i) {
    peer_access_matrix_[i][i] = true; // Device can access itself

    for (size_t j = i + 1; j < num_devices; ++j) {
      int can_access_ij, can_access_ji;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_ij, i, j));
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_ji, j, i));

      peer_access_matrix_[i][j] = (can_access_ij != 0);
      peer_access_matrix_[j][i] = (can_access_ji != 0);

      if (can_access_ij || can_access_ji) {
        LOGDEBUG("P2P access between devices {} and {}: {} <-> {}", i, j,
                 can_access_ij ? "yes" : "no", can_access_ji ? "yes" : "no");
      }
    }
  }
}

void Manager::enable_peer_access() {
  if (!initialized_) {
    ARBD_Exception(ExceptionType::ValueError, "CUDA Manager not initialized");
  }

  size_t num_devices = device_properties_.size();

  for (size_t i = 0; i < num_devices; ++i) {
    CUDA_CHECK(cudaSetDevice(i));

    for (size_t j = 0; j < num_devices; ++j) {
      if (i != j && peer_access_matrix_[i][j]) {
        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);

        // Ignore if already enabled
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
          cudaGetLastError(); // Clear the error state
        } else if (err != cudaSuccess) {
          CUDA_CHECK(err);
        }

        LOGDEBUG("Enabled P2P access: device {} -> device {}", i, j);
      }
    }
  }
}

bool Manager::can_access_peer(int device1, int device2) {
  if (!initialized_) {
    return false;
  }

  if (device1 < 0 || device2 < 0 ||
      device1 >= static_cast<int>(peer_access_matrix_.size()) ||
      device2 >= static_cast<int>(peer_access_matrix_.size())) {
    return false;
  }

  return peer_access_matrix_[device1][device2];
}

int Manager::device_count() {
  if (!initialized_) {
    return 0;
  }
  return static_cast<int>(device_properties_.size());
}

cudaDeviceProp Manager::get_device_properties(int device_id) {
  if (!initialized_) {
    ARBD_Exception(ExceptionType::ValueError, "CUDA Manager not initialized");
  }

  if (device_id < 0 ||
      device_id >= static_cast<int>(device_properties_.size())) {
    ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}",
                   device_id);
  }

  return device_properties_[device_id];
}
void Manager::init_for_rank(int local_rank, int ranks_per_node,
                            int threads_per_rank, bool verbose) {
  std::lock_guard<std::mutex> lock(mtx_);

  // First, do standard initialization if not already done
  if (device_properties_.empty()) {
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

  // Get total number of GPUs
  int num_gpus = static_cast<int>(device_properties_.size());
  if (num_gpus == 0) {
    ARBD_Exception(ExceptionType::ValueError,
                   "No CUDA devices available for rank {}", local_rank);
  }

  // Determine GPU assignment for this rank
  rank_devices_.clear();

  if (ranks_per_node <= 0) {
    ranks_per_node = 1; // Default to single rank
  }

  if (ranks_per_node == 1) {
    // Single rank gets all GPUs
    for (int i = 0; i < num_gpus; ++i) {
      rank_devices_.push_back(i);
    }
    LOGINFO("Single rank mode: assigned all {} GPU(s)", num_gpus);

  } else if (ranks_per_node <= num_gpus) {
    // One or more GPUs per rank
    int gpus_per_rank = num_gpus / ranks_per_node;
    int remainder = num_gpus % ranks_per_node;

    int start_gpu = local_rank * gpus_per_rank;
    if (local_rank < remainder) {
      start_gpu += local_rank;
      gpus_per_rank += 1;
    } else {
      start_gpu += remainder;
    }

    for (int i = 0; i < gpus_per_rank; ++i) {
      rank_devices_.push_back(start_gpu + i);
    }

    LOGINFO("Rank {} assigned to {} GPU(s): [{}]", local_rank,
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
    // More ranks than GPUs: round-robin assignment
    int gpu_id = local_rank % num_gpus;
    rank_devices_.push_back(gpu_id);
    LOGWARN("Rank {} sharing GPU {} (oversubscription: {} ranks, {} GPUs)",
            local_rank, gpu_id, ranks_per_node, num_gpus);
  }

  // Setup OpenMP thread to GPU mapping
  setup_omp_gpu_mapping();

  if (verbose) {
    LOGINFO("Rank {} configuration:", local_rank);
    LOGINFO("  OpenMP threads: {}", omp_threads_);
    LOGINFO("  Assigned GPUs: {}", rank_devices_.size());
    for (int gpu_id : rank_devices_) {
      const auto &dev = device_properties_[gpu_id];
      LOGINFO("    GPU {}: {} (SM {}.{}, {:.1f}GB)", gpu_id, dev.name,
              dev.major, dev.minor,
              dev.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    }

    // Show thread-GPU mapping
    LOGINFO("  Thread-GPU mapping (strategy: {}):", gpu_affinity_strategy_);
    for (int t = 0; t < std::min(omp_threads_, 8); ++t) {
      LOGINFO("    Thread {} -> GPU {}", t, thread_gpu_map_[t]);
    }
    if (omp_threads_ > 8) {
      LOGINFO("    ... ({} more threads)", omp_threads_ - 8);
    }
  }

  // Initialize the assigned devices for this rank
  for (int gpu_id : rank_devices_) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Set the current device to the first assigned GPU
  CUDA_CHECK(cudaSetDevice(rank_devices_[0]));

  // DISABLED: Enable peer access between assigned GPUs if multiple (causes
  // hangs)
  if (false && rank_devices_.size() > 1) {
    for (size_t i = 0; i < rank_devices_.size(); ++i) {
      for (size_t j = i + 1; j < rank_devices_.size(); ++j) {
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, rank_devices_[i],
                                           rank_devices_[j]));
        if (can_access) {
          CUDA_CHECK(cudaSetDevice(rank_devices_[i]));
          cudaError_t err = cudaDeviceEnablePeerAccess(rank_devices_[j], 0);
          if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
            // Clear the error if peer access was already enabled
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
              cudaGetLastError(); // Clear the error state
            }
            LOGDEBUG("Peer access enabled: GPU {} <-> GPU {}", rank_devices_[i],
                     rank_devices_[j]);
          } else {
            LOGWARN("Failed to enable peer access GPU {} -> GPU {}: {}",
                    rank_devices_[i], rank_devices_[j],
                    cudaGetErrorString(err));
          }

          CUDA_CHECK(cudaSetDevice(rank_devices_[j]));
          err = cudaDeviceEnablePeerAccess(rank_devices_[i], 0);
          if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
            // Clear the error if peer access was already enabled
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
              cudaGetLastError(); // Clear the error state
            }
            LOGDEBUG("Peer access enabled: GPU {} <-> GPU {}", rank_devices_[j],
                     rank_devices_[i]);
          } else {
            LOGWARN("Failed to enable peer access GPU {} -> GPU {}: {}",
                    rank_devices_[j], rank_devices_[i],
                    cudaGetErrorString(err));
          }
        }
      }
    }
    CUDA_CHECK(cudaSetDevice(rank_devices_[0]));
  }

  // DISABLED: Configure for OpenMP usage (causes hangs with cudaSetDevice in
  // parallel)
#ifdef _OPENMP
  if (false && omp_threads_ > 1) {
// Set thread affinity for NUMA awareness
#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int assigned_gpu = thread_gpu_map_[thread_id];

      // Each thread sets its default GPU
      cudaSetDevice(assigned_gpu);
    }

    LOGINFO("OpenMP configuration complete: {} threads across {} GPU(s)",
            omp_threads_, rank_devices_.size());
  }
#endif

  // Instead, let each OpenMP thread handle its own GPU setup when needed
  LOGINFO("OpenMP thread-GPU mapping configured: {} threads across {} GPU(s)",
          omp_threads_, rank_devices_.size());

  LOGINFO("CUDA Manager initialized for rank {} with {} GPU(s) and {} OpenMP "
          "thread(s)",
          local_rank, rank_devices_.size(), omp_threads_);
}
void Manager::setup_omp_gpu_mapping() {
  thread_gpu_map_.resize(omp_threads_);

  int num_rank_gpus = static_cast<int>(rank_devices_.size());

  if (gpu_affinity_strategy_ == "block") {
    // Block distribution: consecutive threads use same GPU
    int threads_per_gpu = (omp_threads_ + num_rank_gpus - 1) / num_rank_gpus;

    for (int t = 0; t < omp_threads_; ++t) {
      int gpu_idx = t / threads_per_gpu;
      if (gpu_idx >= num_rank_gpus)
        gpu_idx = num_rank_gpus - 1;
      thread_gpu_map_[t] = rank_devices_[gpu_idx];
    }

  } else if (gpu_affinity_strategy_ == "cyclic") {
    // Cyclic distribution: round-robin threads across GPUs
    for (int t = 0; t < omp_threads_; ++t) {
      int gpu_idx = t % num_rank_gpus;
      thread_gpu_map_[t] = rank_devices_[gpu_idx];
    }

  } else {
    // Default to block
    LOGWARN("Unknown GPU affinity strategy '{}', using 'block'",
            gpu_affinity_strategy_);
    gpu_affinity_strategy_ = "block";
    setup_omp_gpu_mapping();
    return;
  }
}
void Manager::set_omp_gpu_affinity(const std::string &strategy) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (strategy != "block" && strategy != "cyclic") {
    LOGWARN("Invalid GPU affinity strategy '{}'. Use 'block' or 'cyclic'",
            strategy);
    return;
  }

  gpu_affinity_strategy_ = strategy;
  setup_omp_gpu_mapping();

  LOGINFO("GPU affinity strategy set to '{}'", strategy);
}

void Manager::init_for_omp_thread() {
#ifdef _OPENMP
  int thread_id = omp_get_thread_num();
#else
  int thread_id = 0;
#endif

  if (thread_id >= static_cast<int>(thread_gpu_map_.size())) {
    LOGWARN("Thread {} has no GPU mapping. Using GPU 0", thread_id);
    cudaSetDevice(0);
    return;
  }

  int assigned_gpu = thread_gpu_map_[thread_id];
  cudaSetDevice(assigned_gpu);

  LOGTRACE("OpenMP thread {} using GPU {}", thread_id, assigned_gpu);
}

int Manager::get_thread_gpu() {
#ifdef _OPENMP
  int thread_id = omp_get_thread_num();
#else
  int thread_id = 0;
#endif

  if (thread_id >= static_cast<int>(thread_gpu_map_.size())) {
    return 0; // Default to GPU 0
  }

  return thread_gpu_map_[thread_id];
}

void Manager::finalize(bool reset_devices) {
  if (!initialized_) {
    return;
  }

  LOGDEBUG("Finalizing CUDA Manager...");

  // Disable peer access before cleanup
  for (size_t i = 0; i < rank_devices_.size(); ++i) {
    for (size_t j = 0; j < rank_devices_.size(); ++j) {
      if (i != j) {
        cudaSetDevice(rank_devices_[i]);
        cudaError_t err = cudaDeviceDisablePeerAccess(rank_devices_[j]);
        // Ignore errors - peer access might not be enabled
        (void)err;
      }
    }
  }

  // Reset CUDA devices (optional, expensive operation)
  if (reset_devices) {
    for (size_t i = 0; i < device_properties_.size(); ++i) {
      cudaSetDevice(i);
      cudaDeviceReset();
    }
  }

  // Only clear device info and mark uninitialized if doing full reset
  if (reset_devices) {
    device_properties_.clear();
    peer_access_matrix_.clear();
    initialized_ = false;
  }
  // If not resetting devices, keep initialized_ = true so device_count() works

  // Always clear rank-specific info to allow reinitialization
  rank_devices_.clear();
  thread_gpu_map_.clear();
  multi_rank_mode_ = false;
  rank_id_ = -1;
  omp_threads_ = 1;
  gpu_affinity_strategy_ = "block";

  LOGINFO("CUDA Manager finalized");
}

void Manager::load_info() {
  if (!initialized_) {
    init();
  }

  LOGINFO("CUDA Manager Information:");
  LOGINFO("  Device count: {}", device_count());
  for (int i = 0; i < device_count(); ++i) {
    const auto &props = get_device_properties(i);
    LOGINFO("  Device {}: {} (Compute {}.{}, Memory: {:.2f} GB)", i, props.name,
            props.major, props.minor,
            props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }
}

} // namespace CUDA
} // namespace ARBD
#endif
