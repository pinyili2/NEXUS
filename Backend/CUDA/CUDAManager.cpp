// src/Backend/CUDA/CUDAManager.cpp
#ifdef USE_CUDA
#include "CUDAManager.h"

namespace ARBD {
namespace CUDA {

// Static member initialization
std::vector<cudaDeviceProp> Manager::device_properties_;
std::vector<std::vector<bool>> Manager::peer_access_matrix_;
bool Manager::initialized_ = false;

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
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
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

void Manager::finalize() {
  if (!initialized_) {
    return;
  }

  LOGDEBUG("Finalizing CUDA Manager...");

  // Reset CUDA devices
  for (size_t i = 0; i < device_properties_.size(); ++i) {
    cudaSetDevice(i);
    cudaDeviceReset();
  }

  device_properties_.clear();
  peer_access_matrix_.clear();
  initialized_ = false;

  LOGINFO("CUDA Manager finalized");
}

} // namespace CUDA
} // namespace ARBD
#endif
