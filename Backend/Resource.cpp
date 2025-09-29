// src/Backend/Resource.cpp - New implementation
#include "Resource.h"

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#endif

namespace ARBD {

// ============================================================================
// Device Context Management
// ============================================================================
void *Resource::get_stream_impl(StreamType stream_type) const {
  if (type_ == ResourceType::CPU)
    return nullptr;

  ensure_context();
  ensure_queues_initialized();

  short stream_id = static_cast<short>(stream_type);

#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA) {
    return reinterpret_cast<void *>(streams_->get_stream(stream_id));
  }
#endif

#ifdef USE_SYCL
  if (type_ == ResourceType::SYCL) {
    return &streams_->get_queue(stream_id);
  }
#endif

  return nullptr;
}
void *Resource::get_stream_impl(size_t stream_id,
                                StreamType stream_type) const {
  if (type_ == ResourceType::CPU)
    return nullptr;

  ensure_context();
  ensure_queues_initialized();

#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA) {
    return reinterpret_cast<void *>(streams_->get_stream(stream_id));
  }
#endif

#ifdef USE_SYCL
  if (type_ == ResourceType::SYCL) {
    return &streams_->get_queue(stream_id);
  }
#endif

  return nullptr;
}
void Resource::ensure_context() const {
  if (!device_verified_) {
    device_available_ = verify_device();
    device_verified_ = true;
  }

  if (!device_available_) {
    ARBD_Exception(ExceptionType::ValueError,
                   "Device {} of type {} is not available", id_,
                   getTypeString());
  }

  // Activate device for all operations
  activate();
}

void Resource::activate() const {
  if (type_ == ResourceType::CPU)
    return;

#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA) {
    CUDA_CHECK(cudaSetDevice(static_cast<int>(id_)));
    return;
  }
#endif

#ifdef USE_SYCL
  if (type_ == ResourceType::SYCL) {

    return;
  }
#endif
}

bool Resource::verify_device() const {
  if (type_ == ResourceType::CPU)
    return true;

#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA) {
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess)
      return false;
    return id_ < device_count;
  }
#endif

#ifdef USE_SYCL
  if (type_ == ResourceType::SYCL) {
    return id_ < static_cast<short>(SYCL::Manager::device_count());
  }
#endif

  return false;
}

void Resource::ensure_queues_initialized() const {
#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA && !streams_) {
    // Device context is already activated by ensure_context()
    streams_ = std::make_shared<CUDA::InitStreams>(static_cast<int>(id_));
  }
#endif

#ifdef USE_SYCL
  if (type_ == ResourceType::SYCL && !streams_) {
    auto device = SYCL::Manager::get_device_by_id(id_);
    streams_ = std::make_shared<SYCL::InitQueues>(device);
  }
#endif
}

void Resource::synchronize_streams() const {
  if (type_ == ResourceType::CPU)
    return;
  else {
    streams_->synchronize_all();
  }
}

// ============================================================================
// Factory Methods
// ============================================================================

Resource Resource::create_cuda_device(short device_id) {
#ifdef USE_CUDA
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_id >= device_count) {
    ARBD_Exception(ExceptionType::ValueError,
                   "CUDA device {} not available (found {} devices)", device_id,
                   device_count);
  }
  return Resource{ResourceType::CUDA, device_id};
#else
  ARBD_Exception(ExceptionType::ValueError, "CUDA not available");
#endif
}

Resource Resource::create_sycl_device(short device_id) {
#ifdef USE_SYCL
  if (device_id >= static_cast<short>(SYCL::Manager::device_count())) {
    ARBD_Exception(ExceptionType::ValueError, "SYCL device {} not available",
                   device_id);
  }
  return Resource{ResourceType::SYCL, device_id};
#else
  ARBD_Exception(ExceptionType::ValueError, "SYCL not available");
#endif
}

// ============================================================================
// Peer Access
// ============================================================================

bool Resource::can_access_peer(const Resource &other) const {
  // Same resource can always access itself
  if (type_ == other.type_ && id_ == other.id_) {
    return true;
  }

  // CPU can access all resources (through host memory)
  if (type_ == ResourceType::CPU || other.type_ == ResourceType::CPU) {
    return true;
  }

  // Cross-backend peer access is not supported
  if (type_ != other.type_) {
    return false;
  }

#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA) {
    int can_access;
    cudaDeviceCanAccessPeer(&can_access, id_, other.id_);
    return can_access != 0;
  }
#endif

  // For SYCL and Metal, assume no peer access for now
  return false;
}

// ============================================================================
// Validation
// ============================================================================

void Resource::validate() const {
  if (type_ == ResourceType::CPU) {
    return; // CPU resources are always valid
  }

#ifdef USE_CUDA
  if (type_ == ResourceType::CUDA) {
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        id_ >= device_count) {
      ARBD_Exception(ExceptionType::ValueError,
                     "CUDA device {} does not exist (count: {})", id_,
                     device_count);
    }
    return;
  }
#endif

#ifdef USE_SYCL
  if (type_ == ResourceType::SYCL) {
    if (id_ >= static_cast<short>(SYCL::Manager::device_count())) {
      ARBD_Exception(ExceptionType::ValueError, "SYCL device {} does not exist",
                     id_);
    }
    return;
  }
#endif

  ARBD_Exception(ExceptionType::ValueError, "Unsupported resource type {}",
                 static_cast<int>(type_));
}

} // namespace ARBD
