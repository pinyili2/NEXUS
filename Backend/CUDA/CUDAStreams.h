// src/Backend/CUDA/CUDAStream.h
#pragma once
#ifdef USE_CUDA
#include "ARBDException.h"
#include "Header.h"
#include <array>
#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include "CUDAManager.h"

namespace ARBD::CUDA {

class InitStreams {
private:
  static constexpr size_t NUM_STREAMS = NUM_QUEUES;
  std::array<cudaStream_t, NUM_STREAMS> streams_;
  std::atomic<size_t> next_index_{0};
  int device_id_;
  bool initialized_{false};

public:
  explicit InitStreams(int device_id) : device_id_(device_id) {
    initialize_streams();
  }

  ~InitStreams() { cleanup_streams(); }

  // Get next stream in round-robin fashion
  cudaStream_t get_next_stream() {
    if (!initialized_) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "InitStreams not initialized for device {}", device_id_);
    }
    size_t idx = next_index_.fetch_add(1) % NUM_STREAMS;
    return streams_[idx];
  }

  // Get specific stream by ID
  cudaStream_t get_stream(size_t stream_id) {
    if (!initialized_) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "InitStreams not initialized for device {}", device_id_);
    }
    return streams_[stream_id % NUM_STREAMS];
  }

  // Synchronize all streams
  void synchronize_all() {
    if (!initialized_)
      return;

    // Set device context for synchronization
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaSetDevice(device_id_));

    for (auto &stream : streams_) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Restore previous device
    CUDA_CHECK(cudaSetDevice(current_device));
  }

private:
  void initialize_streams() {
    // Save current device
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    // Switch to target device
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Create streams on target device
    for (auto &stream : streams_) {
      CUDA_CHECK(cudaStreamCreate(&stream));
    }

    initialized_ = true;

    // Restore previous device
    CUDA_CHECK(cudaSetDevice(current_device));
  }

  void cleanup_streams() {
    if (!initialized_)
      return;

    int current_device;
    if (cudaGetDevice(&current_device) == cudaSuccess) {
      cudaSetDevice(device_id_);

      for (auto &stream : streams_) {
        cudaStreamDestroy(stream);
      }

      cudaSetDevice(current_device);
    }

    initialized_ = false;
  }

  // Prevent copying
  InitStreams(const InitStreams &) = delete;
  InitStreams &operator=(const InitStreams &) = delete;

  // Allow moving
  InitStreams(InitStreams &&other) noexcept
      : streams_(std::move(other.streams_)),
        next_index_(other.next_index_.load()), device_id_(other.device_id_),
        initialized_(other.initialized_) {
    other.initialized_ = false;
  }
};

} // namespace ARBD::CUDA
#endif
