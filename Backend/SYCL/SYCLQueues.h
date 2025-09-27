// src/Backend/SYCL/SYCLQueue.h
#pragma once
#include "ARBDException.h"
#include "Header.h"
#ifdef USE_SYCL
#include <array>
#include <atomic>
#include <sycl/sycl.hpp>

namespace ARBD::SYCL {

class InitQueues {
private:
  std::array<sycl::queue, NUM_QUEUES> queues_;
  std::atomic<size_t> next_index_{0};
  sycl::device device_;
  bool initialized_{false};
  void initialize_queues() {
    try {
      sycl::property_list props{sycl::property::queue::enable_profiling{}};

      // Initialize each queue in the array
      for (size_t i = 0; i < NUM_QUEUES; ++i) {
        // Use placement construction or assignment after default construction
        queues_[i] = sycl::queue(device_, props);
      }

      initialized_ = true;
    } catch (const sycl::exception &e) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to create SYCL queues: {}", e.what());
    }
  };

public:
  InitQueues(const sycl::device &device) : device_(device) {
    initialize_queues();
  }

  sycl::queue &get_next_queue() {
    // Same round-robin logic as Manager::Device
    size_t idx = next_index_.fetch_add(1) % NUM_QUEUES;
    return queues_[idx];
  }

  sycl::queue &get_queue(size_t queue_id) {
    return queues_[queue_id % NUM_QUEUES];
  }

  void synchronize_all() {
    for (auto &queue : queues_) {
      queue.wait();
    }
  }
};

} // namespace ARBD::SYCL
#endif
