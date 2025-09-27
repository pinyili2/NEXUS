// src/Backend/SYCL/SYCLQueue.h
#pragma once
#include "ARBDException.h"
#include "Header.h"
#ifdef USE_SYCL
#include <array>
#include <atomic>
#include <sycl/sycl.hpp>

namespace ARBD::SYCL {

/**
 * @brief RAII SYCL queue wrapper with proper resource management
 *
 * This class provides a safe RAII wrapper around sycl::queue with
 * guaranteed valid state and automatic resource cleanup.
 *
 * Features:
 * - Guaranteed valid state (no optional/uninitialized queues)
 * - Automatic resource management (RAII)
 * - Exception safety
 * - Move semantics support
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a queue for a specific device - always valid after construction
 * ARBD::Queue queue(device);
 *
 * // Submit work - no need to check if queue is valid
 * queue.submit([&](sycl::handler& h) {
 * // kernel code
 * });
 * queue.synchronize();
 * ```
 *
 * @note The queue is automatically cleaned up when the Queue object is
 * destroyed
 */
class Queue {
private:
  sycl::queue queue_;
  sycl::device device_;
  bool valid_{false}; // Track if queue is valid

public:
  // Default constructor creates invalid queue (for std::array)
  Queue() : queue_(), device_(), valid_(false) {}

  explicit Queue(const sycl::device &dev)
      : queue_(sycl::context({dev}), dev), device_(dev), valid_(true) {}

  explicit Queue(const sycl::device &dev, const sycl::property_list &props)
      : queue_(sycl::context({dev}), dev, props), device_(dev), valid_(true) {}

  ~Queue() {}

  // Prevent copying to avoid resource management complexity
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;

  // Allow moving for efficiency
  Queue(Queue &&other) noexcept
      : queue_(std::move(other.queue_)), device_(std::move(other.device_)),
        valid_(other.valid_) {
    other.valid_ = false;
  }

  Queue &operator=(Queue &&other) noexcept {
    if (this != &other) {
      queue_ = std::move(other.queue_);
      device_ = std::move(other.device_);
      valid_ = other.valid_;
      other.valid_ = false;
    }
    return *this;
  }

  // Check if queue is valid before use
  [[nodiscard]] bool is_valid() const noexcept { return valid_; }

  [[nodiscard]] bool is_in_order() const noexcept {
    return valid_ && queue_.is_in_order();
  }

  [[nodiscard]] sycl::context get_context() const {
    if (!valid_) {
      throw std::runtime_error("Attempting to get context from invalid queue");
    }
    return queue_.get_context();
  }

  [[nodiscard]] const sycl::device &get_device() const {
    if (!valid_) {
      throw std::runtime_error("Attempting to get device from invalid queue");
    }
    return device_;
  }

  [[nodiscard]] sycl::queue &get() noexcept { return queue_; }

  [[nodiscard]] const sycl::queue &get() const noexcept { return queue_; }

  // Implicit conversion operators for convenience
  operator sycl::queue &() noexcept { return queue_; }

  operator const sycl::queue &() const noexcept { return queue_; }
};

class QueuePool {
private:
  std::array<Queue, NUM_QUEUES> queues_;
  std::atomic<size_t> next_index_{0};
  sycl::device device_;
  bool initialized_{false};
  void initialize_queues() {
    try {
      sycl::property_list props{sycl::property::queue::enable_profiling{}};

      // Initialize each queue in the array
      for (size_t i = 0; i < NUM_QUEUES; ++i) {
        // Use placement construction or assignment after default construction
        queues_[i] = Queue(device_, props);
      }

      initialized_ = true;
    } catch (const sycl::exception &e) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to create SYCL queues: {}", e.what());
    }
  };

public:
  QueuePool(const sycl::device &device) : device_(device) {
    initialize_queues();
  }

  Queue &get_next_queue() {
    // Same round-robin logic as Manager::Device
    size_t idx = next_index_.fetch_add(1) % NUM_QUEUES;
    return queues_[idx];
  }

  Queue &get_queue(size_t queue_id) { return queues_[queue_id % NUM_QUEUES]; }

  void synchronize_all() {
    for (auto &queue : queues_) {
      queue.get().wait();
    }
  }
};

} // namespace ARBD::SYCL
#endif
