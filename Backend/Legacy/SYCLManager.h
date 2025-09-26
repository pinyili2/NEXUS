#pragma once

#ifdef USE_SYCL
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Header.h"
#include <array>
#include <chrono>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace ARBD {
namespace SYCL {
inline void check_sycl_error(const sycl::exception &e, std::string_view file,
                             int line) {
  ARBD_Exception(ExceptionType::SYCLRuntimeError, "SYCL error at {}:{}: {}",
                 file, line, e.what());
}

#define SYCL_CHECK(call)                                                       \
  try {                                                                        \
    call;                                                                      \
  } catch (const sycl::exception &e) {                                         \
    check_sycl_error(e, __FILE__, __LINE__);                                   \
  }

/**
 * @brief Modern SYCL device management system
 *
 * This class provides a comprehensive SYCL device management system with
 * support for multiple devices, queue management, and device selection. It
 * handles device initialization, selection, and provides utilities for
 * multi-device operations.
 *
 * Features:
 * - Multi-device support (GPU, CPU, accelerators)
 * - Automatic queue management
 * - Device selection and synchronization
 * - Performance monitoring
 * - Exception handling integration
 *
 * @example Basic Usage:
 * ```cpp
 * // Initialize SYCL system
 * ARBD::Manager::init();
 *
 * // Select specific devices
 * std::vector<unsigned int> device_ids = {0, 1};
 * ARBD::Manager::select_devices(device_ids);
 *
 * // Use a specific device
 * ARBD::Manager::use(0);
 *
 * // Get current queue
 * auto& queue = ARBD::Manager::get_current_queue();
 *
 * // Synchronize all devices
 * ARBD::Manager::sync();
 * ```
 *
 * @example Multi-Device Operations:
 * ```cpp
 * // Get device properties
 * const auto& device = ARBD::Manager::devices[0];
 * const auto& props = device.properties();
 *
 * // Submit work to specific device
 * auto& queue = device.get_queue();
 * queue.submit([&](sycl::handler& h) {
 * // kernel code
 * });
 * ```
 *
 * @note The class uses static methods for global device management.
 * All operations are thread-safe and exception-safe.
 */
class Manager {
public:
  static constexpr idx_t NUM_QUEUES = 8;

  /**
   * @brief Individual SYCL device management class
   *
   * This nested class represents a single SYCL device and manages its
   * resources, including queues and device properties.
   *
   * Features:
   * - Queue management
   * - Device property access
   * - Performance monitoring
   * - Safe resource cleanup
   *
   * @example Basic Usage:
   * ```cpp
   * // Get device properties
   * const auto& device = ARBD::Manager::devices[0];
   * const auto& props = device.properties();
   *
   * // Get a queue
   * auto& queue = device.get_queue(0);
   *
   * // Get next available queue
   * auto& next_queue = device.get_next_queue();
   * ```
   */
  class Device {
  public:
    explicit Device(const sycl::device &dev, unsigned int id);
    ~Device() = default;

    // Delete copy constructor and copy assignment operator
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    // Allow moving
    Device(Device &&) = default;
    Device &operator=(Device &&) = default;

    [[nodiscard]] Queue &get_queue(idx_t queue_id) {
      return queues_[queue_id % NUM_QUEUES];
    }

    [[nodiscard]] const Queue &get_queue(idx_t queue_id) const {
      return queues_[queue_id % NUM_QUEUES];
    }

    [[nodiscard]] Queue &get_next_queue() {
      last_queue_ = (last_queue_ + 1) % NUM_QUEUES;
      return queues_[last_queue_];
    }

    [[nodiscard]] unsigned int id() const noexcept { return id_; }
    void set_id(unsigned int new_id) noexcept {
      id_ = new_id;
    } // Add setter for ID
    [[nodiscard]] const sycl::device &get_device() const noexcept {
      return device_;
    }
    [[nodiscard]] const std::string &name() const noexcept { return name_; }
    [[nodiscard]] const std::string &vendor() const noexcept { return vendor_; }
    [[nodiscard]] const std::string &version() const noexcept {
      return version_;
    }
    [[nodiscard]] idx_t max_work_group_size() const noexcept {
      return max_work_group_size_;
    }
    [[nodiscard]] idx_t max_compute_units() const noexcept {
      return max_compute_units_;
    }
    [[nodiscard]] idx_t global_mem_size() const noexcept {
      return global_mem_size_;
    }
    [[nodiscard]] idx_t local_mem_size() const noexcept {
      return local_mem_size_;
    }
    [[nodiscard]] bool is_cpu() const noexcept { return is_cpu_; }
    [[nodiscard]] bool is_gpu() const noexcept { return is_gpu_; }
    [[nodiscard]] bool is_accelerator() const noexcept {
      return is_accelerator_;
    }

    void synchronize_all_queues();

  private:
    void query_device_properties();

    // Helper function to create all queues for a device
    static std::array<ARBD::SYCL::Queue, NUM_QUEUES>
    create_queues(const sycl::device &dev, unsigned int id);

    unsigned int id_;
    sycl::device device_;
    std::array<Queue, NUM_QUEUES> queues_;
    int last_queue_{-1};

    // Device properties
    std::string name_;
    std::string vendor_;
    std::string version_;
    idx_t max_work_group_size_;
    idx_t max_compute_units_;
    idx_t global_mem_size_;
    idx_t local_mem_size_;
    bool is_cpu_;
    bool is_gpu_;
    bool is_accelerator_;

    // Friend class to allow Manager to access private members
    friend class Manager;
  };

  // Static interface
  static void init();
  static void load_info();
  static void select_devices(std::span<const unsigned int> device_ids);
  static void use(int device_id);
  static void sync(int device_id);
  static void sync();
  static int current();
  static void prefer_device_type(sycl::info::device_type type);
  static void finalize();

  [[nodiscard]] static idx_t all_device_size() noexcept {
    return all_devices_.size();
  }
  [[nodiscard]] static const std::vector<Device> &all_devices() noexcept {
    return all_devices_;
  }
  [[nodiscard]] static const std::vector<Device> &devices() noexcept {
    return devices_;
  }
  [[nodiscard]] static Queue &get_current_queue() {
    return devices_[current_device_].get_next_queue();
  }
  [[nodiscard]] static Device &get_current_device() {
    return devices_[current_device_];
  }

  // Device filtering utilities
  [[nodiscard]] static std::vector<unsigned int> get_gpu_device_ids();
  [[nodiscard]] static std::vector<unsigned int> get_cpu_device_ids();
  [[nodiscard]] static std::vector<unsigned int> get_accelerator_device_ids();

  // Add missing static method declaration
  [[nodiscard]] static Device &get_device(unsigned int device_id) {
    if (device_id >= devices_.size()) {
      ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}",
                     device_id);
    }
    return devices_[device_id];
  }

private:
  static void init_devices();
  static void discover_devices();

  static std::vector<Device> all_devices_;
  static std::vector<Device> devices_;
  static int current_device_;
  static sycl::info::device_type preferred_type_;
};

} // namespace SYCL
} // namespace ARBD
#endif // PROJECT_USES_SYCL
