#ifdef USE_SYCL
#include "SYCLManager.h"

namespace ARBD {
namespace SYCL {

// Static member initialization
std::vector<sycl::device> Manager::all_devices_;
bool Manager::initialized_ = false;

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

} // namespace SYCL
} // namespace ARBD
#endif
