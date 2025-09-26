#ifdef USE_SYCL
#include "SYCLManager.h"
#include "ARBDLogger.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <span>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

namespace ARBD {
namespace SYCL {
// Static member initialization
std::vector<Manager::Device> Manager::all_devices_;
std::vector<Manager::Device> Manager::devices_;
int Manager::current_device_{0};
sycl::info::device_type Manager::preferred_type_{sycl::info::device_type::cpu};

// Device class implementation
Manager::Device::Device(const sycl::device &dev, unsigned int id)
    : id_(id), device_(dev), queues_(create_queues(dev, id)) {

  // Query device properties after construction
  query_device_properties();

  LOGDEBUG("Device {} initialized: {} ({})", id_, name_.c_str(),
           vendor_.c_str());
  LOGDEBUG("  Compute units: {}, Global memory: {:.1f}GB, Max work group: {}",
           max_compute_units_,
           static_cast<float>(global_mem_size_) / (1024.0f * 1024.0f * 1024.0f),
           max_work_group_size_);
}

// Helper function to create all queues for a device with proper RAII
std::array<ARBD::SYCL::Queue, Manager::NUM_QUEUES>
Manager::Device::create_queues(const sycl::device &dev, unsigned int id) {
  try {
    // Test if device can create a basic queue first with explicit single-device
    // context
    sycl::queue test_queue(sycl::context({dev}), dev);

    // If successful, create all our wrapped queues
    // Note: We need to construct each queue individually since Queue() is
    // deleted
    return std::array<Queue, Manager::NUM_QUEUES>{
        {Queue(dev), Queue(dev), Queue(dev), Queue(dev), Queue(dev), Queue(dev),
         Queue(dev), Queue(dev)}};

  } catch (const sycl::exception &e) {
    LOGERROR("SYCL exception creating queues for device {}: {}", id, e.what());

    // Try fallback with empty properties
    try {
      sycl::property_list empty_props;
      return std::array<Queue, Manager::NUM_QUEUES>{
          {Queue(dev, empty_props), Queue(dev, empty_props),
           Queue(dev, empty_props), Queue(dev, empty_props),
           Queue(dev, empty_props), Queue(dev, empty_props),
           Queue(dev, empty_props), Queue(dev, empty_props)}};
    } catch (const sycl::exception &e2) {
      LOGERROR("Failed to create fallback queues for device {}: {}", id,
               e2.what());
      throw; // Re-throw if we can't create any queues
    }
  }
}

void Manager::Device::query_device_properties() {
  try {
    // Set default values first
    name_ = "Unknown Device";
    vendor_ = "Unknown Vendor";
    version_ = "Unknown Version";
    max_work_group_size_ = 1;
    max_compute_units_ = 1;
    global_mem_size_ = 0;
    local_mem_size_ = 0;
    is_cpu_ = false;
    is_gpu_ = false;
    is_accelerator_ = false;

    // Try to get device type first (most critical)
    try {
      auto device_type = device_.get_info<sycl::info::device::device_type>();
      is_cpu_ = (device_type == sycl::info::device_type::cpu);
      is_gpu_ = (device_type == sycl::info::device_type::gpu);
      is_accelerator_ = (device_type == sycl::info::device_type::accelerator);
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query device type for device {}: {}", id_, e.what());
    }

    // Try to get basic device info
    try {
      name_ = device_.get_info<sycl::info::device::name>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query device name for device {}: {}", id_, e.what());
    }

    try {
      vendor_ = device_.get_info<sycl::info::device::vendor>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query device vendor for device {}: {}", id_, e.what());
    }

    try {
      version_ = device_.get_info<sycl::info::device::version>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query device version for device {}: {}", id_,
              e.what());
    }

    // Try to get performance characteristics
    try {
      max_work_group_size_ =
          device_.get_info<sycl::info::device::max_work_group_size>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query max work group size for device {}: {}", id_,
              e.what());
    }

    try {
      max_compute_units_ =
          device_.get_info<sycl::info::device::max_compute_units>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query max compute units for device {}: {}", id_,
              e.what());
    }

    try {
      global_mem_size_ =
          device_.get_info<sycl::info::device::global_mem_size>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query global memory size for device {}: {}", id_,
              e.what());
    }

    try {
      local_mem_size_ = device_.get_info<sycl::info::device::local_mem_size>();
    } catch (const sycl::exception &e) {
      LOGWARN("Failed to query local memory size for device {}: {}", id_,
              e.what());
    }

  } catch (const sycl::exception &e) {
    std::cerr << "!!! query_device_properties caught sycl::exception: "
              << e.what() << " for device (name might be uninit): " << name_
              << std::endl;
    LOGERROR("Critical error querying device properties for device {}: {}", id_,
             e.what());
    // Keep default values set above
  }
}

void Manager::Device::synchronize_all_queues() {
  for (auto &queue : queues_) {
    try {
      queue.synchronize();
    } catch (const sycl::exception &e) {
      // Log SYCL errors but continue with other queues
      std::cerr << "Warning: SYCL error during queue synchronization: "
                << e.what() << std::endl;
    } catch (const std::exception &e) {
      // Log other errors but continue with other queues
      std::cerr << "Warning: Error during queue synchronization: " << e.what()
                << std::endl;
    }
  }
}

// Manager static methods implementation
void Manager::init() {
  LOGDEBUG("Initializing SYCL Manager...");

  all_devices_.clear();
  devices_.clear();
  current_device_ = 0;

  discover_devices();

  if (all_devices_.empty()) {
    ARBD_Exception(ExceptionType::ValueError, "No SYCL devices found");
  }

  LOGINFO("Found {} SYCL device(s)", all_devices_.size());
}

// Helper function to check if OpenMP backend should be preferred via
// environment variables
static bool should_prefer_openmp_backend() {
  // Check ONEAPI_DEVICE_SELECTOR for OpenMP preference
  const char *oneapi_selector = std::getenv("ONEAPI_DEVICE_SELECTOR");
  if (oneapi_selector) {
    std::string selector_str(oneapi_selector);
    if (selector_str.find("omp:") != std::string::npos ||
        selector_str.find("openmp:") != std::string::npos) {
      return true;
    }
  }

  // Check SYCL_DEVICE_FILTER for OpenMP backend
  const char *sycl_filter = std::getenv("SYCL_DEVICE_FILTER");
  if (sycl_filter) {
    std::string filter_str(sycl_filter);
    if (filter_str.find("omp") != std::string::npos ||
        filter_str.find("openmp") != std::string::npos) {
      return true;
    }
  }

  return false;
}

void Manager::discover_devices() {
  try {
    // Check if OpenMP backend is explicitly requested via environment variables
    bool env_prefers_openmp = should_prefer_openmp_backend();
    if (env_prefers_openmp) {
      LOGINFO("Environment variables indicate OpenMP backend preference");
    }

    // Get all platforms
    auto platforms = sycl::platform::get_platforms();

    // First pass: collect valid devices information without constructing Device
    // objects
    struct DeviceInfo {
      sycl::device device;
      unsigned int id;
      sycl::info::device_type type;
      std::string platform_name;
      bool is_openmp_backend;
    };

    std::vector<DeviceInfo> potential_device_infos;
    unsigned int device_id = 0;

    for (const auto &platform : platforms) {
      std::string platform_name =
          platform.get_info<sycl::info::platform::name>();
      std::string platform_vendor =
          platform.get_info<sycl::info::platform::vendor>();

      LOGDEBUG("Platform: {} ({})", platform_name.c_str(),
               platform_vendor.c_str());

      // Check if this platform supports OpenMP backend
      // Look for OpenMP indicators in platform name or vendor
      bool is_openmp_platform =
          (platform_name.find("OpenMP") != std::string::npos) ||
          (platform_name.find("omp") != std::string::npos) ||
          (platform_name.find("OMP") != std::string::npos) ||
          (platform_vendor.find("OpenMP") != std::string::npos) ||
          (platform_vendor.find("omp") != std::string::npos) ||
          // Check for common OpenMP backend implementations
          (platform_name.find("hipSYCL") != std::string::npos &&
           env_prefers_openmp) ||
          (platform_name.find("AdaptiveCpp") != std::string::npos &&
           env_prefers_openmp);

      // Get all devices for this platform
      auto platform_devices = platform.get_devices();

      for (const auto &device : platform_devices) {
        try {
          // Test sycl::device copy construction explicitly
          sycl::device temp_device_copy(device);
          auto dev_type =
              temp_device_copy.get_info<sycl::info::device::device_type>();
          LOGDEBUG(
              "Successfully test-copied sycl::device: {}",
              temp_device_copy.get_info<sycl::info::device::name>().c_str());

          // Store device info for later construction
          potential_device_infos.push_back({std::move(temp_device_copy),
                                            device_id, dev_type, platform_name,
                                            is_openmp_platform});
          device_id++;

        } catch (const sycl::exception &e) {
          LOGWARN("SYCL exception during device discovery for device id {}: {}",
                  device_id, e.what());
        } catch (const ARBD::Exception &e) {
          LOGWARN(
              "ARBD::Exception during device discovery for device id {}: {}",
              device_id, e.what());
        } catch (const std::exception &e) {
          LOGWARN("Std::exception during device discovery for device id {}: {}",
                  device_id, e.what());
        }
      }
    }

    // Filter devices with smart preference: GPU > OpenMP CPU > regular CPU
    std::vector<DeviceInfo> selected_device_infos;

    // First, look for GPU devices (highest priority)
    for (const auto &device_info : potential_device_infos) {
      if (device_info.type == sycl::info::device_type::gpu) {
        selected_device_infos.push_back(device_info);
      }
    }

    // If no GPU devices found, look for CPU devices with OpenMP preference
    if (selected_device_infos.empty()) {
      // Look for OpenMP CPU devices first
      for (const auto &device_info : potential_device_infos) {
        if (device_info.type == sycl::info::device_type::cpu &&
            device_info.is_openmp_backend) {
          selected_device_infos.push_back(device_info);
        }
      }

      // If no OpenMP CPU devices, fall back to regular CPU devices
      if (selected_device_infos.empty()) {
        LOGWARN("No OpenMP CPU devices found, using regular CPU devices");
        for (const auto &device_info : potential_device_infos) {
          if (device_info.type == sycl::info::device_type::cpu) {
            selected_device_infos.push_back(device_info);
          }
        }
      } else {
        LOGINFO("Found {} OpenMP CPU device(s)", selected_device_infos.size());
      }
    } else {
      LOGINFO("Found {} GPU device(s)", selected_device_infos.size());
    }

    // If still no devices, use whatever is available
    if (selected_device_infos.empty()) {
      LOGWARN("No GPU or CPU devices found, using all available devices");
      selected_device_infos = std::move(potential_device_infos);
    }

    // Sort device infos by preference: GPU > OpenMP CPU > regular CPU
    std::stable_sort(selected_device_infos.begin(), selected_device_infos.end(),
                     [](const DeviceInfo &a, const DeviceInfo &b) {
                       // GPU devices have highest priority
                       if (a.type == sycl::info::device_type::gpu &&
                           b.type != sycl::info::device_type::gpu) {
                         return true;
                       }
                       if (b.type == sycl::info::device_type::gpu &&
                           a.type != sycl::info::device_type::gpu) {
                         return false;
                       }

                       // Among CPU devices, prefer OpenMP
                       if (a.type == sycl::info::device_type::cpu &&
                           b.type == sycl::info::device_type::cpu) {
                         if (a.is_openmp_backend && !b.is_openmp_backend) {
                           return true;
                         }
                         if (b.is_openmp_backend && !a.is_openmp_backend) {
                           return false;
                         }
                       }

                       // Default to device ID ordering
                       return a.id < b.id;
                     });

    // Now construct Device objects in place
    all_devices_.clear();
    all_devices_.reserve(selected_device_infos.size());

    for (size_t i = 0; i < selected_device_infos.size(); ++i) {
      const auto &device_info = selected_device_infos[i];
      all_devices_.emplace_back(device_info.device,
                                static_cast<unsigned int>(i));
    }

  } catch (const sycl::exception &e) {
    check_sycl_error(e, __FILE__, __LINE__);
  }
}

void Manager::load_info() {
  init();

  // For single device case (Mac), explicitly select only the first device
  // to avoid multi-device context issues
  if (all_devices_.size() == 1) {
    LOGINFO("Single device detected, selecting device 0 for single-device "
            "operation");
    // Use array index 0, not the device's internal ID
    unsigned int device_index = 0;
    select_devices(std::span<const unsigned int>{&device_index, 1});
  } else {
    // Multi-device case - use all discovered devices
    // Copy all_devices_ to devices_ instead of moving to preserve all_devices_
    // for select_devices()
    devices_.clear();
    devices_.reserve(all_devices_.size());
    for (size_t i = 0; i < all_devices_.size(); ++i) {
      devices_.emplace_back(all_devices_[i].get_device(),
                            static_cast<unsigned int>(i));
    }
    init_devices();
  }
}

void Manager::init_devices() {
  LOGINFO("Initializing SYCL devices...");
  std::string msg;

  for (size_t i = 0; i < devices_.size(); i++) {
    if (i > 0) {
      if (i == devices_.size() - 1) {
        msg += " and ";
      } else {
        msg += ", ";
      }
    }
    msg += std::to_string(devices_[i].id());

    // Devices are already initialized in constructor
    // Just log that they're ready
    LOGDEBUG("Device {} ready: {}", devices_[i].id(),
             devices_[i].name().c_str());
  }

  LOGINFO("Initialized SYCL devices: {}", msg.c_str());
  current_device_ = 0;
}

void Manager::select_devices(std::span<const unsigned int> device_ids) {
  devices_.clear();
  devices_.reserve(device_ids.size()); // Reserve space to avoid reallocations

  for (unsigned int id : device_ids) {
    if (id >= all_devices_.size()) {
      ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}", id);
    }
    // Create a new Device by copying the sycl::device and id
    devices_.emplace_back(all_devices_[id].get_device(), id);
  }
  init_devices();
}

void Manager::use(int device_id) {
  if (devices_.empty()) {
    ARBD_Exception(ExceptionType::ValueError, "No devices selected");
  }
  current_device_ = device_id % static_cast<int>(devices_.size());
}

void Manager::sync(int device_id) {
  if (device_id >= static_cast<int>(devices_.size())) {
    ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}",
                   device_id);
  }
  devices_[device_id].synchronize_all_queues();
}

void Manager::sync() {
  for (auto &device : devices_) {
    try {
      device.synchronize_all_queues();
    } catch (const std::exception &e) {
      // Log but continue with other devices
      std::cerr << "Warning: Error synchronizing device " << device.id() << ": "
                << e.what() << std::endl;
    }
  }
}

void Manager::finalize() {
  try {
    // First, synchronize all devices to ensure all operations complete
    if (!devices_.empty()) {
      sync();
    }

    // Minimal delay to let HipSYCL runtime stabilize
    // This prevents worker thread crashes without performance degradation
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } catch (...) {
      // Ignore any errors during the wait
    }

    // Clear current device reference first to prevent access during cleanup
    current_device_ = 0;

    // Clear devices explicitly which will call Device destructors
    // in a controlled manner
    devices_.clear();

    // Now clear all_devices_
    all_devices_.clear();

    // Reset to default state
    preferred_type_ = sycl::info::device_type::cpu;

  } catch (const std::exception &e) {
    // Log but don't throw during finalization to prevent cascading errors
    std::cerr << "Warning: Error during SYCL finalization: " << e.what()
              << std::endl;
  } catch (...) {
    // Catch any other exceptions during cleanup
    std::cerr << "Warning: Unknown error during SYCL finalization" << std::endl;
  }
}

int Manager::current() { return current_device_; }

void Manager::prefer_device_type(sycl::info::device_type type) {
  preferred_type_ = type;

  if (!all_devices_.empty()) {
    std::sort(all_devices_.begin(), all_devices_.end(),
              [type](const Device &a, const Device &b) {
                auto a_type =
                    a.get_device().get_info<sycl::info::device::device_type>();
                auto b_type =
                    b.get_device().get_info<sycl::info::device::device_type>();

                if ((a_type == type) != (b_type == type)) {
                  return a_type == type;
                }
                return a.id() < b.id();
              });

    // Reassign IDs after sorting
    for (size_t i = 0; i < all_devices_.size(); ++i) {
      const_cast<unsigned int &>(all_devices_[i].id_) =
          static_cast<unsigned int>(i);
    }
  }
}

std::vector<unsigned int> Manager::get_gpu_device_ids() {
  std::vector<unsigned int> gpu_ids;
  for (const auto &device : all_devices_) {
    if (device.is_gpu()) {
      gpu_ids.push_back(device.id());
    }
  }
  return gpu_ids;
}

std::vector<unsigned int> Manager::get_cpu_device_ids() {
  std::vector<unsigned int> cpu_ids;
  for (const auto &device : all_devices_) {
    if (device.is_cpu()) {
      cpu_ids.push_back(device.id());
    }
  }
  return cpu_ids;
}

std::vector<unsigned int> Manager::get_accelerator_device_ids() {
  std::vector<unsigned int> accel_ids;
  for (const auto &device : all_devices_) {
    if (device.is_accelerator()) {
      accel_ids.push_back(device.id());
    }
  }
  return accel_ids;
}

} // namespace SYCL
} // namespace ARBD

#endif
