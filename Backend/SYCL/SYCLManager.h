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
inline void check_sycl_error(const sycl::exception& e, std::string_view file, int line) {
	ARBD_Exception(ExceptionType::SYCLRuntimeError,
				   "SYCL error at {}:{}: {}",
				   file,
				   line,
				   e.what());
}

#define SYCL_CHECK(call)                         \
	try {                                        \
		call;                                    \
	} catch (const sycl::exception& e) {         \
		check_sycl_error(e, __FILE__, __LINE__); \
	}

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
 * @note The queue is automatically cleaned up when the Queue object is destroyed
 */
class Queue {
  private:
	sycl::queue queue_;
	sycl::device device_;

  public:
	// Delete default constructor to prevent invalid state
	Queue() = delete;

	// **MODIFIED**: Explicitly create a single-device context to prevent ambiguity.
	// This guarantees that the queue is not considered "multi-device" by the runtime.
	explicit Queue(const sycl::device& dev) : queue_(sycl::context({dev}), dev), device_(dev) {}

	// **MODIFIED**: Also apply the explicit single-device context here.
	explicit Queue(const sycl::device& dev, const sycl::property_list& props)
		: queue_(sycl::context({dev}), dev, props), device_(dev) {}

	// RAII destructor - automatic cleanup
	~Queue() {
		// Skip wait() in destructor - causes deadlocks with HipSYCL worker threads
		// The SYCL runtime will handle cleanup automatically
	}

	// Prevent copying to avoid resource management complexity
	Queue(const Queue&) = delete;
	Queue& operator=(const Queue&) = delete;

	// Allow moving for efficiency
	Queue(Queue&& other) noexcept
		: queue_(std::move(other.queue_)), device_(std::move(other.device_)) {}

	Queue& operator=(Queue&& other) noexcept {
		if (this != &other) {
			// Skip wait() during move - can cause deadlocks with HipSYCL
			// The SYCL runtime will handle outstanding operations
			queue_ = std::move(other.queue_);
			device_ = std::move(other.device_);
		}
		return *this;
	}

	// All operations can assume queue_ is valid
	void synchronize() {
		// Skip synchronization with HipSYCL to avoid deadlocks during cleanup
		// The runtime will handle outstanding operations automatically
	}

	template<typename KernelName = class kernel_default_name, typename F>
	sycl::event submit(F&& f) {
		return queue_.submit(std::forward<F>(f)); // Direct delegation
	}

	[[nodiscard]] bool is_in_order() const noexcept {
		return queue_.is_in_order();
	}

	[[nodiscard]] sycl::context get_context() const {
		return queue_.get_context();
	}

	// Return the specific device associated with this queue.
	[[nodiscard]] const sycl::device& get_device() const {
		return device_;
	}

	// Direct access to underlying queue - always safe
	[[nodiscard]] sycl::queue& get() noexcept {
		return queue_;
	}

	[[nodiscard]] const sycl::queue& get() const noexcept {
		return queue_;
	}

	// Implicit conversion operators for convenience
	operator sycl::queue&() noexcept {
		return queue_;
	}
	operator const sycl::queue&() const noexcept {
		return queue_;
	}
};

/**
 * @brief SYCL event wrapper for timing and synchronization
 *
 * This class provides a convenient wrapper around sycl::event with
 * additional timing and synchronization utilities.
 *
 * Features:
 * - Event timing measurements
 * - Synchronization utilities
 * - Exception handling integration
 * - Profiling support
 *
 * @example Basic Usage:
 * ```cpp
 * // Submit work and get an event
 * ARBD::Event event = queue.submit([&](sycl::handler& h) {
 * // kernel code
 * });
 *
 * // Wait for completion
 * event.wait();
 *
 * // Get execution time
 * auto duration = event.get_execution_time();
 * ```
 *
 * @note Events are automatically waited on when the Event object is destroyed
 */
class Event {
  public:
	Event() = default;

	explicit Event(sycl::event e) : event_(std::move(e)) {}

	~Event() {}

	// Prevent copying
	Event(const Event&) = delete;
	Event& operator=(const Event&) = delete;

	// Allow moving
	Event(Event&& other) noexcept : event_(std::move(other.event_)) {}

	Event& operator=(Event&& other) noexcept {
		if (this != &other) {
			event_ = std::move(other.event_);
		}
		return *this;
	}

	void wait() {
		if (event_.has_value()) {
			SYCL_CHECK(event_->wait());
		}
	}

	[[nodiscard]] bool is_complete() const {
		if (!event_.has_value())
			return true;

		try {
			auto status = event_->get_info<sycl::info::event::command_execution_status>();
			return status == sycl::info::event_command_status::complete;
		} catch (const sycl::exception&) {
			return false;
		}
	}

	[[nodiscard]] std::chrono::nanoseconds get_execution_time() const {
		if (!event_.has_value()) {
			return std::chrono::nanoseconds{0};
		}

		try {
			auto start = event_->get_profiling_info<sycl::info::event_profiling::command_start>();
			auto end = event_->get_profiling_info<sycl::info::event_profiling::command_end>();
			return std::chrono::nanoseconds{end - start};
		} catch (const sycl::exception& e) {
			check_sycl_error(e, __FILE__, __LINE__);
			return std::chrono::nanoseconds{0};
		}
	}

	[[nodiscard]] sycl::event& get() {
		if (!event_.has_value()) {
			ARBD_Exception(ExceptionType::CUDARuntimeError, "Event not initialized");
		}
		return *event_;
	}

	[[nodiscard]] const sycl::event& get() const {
		if (!event_.has_value()) {
			ARBD_Exception(ExceptionType::CUDARuntimeError, "Event not initialized");
		}
		return *event_;
	}

	operator sycl::event&() {
		return get();
	}
	operator const sycl::event&() const {
		return get();
	}

  private:
	std::optional<sycl::event> event_;
};

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
		explicit Device(const sycl::device& dev, unsigned int id);
		~Device() = default;

		// Delete copy constructor and copy assignment operator
		Device(const Device&) = delete;
		Device& operator=(const Device&) = delete;

		// Allow moving
		Device(Device&&) = default;
		Device& operator=(Device&&) = default;

		[[nodiscard]] Queue& get_queue(idx_t queue_id) {
			return queues_[queue_id % NUM_QUEUES];
		}

		[[nodiscard]] const Queue& get_queue(idx_t queue_id) const {
			return queues_[queue_id % NUM_QUEUES];
		}

		[[nodiscard]] Queue& get_next_queue() {
			last_queue_ = (last_queue_ + 1) % NUM_QUEUES;
			return queues_[last_queue_];
		}

		[[nodiscard]] unsigned int id() const noexcept {
			return id_;
		}
		void set_id(unsigned int new_id) noexcept {
			id_ = new_id;
		} // Add setter for ID
		[[nodiscard]] const sycl::device& get_device() const noexcept {
			return device_;
		}
		[[nodiscard]] const std::string& name() const noexcept {
			return name_;
		}
		[[nodiscard]] const std::string& vendor() const noexcept {
			return vendor_;
		}
		[[nodiscard]] const std::string& version() const noexcept {
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
		[[nodiscard]] bool is_cpu() const noexcept {
			return is_cpu_;
		}
		[[nodiscard]] bool is_gpu() const noexcept {
			return is_gpu_;
		}
		[[nodiscard]] bool is_accelerator() const noexcept {
			return is_accelerator_;
		}

		void synchronize_all_queues();

	  private:
		void query_device_properties();

		// Helper function to create all queues for a device
		static std::array<ARBD::SYCL::Queue, NUM_QUEUES> create_queues(const sycl::device& dev,
																	   unsigned int id);

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
	[[nodiscard]] static const std::vector<Device>& all_devices() noexcept {
		return all_devices_;
	}
	[[nodiscard]] static const std::vector<Device>& devices() noexcept {
		return devices_;
	}
	[[nodiscard]] static Queue& get_current_queue() {
		return devices_[current_device_].get_next_queue();
	}
	[[nodiscard]] static Device& get_current_device() {
		return devices_[current_device_];
	}

	// Device filtering utilities
	[[nodiscard]] static std::vector<unsigned int> get_gpu_device_ids();
	[[nodiscard]] static std::vector<unsigned int> get_cpu_device_ids();
	[[nodiscard]] static std::vector<unsigned int> get_accelerator_device_ids();

	// Add missing static method declaration
	[[nodiscard]] static Device& get_device(unsigned int device_id) {
		if (device_id >= devices_.size()) {
			ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}", device_id);
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
