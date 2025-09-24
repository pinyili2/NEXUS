#pragma once
#ifdef USE_METAL
#ifndef __METAL_VERSION__
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Header.h"
#include "Metal/Metal.hpp"
#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare metal-cpp types. This avoids having to include the full
// Metal.hpp header in every file that includes the manager.
namespace MTL {
class Device;
class Library;
class Function;
class ComputePipelineState;
} // namespace MTL
namespace MTL4 {
class CommandQueue;
} // namespace MTL4

namespace ARBD {
namespace METAL {

inline void check_metal_error(void* object, std::string_view file, int line) {
	if (object == nullptr) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Metal error at {}:{}: Object is null",
					   file,
					   line);
	}
}

#define METAL_CHECK(call)                              \
	do {                                               \
		auto result = call;                            \
		check_metal_error(result, __FILE__, __LINE__); \
	} while (0)

/**
 * @brief Modern RAII wrapper for Metal device memory
 *
 * This class provides a safe and efficient way to manage Metal device memory with RAII semantics.
 * It handles memory allocation, deallocation, and data transfer between host and device memory.
 *
 * Features:
 * - Automatic memory management (RAII)
 * - Move semantics support
 * - Safe copy operations using std::span
 * - Exception handling for Metal errors
 *
 * @tparam T The type of data to store in device memory
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a device and allocate memory for 1000 integers
 * auto device = MTLCreateSystemDefaultDevice();
 * ARBD::DeviceMemory<int> device_mem(device, 1000);
 *
 * // Copy data from host to device
 * std::vector<int> host_data(1000, 42);
 * device_mem.copyFromHost(host_data);
 *
 * // Copy data back to host
 * std::vector<int> result(1000);
 * device_mem.copyToHost(result);
 * ```
 *
 * @example Move Semantics:
 * ```cpp
 * ARBD::DeviceMemory<float> mem1(device, 1000);
 * ARBD::DeviceMemory<float> mem2 = std::move(mem1); // mem1 is now empty
 * ```
 *
 * @note The class prevents copying to avoid accidental memory leaks.
 *       Use move semantics when transferring ownership.
 */
template<typename T>
class DeviceMemory {
  private:
	MTL::Buffer* buffer_{nullptr};
	idx_t size_{0};
	MTL::Device* device_{nullptr};

  public:
	DeviceMemory() = default;

	explicit DeviceMemory(void* device, idx_t count);
	~DeviceMemory();

	// Prevent copying
	DeviceMemory(const DeviceMemory&) = delete;
	DeviceMemory& operator=(const DeviceMemory&) = delete;

	// Allow moving
	DeviceMemory(DeviceMemory&& other) noexcept;
	DeviceMemory& operator=(DeviceMemory&& other) noexcept;

	// Modern copy operations using std::span
	void copyFromHost(std::span<const T> host_data);
	void copyToHost(std::span<T> host_data) const;

	// Accessors
	[[nodiscard]] T* get() noexcept;
	[[nodiscard]] const T* get() const noexcept;
	[[nodiscard]] idx_t size() const noexcept {
		return size_;
	}
	[[nodiscard]] void* device() const noexcept {
		return device_;
	}
	[[nodiscard]] void* buffer() const noexcept {
		return buffer_;
	}

	// Conversion operators
	operator T*() noexcept {
		return get();
	}
	operator const T*() const noexcept {
		return get();
	}
};

/**
 * @brief Metal command queue wrapper with additional functionality
 *
 * This class extends MTLCommandQueue with additional convenience methods
 * and RAII semantics for better integration with the ARBD framework.
 *
 * Features:
 * - Automatic queue creation and management
 * - Synchronization utilities
 * - Exception handling integration
 * - Performance profiling support
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a queue for a specific device
 * ARBD::Queue queue(device);
 *
 * // Create command buffer and commit
 * auto cmd_buffer = queue.create_command_buffer();
 * // ... encode commands ...
 * queue.commit_and_wait(cmd_buffer);
 *
 * // Check if queue is available
 * bool available = queue.is_available();
 * ```
 *
 * @note The queue is automatically synchronized when the Queue object is destroyed
 */
class Queue {
  private:
	void* queue_{nullptr};	// id<MTLCommandQueue>
	void* device_{nullptr}; // id<MTLDevice>

  public:
	Queue() = default;
	explicit Queue(void* device);
	~Queue();

	// Prevent copying
	Queue(const Queue&) = delete;
	Queue& operator=(const Queue&) = delete;

	// Allow moving
	Queue(Queue&& other) noexcept;
	Queue& operator=(Queue&& other) noexcept;

	void synchronize();
	void* create_command_buffer();				// Returns id<MTLCommandBuffer>
	void commit_and_wait(void* command_buffer); // id<MTLCommandBuffer>

	[[nodiscard]] bool is_available() const;
	[[nodiscard]] void* get() noexcept {
		return queue_;
	}
	[[nodiscard]] const void* get() const noexcept {
		return queue_;
	}
	[[nodiscard]] void* get_device() const noexcept {
		return device_;
	}

	operator void*() noexcept {
		return queue_;
	}
	operator const void*() const noexcept {
		return queue_;
	}
};

/**
 * @brief Metal command buffer wrapper for timing and synchronization
 *
 * This class provides a convenient wrapper around MTLCommandBuffer with
 * additional timing and synchronization utilities.
 *
 * Features:
 * - Command buffer timing measurements
 * - Synchronization utilities
 * - Exception handling integration
 * - Profiling support
 *
 * @example Basic Usage:
 * ```cpp
 * // Create and use a command buffer
 * ARBD::Event event = queue.create_command_buffer();
 * // ... encode commands ...
 * event.commit();
 *
 * // Wait for completion
 * event.wait();
 *
 * // Get execution time (if available)
 * auto duration = event.get_execution_time();
 * ```
 *
 * @note Command buffers are automatically waited on when the Event object is destroyed
 */
class Event {
  private:
	void* command_buffer_{nullptr}; // id<MTLCommandBuffer>
	std::chrono::high_resolution_clock::time_point start_time_;
	std::chrono::high_resolution_clock::time_point end_time_;
	bool timing_available_{false};

  public:
	Event() = default;
	explicit Event(void* command_buffer); // id<MTLCommandBuffer>
	~Event();

	// Prevent copying
	Event(const Event&) = delete;
	Event& operator=(const Event&) = delete;

	// Allow moving
	Event(Event&& other) noexcept;
	Event& operator=(Event&& other) noexcept;

	void commit();
	void wait();

	[[nodiscard]] bool is_complete() const;
	[[nodiscard]] std::chrono::nanoseconds get_execution_time() const;

	[[nodiscard]] void* get() noexcept {
		return command_buffer_;
	}
	[[nodiscard]] const void* get() const noexcept {
		return command_buffer_;
	}

	operator void*() noexcept {
		return command_buffer_;
	}
	operator const void*() const noexcept {
		return command_buffer_;
	}
};

// Custom deleters for Metal objects
struct MTLLibraryDeleter {
	void operator()(MTL::Library* lib) const noexcept;
};

struct MTLFunctionDeleter {
	void operator()(MTL::Function* func) const noexcept;
};

struct MTLPipelineStateDeleter {
	void operator()(MTL::ComputePipelineState* pipeline) const noexcept;
};

struct BufferDeleter {
	void operator()(MTL::Buffer* buffer) const noexcept;
};

// Smart pointer aliases
using MTLLibraryPtr = std::unique_ptr<MTL::Library, MTLLibraryDeleter>;
using MTLFunctionPtr = std::unique_ptr<MTL::Function, MTLFunctionDeleter>;
using MTLPipelineStatePtr = std::unique_ptr<MTL::ComputePipelineState, MTLPipelineStateDeleter>;
using MTLBufferPtr = std::unique_ptr<MTL::Buffer, BufferDeleter>;
static std::unordered_map<void*, MTLBufferPtr> raw_buffer_map_;
static std::mutex buffer_map_mutex_;

/**
 * @brief Main Metal manager class
 *
 * This class provides a high-level interface for managing Metal devices,
 * command queues, and other resources.
 *
 * Features:
 * - Automatic device discovery and selection
 * - Device property querying
 * - Command queue management
 * - Synchronization utilities
 * - Power preference settings (low power vs. high performance)
 *
 * @example Basic Usage:
 * ```cpp
 * // Initialize the Metal manager
 * ARBD::METAL::Manager::init();
 *
 * // Get the current device
 * auto& device = ARBD::METAL::Manager::get_current_device();
 *
 * // Use the device...
 *
 * // Finalize the manager when done
 * ARBD::METAL::Manager::finalize();
 * ```
 */
class Manager {
  public:
	/**
	 * @brief Represents a single Metal device
	 */
	class Device {
	  private:
		friend class Manager;
		unsigned int id_{0};
		void* device_{nullptr};		  // This will internally hold an id<MTLDevice>
		std::array<Queue, 3> queues_; // e.g., for compute, blit, render
		idx_t next_queue_{0};

		// Device properties
		std::string name_;
		idx_t max_threads_per_group_{1};
		uint64_t recommended_max_working_set_size_{0};
		bool has_unified_memory_{false};
		bool is_low_power_{false};
		bool is_removable_{false};
		bool supports_compute_{false};
		void query_device_properties();
		bool prefer_low_power_{false};

	  public:
		MTL::Device* metal_device() const;
		explicit Device(void* device, unsigned int id); // id<MTLDevice>

		// Copy constructor and assignment operator
		Device(const Device& other);
		Device& operator=(const Device& other);

		void synchronize_all_queues() const;

		// Accessors
		[[nodiscard]] const Queue& get_queue(idx_t queue_id) const {
			if (queue_id >= queues_.size()) {
				ARBD_Exception(ExceptionType::ValueError, "Invalid queue ID: {}", queue_id);
			}
			return queues_[queue_id];
		}

		Queue& get_next_queue() {
			Queue& queue = queues_[next_queue_];
			next_queue_ = (next_queue_ + 1) % queues_.size();
			return queue;
		}

		[[nodiscard]] unsigned int id() const noexcept {
			return id_;
		}
		[[deprecated("Use id() instead")]] [[nodiscard]] unsigned int get_id() const noexcept {
			return id_;
		}
		void set_id(unsigned int new_id) noexcept {
			id_ = new_id;
		}
		[[nodiscard]] const std::string& name() const noexcept {
			return name_;
		}
		[[nodiscard]] idx_t max_threads_per_group() const noexcept {
			return max_threads_per_group_;
		}
		[[nodiscard]] uint64_t recommended_max_working_set_size() const noexcept {
			return recommended_max_working_set_size_;
		}
		[[nodiscard]] bool has_unified_memory() const noexcept {
			return has_unified_memory_;
		}
		[[nodiscard]] bool is_low_power() const noexcept {
			return is_low_power_;
		}
		[[nodiscard]] bool is_removable() const noexcept {
			return is_removable_;
		}
		[[nodiscard]] bool supports_compute() const noexcept {
			return supports_compute_;
		}
	};

	// --- Static API ---
	static void init();
	static void load_info();
	static void select_devices(std::span<const unsigned int> device_ids);
	static void use(int device_id);
	static void sync(int device_id);
	static void sync();
	static int current();
	static void prefer_low_power(bool prefer);
	static void finalize();
	static void preload_all_functions();

	template<typename T>
	static MTLBufferPtr
	create_buffer(idx_t count, MTL::ResourceOptions options = MTL::ResourceStorageModeShared) {
		auto* device = get_current_device().metal_device();
		auto* buffer = device->newBuffer(count * sizeof(T), options);

		if (!buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "Failed to allocate buffer for {} elements of type {}",
						   count,
						   typeid(T).name());
		}

		return MTLBufferPtr(buffer);
	}
	static MTLBufferPtr
	create_raw_buffer(idx_t bytes, MTL::ResourceOptions options = MTL::ResourceStorageModeShared) {
		auto* device = get_current_device().metal_device();
		auto* buffer = device->newBuffer(bytes, options);

		if (!buffer) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "Failed to allocate raw buffer of {} bytes",
						   bytes);
		}

		return MTLBufferPtr(buffer);
	}
	static void* allocate_raw(idx_t size,
							  MTL::ResourceOptions options = MTL::ResourceStorageModeShared) {
		auto buffer = create_raw_buffer(size);
		void* contents = buffer->contents();

		// Store buffer for later cleanup using simplified tracking
		std::lock_guard<std::mutex> lock(buffer_map_mutex_);
		raw_buffer_map_[contents] = std::move(buffer);

		return contents;
	}
	static void deallocate_raw(void* ptr) {
		if (!ptr)
			return;
		std::lock_guard<std::mutex> lock(buffer_map_mutex_);
		auto it = raw_buffer_map_.find(ptr);

		if (it != raw_buffer_map_.end()) {
			raw_buffer_map_.erase(it);
		} else {
			LOGWARN("Attempted to deallocate untracked Metal buffer pointer: {}", ptr);
		}
	}
	static MTL::Buffer* get_metal_buffer_from_ptr(void* ptr) {
		std::lock_guard<std::mutex> lock(buffer_map_mutex_);
		LOGDEBUG("Looking for buffer with ptr: {} in map with {} entries",
				ptr,
				raw_buffer_map_.size());
		for (const auto& entry : raw_buffer_map_) {
			LOGDEBUG("Map entry: ptr={}, buffer={}", entry.first, (void*)entry.second.get());
		}
		auto it = raw_buffer_map_.find(ptr);
		if (it != raw_buffer_map_.end()) {
			LOGDEBUG("Found Metal buffer: {} for ptr: {}", (void*)it->second.get(), ptr);
			return it->second.get();
		}
		LOGDEBUG("Metal buffer not found for ptr: {}", ptr);
		return nullptr;
	}

	[[nodiscard]] static Device& get_current_device();
	[[nodiscard]] static MTL::CommandQueue* get_current_queue();

	// --- UPDATED & NEW FUNCTIONS ---
	[[nodiscard]] static MTL::Library* get_library();
	[[nodiscard]] static MTL::ComputePipelineState*
	get_compute_pipeline_state(const std::string& kernelName);
	[[nodiscard]] static idx_t all_device_size() noexcept {
		return all_devices_.size();
	}
	[[nodiscard]] static const std::vector<Device>& all_devices() noexcept {
		return all_devices_;
	}
	[[nodiscard]] static const std::vector<Device>& devices() noexcept {
		return devices_;
	}

	static MTL::Function* get_function(const std::string& function_name);

	// Additional utility methods for completeness
	[[nodiscard]] static Device& get_device(unsigned int device_id);
	[[nodiscard]] static idx_t get_device_count() noexcept;
	[[nodiscard]] static bool has_device(unsigned int device_id);
	static void reset_device_selection();
	static void enable_profiling(bool enable = true);
	[[nodiscard]] static bool is_profiling_enabled() noexcept;

  private:
	[[nodiscard]] static std::vector<unsigned int> get_discrete_gpu_device_ids();
	[[nodiscard]] static std::vector<unsigned int> get_integrated_gpu_device_ids();
	[[nodiscard]] static std::vector<unsigned int> get_low_power_device_ids();
	static std::unordered_map<void*, MTLBufferPtr> raw_buffer_map_;
	static std::mutex buffer_map_mutex_;
	static void init_devices();
	static void discover_devices();
	static std::vector<Device> all_devices_;
	static std::vector<Device> devices_;
	static int current_device_;

	static MTL::Library* library_;
	static std::unordered_map<std::string, MTL::ComputePipelineState*>* pipeline_state_cache_;
	static std::mutex cache_mutex_;
	static bool prefer_low_power_;
	static std::unordered_map<std::string, MTL::Function*>* function_cache_;
};

} // namespace METAL
} // namespace ARBD

#endif
#endif // USE_METAL
