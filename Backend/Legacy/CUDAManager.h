#pragma once

#ifdef USE_CUDA
// Prevent conflicts between CUDA headers and C++ standard library
#ifndef __CUDA_RUNTIME_H__
#define _GLIBCXX_USE_CXX11_ABI 1
#endif

#include "ARBDException.h"
#include <algorithm>
#include <array>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

// Include CUDA headers after standard library headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/**
 * @brief Modern RAII wrapper for CUDA device memory
 */

namespace ARBD {
inline void check_cuda_error(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		ARBD_Exception(ExceptionType::CUDARuntimeError,
					   "CUDA error at %s:%d: %s",
					   file,
					   line,
					   cudaGetErrorString(error));
	}
}

#define CUDA_CHECK(call) check_cuda_error(call, __FILE__, __LINE__)

namespace CUDA {

/**
 * @brief Modern RAII wrapper for CUDA streams
 *
 * This class provides a safe, modern C++ wrapper around CUDA streams with RAII
 * semantics. It automatically manages the lifecycle of CUDA streams, ensuring
 * proper creation and cleanup.
 *
 * Features:
 * - Automatic stream creation and destruction
 * - Move semantics support
 * - Thread-safe stream synchronization
 * - Implicit conversion to cudaStream_t for CUDA API compatibility
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a default stream
 * ARBD::CUDA::Stream stream;
 *
 * // Create a stream with specific flags
 * ARBD::CUDA::Stream non_blocking_stream(cudaStreamNonBlocking);
 *
 * // Synchronize the stream
 * stream.synchronize();
 *
 * // Use with CUDA APIs
 * cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
 * ```
 *
 * @note The stream is automatically destroyed when the Stream object goes out
 * of scope
 */

class Stream {
  public:
	Stream() {
		CUDA_CHECK(cudaGetDevice(&deviceID_)); // Get current device ID
		// Create the stream with a recommended non-blocking flag for performance
		CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
	}

	explicit Stream(int deviceID, unsigned int flags = cudaStreamNonBlocking)
		: deviceID_(deviceID) {
		int originalDeviceID = -1;
		// Save the original device
		CUDA_CHECK(cudaGetDevice(&originalDeviceID));

		// Set the new device if it's different
		if (originalDeviceID != deviceID_) {
			CUDA_CHECK(cudaSetDevice(deviceID_));
		}

		// Create the stream on the now-active device
		CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));

		// Restore the original device if we changed it
		if (originalDeviceID != deviceID_) {
			CUDA_CHECK(cudaSetDevice(originalDeviceID));
		}
	}

	~Stream() {
		if (stream_) {
			// Set the correct device context before destroying the stream
			int originalDeviceID = -1;
			cudaGetDevice(&originalDeviceID);
			if (originalDeviceID != deviceID_) {
				cudaSetDevice(deviceID_);
			}

			cudaStreamDestroy(stream_);

			// Restore the original context if we changed it
			if (originalDeviceID != deviceID_) {
				cudaSetDevice(originalDeviceID);
			}
		}
	}

	// Prevent copying
	Stream(const Stream&) = delete;
	Stream& operator=(const Stream&) = delete;

	// Allow moving
	Stream(Stream&& other) noexcept
		: stream_(std::exchange(other.stream_, nullptr)),
		  deviceID_(std::exchange(other.deviceID_, -1)) {}

	Stream& operator=(Stream&& other) noexcept {
		if (this != &other) {
			if (stream_) {
				// Ensure destruction happens on the correct device
				int originalDeviceID = -1;
				cudaGetDevice(&originalDeviceID);
				if (originalDeviceID != deviceID_)
					cudaSetDevice(deviceID_);

				cudaStreamDestroy(stream_);

				if (originalDeviceID != deviceID_)
					cudaSetDevice(originalDeviceID);
			}
			stream_ = std::exchange(other.stream_, nullptr);
			deviceID_ = std::exchange(other.deviceID_, -1);
		}
		return *this;
	}

	[[nodiscard]] int deviceID() const noexcept {
		return deviceID_;
	}

	void synchronize() {
		CUDA_CHECK(cudaStreamSynchronize(stream_));
	}

	[[nodiscard]] bool query() const {
		cudaError_t result = cudaStreamQuery(stream_);
		if (result == cudaSuccess)
			return true;
		if (result == cudaErrorNotReady)
			return false;
		CUDA_CHECK(result); // Throws on other errors
		return false;
	}

	// --- Getters ---
	[[nodiscard]] cudaStream_t get() const noexcept {
		return stream_;
	}
	operator cudaStream_t() const noexcept {
		return stream_;
	}

  private:
	cudaStream_t stream_{nullptr};
	int deviceID_ = -1;
};

/**
 * @brief Modern RAII wrapper for CUDA events
 *
 * This class provides a safe, modern C++ wrapper around CUDA events with RAII
 * semantics. It manages the lifecycle of CUDA events and provides utilities for
 * timing and synchronization.
 *
 * Features:
 * - Automatic event creation and destruction
 * - Move semantics support
 * - Event recording and synchronization
 * - Timing measurements between events
 * - Implicit conversion to cudaEvent_t for CUDA API compatibility
 *
 * @example Basic Usage:
 * ```cpp
 * // Create events
 * ARBD::CUDA::Event start_event;
 * ARBD::CUDA::Event end_event;
 *
 * // Record events on a stream
 * start_event.record(stream);
 * // ... perform operations ...
 * end_event.record(stream);
 *
 * // Get elapsed time
 * float elapsed_ms = end_event.elapsed(start_event);
 *
 * // Check if event is completed
 * if (end_event.query()) {
 *     // Event is completed
 * }
 * ```
 *
 * @note Events are automatically destroyed when the Event object goes out of
 * scope
 */
class Event {
  public:
	Event() {
		CUDA_CHECK(cudaEventCreate(&event_));
	}

	explicit Event(unsigned int flags) {
		CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
	}

	~Event() {
		if (event_) {
			cudaEventDestroy(event_);
		}
	}

	// Prevent copying
	Event(const Event&) = delete;
	Event& operator=(const Event&) = delete;

	// Allow moving
	Event(Event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

	Event& operator=(Event&& other) noexcept {
		if (this != &other) {
			if (event_)
				cudaEventDestroy(event_);
			event_ = std::exchange(other.event_, nullptr);
		}
		return *this;
	}

	void record(cudaStream_t stream = nullptr) {
		CUDA_CHECK(cudaEventRecord(event_, stream));
	}

	void synchronize() {
		CUDA_CHECK(cudaEventSynchronize(event_));
	}

	[[nodiscard]] bool query() const {
		cudaError_t result = cudaEventQuery(event_);
		if (result == cudaSuccess)
			return true;
		if (result == cudaErrorNotReady)
			return false;
		CUDA_CHECK(result);
		return false;
	}

	[[nodiscard]] float elapsed(const Event& start) const {
		float time;
		CUDA_CHECK(cudaEventElapsedTime(&time, start.event_, event_));
		return time;
	}

	[[nodiscard]] cudaEvent_t get() const noexcept {
		return event_;
	}
	operator cudaEvent_t() const noexcept {
		return event_;
	}

  private:
	cudaEvent_t event_{nullptr};
};

/**
 * @brief Modern device management system
 *
 * This class provides a comprehensive device management system with support for
 * multiple devices, stream management, and device selection. It handles device
 * initialization, selection, and provides utilities for multi-device
 * operations.
 *
 * Features:
 * - Automatic stream management
 * - Device selection and synchronization
 * - Peer-to-peer memory access
 * - Safe device timeout handling
 *
 * @example Basic Usage:
 * ```cpp
 * // Initialize device system
 * ARBD::CUDA::Manager::init();
 *
 * // Use a specific device
 * ARBD::CUDA::Manager::use(0);
 *
 * // Get current device
 * auto& device = ARBD::CUDA::Manager::get_current_device();
 *
 * // Finalize when done
 * ARBD::CUDA::Manager::finalize();
 * ```
 *
 * @note The class uses static methods for global device management.
 *       All operations are thread-safe and exception-safe.
 */
class Manager {
  public:
	static constexpr size_t NUM_STREAMS = 8;

	/**
	 * @brief Individual device management class
	 *
	 * This nested class represents a single CUDA device and manages its
	 * resources, including streams and device properties.
	 *
	 * Features:
	 * - Stream management
	 * - Device property access
	 * - Timeout detection
	 * - Safe resource cleanup
	 *
	 * @example Basic Usage:
	 * ```cpp
	 * // Get device properties
	 * const auto& device = ARBD::CUDA::Manager::devices()[0];
	 * const auto& props = device.properties();
	 *
	 * // Get a stream
	 * cudaStream_t stream = device.get_stream(0);
	 *
	 * // Get next available stream
	 * cudaStream_t next_stream = device.get_next_stream();
	 * ```
	 */
	class Device {
	  public:
		explicit Device(unsigned int id);
		~Device();

		// Prevent copying, allow moving
		Device(const Device&) = delete;
		Device& operator=(const Device&) = delete;

		// Custom move constructor and assignment
		Device(Device&& other) noexcept
			: id_(other.id_), may_timeout_(other.may_timeout_), streams_(std::move(other.streams_)),
			  last_stream_(other.last_stream_), streams_created_(other.streams_created_),
			  properties_(other.properties_) {
			// Reset the moved-from object
			other.streams_created_ = false;
		}

		Device& operator=(Device&& other) noexcept {
			if (this != &other) {
				id_ = other.id_;
				may_timeout_ = other.may_timeout_;
				streams_ = std::move(other.streams_);
				last_stream_ = other.last_stream_;
				streams_created_ = other.streams_created_;
				properties_ = other.properties_;

				// Reset the moved-from object
				other.streams_created_ = false;
			}
			return *this;
		}

		/**
		 * @brief Get a specific stream by index
		 * @param stream_id Stream index (0 to NUM_STREAMS-1)
		 * @return CUDA stream handle
		 */
		[[nodiscard]] cudaStream_t get_stream(size_t stream_id) const { // Make const
			if (stream_id >= NUM_STREAMS) {
				throw std::out_of_range("Stream ID out of range");
			}
			return streams_[stream_id].get();
		}

		/**
		 * @brief Get the next stream in round-robin fashion
		 * @return CUDA stream handle
		 */
		[[nodiscard]] cudaStream_t get_next_stream() const {
			std::lock_guard<std::mutex> lock(stream_mtx_);
			last_stream_ = (last_stream_ + 1) % NUM_STREAMS;
			return streams_[last_stream_].get();
		}

		[[nodiscard]] unsigned int id() const noexcept {
			return id_;
		}
		[[nodiscard]] bool may_timeout() const noexcept {
			return may_timeout_;
		}
		[[nodiscard]] const cudaDeviceProp& properties() const noexcept {
			return properties_;
		}

		// Convenience property accessors
		[[nodiscard]] size_t total_memory() const noexcept {
			return properties_.totalGlobalMem;
		}
		[[nodiscard]] int compute_capability_major() const noexcept {
			return properties_.major;
		}
		[[nodiscard]] int compute_capability_minor() const noexcept {
			return properties_.minor;
		}
		[[nodiscard]] int multiprocessor_count() const noexcept {
			return properties_.multiProcessorCount;
		}
		[[nodiscard]] int max_threads_per_block() const noexcept {
			return properties_.maxThreadsPerBlock;
		}
		[[nodiscard]] bool supports_managed_memory() const noexcept {
			return properties_.managedMemory;
		}
		[[nodiscard]] const char* name() const noexcept {
			return properties_.name;
		}

		void synchronize_all_streams();

	  private:
		void create_streams();
		void destroy_streams();

		unsigned int id_;
		bool may_timeout_;
		std::array<Stream, NUM_STREAMS> streams_;
		mutable int last_stream_{-1};
		bool streams_created_{false};
		cudaDeviceProp properties_;
		mutable std::mutex stream_mtx_;
	};

	// Static interface
	static void init();
	static void load_info();
	static void use(int device_id);
	static void sync(int device_id);
	static void sync();
	static int current();
	static void prefer_safe_devices(bool safe = true);
	static int get_safest_device();
	static void finalize();

	// Current device access
	static Device& get_current_device();
	static Device& get_device(int device_id);

	// Peer-to-peer operations
	static void enable_peer_access();
	static bool can_access_peer(int device1, int device2);

	// Cache configuration
	static void set_cache_config(cudaFuncCache config = cudaFuncCachePreferL1);

	// Advanced stream access
	static cudaStream_t get_stream(int device_id, size_t stream_id);

	[[nodiscard]] static size_t all_device_size() noexcept {
		return all_devices_.size();
	}
	[[deprecated("Use all_device_size() instead")]] [[nodiscard]] static size_t
	allGpuSize() noexcept {
		return all_devices_.size();
	}
	[[nodiscard]] static bool safe() noexcept {
		return prefer_safe_;
	}
	[[nodiscard]] static cudaStream_t get_next_stream() {
		if (devices_.empty()) {
			ARBD_Exception(ExceptionType::ValueError, "No devices available");
		}
		return devices_[0].get_next_stream();
	}
	[[nodiscard]] static const std::vector<Device>& devices() noexcept {
		return devices_;
	}
	[[nodiscard]] static const std::vector<Device>& all_devices() noexcept {
		return all_devices_;
	}
	[[nodiscard]] static const std::vector<Device>& safe_devices() noexcept {
		return safe_devices_;
	}
	[[nodiscard]] static const std::vector<std::vector<bool>>& peer_access_matrix() noexcept {
		return peer_access_matrix_;
	}

  private:
	static void init_devices();
	static void init_impl();
	static void init_devices_impl();
	static void use_impl(int device_id);
	static void sync_impl(int device_id);
	static void finalize_impl();
	static bool can_access_peer_impl(int device1, int device2);
	static void query_peer_access_impl();
	static void query_peer_access();
	static std::vector<Device> all_devices_;
	static std::vector<Device> devices_;
	static std::vector<Device> safe_devices_;
	static std::vector<std::vector<bool>> peer_access_matrix_;
	static bool prefer_safe_;
	static int current_device_;
	static std::mutex mtx_;
};

/**
 * @brief Queue alias for Stream to maintain consistency with industry standard terminology
 *
 * This alias allows CUDA code to use the same Queue naming convention as SYCL/Metal/OpenCL
 * while maintaining the native CUDA Stream class functionality.
 *
 * @example Usage:
 * ```cpp
 * ARBD::CUDA::Queue queue;        // Industry standard naming
 * ARBD::CUDA::Stream stream;      // Native CUDA naming
 * ```
 */
using Queue = Stream;

// ============================================================================
// Low-Level CUDA Device Utilities
// ============================================================================

/**
 * @brief Modern C++20 warp-level broadcast primitive
 *
 * Broadcasts a value from one thread (leader) to all threads in the warp.
 * This function adapts to different CUDA compute capabilities automatically.
 *
 * @param value Value to broadcast (only meaningful on leader thread)
 * @param leader Lane ID of the thread to broadcast from (0-31)
 * @return The broadcast value on all threads in the warp
 *
 * @note This function must be called by all active threads in a warp
 * @note For compute capability < 7.0, requires external shared memory setup
 *
 * @example Usage in a kernel:
 * ```cpp
 * __global__ void my_kernel() {
 *     int lane_id = threadIdx.x % 32;
 *     int value = (lane_id == 0) ? 42 : 0;
 *     int broadcast_value = ARBD::CUDA::warp_broadcast(value, 0);
 *     // Now all threads in warp have broadcast_value == 42
 * }
 * ```
 */
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 700
// For very old architectures, require shared memory
extern __shared__ int warp_broadcast_shared[];
__device__ inline int warp_broadcast(int value, int leader) noexcept {
	const int tid = threadIdx.x;
	const int warp_lane = tid % 32;
	const int warp_id = tid / 32;

	if (warp_lane == leader) {
		warp_broadcast_shared[warp_id] = value;
	}
	__syncwarp();
	return warp_broadcast_shared[warp_id];
}
#else
// Volta and later: use __shfl_sync with mask
__device__ constexpr int warp_broadcast(int value, int leader) noexcept {
	return __shfl_sync(0xffffffff, value, leader);
}
#endif
#else
// Host-side stub (should not be called)
inline int warp_broadcast(int value, int leader) noexcept {
	return value; // Just return the input value
}
#endif

/**
 * @brief Warp-aggregated atomic increment for improved performance
 *
 * This function performs an atomic increment using warp-level aggregation,
 * which can significantly improve performance when many threads in a warp
 * need to increment the same counter.
 *
 * The algorithm:
 * 1. All active threads vote to participate
 * 2. One thread (leader) performs a single atomic add for the entire warp
 * 3. Each thread gets its unique position within the warp's contribution
 *
 * @param counter Pointer to the counter to increment
 * @param warp_lane Lane ID of calling thread (threadIdx.x % 32)
 * @return Unique value for this thread (like atomicAdd would return)
 *
 * @note Significantly faster than individual atomicAdd calls when most
 *       threads in a warp need to increment the same counter
 *
 * @example Usage in a kernel:
 * ```cpp
 * __global__ void count_kernel(int* global_counter, int* thread_ids, int n) {
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (tid < n) {
 *         int warp_lane = threadIdx.x % 32;
 *         int my_id = ARBD::CUDA::warp_aggregated_atomic_inc(global_counter,
 * warp_lane); thread_ids[tid] = my_id;
 *     }
 * }
 * ```
 *
 * @see
 * https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__device__ inline int warp_aggregated_atomic_inc(int* counter, int warp_lane) noexcept {
	// Get mask of active threads in warp
	unsigned int mask = __ballot_sync(0xffffffff, 1);

	// Find the leader (first active thread)
	int leader = __ffs(mask) - 1;

	// Count how many threads are participating
	int warp_count = __popc(mask);

	int result;
	if (warp_lane == leader) {
		// Leader performs the atomic operation for the entire warp
		result = atomicAdd(counter, warp_count);
	}

	// Broadcast the result to all threads in the warp
	result = warp_broadcast(result, leader);

	// Calculate this thread's unique position within the warp's contribution
	unsigned int rank_mask = mask & ((1u << warp_lane) - 1u);
	return result + __popc(rank_mask);
}
#else
// Host-side or very old GPU stub
inline int warp_aggregated_atomic_inc(int* counter, int warp_lane) noexcept {
	return (*counter)++; // Not thread-safe, but shouldn't be called anyway
}
#endif

/**
 * @brief Template version of warp_broadcast for different types
 *
 * @tparam T Type to broadcast (must be 32-bit or smaller)
 * @param value Value to broadcast
 * @param leader Lane ID to broadcast from
 * @return Broadcast value
 */
template<typename T>
	requires(sizeof(T) <= sizeof(int))
__device__ constexpr T warp_broadcast(T value, int leader) noexcept {
	if constexpr (std::is_same_v<T, int>) {
		return warp_broadcast(value, leader);
	} else {
		// Cast to int, broadcast, then cast back
		int int_value = *reinterpret_cast<const int*>(&value);
		int result = warp_broadcast(int_value, leader);
		return *reinterpret_cast<const T*>(&result);
	}
}

} // namespace CUDA
} // namespace ARBD

#define WITH_DEVICE(device_id, code)                                                          \
	do {                                                                                      \
		int _wd_current_device;                                                               \
		ARBD::CUDA::check_cuda_error(cudaGetDevice(&_wd_current_device), __FILE__, __LINE__); \
		ARBD::CUDA::check_cuda_error(cudaSetDevice(device_id), __FILE__, __LINE__);           \
		code;                                                                                 \
		ARBD::CUDA::check_cuda_error(cudaSetDevice(_wd_current_device), __FILE__, __LINE__);  \
	} while (0)

// Legacy compatibility macro (deprecated)
#define WITH_GPU(gpu_id, code) WITH_DEVICE(gpu_id, code)

/**
 * @brief Legacy error checking macro (for backward compatibility)
 * Use CUDA_CHECK in new code instead
 */
#define gpuErrchk(call) ARBD::CUDA::check_cuda_error(call, __FILE__, __LINE__)

#endif
