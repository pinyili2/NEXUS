#pragma once
#ifndef __METAL_VERSION__
#include "ARBDLogger.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Backend-specific profiling headers
#ifdef USE_CUDA
#include <cuda_runtime.h>
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif
#include <cuda_profiler_api.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Metal/Metal.hpp"
#include <QuartzCore/QuartzCore.h>
#endif

namespace ARBD {
namespace Profiling {

// ============================================================================
// Profiling Configuration
// ============================================================================

struct ProfilingConfig {
	bool enable_timing = true;
	bool enable_memory_tracking = true;
	bool enable_kernel_profiling = true;
	bool enable_backend_markers = true;
	std::string output_file = "arbd_profile.json";
	size_t max_events = 10000;
};

// ============================================================================
// Profiling Event Data
// ============================================================================

struct ProfileEvent {
	std::string name;
	std::string category;
	ResourceType backend;
	std::chrono::high_resolution_clock::time_point start_time;
	std::chrono::high_resolution_clock::time_point end_time;
	size_t memory_used = 0;
	size_t device_id = 0;
	std::unordered_map<std::string, std::string> metadata;

	double duration_ms() const {
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		return static_cast<double>(duration.count()) / 1000.0;
	}
};

// Generic Timer class for both CUDA and SYCL
class Timer {
  private:
	using clock_t = std::chrono::high_resolution_clock;
	clock_t::time_point start_time_;
	std::string name_;

#ifdef USE_CUDA
	cudaEvent_t cuda_start_ = nullptr;
	cudaEvent_t cuda_stop_ = nullptr;
	cudaStream_t cuda_queue_ = nullptr;
#endif

#ifdef USE_SYCL
	sycl::event sycl_event_;
#endif

  public:
	Timer(const std::string& name = "", void* queue = nullptr) : name_(name) {

#ifdef USE_CUDA
		if (queue) {
			cuda_queue_ = static_cast<cudaStream_t>(queue);
			cudaEventCreate(&cuda_start_);
			cudaEventCreate(&cuda_stop_);
			cudaEventRecord(cuda_start_, cuda_queue_);
		} else
#endif
		{
			start_time_ = clock_t::now();
		}

		if (!name_.empty()) {
			LOGINFO("{} ... ", name_);
		}
	}

	~Timer() {
#ifdef USE_CUDA
		if (cuda_start_)
			cudaEventDestroy(cuda_start_);
		if (cuda_stop_)
			cudaEventDestroy(cuda_stop_);
#endif
	}

	void start() {
#ifdef USE_CUDA
		if (cuda_queue_) {
			cudaEventRecord(cuda_start_, cuda_queue_);
		} else
#endif
		{
			start_time_ = clock_t::now();
		}
	}

	float elapsed() {
#ifdef USE_CUDA
		if (cuda_queue_) {
			cudaEventRecord(cuda_stop_, cuda_queue_);
			cudaEventSynchronize(cuda_stop_);
			float msec = 0.0f;
			cudaEventElapsedTime(&msec, cuda_start_, cuda_stop_);
			return msec;
		}
#endif

		auto end_time = clock_t::now();
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
		return duration.count() / 1000.0f; // Return milliseconds
	}

	void stop() {
		float ms = elapsed();
		if (!name_.empty()) {
			LOGINFO("{} completed in {:.3f} ms", name_, ms);
		}
	}

	void restart(const std::string& msg) {
		stop();
		name_ = msg;
		start();
		if (!name_.empty()) {
			LOGINFO("{} ... ", name_);
		}
	}
};

// Convenience macros

// ============================================================================
// CUDA Profiling Implementation
// ============================================================================

#ifdef USE_CUDA
class CUDAProfiler {
  public:
	struct CUDAEvent {
		cudaEvent_t start_event;
		cudaEvent_t end_event;
		std::string name;
		bool is_active = false;

		CUDAEvent(const std::string& event_name) : name(event_name) {
			cudaEventCreate(&start_event);
			cudaEventCreate(&end_event);
		}

		~CUDAEvent() {
			if (start_event)
				cudaEventDestroy(start_event);
			if (end_event)
				cudaEventDestroy(end_event);
		}

		void start() {
			cudaEventRecord(start_event);
			is_active = true;
		}

		void end() {
			if (is_active) {
				cudaEventRecord(end_event);
				is_active = false;
			}
		}

		float elapsed_time_ms() const {
			if (is_active)
				return 0.0f;
			float time_ms;
			cudaEventElapsedTime(&time_ms, start_event, end_event);
			return time_ms;
		}
	};

	static void init() {
		LOGINFO("Initializing CUDA profiler");
#ifdef USE_NVTX
		nvtxInitialize(nullptr);
#endif
	}

	static void finalize() {
		LOGINFO("Finalizing CUDA profiler");
		// Cleanup any remaining events
		active_events_.clear();
	}

	static void start_range(const std::string& name, uint32_t color = 0xFF00FF00) {
#ifdef USE_NVTX
		nvtxEventAttributes_t eventAttrib = {0};
		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = color;
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = name.c_str();
		nvtxRangePushEx(&eventAttrib);
#endif
	}

	static void end_range() {
#ifdef USE_NVTX
		nvtxRangePop();
#endif
	}

	static void mark(const std::string& message) {
#ifdef USE_NVTX
		nvtxMarkA(message.c_str());
#endif
	}

	static std::shared_ptr<CUDAEvent> create_event(const std::string& name) {
		auto event = std::make_shared<CUDAEvent>(name);
		active_events_[name] = event;
		return event;
	}

	static void remove_event(const std::string& name) {
		active_events_.erase(name);
	}

	static void profile_memory_usage() {
		size_t free_bytes, total_bytes;
		cudaMemGetInfo(&free_bytes, &total_bytes);
		size_t used_bytes = total_bytes - free_bytes;

		LOGINFO("CUDA Memory: {} MB used / {} MB total",
				used_bytes / (1024 * 1024),
				total_bytes / (1024 * 1024));
	}

	// RAII wrapper for CUDA profiling ranges
	class ScopedRange {
	  public:
		ScopedRange(const std::string& name, uint32_t color = 0xFF00FF00) {
			start_range(name, color);
		}
		~ScopedRange() {
			end_range();
		}
	};

  private:
	static inline std::unordered_map<std::string, std::shared_ptr<CUDAEvent>> active_events_;
};

// Convenience macros for CUDA profiling
#define CUDA_PROFILE_RANGE(name) ARBD::Profiling::CUDAProfiler::ScopedRange _prof_range(name)
#define CUDA_PROFILE_RANGE_COLOR(name, color) \
	ARBD::Profiling::CUDAProfiler::ScopedRange _prof_range(name, color)
#define CUDA_PROFILE_MARK(msg) ARBD::Profiling::CUDAProfiler::mark(msg)

#else
// Dummy implementations when CUDA is not available
class CUDAProfiler {
  public:
	static void init() {}
	static void finalize() {}
	static void start_range(const std::string&, uint32_t = 0) {}
	static void end_range() {}
	static void mark(const std::string&) {}
	static void profile_memory_usage() {}
	class ScopedRange {
	  public:
		ScopedRange(const std::string&, uint32_t = 0) {}
	};
};
#define CUDA_PROFILE_RANGE(name)
#define CUDA_PROFILE_RANGE_COLOR(name, color)
#define CUDA_PROFILE_MARK(msg)
#endif

// ============================================================================
// SYCL Profiling Implementation
// ============================================================================

#ifdef USE_SYCL
class SYCLProfiler {
  public:
	struct SYCLEvent {
		sycl::event event;
		std::string name;
		std::chrono::high_resolution_clock::time_point host_start;
		std::chrono::high_resolution_clock::time_point host_end;
		bool host_timing_valid = false;

		SYCLEvent(const std::string& event_name) : name(event_name) {}

		void start_host_timing() {
			host_start = std::chrono::high_resolution_clock::now();
		}

		void end_host_timing() {
			host_end = std::chrono::high_resolution_clock::now();
			host_timing_valid = true;
		}

		double elapsed_time_ms() const {
			if (host_timing_valid) {
				auto duration =
					std::chrono::duration_cast<std::chrono::microseconds>(host_end - host_start);
				return static_cast<double>(duration.count()) / 1000.0;
			}

			// Try to get device timing if available
			try {
				if (event.get_info<sycl::info::event::command_execution_status>() ==
					sycl::info::event_command_status::complete) {

					auto start_time =
						event.get_profiling_info<sycl::info::event_profiling::command_start>();
					auto end_time =
						event.get_profiling_info<sycl::info::event_profiling::command_end>();

					return static_cast<double>(end_time - start_time) / 1000000.0; // ns to ms
				}
			} catch (const sycl::exception& e) {
				LOGWARN("Failed to get SYCL profiling info: {}", e.what());
			}
			return 0.0;
		}
	};

	static void init() {
		LOGINFO("Initializing SYCL profiler");
		// Enable profiling on queues
		profiling_enabled_ = true;
	}

	static void finalize() {
		LOGINFO("Finalizing SYCL profiler");
		active_events_.clear();
		profiling_enabled_ = false;
	}

	static void start_range(const std::string& name) {
		if (!profiling_enabled_)
			return;

		auto event_obj = std::make_shared<SYCLEvent>(name);
		event_obj->start_host_timing();
		active_events_[name] = event_obj;

		LOGTRACE("SYCL: Started profiling range '{}'", name);
	}

	static void end_range(const std::string& name) {
		if (!profiling_enabled_)
			return;

		auto it = active_events_.find(name);
		if (it != active_events_.end()) {
			it->second->end_host_timing();
			double elapsed = it->second->elapsed_time_ms();

			LOGINFO("SYCL Range '{}': {:.3f} ms", name, elapsed);
			active_events_.erase(it);
		}
	}

	static void mark(const std::string& message) {
		if (profiling_enabled_) {
			LOGINFO("SYCL Mark: {}", message);
		}
	}

	static std::shared_ptr<SYCLEvent> create_event(const std::string& name) {
		auto event = std::make_shared<SYCLEvent>(name);
		active_events_[name] = event;
		return event;
	}

	static void profile_memory_usage(sycl::queue& queue) {
		try {
			auto device = queue.get_device();

			if (device.has(sycl::aspect::usm_device_allocations)) {
				// Get device memory info if available
				auto global_mem = device.get_info<sycl::info::device::global_mem_size>();
				LOGINFO("SYCL Device Memory: {} MB total", global_mem / (1024 * 1024));
			}
		} catch (const sycl::exception& e) {
			LOGWARN("Failed to get SYCL memory info: {}", e.what());
		}
	}

	// RAII wrapper for SYCL profiling ranges
	class ScopedRange {
	  private:
		std::string name_;

	  public:
		ScopedRange(const std::string& name) : name_(name) {
			start_range(name_);
		}
		~ScopedRange() {
			end_range(name_);
		}
	};

  private:
	static inline std::unordered_map<std::string, std::shared_ptr<SYCLEvent>> active_events_;
	static inline bool profiling_enabled_ = false;
};

// Convenience macros for SYCL profiling
#define SYCL_PROFILE_RANGE(name) ARBD::Profiling::SYCLProfiler::ScopedRange _prof_range(name)
#define SYCL_PROFILE_MARK(msg) ARBD::Profiling::SYCLProfiler::mark(msg)

#else
// Dummy implementations when SYCL is not available
class SYCLProfiler {
  public:
	static void init() {}
	static void finalize() {}
	static void start_range(const std::string&) {}
	static void end_range(const std::string&) {}
	static void mark(const std::string&) {}
	static void profile_memory_usage(void*) {}
	class ScopedRange {
	  public:
		ScopedRange(const std::string&) {}
	};
};
#define SYCL_PROFILE_RANGE(name)
#define SYCL_PROFILE_MARK(msg)
#endif

// ============================================================================
// Metal Profiling Implementation
// ============================================================================

#ifdef USE_METAL
class METALProfiler {
  public:
	struct METALEvent {
		std::string name;
		std::chrono::high_resolution_clock::time_point host_start;
		std::chrono::high_resolution_clock::time_point host_end;
		bool host_timing_valid = false;
		void* command_buffer = nullptr; // MTL::CommandBuffer*

		METALEvent(const std::string& event_name) : name(event_name) {}

		void start_host_timing() {
			host_start = std::chrono::high_resolution_clock::now();
		}

		void end_host_timing() {
			host_end = std::chrono::high_resolution_clock::now();
			host_timing_valid = true;
		}

		double elapsed_time_ms() const {
			if (host_timing_valid) {
				auto duration =
					std::chrono::duration_cast<std::chrono::microseconds>(host_end - host_start);
				return static_cast<double>(duration.count()) / 1000.0;
			}

			// Try to get GPU timing if available
			if (command_buffer) {
				MTL::CommandBuffer* cmd_buf = static_cast<MTL::CommandBuffer*>(command_buffer);

				if (cmd_buf->status() == MTL::CommandBufferStatusCompleted) {
					double gpu_start = cmd_buf->GPUStartTime();
					double gpu_end = cmd_buf->GPUEndTime();
					return (gpu_end - gpu_start) * 1000.0; // Convert to ms
				}
			}

			return 0.0;
		}
	};

	static void init() {
		LOGINFO("Initializing Metal profiler");
		profiling_enabled_ = true;

// Enable GPU timeline capture if available
#if TARGET_OS_MAC
		// Check for macOS 10.15+ availability using compile-time version check
		if (__MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_10_15) {
			// Metal Performance Shaders Graph profiling setup could go here
		}
#endif
	}

	static void finalize() {
		LOGINFO("Finalizing Metal profiler");
		active_events_.clear();
		profiling_enabled_ = false;
	}

	static void start_range(const std::string& name) {
		if (!profiling_enabled_)
			return;

		auto event_obj = std::make_shared<METALEvent>(name);
		event_obj->start_host_timing();
		active_events_[name] = event_obj;

		LOGTRACE("Metal: Started profiling range '{}'", name);
	}

	static void end_range(const std::string& name) {
		if (!profiling_enabled_)
			return;

		auto it = active_events_.find(name);
		if (it != active_events_.end()) {
			it->second->end_host_timing();
			double elapsed = it->second->elapsed_time_ms();

			LOGINFO("Metal Range '{}': {:.3f} ms", name, elapsed);
			active_events_.erase(it);
		}
	}

	static void mark(const std::string& message) {
		if (profiling_enabled_) {
			LOGINFO("Metal Mark: {}", message);
		}
	}

	static std::shared_ptr<METALEvent> create_event(const std::string& name,
													void* cmd_buffer = nullptr) {
		auto event = std::make_shared<METALEvent>(name);
		event->command_buffer = cmd_buffer;
		active_events_[name] = event;
		return event;
	}

	static void profile_memory_usage(void* device_ptr) {
		if (!device_ptr)
			return;

		MTL::Device* device = static_cast<MTL::Device*>(device_ptr);

		// Get device memory information
		uint64_t recommended_working_set = device->recommendedMaxWorkingSetSize();
		bool has_unified_memory = device->hasUnifiedMemory();

		LOGINFO("Metal Device Memory: Recommended working set {} MB, Unified Memory: {}",
				recommended_working_set / (1024 * 1024),
				has_unified_memory);
	}

	// RAII wrapper for Metal profiling ranges
	class ScopedRange {
	  private:
		std::string name_;

	  public:
		ScopedRange(const std::string& name) : name_(name) {
			start_range(name_);
		}
		~ScopedRange() {
			end_range(name_);
		}
	};

  private:
	static inline std::unordered_map<std::string, std::shared_ptr<METALEvent>> active_events_;
	static inline bool profiling_enabled_ = false;
};

// Convenience macros for Metal profiling
#define METAL_PROFILE_RANGE(name) ARBD::Profiling::METALProfiler::ScopedRange _prof_range(name)
#define METAL_PROFILE_MARK(msg) ARBD::Profiling::METALProfiler::mark(msg)

#else
// Dummy implementations when Metal is not available
class METALProfiler {
  public:
	static void init() {}
	static void finalize() {}
	static void start_range(const std::string&) {}
	static void end_range(const std::string&) {}
	static void mark(const std::string&) {}
	static void profile_memory_usage(void*) {}
	class ScopedRange {
	  public:
		ScopedRange(const std::string&) {}
	};
};
#define METAL_PROFILE_RANGE(name)
#define METAL_PROFILE_MARK(msg)
#endif

/**
 * @class ProfileManager
 * @brief Unified interface for profiling across all supported backends (CUDA, SYCL, Metal).
 *
 * The ProfileManager class provides a backend-agnostic API for initializing, finalizing,
 * and managing profiling events, memory usage, and markers. It coordinates profiling
 * across the enabled backend (CUDA, SYCL, or Metal), and collects timing and memory
 * usage data for performance analysis.
 *
 * Usage:
 *   - Call ProfileManager::init() at the start of profiling.
 *   - Use PROFILE_RANGE, PROFILE_MARK, and PROFILE_MEMORY macros to annotate code regions.
 *   - Call ProfileManager::finalize() to end profiling and optionally save results.
 *
 * Features:
 *   - Supports enabling/disabling timing, memory tracking, and kernel profiling.
 *   - Aggregates profiling events and outputs summary statistics.
 *   - Provides RAII-based scoped profiling ranges for easy instrumentation.
 *   - Backend-specific profiling is handled transparently based on compile-time flags.
 *
 * Note:
 *   - Only one backend should be enabled at a time (CUDA, SYCL, or Metal).
 *   - Profiling configuration is controlled via the ProfilingConfig struct.
 */

class ProfileManager {
  public:
	static void init(const ProfilingConfig& config = {}) {
		config_ = config;

		if (!config_.enable_backend_markers)
			return;

		LOGINFO("Initializing profiling for all available backends");

#ifdef USE_CUDA
		CUDAProfiler::init();
#endif

#ifdef USE_SYCL
		SYCLProfiler::init();
#endif

#ifdef USE_METAL
		METALProfiler::init();
#endif

		initialized_ = true;
	}

	static void finalize() {
		if (!initialized_)
			return;

		LOGDEBUG("Finalizing profiling for all backends");

#ifdef USE_CUDA
		CUDAProfiler::finalize();
#endif

#ifdef USE_SYCL
		SYCLProfiler::finalize();
#endif

#ifdef USE_METAL
		METALProfiler::finalize();
#endif

		// Save profile data if requested
		if (!config_.output_file.empty()) {
			save_profile_data();
		}

		events_.clear();
		initialized_ = false;
	}

	static void start_range(const std::string& name, ResourceType backend) {
		if (!initialized_ || !config_.enable_backend_markers)
			return;

		auto event = std::make_shared<ProfileEvent>();
		event->name = name;
		event->backend = backend;
		event->start_time = std::chrono::high_resolution_clock::now();

		// Backend-specific profiling
		switch (backend) {
		case ResourceType::CUDA:
#ifdef USE_CUDA
			CUDAProfiler::start_range(name);
#endif
			break;
		case ResourceType::SYCL:
#ifdef USE_SYCL
			SYCLProfiler::start_range(name);
#endif
			break;
		case ResourceType::METAL:
#ifdef USE_METAL
			METALProfiler::start_range(name);
#endif
			break;
		default:
			break;
		}

		active_ranges_[name] = event;
	}

	static void end_range(const std::string& name, ResourceType backend) {
		if (!initialized_ || !config_.enable_backend_markers)
			return;

		auto it = active_ranges_.find(name);
		if (it != active_ranges_.end()) {
			it->second->end_time = std::chrono::high_resolution_clock::now();

			// Add to completed events
			events_.push_back(it->second);
			active_ranges_.erase(it);

			// Limit event storage
			if (events_.size() > config_.max_events) {
				events_.erase(events_.begin(),
							  events_.begin() + (events_.size() - config_.max_events));
			}
		}

		// Backend-specific profiling
		switch (backend) {
		case ResourceType::CUDA:
#ifdef USE_CUDA
			CUDAProfiler::end_range();
#endif
			break;
		case ResourceType::SYCL:
#ifdef USE_SYCL
			SYCLProfiler::end_range(name);
#endif
			break;
		case ResourceType::METAL:
#ifdef USE_METAL
			METALProfiler::end_range(name);
#endif
			break;
		default:
			break;
		}
	}

	static void mark(const std::string& message, ResourceType backend) {
		if (!initialized_ || !config_.enable_backend_markers)
			return;

		switch (backend) {
		case ResourceType::CUDA:
#ifdef USE_CUDA
			CUDAProfiler::mark(message);
#endif
			break;
		case ResourceType::SYCL:
#ifdef USE_SYCL
			SYCLProfiler::mark(message);
#endif
			break;
		case ResourceType::METAL:
#ifdef USE_METAL
			METALProfiler::mark(message);
#endif
			break;
		default:
			LOGINFO("Profile Mark [CPU]: {}", message);
			break;
		}
	}

	static void profile_memory_usage(ResourceType backend, void* backend_specific_ptr = nullptr) {
		if (!initialized_ || !config_.enable_memory_tracking)
			return;

		switch (backend) {
		case ResourceType::CUDA:
#ifdef USE_CUDA
			CUDAProfiler::profile_memory_usage();
#endif
			break;
		case ResourceType::SYCL:
#ifdef USE_SYCL
			if (backend_specific_ptr) {
				SYCLProfiler::profile_memory_usage(
					*static_cast<sycl::queue*>(backend_specific_ptr));
			}
#endif
			break;
		case ResourceType::METAL:
#ifdef USE_METAL
			METALProfiler::profile_memory_usage(backend_specific_ptr);
#endif
			break;
		default:
			break;
		}
	}
	class ScopedTimer {
		Timer timer_;

	  public:
		ScopedTimer(const std::string& name, void* queue = nullptr) : timer_(name, queue) {}
		~ScopedTimer() {
			timer_.stop();
		}
	};

	// RAII wrapper for unified profiling
	class ScopedRange {
	  private:
		std::string name_;
		ResourceType backend_;

	  public:
		ScopedRange(const std::string& name, ResourceType backend)
			: name_(name), backend_(backend) {
			start_range(name_, backend_);
		}
		~ScopedRange() {
			end_range(name_, backend_);
		}
	};

	static const std::vector<std::shared_ptr<ProfileEvent>>& get_events() {
		return events_;
	}

	static void print_summary() {
		if (events_.empty()) {
			LOGINFO("No profiling events recorded");
			return;
		}

		LOGINFO("=== Profiling Summary ===");
		LOGINFO("Total events: {}", events_.size());

		// Group by backend
		std::unordered_map<ResourceType, std::vector<double>> backend_times;

		for (const auto& event : events_) {
			backend_times[event->backend].push_back(event->duration_ms());
		}

		for (const auto& [backend, times] : backend_times) {
			double total_time = 0.0;
			double min_time = *std::min_element(times.begin(), times.end());
			double max_time = *std::max_element(times.begin(), times.end());

			for (double time : times) {
				total_time += time;
			}

			std::string backend_name;
			switch (backend) {
			case ResourceType::CUDA:
				backend_name = "CUDA";
				break;
			case ResourceType::SYCL:
				backend_name = "SYCL";
				break;
			case ResourceType::METAL:
				backend_name = "Metal";
				break;
			default:
				backend_name = "CPU";
				break;
			}

			LOGINFO("{}: {} events, {:.3f} ms total, {:.3f} ms avg, {:.3f}/{:.3f} ms min/max",
					backend_name,
					times.size(),
					total_time,
					total_time / times.size(),
					min_time,
					max_time);
		}
	}

  private:
	static void save_profile_data() {
		// Save profiling data to JSON file (simplified implementation)
		LOGINFO("Saving profile data to {}", config_.output_file);
		// Implementation would write JSON format profile data
	}

	static inline ProfilingConfig config_;
	static inline bool initialized_ = false;
	static inline std::vector<std::shared_ptr<ProfileEvent>> events_;
	static inline std::unordered_map<std::string, std::shared_ptr<ProfileEvent>> active_ranges_;
};

// ============================================================================
// Convenience Macros for Unified Profiling
// ============================================================================

#define PROFILE_RANGE(name, backend) \
	ARBD::Profiling::ProfileManager::ScopedRange _prof_range(name, backend)
#define PROFILE_MARK(msg, backend) ARBD::Profiling::ProfileManager::mark(msg, backend)
#define PROFILE_MEMORY(backend, ptr) \
	ARBD::Profiling::ProfileManager::profile_memory_usage(backend, ptr)

// Backend-specific convenience macros
#define PROFILE_CUDA_RANGE(name) PROFILE_RANGE(name, ARBD::ResourceType::CUDA)
#define PROFILE_SYCL_RANGE(name) PROFILE_RANGE(name, ARBD::ResourceType::SYCL)
#define PROFILE_METAL_RANGE(name) PROFILE_RANGE(name, ARBD::ResourceType::METAL)
#define PROFILE_TIMER(name) ARBD::Profiling::ProfileManager::ScopedTimer _timer(name)
#define PROFILE_TIMER_QUEUE(name, queue) \
	ARBD::Profiling::ProfileManager::ScopedTimer _timer(name, queue)

} // namespace Profiling
} // namespace ARBD
#endif