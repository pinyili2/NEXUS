#ifdef USE_METAL
#include "METALManager.h"
#include "ARBDLogger.h"
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <chrono>

namespace ARBD {
namespace METAL {

// --- Static member initialization ---
std::vector<Manager::Device> Manager::all_devices_;
std::vector<Manager::Device> Manager::devices_;
int Manager::current_device_{0};
MTL::Library* Manager::library_{nullptr};
std::unordered_map<std::string, MTL::ComputePipelineState*>* Manager::pipeline_state_cache_{
	nullptr};
std::unordered_map<std::string, MTL::Function*>* Manager::function_cache_{nullptr};
std::mutex Manager::cache_mutex_;
bool Manager::prefer_low_power_{false};
std::mutex metal_buffer_map_mutex;

// Define static member variables
std::unordered_map<void*, MTLBufferPtr> ARBD::METAL::Manager::raw_buffer_map_;
std::mutex ARBD::METAL::Manager::buffer_map_mutex_;

void MTLLibraryDeleter::operator()(MTL::Library* lib) const noexcept {
	if (lib)
		lib->release();
}

void MTLFunctionDeleter::operator()(MTL::Function* func) const noexcept {
	if (func)
		func->release();
}

void MTLPipelineStateDeleter::operator()(MTL::ComputePipelineState* pipeline) const noexcept {
	if (pipeline)
		pipeline->release();
}
void BufferDeleter::operator()(MTL::Buffer* buffer) const noexcept {
	if (buffer) {
		buffer->release();
		LOGTRACE("Released MTL::Buffer");
	}
}
// Static buffer map to track MTLBuffer objects for raw pointer deallocation
static std::unordered_map<void*, MTL::Buffer*> metal_buffer_map;

template<typename T>
DeviceMemory<T>::DeviceMemory(void* device, size_t count)
	: device_(static_cast<MTL::Device*>(device)), size_(count) {
	if (count > 0 && device_) {
		// Use MTLResourceStorageModeShared for unified memory architectures
		buffer_ = device_->newBuffer(count * sizeof(T), MTL::ResourceStorageModeShared);
		if (!buffer_) {
			ARBD_Exception(ExceptionType::MetalRuntimeError,
						   "Failed to allocate {} elements of type {}",
						   count,
						   typeid(T).name());
		}
	}
}

template<typename T>
DeviceMemory<T>::~DeviceMemory() {
	if (buffer_) {
		static_cast<MTL::Buffer*>(buffer_)->release();
	}
}

template<typename T>
DeviceMemory<T>::DeviceMemory(DeviceMemory&& other) noexcept
	: buffer_(std::exchange(other.buffer_, nullptr)), size_(std::exchange(other.size_, 0)),
	  device_(std::exchange(other.device_, nullptr)) {}

template<typename T>
DeviceMemory<T>& DeviceMemory<T>::operator=(DeviceMemory&& other) noexcept {
	if (this != &other) {
		if (buffer_) {
			static_cast<MTL::Buffer*>(buffer_)->release();
		}
		buffer_ = std::exchange(other.buffer_, nullptr);
		size_ = std::exchange(other.size_, 0);
		device_ = std::exchange(other.device_, nullptr);
	}
	return *this;
}

template<typename T>
void DeviceMemory<T>::copyFromHost(std::span<const T> host_data) {
	if (host_data.size() > size_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Tried to copy {} elements but only {} allocated",
					   host_data.size(),
					   size_);
	}
	if (!buffer_ || host_data.empty())
		return;

	void* buffer_contents = static_cast<MTL::Buffer*>(buffer_)->contents();
	std::memcpy(buffer_contents, host_data.data(), host_data.size_bytes());
}

template<typename T>
void DeviceMemory<T>::copyToHost(std::span<T> host_data) const {
	if (host_data.size() > size_) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Tried to copy to {} elements but only {} allocated",
					   host_data.size(),
					   size_);
	}
	if (!buffer_ || host_data.empty())
		return;

	const void* buffer_contents = static_cast<MTL::Buffer*>(buffer_)->contents();
	std::memcpy(host_data.data(), buffer_contents, host_data.size_bytes());
}

template<typename T>
T* DeviceMemory<T>::get() noexcept {
	if (!buffer_)
		return nullptr;
	return static_cast<T*>(static_cast<MTL::Buffer*>(buffer_)->contents());
}

template<typename T>
const T* DeviceMemory<T>::get() const noexcept {
	if (!buffer_)
		return nullptr;
	return static_cast<const T*>(static_cast<MTL::Buffer*>(buffer_)->contents());
}

// Explicitly instantiate DeviceMemory for common types so the linker can find them.
template class DeviceMemory<float>;
template class DeviceMemory<double>;
template class DeviceMemory<int>;
template class DeviceMemory<unsigned int>;
template class DeviceMemory<char>;
template class DeviceMemory<unsigned char>;

// ===================================================================
// Queue Implementation
// ===================================================================

Queue::Queue(void* device) : device_(device) {
	if (device) {
		MTL::Device* pDevice = static_cast<MTL::Device*>(device);
		queue_ = pDevice->newCommandQueue();
	}
}

Queue::~Queue() {
	if (queue_) {
		static_cast<MTL::CommandQueue*>(queue_)->release();
	}
}

Queue::Queue(Queue&& other) noexcept
	: queue_(std::exchange(other.queue_, nullptr)), device_(std::exchange(other.device_, nullptr)) {
}

Queue& Queue::operator=(Queue&& other) noexcept {
	if (this != &other) {
		if (queue_) {
			static_cast<MTL::CommandQueue*>(queue_)->release();
		}
		queue_ = std::exchange(other.queue_, nullptr);
		device_ = std::exchange(other.device_, nullptr);
	}
	return *this;
}

void Queue::synchronize() {
	if (!queue_) {
		LOGWARN("Attempted to synchronize uninitialized queue");
		return;
	}
	MTL::CommandQueue* pQueue = static_cast<MTL::CommandQueue*>(queue_);
	MTL::CommandBuffer* pSyncBuffer = pQueue->commandBuffer();
	if (pSyncBuffer) {
		pSyncBuffer->commit();
		pSyncBuffer->waitUntilCompleted();
		pSyncBuffer->release();
	}
}

void* Queue::create_command_buffer() {
	if (!queue_) {
		ARBD_Exception(ExceptionType::MetalRuntimeError, "Queue not initialized");
	}
	MTL::CommandQueue* pQueue = static_cast<MTL::CommandQueue*>(queue_);
	MTL::CommandBuffer* pCmdBuffer = pQueue->commandBuffer();

	if (!pCmdBuffer) {
		ARBD_Exception(ExceptionType::MetalRuntimeError, "Failed to create command buffer");
	}

	// The caller of this function is now responsible for releasing the command buffer.
	// This is typically done by wrapping it in an Event object.
	return pCmdBuffer;
}

void Queue::commit_and_wait(void* command_buffer) {
	if (!command_buffer)
		return;
	MTL::CommandBuffer* pCmdBuffer = static_cast<MTL::CommandBuffer*>(command_buffer);
	pCmdBuffer->commit();
	pCmdBuffer->waitUntilCompleted();
	pCmdBuffer->release(); // Release after use
}

bool Queue::is_available() const {
	return queue_ != nullptr;
}

// ===================================================================
// Event Implementation
// ===================================================================

Event::Event(void* command_buffer) : command_buffer_(command_buffer) {}

Event::~Event() {
	if (command_buffer_) {
		static_cast<MTL::CommandBuffer*>(command_buffer_)->release();
	}
}

Event::Event(Event&& other) noexcept
	: command_buffer_(std::exchange(other.command_buffer_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
	if (this != &other) {
		if (command_buffer_) {
			static_cast<MTL::CommandBuffer*>(command_buffer_)->release();
		}
		command_buffer_ = std::exchange(other.command_buffer_, nullptr);
	}
	return *this;
}

void Event::commit() {
	if (!command_buffer_)
		return;
	static_cast<MTL::CommandBuffer*>(command_buffer_)->commit();
}

void Event::wait() {
	if (!command_buffer_)
		return;
	MTL::CommandBuffer* pCmdBuffer = static_cast<MTL::CommandBuffer*>(command_buffer_);
	if (pCmdBuffer->status() != MTL::CommandBufferStatusCompleted &&
		pCmdBuffer->status() != MTL::CommandBufferStatusError) {
		pCmdBuffer->waitUntilCompleted();
	}
}

bool Event::is_complete() const {
	if (!command_buffer_)
		return true;
	MTL::CommandBuffer* pCmdBuffer = static_cast<MTL::CommandBuffer*>(command_buffer_);
	auto status = pCmdBuffer->status();
	return status == MTL::CommandBufferStatusCompleted || status == MTL::CommandBufferStatusError;
}

std::chrono::nanoseconds Event::get_execution_time() const {
	if (!timing_available_) {
		return std::chrono::nanoseconds::zero();
	}
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
}

// ===================================================================
// Device Class Implementation
// ===================================================================

Manager::Device::Device(void* device, unsigned int id) : id_(id), device_(device) {
	query_device_properties();
	LOGDEBUG("Metal Device {} initialized: {}", id_, name_.c_str());

	for (size_t i = 0; i < queues_.size(); ++i) {
		queues_[i] = Queue(device_);
	}
}

Manager::Device::Device(const Device& other)
	: id_(other.id_), device_(other.device_), next_queue_(other.next_queue_), name_(other.name_),
	  max_threads_per_group_(other.max_threads_per_group_),
	  recommended_max_working_set_size_(other.recommended_max_working_set_size_),
	  has_unified_memory_(other.has_unified_memory_), is_low_power_(other.is_low_power_),
	  is_removable_(other.is_removable_), supports_compute_(other.supports_compute_),
	  prefer_low_power_(other.prefer_low_power_) {
	// Retain the Metal device
	if (device_) {
		static_cast<MTL::Device*>(device_)->retain();
	}
	// Initialize queues
	for (size_t i = 0; i < queues_.size(); ++i) {
		queues_[i] = Queue(device_);
	}
}

Manager::Device& Manager::Device::operator=(const Device& other) {
	if (this != &other) {
		// Release current device if any
		if (device_) {
			static_cast<MTL::Device*>(device_)->release();
		}

		// Copy all members
		id_ = other.id_;
		device_ = other.device_;
		next_queue_ = other.next_queue_;
		name_ = other.name_;
		max_threads_per_group_ = other.max_threads_per_group_;
		recommended_max_working_set_size_ = other.recommended_max_working_set_size_;
		has_unified_memory_ = other.has_unified_memory_;
		is_low_power_ = other.is_low_power_;
		is_removable_ = other.is_removable_;
		supports_compute_ = other.supports_compute_;
		prefer_low_power_ = other.prefer_low_power_;

		// Retain the new device
		if (device_) {
			static_cast<MTL::Device*>(device_)->retain();
		}

		// Initialize queues
		for (size_t i = 0; i < queues_.size(); ++i) {
			queues_[i] = Queue(device_);
		}
	}
	return *this;
}

void Manager::Device::synchronize_all_queues() const {
	for (const auto& queue : queues_) {
		const_cast<Queue&>(queue).synchronize();
	}
}

MTL::Device* Manager::Device::metal_device() const {
	return static_cast<MTL::Device*>(device_);
}

void Manager::Device::query_device_properties() {
	if (!device_)
		return;
	MTL::Device* pDevice = metal_device();

	// Basic device properties
	name_ = pDevice->name()->utf8String();
	max_threads_per_group_ = pDevice->maxThreadsPerThreadgroup().width;
	has_unified_memory_ = pDevice->hasUnifiedMemory();
	is_low_power_ = pDevice->isLowPower();
	is_removable_ = pDevice->isRemovable();
	supports_compute_ = true;

	// Get additional device properties
	recommended_max_working_set_size_ = pDevice->recommendedMaxWorkingSetSize();

	// Log device information
	LOGDEBUG("Metal Device {}: {} (Unified Memory: {}, Low Power: {}, Removable: {})",
			id_,
			name_,
			has_unified_memory_,
			is_low_power_,
			is_removable_);
}

// ===================================================================
// Manager Static Methods Implementation
// ===================================================================

void Manager::init() {
	LOGDEBUG("Initializing Metal Manager...");

	// Initialize caches if not already initialized
	if (!function_cache_) {
		function_cache_ = new std::unordered_map<std::string, MTL::Function*>();
	}
	if (!pipeline_state_cache_) {
		pipeline_state_cache_ = new std::unordered_map<std::string, MTL::ComputePipelineState*>();
	}

	function_cache_->clear();
	pipeline_state_cache_->clear();

	discover_devices();

	if (all_devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No Metal devices found");
	}

	// Inform about Metal's unified memory architecture
	LOGDEBUG("Metal initialized with unified memory architecture - host and device memory share the "
			"same address space");

	MTL::Device* pDefaultDevice = all_devices_[0].metal_device();
	NS::Error* pError = nullptr;

	// Try loading from file first - try standard locations
	std::vector<std::string> possible_paths = {
		"default.metallib",           // Current directory (from CMake copy)
		"./default.metallib",         // Explicit current directory
		"test_unit/default.metallib"  // Test directory (fallback)
	};
	
	LOGDEBUG("Attempting to load Metal library from {} possible paths", possible_paths.size());
	
	bool library_loaded = false;
	for (const auto& path_str : possible_paths) {
		LOGDEBUG("Trying to load Metal library from: {}", path_str);
		NS::String* path = NS::String::string(path_str.c_str(), NS::UTF8StringEncoding);
		if (auto* file_lib = pDefaultDevice->newLibrary(path, &pError)) {
			library_ = file_lib;
			LOGDEBUG("Successfully loaded Metal library from file: {}", path_str);
			library_loaded = true;
			break;
		} else {
			LOGDEBUG("Failed to load Metal library from: {} (error: {})", 
				path_str, 
				pError ? pError->localizedDescription()->utf8String() : "Not found");
		}
		// Reset error for next iteration
		if (pError) {
			pError->release();
			pError = nullptr;
		}
	}
	
	// If custom library not found, try default library
	if (!library_loaded) {
		if (auto* default_lib = pDefaultDevice->newDefaultLibrary()) {
			library_ = default_lib;
			LOGDEBUG("Loaded default Metal library");
		} else {
			LOGDEBUG("No Metal compute library found. Memory management and basic operations are "
					"available. "
					"Compile .metal shaders to enable compute kernels. ({})",
					pError ? pError->localizedDescription()->utf8String() : "No library found");
		}
	}

	if (library_) {
		preload_all_functions();
	}

	LOGDEBUG("Found {} Metal device(s). Metal Manager initialized.", all_devices_.size());
}

Manager::Device& Manager::get_current_device() {
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No Metal device is active.");
	}
	if (current_device_ < 0 || current_device_ >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Invalid current device index: {}",
					   current_device_);
	}
	return devices_[current_device_];
}

void Manager::discover_devices() {
	NS::Array* mtl_devices = MTL::CopyAllDevices();

	if (mtl_devices == nullptr || mtl_devices->count() == 0) {
		LOGWARN("No Metal devices found on this system");
		if (mtl_devices) {
			mtl_devices->release();
		}
		return;
	}

	LOGDEBUG("Discovering Metal devices...");
	all_devices_.clear();

	for (NS::UInteger i = 0; i < mtl_devices->count(); ++i) {
		// The void* stored in our Device class is the Objective-C id<MTLDevice>
		void* device_ptr = (void*)mtl_devices->object(i);
		try {
			all_devices_.emplace_back(device_ptr, static_cast<unsigned int>(i));
		} catch (const std::exception& e) {
			LOGWARN("Failed to initialize Metal device {}: {}", i, e.what());
		}
	}

	if (all_devices_.empty()) {
		LOGWARN("No valid Metal devices could be initialized");
		mtl_devices->release();
		return;
	}

	// Sort devices based on preference (e.g., prefer high-performance)
	std::stable_sort(all_devices_.begin(),
					 all_devices_.end(),
					 [](const Device& a, const Device& b) {
						 if (prefer_low_power_) {
							 return a.is_low_power() && !b.is_low_power();
						 } else {
							 return !a.is_low_power() && b.is_low_power();
						 }
					 });

	// Reassign IDs after sorting
	for (size_t i = 0; i < all_devices_.size(); ++i) {
		all_devices_[i].set_id(i);
	}

	LOGDEBUG("Discovered {} Metal device(s)", all_devices_.size());

	// Inform about Metal's single-device nature
	if (all_devices_.size() > 1) {
		LOGWARN("Multiple Metal devices detected ({} devices). This is unusual for Apple Silicon "
				"which typically has one unified GPU.",
				all_devices_.size());
	} 

	// Release the array now that we are done with it.
	mtl_devices->release();
}

void Manager::load_info() {
	init();
	// For Metal, select all available devices by default
	std::vector<unsigned int> device_ids;
	for (const auto& device : all_devices_) {
		device_ids.push_back(device.id());
	}
	select_devices(device_ids);
}

void Manager::init_devices() {
	if (devices_.empty()) {
		LOGWARN("No Metal devices selected for use.");
		return;
	}
	LOGDEBUG("Initializing {} selected Metal devices...", devices_.size());
	current_device_ = 0;
}

void Manager::finalize() {
	sync();
	if (library_) {
		library_->release();
		library_ = nullptr;
	}
	if (pipeline_state_cache_) {
		for (auto const& [key, val] : *pipeline_state_cache_) {
			val->release();
		}
		delete pipeline_state_cache_;
		pipeline_state_cache_ = nullptr;
	}
	if (function_cache_) {
		for (auto const& [key, val] : *function_cache_) {
			val->release();
		}
		delete function_cache_;
		function_cache_ = nullptr;
	}
	std::lock_guard<std::mutex> lock(buffer_map_mutex_);
	if (!raw_buffer_map_.empty()) {
		LOGWARN("Cleaning up {} leaked Metal buffers during finalization", raw_buffer_map_.size());
		raw_buffer_map_.clear();
	}
	devices_.clear();
	all_devices_.clear();
	LOGDEBUG("Metal Manager finalized.");
}

MTL::CommandQueue* Manager::get_current_queue() {
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No Metal device is active.");
	}
	return static_cast<MTL::CommandQueue*>(devices_[current_device_].get_next_queue().get());
}

MTL::Library* Manager::get_library() {
	if (!library_) {
		LOGDEBUG("Metal compute library not available. Call load_info() and compile .metal shaders "
				 "first.");
		return nullptr;
	}
	return library_;
}

MTL::ComputePipelineState* Manager::get_compute_pipeline_state(const std::string& kernelName) {
	std::lock_guard<std::mutex> lock(cache_mutex_);

	auto it = pipeline_state_cache_->find(kernelName);
	if (it != pipeline_state_cache_->end()) {
		return it->second;
	}

	MTL::Library* library = get_library();
	if (!library) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Cannot create compute pipeline state '{}': Metal library is not loaded",
					   kernelName);
	}

	NS::Error* pError = nullptr;
	MTL::Function* pFunction =
		library->newFunction(NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding));
	if (!pFunction) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Failed to find kernel function: %s",
					   kernelName.c_str());
	}

	MTL::ComputePipelineDescriptor* pDesc = MTL::ComputePipelineDescriptor::alloc()->init();
	pDesc->setComputeFunction(pFunction);

	MTL::ComputePipelineState* pPipelineState =
		get_current_device().metal_device()->newComputePipelineState(pDesc,
																	 MTL::PipelineOptionNone,
																	 nullptr,
																	 &pError);
	pDesc->release();
	if (!pPipelineState || pError) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Failed to create compute pipeline state for %s. Error: %s",
					   kernelName.c_str(),
					   pError ? pError->localizedDescription()->utf8String() : "Unknown");
	}

	pFunction->release();
	(*pipeline_state_cache_)[kernelName] =
		pPipelineState; // pPipelineState is autoreleased, but we want to hold on to it.
	pPipelineState->retain();

	return pPipelineState;
}

void Manager::sync() {
	for (auto& device : devices_) {
		device.synchronize_all_queues();
	}
}

int Manager::current() {
	return current_device_;
}

void Manager::select_devices(std::span<const unsigned int> device_ids) {
	if (all_devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError,
					   "No Metal devices discovered. Call init() first.");
	}

	devices_.clear();

	if (device_ids.empty()) {
		// If no specific devices requested, use all available devices
		devices_ = all_devices_;
	} else {
		// Select only the requested devices
		for (unsigned int id : device_ids) {
			if (id >= all_devices_.size()) {
				LOGWARN("Requested device ID {} is out of range. Skipping.", id);
				continue;
			}
			devices_.push_back(all_devices_[id]);
		}
	}

	if (devices_.empty()) {
		LOGWARN("No valid devices selected. Using default device.");
		if (!all_devices_.empty()) {
			devices_.push_back(all_devices_[0]);
		}
	}

	// Warn about multi-device usage on Metal
	if (devices_.size() > 1) {
		LOGWARN("Multiple Metal devices selected ({} devices). Metal typically supports only one "
				"unified device on Apple Silicon. Multi-device operations may cause issues.",
				devices_.size());
	}

	current_device_ = 0;
	init_devices();
	LOGDEBUG("Selected {} Metal device(s)", devices_.size());
}

void Manager::use(int device_id) {
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError,
					   "No Metal devices are active. Call select_devices() first.");
	}

	if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Device ID {} is out of range. Available devices: 0-{}",
					   device_id,
					   devices_.size() - 1);
	}

	int old_device = current_device_;
	current_device_ = device_id;

	// Warn about device switching on Metal
	if (devices_.size() > 1 && old_device != device_id) {
		LOGWARN("Switching Metal devices from {} to {}. Metal typically uses a unified device "
				"architecture.",
				old_device,
				device_id);
	}

	LOGDEBUG("Switched from Metal device {} to {}: {}",
			old_device,
			device_id,
			devices_[device_id].name());
}

void Manager::sync(int device_id) {
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No Metal devices are active.");
	}

	if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Device ID {} is out of range. Available devices: 0-{}",
					   device_id,
					   devices_.size() - 1);
	}

	devices_[device_id].synchronize_all_queues();
}

void Manager::prefer_low_power(bool prefer) {
	prefer_low_power_ = prefer;

	// Re-sort devices based on new preference
	if (!all_devices_.empty()) {
		std::sort(all_devices_.begin(), all_devices_.end(), [](const Device& a, const Device& b) {
			if (prefer_low_power_) {
				if (a.is_low_power() != b.is_low_power()) {
					return a.is_low_power();
				}
			} else {
				if (a.is_low_power() != b.is_low_power()) {
					return !a.is_low_power();
				}
			}
			return a.id() < b.id();
		});

		// Reassign IDs after sorting
		for (size_t i = 0; i < all_devices_.size(); ++i) {
			all_devices_[i].set_id(static_cast<unsigned int>(i));
		}
	}
}

std::vector<unsigned int> Manager::get_discrete_gpu_device_ids() {
	std::vector<unsigned int> discrete_ids;
	for (const auto& device : all_devices_) {
		if (!device.is_low_power() && !device.has_unified_memory()) {
			discrete_ids.push_back(device.id());
		}
	}
	return discrete_ids;
}

std::vector<unsigned int> Manager::get_integrated_gpu_device_ids() {
	std::vector<unsigned int> integrated_ids;
	for (const auto& device : all_devices_) {
		if (device.has_unified_memory()) {
			integrated_ids.push_back(device.id());
		}
	}
	return integrated_ids;
}

std::vector<unsigned int> Manager::get_low_power_device_ids() {
	std::vector<unsigned int> low_power_ids;
	for (const auto& device : all_devices_) {
		if (device.is_low_power()) {
			low_power_ids.push_back(device.id());
		}
	}
	return low_power_ids;
}


void Manager::preload_all_functions() {
	if (!library_) {
		LOGWARN("Cannot preload functions: Metal library not available");
		return;
	}

	LOGDEBUG("Preloading Metal compute functions...");

	// Get all function names from the library
	NS::Array* function_names = library_->functionNames();
	if (!function_names) {
		LOGWARN("No functions found in Metal library");
		return;
	}

	std::lock_guard<std::mutex> lock(cache_mutex_);

	for (NS::UInteger i = 0; i < function_names->count(); ++i) {
		NS::String* name = static_cast<NS::String*>(function_names->object(i));
		if (name) {
			std::string func_name = name->utf8String();
			LOGDEBUG("Found function: {}", func_name);
			MTL::Function* function = library_->newFunction(name);
			if (function) {
				(*function_cache_)[func_name] = function;
			}
		}
	}

	LOGDEBUG("Preloaded {} Metal functions", function_cache_->size());
}

MTL::Function* Manager::get_function(const std::string& function_name) {
	if (!function_cache_) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Function cache not initialized. Call init() first.");
	}

	std::lock_guard<std::mutex> lock(cache_mutex_);

	auto it = function_cache_->find(function_name);
	if (it != function_cache_->end()) {
		return it->second;
	}

	// If not in cache, try to load it from library
	if (!library_) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Cannot load function '{}': Metal library not available",
					   function_name);
	}

	MTL::Function* function =
		library_->newFunction(NS::String::string(function_name.c_str(), NS::UTF8StringEncoding));

	if (!function) {
		ARBD_Exception(ExceptionType::MetalRuntimeError,
					   "Function '{}' not found in Metal library",
					   function_name);
	}

	(*function_cache_)[function_name] = function;
	return function;
}

// ===================================================================
// Additional Utility Methods Implementation
// ===================================================================

Manager::Device& Manager::get_device(unsigned int device_id) {
	if (device_id >= devices_.size()) {
		ARBD_Exception(ExceptionType::ValueError, "Invalid device ID: {}", device_id);
	}
	return devices_[device_id];
}

size_t Manager::get_device_count() noexcept {
	return devices_.size();
}

bool Manager::has_device(unsigned int device_id) {
	return device_id < devices_.size();
}

void Manager::reset_device_selection() {
	if (!all_devices_.empty()) {
		devices_ = all_devices_;
		current_device_ = 0;
		LOGDEBUG("Reset device selection to all available devices");
	}
}

void Manager::enable_profiling(bool enable) {
	// Metal profiling is handled at the command buffer level
	// This is a placeholder for future profiling implementation
	LOGDEBUG("Metal profiling {} (implementation pending)", enable ? "enabled" : "disabled");
}

bool Manager::is_profiling_enabled() noexcept {
	// Placeholder implementation
	return false;
}

} // namespace METAL
} // namespace ARBD
#endif // USE_METAL