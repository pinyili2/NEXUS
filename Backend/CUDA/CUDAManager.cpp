#include "ARBDException.h"
#include "ARBDLogger.h"
#include <mutex>
#include <string>
#include <vector>
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
namespace ARBD {
namespace CUDA {

// Static member initialization
std::vector<Manager::Device> Manager::all_devices_;
std::vector<Manager::Device> Manager::devices_;
std::vector<Manager::Device> Manager::safe_devices_;
std::vector<std::vector<bool>> Manager::peer_access_matrix_;
std::mutex Manager::mtx_;
bool Manager::prefer_safe_{false};
int Manager::current_device_{0};

// Device class implementation
Manager::Device::Device(unsigned int id) : id_(id) {
	CUDA_CHECK(cudaSetDevice(id_));

#if CUDART_VERSION >= 12000
	CUDA_CHECK(cudaGetDeviceProperties_v2(&properties_, id_));
#else
	CUDA_CHECK(cudaGetDeviceProperties(&properties_, id_));
#endif

	may_timeout_ = properties_.kernelExecTimeoutEnabled;

	// Enhanced logging to match old naming format
	const char* timeout_str = may_timeout_ ? "(may timeout) " : "";
	LOGDEBUG("[{}] {} {}| SM {}.{} {:.2f}GHz, {:.1f}GB RAM",
			 id_,
			 properties_.name,
			 timeout_str,
			 properties_.major,
			 properties_.minor,
			 static_cast<float>(properties_.clockRate) * 1e-6f,
			 static_cast<float>(properties_.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));

	create_streams();
}

Manager::Device::~Device() {
	destroy_streams();
}

void Manager::Device::create_streams() {
	int curr;
	CUDA_CHECK(cudaGetDevice(&curr));
	CUDA_CHECK(cudaSetDevice(id_));

	if (streams_created_)
		destroy_streams();
	last_stream_ = -1;

	// Create streams using CUDA stream creation (matching old implementation)
	for (size_t i = 0; i < NUM_STREAMS; ++i) {
		// Note: Stream constructor calls cudaStreamCreate internally
		streams_[i] = Stream();
	}

	streams_created_ = true;
	CUDA_CHECK(cudaSetDevice(curr));
}

void Manager::Device::destroy_streams() {
	int curr;
	LOGTRACE("Destroying streams for device %d", id_);

	if (cudaGetDevice(&curr) == cudaSuccess) { // Avoid errors during shutdown
		CUDA_CHECK(cudaSetDevice(id_));
		streams_created_ = false;
		CUDA_CHECK(cudaSetDevice(curr));
	}
}

void Manager::Device::synchronize_all_streams() {
	int curr;
	CUDA_CHECK(cudaGetDevice(&curr));
	CUDA_CHECK(cudaSetDevice(id_));
	for (auto& stream : streams_) {
		stream.synchronize();
	}

	CUDA_CHECK(cudaSetDevice(curr));
}

// Manager static methods implementation
void Manager::init_impl() {

	int num_devices;
	CUDA_CHECK(cudaGetDeviceCount(&num_devices));
	LOGDEBUG("Found {} CUDA device(s)", num_devices);

	all_devices_.clear();
	safe_devices_.clear();

	for (int dev = 0; dev < num_devices; dev++) {
		all_devices_.emplace_back(dev);
		if (!all_devices_.back().may_timeout()) {
			safe_devices_.emplace_back(dev);
		}
	}

	if (all_devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No CUDA devices found");
	}

	// Initialize peer access matrix
	query_peer_access_impl();
}

void Manager::query_peer_access_impl() {
	size_t num_devices = all_devices_.size();
	peer_access_matrix_.resize(num_devices, std::vector<bool>(num_devices, false));

	for (size_t i = 0; i < num_devices; ++i) {
		for (size_t j = 0; j < num_devices; ++j) {
			if (i != j) {
				int can_access;
				CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access,
												   all_devices_[i].id(),
												   all_devices_[j].id()));
				peer_access_matrix_[i][j] = (can_access == 1);

				if (can_access) {
					LOGDEBUG("Peer access available: device {} -> device {}",
							 all_devices_[i].id(),
							 all_devices_[j].id());
				}
			}
		}
	}
}

void Manager::init_devices_impl() {
	LOGDEBUG("Initializing CUDA devices...");
	std::string msg;

	// Build message string like the old implementation
	for (size_t i = 0; i < devices_.size(); i++) {
		if (i != devices_.size() - 1 && devices_.size() > 1) {
			msg += std::to_string(devices_[i].id()) + ", ";
		} else if (devices_.size() > 1) {
			msg += "and " + std::to_string(devices_[i].id());
		} else {
			msg += std::to_string(devices_[i].id());
		}

		use_impl(static_cast<int>(i));
		CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	}

	LOGINFO("Initializing devices: {}", msg.c_str());
	use_impl(0);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void Manager::use_impl(int device_id) {
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No devices selected");
	}
	device_id = device_id % static_cast<int>(devices_.size());
	current_device_ = device_id;
	CUDA_CHECK(cudaSetDevice(devices_[device_id].id()));
}

void Manager::sync_impl(int device_id) {
	if (device_id >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError, "Invalid device index: {}", device_id);
	}

	int curr;
	CUDA_CHECK(cudaGetDevice(&curr));
	CUDA_CHECK(cudaSetDevice(devices_[device_id].id()));
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaSetDevice(curr));
}

bool Manager::can_access_peer_impl(int device1, int device2) {
	if (device1 >= static_cast<int>(peer_access_matrix_.size()) ||
		device2 >= static_cast<int>(peer_access_matrix_[device1].size())) {
		return false;
	}
	return peer_access_matrix_[device1][device2];
}

void Manager::finalize_impl() {
	LOGDEBUG("Finalizing CUDA manager...");

	// Synchronize all devices
	for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
		sync_impl(i);
	}

	// Clear device vectors
	devices_.clear();
	safe_devices_.clear();
	all_devices_.clear();
	peer_access_matrix_.clear();

	// Reset state
	current_device_ = 0;
	prefer_safe_ = false;

	LOGDEBUG("CUDA manager finalized");
}

void Manager::init() {
	std::lock_guard<std::mutex> lock(mtx_);
	init_impl();
}

void Manager::load_info() {
	std::lock_guard<std::mutex> lock(mtx_);
	init_impl();
	devices_.clear();
	if (prefer_safe_) {
		for (const auto& device : safe_devices_) {
			devices_.emplace_back(device.id());
		}
	} else {
		for (const auto& device : all_devices_) {
			devices_.emplace_back(device.id());
		}
	}
	if (!devices_.empty()) {
		init_devices_impl();
	} else {
		LOGWARN("No devices selected for initialization");
	}
}

void Manager::use(int device_id) {
	std::lock_guard<std::mutex> lock(mtx_);
	use_impl(device_id);
}

void Manager::sync(int device_id) {
	std::lock_guard<std::mutex> lock(mtx_);
	sync_impl(device_id);
}

void Manager::sync() {
	std::lock_guard<std::mutex> lock(mtx_);
	for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
		sync_impl(i);
	}
}

int Manager::current() {
	std::lock_guard<std::mutex> lock(mtx_);
	return current_device_;
}

Manager::Device& Manager::get_current_device() {
	std::lock_guard<std::mutex> lock(mtx_);
	if (devices_.empty()) {
		ARBD_Exception(ExceptionType::ValueError, "No devices available");
	}
	if (current_device_ >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Current device index {} out of range",
					   current_device_);
	}
	return devices_[current_device_];
}

Manager::Device& Manager::get_device(int device_id) {
	std::lock_guard<std::mutex> lock(mtx_);
	return all_devices_[device_id];
}

void Manager::prefer_safe_devices(bool safe) {
	std::lock_guard<std::mutex> lock(mtx_);
	prefer_safe_ = safe;

	if (safe && safe_devices_.empty()) {
		LOGWARN("No safe devices available (no devices without timeout enabled)");
		return;
	}

	// Update current device selection
	devices_.clear();
	if (prefer_safe_) {
		for (const auto& device : safe_devices_) {
			devices_.emplace_back(device.id());
		}
	} else {
		for (const auto& device : all_devices_) {
			devices_.emplace_back(device.id());
		}
	}
	if (!devices_.empty()) {
		init_devices_impl();
	}
}

int Manager::get_safest_device() {
	std::lock_guard<std::mutex> lock(mtx_);
	if (!safe_devices_.empty()) {
		return safe_devices_[0].id();
	}

	if (!all_devices_.empty()) {
		LOGWARN("No safe devices available, returning first available device");
		return all_devices_[0].id();
	}

	ARBD_Exception(ExceptionType::ValueError, "No devices available");
}
void Manager::finalize() {
	std::lock_guard<std::mutex> lock(mtx_);
	finalize_impl();
}
bool Manager::can_access_peer(int device1, int device2) {
	std::lock_guard<std::mutex> lock(mtx_);
	return can_access_peer_impl(device1, device2);
}
void Manager::set_cache_config(cudaFuncCache config) {
	std::lock_guard<std::mutex> lock(mtx_);
	for (size_t i = 0; i < devices_.size(); ++i) {
		use_impl(static_cast<int>(i)); // Call non-locking helper
		CUDA_CHECK(cudaDeviceSetCacheConfig(config));
	}
}
void Manager::enable_peer_access() {
	std::lock_guard<std::mutex> lock(mtx_);
	if (devices_.size() < 2) {
		LOGDEBUG("Peer access not needed with fewer than 2 devices");
		return;
	}

	LOGINFO("Enabling peer access between {} devices", devices_.size());

	for (size_t i = 0; i < devices_.size(); ++i) {
		CUDA_CHECK(cudaSetDevice(devices_[i].id()));

		for (size_t j = 0; j < devices_.size(); ++j) {
			if (i != j && can_access_peer_impl(devices_[i].id(), devices_[j].id())) {
				cudaError_t result = cudaDeviceEnablePeerAccess(devices_[j].id(), 0);
				if (result == cudaSuccess) {
					LOGDEBUG("Enabled peer access: device {} -> device {}",
							 devices_[i].id(),
							 devices_[j].id());
				} else if (result != cudaErrorPeerAccessAlreadyEnabled) {
					LOGWARN("Failed to enable peer access: device {} -> device {}: {}",
							devices_[i].id(),
							devices_[j].id(),
							cudaGetErrorString(result));
				}
			}
		}
	}
}

cudaStream_t Manager::get_stream(int device_id, size_t stream_id) {
	std::lock_guard<std::mutex> lock(mtx_);
	if (device_id >= static_cast<int>(devices_.size())) {
		ARBD_Exception(ExceptionType::ValueError, "Invalid device index: {}", device_id);
	}
	return devices_[device_id].get_stream(stream_id);
}

} // namespace CUDA
} // namespace ARBD
#endif // USE_CUDA
