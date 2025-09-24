#include "Backend/CUDA/NCCLManager.h"

#if defined(USE_CUDA) && defined(USE_NCCL)
#include <algorithm>
#include <vector>

namespace ARBD {
namespace CUDA {

// Static member initialization
std::vector<ncclComm_t> NCCLManager::comms_;
std::vector<int> NCCLManager::gpu_ids_;
bool NCCLManager::initialized_{false};

void NCCLManager::init(const int* gpu_ids, size_t count) {
	if (initialized_) {
		LOGWARN("NCCL already initialized, finalizing first");
		finalize();
	}

	if (count == 0) {
		ARBD_Exception(ExceptionType::ValueError, "No GPU IDs provided for NCCL initialization");
	}

	if (count == 1) {
		LOGINFO("Only one GPU provided, NCCL not needed");
		return;
	}

	LOGINFO("Initializing NCCL for {} GPUs", count);

	// Copy GPU IDs
	gpu_ids_.assign(gpu_ids, gpu_ids + count);

	// Verify all GPUs are available
	int device_count;
	NCCL_CHECK(cudaGetDeviceCount(&device_count));
	for (int gpu_id : gpu_ids_) {
		if (gpu_id >= device_count) {
			ARBD_Exception(ExceptionType::ValueError,
						   "GPU ID {} not available (only {} GPUs found)",
						   gpu_id,
						   device_count);
		}
	}

	// Initialize NCCL communicators
	comms_.resize(gpu_ids_.size());

	try {
		NCCL_CHECK(
			ncclCommInitAll(comms_.data(), static_cast<int>(gpu_ids_.size()), gpu_ids_.data()));

		initialized_ = true;

		LOGINFO("Successfully initialized NCCL for GPUs: [{}]", [&]() {
			std::string result;
			for (size_t i = 0; i < gpu_ids_.size(); ++i) {
				if (i > 0)
					result += ", ";
				result += std::to_string(gpu_ids_[i]);
			}
			return result;
		}());

	} catch (const std::exception& e) {
		comms_.clear();
		gpu_ids_.clear();
		ARBD_Exception(ExceptionType::CUDARuntimeError, "Failed to initialize NCCL: {}", e.what());
	}
}

void NCCLManager::finalize() {
	if (!initialized_) {
		return;
	}

	LOGINFO("Finalizing NCCL communicators");

	// Synchronize all operations before cleanup
	try {
		synchronize();
	} catch (const std::exception& e) {
		LOGWARN("Error during NCCL synchronization before finalize: {}", e.what());
	}

	// Destroy communicators
	for (auto& comm : comms_) {
		if (comm != nullptr) {
			ncclCommDestroy(comm);
		}
	}

	comms_.clear();
	gpu_ids_.clear();
	initialized_ = false;

	LOGINFO("NCCL finalized");
}

void NCCLManager::synchronize() {
	if (!initialized_) {
		return;
	}

	// Synchronize all GPUs
	int current_device;
	NCCL_CHECK(cudaGetDevice(&current_device));

	for (size_t i = 0; i < gpu_ids_.size(); ++i) {
		NCCL_CHECK(cudaSetDevice(gpu_ids_[i]));
		NCCL_CHECK(cudaDeviceSynchronize());
	}

	NCCL_CHECK(cudaSetDevice(current_device));
}

void NCCLManager::sync_all() {
	if (initialized_) {
		// NCCL is initialized - use NCCL-managed GPUs
		LOGDEBUG("NCCLManager::sync_all() using NCCL GPU group ({} GPUs)", gpu_ids_.size());
		synchronize();
	} else {
		// NCCL not initialized - fall back to GPUManager basic synchronization
		LOGDEBUG("NCCLManager::sync_all() falling back to basic GPU synchronization");

		const auto& gpus = Manager::devices();
		if (gpus.size() > 1) {
			int curr;
			CUDA_CHECK(cudaGetDevice(&curr));
			for (const auto& gpu : gpus) {
				CUDA_CHECK(cudaSetDevice(gpu.id()));
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			CUDA_CHECK(cudaSetDevice(curr));
		} else if (gpus.size() == 1) {
			CUDA_CHECK(cudaDeviceSynchronize());
		} else {
			LOGWARN("NCCLManager::sync_all() - No GPUs available for synchronization");
		}
	}
}

int NCCLManager::get_rank(int device_id) {
	auto it = std::find(gpu_ids_.begin(), gpu_ids_.end(), device_id);
	if (it != gpu_ids_.end()) {
		return static_cast<int>(std::distance(gpu_ids_.begin(), it));
	}
	return -1;
}

cudaStream_t NCCLManager::get_stream(int rank, int stream_id) {
	if (rank < 0 || rank >= static_cast<int>(gpu_ids_.size())) {
		ARBD_Exception(ExceptionType::ValueError, "Invalid NCCL rank: {}", rank);
	}

	if (stream_id < 0) {
		return nullptr; // Default stream
	}

	// Get stream from GPUManager
	try {
		return Manager::get_stream(gpu_ids_[rank], static_cast<size_t>(stream_id));
	} catch (const std::exception&) {
		LOGWARN("Failed to get stream {} for GPU {}, using default stream",
				stream_id,
				gpu_ids_[rank]);
		return nullptr;
	}
}

} // namespace CUDA
} // namespace ARBD

#endif // USE_CUDA && USE_NCCL