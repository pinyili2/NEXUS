#include "Backend/Resource.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

namespace ARBD {
void* Resource::get_stream_type(int stream_id) const {
	return get_stream_type(static_cast<StreamType>(stream_id));
}

void* Resource::get_stream_type(StreamType stream_type) const {
	// CPU resources don't have streams
	if (type == ResourceType::CPU) {
		return nullptr;
	}

#ifdef USE_CUDA
	if (type == ResourceType::CUDA) {
		try {
			auto& device = CUDA::Manager::get_device(static_cast<int>(id));
			// Return the raw cudaStream_t as void*
			return reinterpret_cast<void*>(device.get_next_stream());
		} catch (...) {
			return nullptr;
		}
	}
#endif

#ifdef USE_SYCL
	if (type == ResourceType::SYCL) {
		try {
			auto& device = SYCL::Manager::get_device(id);
			// Return pointer to the Queue object
			return &device.get_next_queue();
		} catch (...) {
			return nullptr;
		}
	}
#endif

#ifdef USE_METAL
	if (type == ResourceType::METAL) {
		try {
			auto& device = METAL::Manager::get_device(id);
			// Return pointer to the Queue object
			return device.get_next_queue();
		} catch (...) {
			return nullptr;
		}
	}
#endif

	return nullptr;
}

void Resource::synchronize_streams() const {
	// CPU resources don't need synchronization
	if (type == ResourceType::CPU) {
		return;
	}

#ifdef USE_CUDA
	if (type == ResourceType::CUDA) {
		try {
			auto& device = CUDA::Manager::get_device(static_cast<int>(id));
			// Synchronize all streams for this device
			for (size_t i = 0; i < CUDA::Manager::NUM_STREAMS; ++i) {
				cudaStreamSynchronize(device.get_stream(i));
			}
		} catch (...) {
			// Ignore errors during synchronization
		}
		return;
	}
#endif

#ifdef USE_SYCL
	if (type == ResourceType::SYCL) {
		try {
			auto& device = SYCL::Manager::get_device(id);
			// Use the device's built-in synchronization method
			device.synchronize_all_queues();
		} catch (...) {
			// Ignore errors during synchronization
		}
		return;
	}
#endif

#ifdef USE_METAL
	if (type == ResourceType::METAL) {
		try {
			auto& device = METAL::Manager::get_device(id);
			// Synchronize all queues for this device
			device.synchronize_all_queues();
		} catch (...) {
			// Ignore errors during synchronization
		}
		return;
	}
#endif
}

} // namespace ARBD
