#pragma once
#ifdef USE_CUDA
#include "ARBDException.h"
#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Resource.h"
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>

namespace ARBD {
namespace CUDA {
struct Policy {
	// queue is a cuda stream
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}

		// Thread-safe device context management
		int old_device;
		CUDA_CHECK(cudaGetDevice(&old_device));
		CUDA_CHECK(cudaSetDevice(static_cast<int>(resource.id)));

		void* ptr = nullptr;

		CUDA_CHECK(cudaMalloc(&ptr, bytes));

		// Restore previous device context
		CUDA_CHECK(cudaSetDevice(old_device));

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (ptr) {
			// Use synchronous deallocation for reliable cleanup
			// This ensures proper cleanup without stream dependencies
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void copy_to_host(void* host_dst,
							 const void* device_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		if (!host_dst || !device_src || bytes == 0)
			return;

		if (sync) {
			CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDefault));
		} else {
			// Use the provided queue if available, otherwise get a new stream
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
			CUDA_CHECK(cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDefault, stream));
		}
	}

	static void copy_from_host(void* device_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		if (!device_dst || !host_src || bytes == 0)
			return;

		if (sync) {
			CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
			CUDA_CHECK(cudaMemcpyAsync(device_dst, host_src, bytes, cudaMemcpyDefault, stream));
		}
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		if (!dst || !src || bytes == 0)
			return;

		if (sync) {
			CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
		} else {
			// Use the provided queue if available, otherwise get a new stream
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
		}
	}
};
struct PinnedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}
		void* ptr = nullptr;
		CUDA_CHECK(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped));

		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		// Pinned memory deallocation doesn't need queue or sync parameters
		(void)queue;
		(void)sync;
		if (ptr) {
			CUDA_CHECK(cudaFreeHost(ptr));
		}
	}
	static void upload_to_device(void* device_dst,
								 const void* pinned_src,
								 size_t bytes,
								 const Resource& resource,
								 void* queue = nullptr) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
									: Manager::get_device(resource.id).get_next_stream();
		CUDA_CHECK(cudaMemcpyAsync(device_dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
	}

	static void download_from_device(void* pinned_dst,
									 const void* device_src,
									 size_t bytes,
									 const Resource& resource,
									 void* queue = nullptr) {
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
									: Manager::get_device(resource.id).get_next_stream();
		CUDA_CHECK(cudaMemcpyAsync(pinned_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream));
	}
};
struct UnifiedPolicy {
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDA Policy requires CUDA resource, got {}",
						   resource.type);
		}
		void* ptr = nullptr;
		CUDA_CHECK(cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal));
		return ptr;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		// Unified memory deallocation doesn't need queue or sync parameters
		(void)queue;
		(void)sync;
		if (ptr) {
			CUDA_CHECK(cudaFree(ptr));
		}
	}
	static void prefetch(void* ptr, size_t bytes, int device_id, void* queue = nullptr) {
		cudaStream_t stream =
			queue ? static_cast<cudaStream_t>(queue)
				  : (device_id >= 0 ? Manager::get_device(device_id).get_next_stream() : 0);
		CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device_id, stream));
	}

	static void mem_advise(void* ptr, size_t bytes, int advice, int device_id) {
		// Validate parameters before calling cudaMemAdvise
		if (!ptr || bytes == 0) {
			return; // No-op for invalid pointers or zero bytes
		}

		cudaMemoryAdvise cuda_advice =
			(advice == 0) ? cudaMemAdviseSetReadMostly : static_cast<cudaMemoryAdvise>(advice);

		CUDA_CHECK(cudaMemAdvise(ptr, bytes, cuda_advice, device_id));
	}

	static void copy_from_host(void* unified_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		std::memcpy(unified_dst, host_src, bytes);
		// Optionally prefetch to the current device to warm it up
		int device;
		cudaGetDevice(&device);
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;
		CUDA_CHECK(cudaMemPrefetchAsync(unified_dst, bytes, device, stream));
	}

	static void copy_to_host(void* host_dst,
							 const void* unified_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		// Prefetch to the host to ensure data is resident, then copy
		cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue) : 0;
		CUDA_CHECK(
			cudaMemPrefetchAsync(const_cast<void*>(unified_src), bytes, cudaCpuDeviceId, stream));
		if (stream) {
			CUDA_CHECK(cudaStreamSynchronize(stream));
		} else {
			CUDA_CHECK(cudaDeviceSynchronize()); // Sync if default stream
		}
		std::memcpy(host_dst, unified_src, bytes);
	}

	static void copy_device_to_device(void* dst,
									  const void* src,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		if (sync) {
			// cudaMemcpyDefault handles peer-to-peer automatically
			CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
		} else {
			cudaStream_t stream = queue ? static_cast<cudaStream_t>(queue)
										: Manager::get_current_device().get_next_stream();
			CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
		}
	}
};
struct TexturePolicy {
	struct Texture {
		cudaArray_t array{nullptr};
		cudaTextureObject_t textureObject{0};
	};

	static void* allocate(const Resource& resource,
						  size_t width,
						  size_t height,
						  size_t depth,
						  cudaChannelFormatDesc channelDesc,
						  const void* src_data = nullptr,
						  bool is_host_data = false) {
		if (resource.type != ResourceType::CUDA) {
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDATexturePolicy requires CUDA resource, got {}",
						   resource.type);
		}

		Texture* texture = new Texture();

		const unsigned int bits_x = channelDesc.x < 0 ? static_cast<unsigned int>(-channelDesc.x)
													  : static_cast<unsigned int>(channelDesc.x);
		const unsigned int bits_y = channelDesc.y < 0 ? static_cast<unsigned int>(-channelDesc.y)
													  : static_cast<unsigned int>(channelDesc.y);
		const unsigned int bits_z = channelDesc.z < 0 ? static_cast<unsigned int>(-channelDesc.z)
													  : static_cast<unsigned int>(channelDesc.z);
		const unsigned int bits_w = channelDesc.w < 0 ? static_cast<unsigned int>(-channelDesc.w)
													  : static_cast<unsigned int>(channelDesc.w);
		const size_t element_size_bytes =
			(static_cast<size_t>(bits_x + bits_y + bits_z + bits_w)) / 8u;
		const size_t effective_height = (height > 0 ? height : 1);
		const size_t effective_depth = (depth > 0 ? depth : 1);

		cudaResourceDesc resDesc;
		std::memset(&resDesc, 0, sizeof(resDesc));
		if (is_host_data && src_data) {
			resDesc.resType = cudaResourceTypeArray;
			// Note: cudaExtent's depth defaults to 0, which is correct for 1D/2D
			cudaExtent extent = make_cudaExtent(width, height, depth);

			if (depth > 0) {
				CUDA_CHECK(cudaMalloc3DArray(&texture->array, &channelDesc, extent, 0));
			} else if (height > 0) {
				CUDA_CHECK(cudaMallocArray(&texture->array, &channelDesc, width, height, 0));
			} else {
				// Correct way to allocate a 1D array is with height = 0
				CUDA_CHECK(cudaMallocArray(&texture->array, &channelDesc, width, 0, 0));
			}
			resDesc.res.array.array = texture->array;

			cudaMemcpy3DParms copyParams;
			std::memset(&copyParams, 0, sizeof(copyParams));
			copyParams.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src_data),
													width * element_size_bytes,
													width,
													effective_height);
			copyParams.extent = make_cudaExtent(width, effective_height, effective_depth);
			copyParams.dstArray = texture->array;
			copyParams.kind = cudaMemcpyHostToDevice;
			CUDA_CHECK(cudaMemcpy3D(&copyParams));
		} else if (src_data) {
			resDesc.resType = cudaResourceTypeLinear;
			resDesc.res.linear.devPtr = const_cast<void*>(src_data);
			resDesc.res.linear.desc = channelDesc;
			resDesc.res.linear.sizeInBytes =
				width * effective_height * effective_depth * element_size_bytes;
		} else {
			delete texture;
			ARBD_Exception(ExceptionType::ValueError,
						   "CUDATexturePolicy allocation requires source data.");
			return nullptr;
		}

		cudaTextureDesc texDesc;
		std::memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA_CHECK(cudaCreateTextureObject(&texture->textureObject, &resDesc, &texDesc, nullptr));
		return static_cast<void*>(texture);
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		(void)queue;
		(void)sync;
		if (!ptr)
			return;
		Texture* texture = static_cast<Texture*>(ptr);
		if (texture->textureObject) {
			CUDA_CHECK(cudaDestroyTextureObject(texture->textureObject));
			texture->textureObject = 0;
		}
		if (texture->array) {
			CUDA_CHECK(cudaFreeArray(texture->array));
			texture->array = nullptr;
		}
		delete texture;
	}

	static void copy_from_buffer(void* texture_ptr,
								 const void* buffer_ptr,
								 size_t width,
								 size_t height,
								 size_t depth,
								 size_t element_size_bytes) {
		if (!texture_ptr || !buffer_ptr)
			return;

		// Cast the void pointer back to our internal texture struct
		Texture* texture = static_cast<Texture*>(texture_ptr);

		// We can only copy to textures that are backed by a cudaArray
		if (!texture->array) {
			ARBD_Exception(
				ExceptionType::RuntimeError,
				"Cannot copy to a texture that was not created with a cudaArray. Textures "
				"created from linear memory are read-only views.");
			return;
		}

		// Use cudaMemcpy2DToArray for 1D and 2D copies. A 3D copy would use cudaMemcpy3D.
		// This is robust for both 1D (height=0) and 2D cases.
		CUDA_CHECK(cudaMemcpy2DToArray(
			texture->array, // Destination cudaArray
			0,
			0,							// Destination x, y offsets in array
			buffer_ptr,					// Source linear device memory
			width * element_size_bytes, // Pitch of source memory (bytes per row)
			width * element_size_bytes, // Width of copy in bytes
			(height > 0 ? height : 1),	// Height of copy
			cudaMemcpyDeviceToDevice));
	}
};
} // namespace CUDA
} // namespace ARBD
#endif
