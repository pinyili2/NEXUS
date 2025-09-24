#pragma once
#ifdef USE_METAL

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "METALManager.h"

#include <Metal/Metal.hpp>

namespace ARBD {
namespace METAL {
// Base policy that interfaces with your existing METAL::Manager
struct Policy {
	/**
	 * @brief Allocates a Metal buffer using the manager's tracking system.
	 */
	static void*
	allocate_with_options(const Resource& resource, size_t bytes, MTL::ResourceOptions options) {
		if (bytes == 0)
			return nullptr;

		// Use your manager's RAII-based buffer creation
		// We create the buffer and then get its contents pointer.
		// The manager's map will keep the smart pointer alive.
		auto mtl_buffer_ptr = Manager::create_raw_buffer(bytes, options);
		void* contents = mtl_buffer_ptr->contents();

		return Manager::allocate_raw(bytes, options);
	}

	/**
	 * @brief Default allocation for DeviceBuffer (uses Shared storage).
	 */
	static void*
	allocate(const Resource& resource, size_t bytes, void* queue = nullptr, bool sync = true) {
		// Shared mode is the versatile, high-performance default on Apple Silicon.
		// The resource parameter is implicitly handled by the manager's 'current_device'.
		return Manager::allocate_raw(bytes, MTL::ResourceStorageModeShared);
	}

	/**
	 * @brief Deallocates a Metal buffer via the manager.
	 */
	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
		if (!ptr)
			return;
		// Your manager's deallocate_raw function will find the smart pointer
		// in its map and let it go out of scope, releasing the MTL::Buffer.
		Manager::deallocate_raw(ptr);
	}

	/**
	 * @brief Copies data to host. For Shared memory, this is a memcpy.
	 */
	static void copy_to_host(void* host_dst,
							 const void* device_src,
							 size_t bytes,
							 void* queue = nullptr,
							 bool sync = false) {
		// Assumes the source buffer is Shared or Managed.
		// A copy from a Private buffer would require a blit encoder and a staging buffer.
		if (device_src) {
			std::memcpy(host_dst, device_src, bytes);
		}
	}

	/**
	 * @brief Copies data from host. For Shared memory, this is a memcpy.
	 */
	static void copy_from_host(void* device_dst,
							   const void* host_src,
							   size_t bytes,
							   void* queue = nullptr,
							   bool sync = false) {
		// Assumes the destination buffer is Shared or Managed.
		if (device_dst) {
			std::memcpy(device_dst, host_src, bytes);
		}
	}

	/**
	 * @brief Copies data between two Metal buffers using a high-performance blit encoder.
	 */
	static void copy_device_to_device(void* dst_ptr,
									  const void* src_ptr,
									  size_t bytes,
									  void* queue = nullptr,
									  bool sync = false) {
		if (!dst_ptr || !src_ptr || bytes == 0) {
			return; // Nothing to copy
		}

		// For Metal's unified memory architecture, use memcpy as the primary method
		// This is more reliable than Metal blit encoder for device-to-device copies
		// and avoids buffer tracking issues during resize operations
		std::memcpy(dst_ptr, src_ptr, bytes);
	}
};

// PinnedPolicy on Metal maps to Shared memory, as it achieves the same goal:
// efficient CPU-GPU data exchange.
struct PinnedPolicy : public Policy {

	static void upload_to_device(void* device_dst,
								 const void* pinned_src,
								 size_t bytes,
								 const Resource& resource,
								 void* queue = nullptr) {
		Policy::copy_from_host(device_dst, pinned_src, bytes, queue,true);

	}

	static void download_from_device(void* pinned_dst,
									 const void* device_src,
									 size_t bytes,
									 const Resource& resource,
									 void* queue = nullptr) {
		Policy::copy_to_host(pinned_dst, device_src, bytes, queue, true);
	}
};

// UnifiedPolicy on Metal also maps to Shared memory.
// Performance hints like prefetch/mem_advise are implemented as no-ops.
struct UnifiedPolicy : public Policy {
	// Inherits allocate, deallocate, and copy methods.

	static void prefetch(void* ptr, size_t bytes, int device_id, void* queue = nullptr) {
		(void)ptr;
		(void)bytes;
		(void)device_id;
		(void)queue; // Suppress unused parameter warnings
	}

	static void
	mem_advise(void* ptr, size_t bytes, int advice, int device_id, void* queue = nullptr) {
		(void)ptr;
		(void)bytes;
		(void)advice;
		(void)device_id;
		(void)queue; // Suppress unused parameter warnings
	}
};
struct TexturePolicy {
	static void* allocate(const Resource& resource, size_t width, size_t height, size_t depth, MTL::PixelFormat pixelFormat) {
			auto& device = Manager::get_device(resource.id);
			auto* mtl_device = device.get_native_device();

			auto* desc = MTL::TextureDescriptor::alloc()->init();
			if (depth > 0) {
					desc->setTextureType(MTL::TextureType3D);
					desc->setDepth(depth);
			} else if (height > 0) {
					desc->setTextureType(MTL::TextureType2D);
					desc->setHeight(height);
			} else {
					desc->setTextureType(MTL::TextureType1D);
			}
			desc->setWidth(width);
			desc->setPixelFormat(pixelFormat);
			desc->setStorageMode(MTL::StorageModeShared); // Or Private for dGPU
			desc->setUsage(MTL::TextureUsageShaderRead);

			MTL::Texture* texture = mtl_device->newTexture(desc);
			desc->release();
			return texture;
	}

	static void deallocate(void* ptr, void* queue = nullptr, bool sync = true) {
			if (ptr) {
					static_cast<MTL::Texture*>(ptr)->release();
			}
	}

	static void copy_from_buffer(void* texture_ptr, const void* buffer_ptr, size_t bytes, const Resource& resource) {
			auto& device = Manager::get_device(resource.id);
			auto& cmd_queue = device.get_next_queue();
			auto* cmd_buffer = cmd_queue.commandBuffer();
			auto* blit_encoder = cmd_buffer->blitCommandEncoder();

			auto* mtl_texture = static_cast<MTL::Texture*>(texture_ptr);
			// We need the MTL::Buffer*, not just the void* contents
			auto* mtl_buffer = Manager::get_metal_buffer_from_ptr(buffer_ptr);

			size_t bytes_per_row = mtl_texture->width() * 4 * sizeof(float); // Example for RGBA32Float
			size_t bytes_per_image = bytes_per_row * mtl_texture->height();

			blit_encoder->copyFromBuffer(mtl_buffer, 0, bytes_per_row, bytes_per_image,
																	 MTL::Size(mtl_texture->width(), mtl_texture->height(), mtl_texture->depth()),
																	 mtl_texture, 0, 0, MTL::Origin(0, 0, 0));

			blit_encoder->endEncoding();
			cmd_buffer->commit();
			cmd_buffer->waitUntilCompleted();
	}
};
} // namespace METAL
} // namespace ARBD

#endif
