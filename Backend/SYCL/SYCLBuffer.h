#pragma once
#ifdef USE_SYCL
#include "../Resource.h"
#include "ARBDException.h"
#include "Backend/SYCL/SYCLManager.h"
#include <cstddef>
#include <cstring>
#include <sycl/sycl.hpp>

namespace ARBD {
namespace SYCL {
struct Policy {
  static void *allocate(const Resource &resource, size_t bytes,
                        void *queue = nullptr, bool sync = true) {
    if (resource.type != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCL Policy requires SYCL resource, got {}",
                     resource.toString());
    }

    // Get the specific device queue for this resource
    auto &device_manager = Manager::get_device(resource.id);

    void *ptr = nullptr; // Initialize the pointer
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : device_manager.get_next_queue();
    SYCL_CHECK(ptr = sycl::malloc_device(bytes, q));
    if (!ptr) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to allocate {} bytes on SYCL device {}", bytes,
                     resource.id);
    }

    return ptr;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    if (ptr) {
      // For deallocation, we need to use the current queue context
      // since we don't have the original resource information
      auto &current_queue = Manager::get_current_queue();
      SYCL_CHECK(sycl::free(ptr, current_queue.get()));
    }
  }

  static void copy_to_host(void *host_dst, const void *device_src, size_t bytes,
                           void *queue = nullptr, bool sync = false) {
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : Manager::get_current_queue().get();
    if (sync) {
      SYCL_CHECK(q.memcpy(host_dst, device_src, bytes).wait());
    } else {
      SYCL_CHECK(q.memcpy(host_dst, device_src, bytes));
    }
  }

  static void copy_from_host(void *device_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : Manager::get_current_queue().get();
    if (sync) {
      SYCL_CHECK(q.memcpy(device_dst, host_src, bytes).wait());
    } else {
      SYCL_CHECK(q.memcpy(device_dst, host_src, bytes));
    }
  }

  static void copy_device_to_device(void *dst, const void *src, size_t bytes,
                                    void *queue = nullptr, bool sync = false) {
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : Manager::get_current_queue().get();
    if (sync) {
      SYCL_CHECK(q.memcpy(dst, src, bytes).wait());
    } else {
      SYCL_CHECK(q.memcpy(dst, src, bytes));
    }
  }
};
struct PinnedPolicy {
  static void *allocate(const Resource &resource, size_t bytes,
                        void *queue = nullptr, bool sync = true) {
    if (resource.type != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCL Policy requires SYCL resource, got {}",
                     resource.toString());
    }
    auto &device = SYCL::Manager::get_device(resource.id);
    auto &q =
        queue ? *static_cast<sycl::queue *>(queue) : device.get_next_queue();

    void *ptr = nullptr; // Initialize the pointer
    SYCL_CHECK(ptr = sycl::malloc_host(bytes, q));
    if (!ptr) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Failed to allocate {} bytes of SYCL host memory", bytes);
    }
    return ptr;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    if (ptr) {
      auto &current_queue = SYCL::Manager::get_current_queue();
      SYCL_CHECK(sycl::free(ptr, current_queue.get()));
    }
  }
  static void upload_to_device(void *device_dst, const void *pinned_src,
                               size_t bytes, const Resource &resource,
                               void *queue = nullptr) {
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : Manager::get_current_queue().get();
    SYCL_CHECK(q.memcpy(device_dst, pinned_src, bytes));
    SYCL_CHECK(q.wait_and_throw()); // Ensure operation completes
  }

  static void download_from_device(void *pinned_dst, const void *device_src,
                                   size_t bytes, const Resource &resource,
                                   void *queue = nullptr) {
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : Manager::get_current_queue().get();
    SYCL_CHECK(q.memcpy(pinned_dst, device_src, bytes));
    SYCL_CHECK(q.wait_and_throw()); // Ensure operation completes
  }

  static void copy_from_host(void *pinned_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    std::memcpy(pinned_dst, host_src, bytes);
  }

  // Copies from this pinned buffer to a standard host buffer.
  static void copy_to_host(void *host_dst, const void *pinned_src, size_t bytes,
                           void *queue = nullptr, bool sync = false) {
    std::memcpy(host_dst, pinned_src, bytes);
  }
  static void copy_device_to_device(void *dst, const void *src, size_t bytes,
                                    void *queue = nullptr, bool sync = false) {
    std::memcpy(dst, src, bytes);
  }
};
struct UnifiedPolicy {
  static void *allocate(const Resource &resource, size_t bytes,
                        void *queue = nullptr, bool sync = true) {
    if (resource.type != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCLUnifiedMemoryPolicy requires a SYCL resource.");
    }
    // Get the queue associated with the target SYCL device
    auto &device = SYCL::Manager::get_device(resource.id);
    auto &q =
        queue ? *static_cast<sycl::queue *>(queue) : device.get_next_queue();

    void *ptr = nullptr; // Initialize the pointer
    SYCL_CHECK(ptr = sycl::malloc_shared(bytes, q));
    if (!ptr) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Failed to allocate {} bytes of SYCL shared memory",
                     bytes);
    }
    return ptr;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    if (ptr) {
      auto &queue = SYCL::Manager::get_current_queue();
      SYCL_CHECK(sycl::free(ptr, queue));
    }
  }
  // Should be default async
  static void prefetch(void *ptr, size_t bytes, int device_id,
                       void *queue = nullptr) {
    // SYCL prefetch is not standard and causes deadlocks in HipSYCL
    // This is a performance hint, so it's safe to make it a no-op
    (void)ptr;
    (void)bytes;
    (void)device_id;
    (void)queue; // Suppress unused parameter warnings
  }

  static void mem_advise(void *ptr, size_t bytes, int advice, int device_id,
                         void *queue = nullptr) {
    // SYCL mem_advise is not standard and causes deadlocks in HipSYCL
    // This is a performance hint, so it's safe to make it a no-op
    (void)ptr;
    (void)bytes;
    (void)advice;
    (void)device_id;
    (void)queue; // Suppress unused parameter warnings
  }
  static void copy_from_host(void *unified_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    std::memcpy(unified_dst, host_src, bytes);
    // Skip prefetch - causes deadlocks in HipSYCL
    (void)queue;
    (void)sync; // Suppress unused parameter warnings
  }

  static void copy_to_host(void *host_dst, const void *unified_src,
                           size_t bytes, void *queue = nullptr,
                           bool sync = false) {
    // Skip prefetch - causes deadlocks in HipSYCL
    std::memcpy(host_dst, unified_src, bytes);
    (void)queue;
    (void)sync; // Suppress unused parameter warnings
  }

  static void copy_device_to_device(void *dst, const void *src, size_t bytes,
                                    void *queue = nullptr, bool sync = false) {
    auto &q = queue ? *static_cast<sycl::queue *>(queue)
                    : Manager::get_current_queue().get();
    if (sync) {
      q.memcpy(dst, src, bytes).wait();
    } else {
      q.memcpy(dst, src, bytes);
    }
  }
};
struct TexturePolicy {
  // Stores a single non-templated sycl::image pointer (1D/2D/3D created based
  // on dims)
  struct Texture {
    void *image{nullptr}; // points to sycl::image<1>, <2>, or <3>
    size_t width{0}, height{0}, depth{0};
    sycl::image_channel_order order;
    sycl::image_channel_type type;
  };

  static void *allocate(const Resource &resource, size_t width, size_t height,
                        size_t depth, sycl::image_channel_order order,
                        sycl::image_channel_type type) {
    Texture *tex = new Texture();
    tex->order = order;
    tex->type = type;
    tex->width = width;
    tex->height = (depth == 0 && height == 0) ? 1 : height;
    tex->depth = depth;
    if (depth > 0) {
      sycl::range<3> dims(width, height, depth);
      tex->image = static_cast<void *>(new sycl::image<3>(order, type, dims));
    } else if (tex->height > 0) {
      sycl::range<2> dims(width, height);
      tex->image = static_cast<void *>(new sycl::image<2>(order, type, dims));
    }
    return tex;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    (void)queue;
    (void)sync;
    if (!ptr)
      return;
    Texture *tex = static_cast<Texture *>(ptr);
    if (tex->image) {
      if (tex->depth > 0) {
        delete static_cast<sycl::image<3> *>(tex->image);
      } else if (tex->height > 0) {
        delete static_cast<sycl::image<2> *>(tex->image);
      } else {
        delete static_cast<sycl::image<1> *>(tex->image);
      }
    }
    delete tex;
  }

  // Uploads from host buffer into the image via command copy
  static void copy_from_buffer(void *texture_ptr, const void *buffer_ptr,
                               size_t bytes, const Resource &resource) {
    if (!texture_ptr || !buffer_ptr)
      return;
    Texture *tex = static_cast<Texture *>(texture_ptr);
    const bool is3D = tex->depth > 0;
    const bool is2D = (tex->depth == 0);

    // For now support only RGBA FP32 writes
    if (tex->order != sycl::image_channel_order::rgba ||
        tex->type != sycl::image_channel_type::fp32) {
      ARBD_Exception(
          ExceptionType::RuntimeError,
          "SYCL TexturePolicy currently supports only RGBA FP32 images for "
          "copy_from_buffer.");
      return;
    }

    auto &q = Manager::get_device(resource.id).get_next_queue().get();
    size_t num_pixels = tex->width * tex->height * (is3D ? tex->depth : 1);
    size_t expected_bytes = num_pixels * sizeof(sycl::float4);
    if (bytes < expected_bytes) {
      ARBD_Exception(
          ExceptionType::RuntimeError,
          "Insufficient bytes for RGBA FP32 image upload: have {}, need {}",
          bytes, expected_bytes);
      return;
    }
    sycl::buffer<sycl::float4, 1> src_buf(
        reinterpret_cast<const sycl::float4 *>(buffer_ptr),
        sycl::range<1>(num_pixels), {sycl::property::buffer::use_host_ptr()});

    q.submit([&](sycl::handler &cgh) {
       auto accSrc = src_buf.template get_access<sycl::access::mode::read>(cgh);
       if (is3D) {
         sycl::accessor<sycl::float4, 3, sycl::access::mode::write,
                        sycl::access::target::image>
             accImg(*static_cast<sycl::image<3> *>(tex->image), cgh);
         cgh.parallel_for(
             sycl::range<3>(tex->width, tex->height, tex->depth),
             [=](sycl::id<3> idx) {
               size_t x = idx[0], y = idx[1], z = idx[2];
               size_t linear_index =
                   (z * tex->height * tex->width + y * tex->width + x);
               accImg.write(sycl::int4(x, y, z, 0), accSrc[linear_index]);
             });
       } else if (is2D) {
         sycl::accessor<sycl::float4, 2, sycl::access::mode::write,
                        sycl::access::target::image>
             accImg(*static_cast<sycl::image<2> *>(tex->image), cgh);
         cgh.parallel_for(
             sycl::range<2>(tex->width, tex->height), [=](sycl::id<2> idx) {
               size_t x = idx[0], y = idx[1];
               size_t linear_index = (y * tex->width + x);
               accImg.write(sycl::int2(x, y), accSrc[linear_index]);
             });
       }
     }).wait_and_throw();
  }
  /**
          static void copy_from_buffer2(void* texture_ptr,
                                                                    const void*
     buffer_ptr, size_t bytes, const Resource& resource) { if (!texture_ptr ||
     !buffer_ptr) return;

                  Texture* tex = static_cast<Texture*>(texture_ptr);
                  auto& q = Manager::get_device(resource.id).get_next_queue();
                  // Create a source buffer from the host pointer.
                  sycl::buffer<unsigned char, 1> src_buf(static_cast<const
     unsigned char*>(buffer_ptr), sycl::range<1>(bytes),
                                                                                             {sycl::property::buffer::use_host_ptr()});
                  size_t num_elements = bytes / sizeof(sycl::float4);

                  q.submit([&](sycl::handler& cgh) {
                           // Get a read accessor for the source buffer
                           auto src_acc =
     src_buf.get_access<sycl::access::mode::read>(cgh);

                           if (tex->depth > 0) {
                                   auto* img =
     static_cast<sycl::image<3>*>(tex->image);
                                   // Create a 3D image accessor for writing
                                   sycl::accessor<sycl::float4,
                                                                  3,
                                                                  sycl::access::mode::write,
                                                                  sycl::access::target::image>
                                           dest_acc(*img, cgh);

                                   // Use parallel_for to copy data from buffer
     to image cgh.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1>
     idx) { if (idx[0] < num_elements) {
                                                   // Convert 1D index to 3D
     coordinates if needed
                                                   // For now, assume linear
     mapping const unsigned char* src_ptr = &src_acc[idx *
     sizeof(sycl::float4)]; sycl::float4 data = *reinterpret_cast<const
     sycl::float4*>(src_ptr); sycl::id<3> coord(idx[0] % tex->width, (idx[0] /
     tex->width) % tex->height, idx[0] / (tex->width * tex->height));
                                                   dest_acc[coord] = data;
                                           }
                                   });
                           } else if (tex->height > 0) {
                                   auto* img =
     static_cast<sycl::image<2>*>(tex->image);
                                   // Create a 2D image accessor for writing
                                   sycl::accessor<sycl::float4,
                                                                  2,
                                                                  sycl::access::mode::write,
                                                                  sycl::access::target::image>
                                           dest_acc(*img, cgh);

                                   // Use parallel_for to copy data from buffer
     to image cgh.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1>
     idx) { if (idx[0] < num_elements) { const unsigned char* src_ptr =
     &src_acc[idx * sizeof(sycl::float4)]; sycl::float4 data =
     *reinterpret_cast<const sycl::float4*>(src_ptr); sycl::id<2> coord(idx[0] %
     tex->width, idx[0] / tex->width); dest_acc[coord] = data;
                                           }
                                   });
                           } else {
                                   auto* img =
     static_cast<sycl::image<1>*>(tex->image);
                                   // Create a 1D image accessor for writing
                                   sycl::accessor<sycl::float4,
                                                                  1,
                                                                  sycl::access::mode::write,
                                                                  sycl::access::target::image>
                                           dest_acc(*img, cgh);

                                   // Use parallel_for to copy data from buffer
     to image cgh.parallel_for(sycl::range<1>(num_elements), [=](sycl::id<1>
     idx) { if (idx[0] < num_elements) { const unsigned char* src_ptr =
     &src_acc[idx * sizeof(sycl::float4)]; sycl::float4 data =
     *reinterpret_cast<const sycl::float4*>(src_ptr); sycl::id<1> coord(idx[0]);
                                                   dest_acc[coord] = data;
                                           }
                                   });
                           }
                   }).wait_and_throw();
          }
          */
};
} // namespace SYCL
} // namespace ARBD
#endif
