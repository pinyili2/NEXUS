// src/Backend/SYCL/SYCLBuffer.h
#pragma once
#ifdef USE_SYCL
#include "../Resource.h"
#include "ARBDException.h"
#include "SYCLManager.h"
#include <cstddef>
#include <cstring>
#include <sycl/sycl.hpp>

namespace ARBD {
namespace SYCL {

struct Policy {
  static void *allocate(const Resource &resource, size_t bytes,
                        void *queue = nullptr, bool sync = true) {
    if (resource.type() != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCL Policy requires SYCL resource, got {}",
                     resource.getTypeString());
    }

    // Get queue from Resource (Resource owns queues now)
    void *ptr = nullptr;
    auto *q = queue ? static_cast<sycl::queue *>(queue)
                    : static_cast<sycl::queue *>(
                          resource.get_stream(StreamType::Memory));

    if (!q) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to get queue for SYCL device {}", resource.id());
    }

    ptr = sycl::malloc_device(bytes, *q);
    if (!ptr) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to allocate {} bytes on SYCL device {}", bytes,
                     resource.id());
    }

    return ptr;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    if (ptr) {
      if (!queue) {
        ARBD_Exception(ExceptionType::RuntimeError,
                       "SYCL deallocation requires a queue");
      }
      sycl::free(ptr, *static_cast<sycl::queue *>(queue));
    }
  }

  static void copy_to_host(void *host_dst, const void *device_src, size_t bytes,
                           void *queue = nullptr, bool sync = false) {
    if (!queue) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "SYCL copy_to_host requires a queue");
    }
    auto &q = *static_cast<sycl::queue *>(queue);

    if (sync) {
      q.memcpy(host_dst, device_src, bytes).wait();
    } else {
      q.memcpy(host_dst, device_src, bytes);
    }
  }

  static void copy_from_host(void *device_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    if (!queue) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "SYCL copy_from_host requires a queue");
    }
    auto &q = *static_cast<sycl::queue *>(queue);

    if (sync) {
      q.memcpy(device_dst, host_src, bytes).wait();
    } else {
      q.memcpy(device_dst, host_src, bytes);
    }
  }

  static void copy_device_to_device(void *dst, const void *src, size_t bytes,
                                    void *queue = nullptr, bool sync = false) {
    if (!queue) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "SYCL copy_device_to_device requires a queue");
    }
    auto &q = *static_cast<sycl::queue *>(queue);

    if (sync) {
      q.memcpy(dst, src, bytes).wait();
    } else {
      q.memcpy(dst, src, bytes);
    }
  }
  template <typename T>
  static void fill(void *dst, T value, size_t num_elements,
                   void *queue = nullptr, bool sync = false) {
    auto &q = *static_cast<sycl::queue *>(queue);
    if (sync) {
      q.fill(static_cast<T *>(dst), value, num_elements).wait();
    } else {
      q.fill(static_cast<T *>(dst), value, num_elements);
    }
  }
};

struct PinnedPolicy {
  static void *allocate(const Resource &resource, size_t bytes,
                        void *queue = nullptr, bool sync = true) {
    if (resource.type() != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCL PinnedPolicy requires SYCL resource, got {}",
                     resource.getTypeString());
    }

    auto *q = queue ? static_cast<sycl::queue *>(queue)
                    : static_cast<sycl::queue *>(
                          resource.get_stream(StreamType::Memory));

    if (!q) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to get queue for SYCL device {}", resource.id());
    }

    void *ptr = sycl::malloc_host(bytes, *q);
    if (!ptr) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to allocate {} bytes of SYCL host memory", bytes);
    }
    return ptr;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    if (ptr && queue) {
      sycl::free(ptr, *static_cast<sycl::queue *>(queue));
    }
  }

  static void upload_to_device(void *device_dst, const void *pinned_src,
                               size_t bytes, const Resource &resource,
                               void *queue = nullptr) {
    auto *q = queue ? static_cast<sycl::queue *>(queue)
                    : static_cast<sycl::queue *>(
                          resource.get_stream(StreamType::Memory));

    if (!q) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to get queue for upload");
    }

    q->memcpy(device_dst, pinned_src, bytes).wait();
  }

  static void download_from_device(void *pinned_dst, const void *device_src,
                                   size_t bytes, const Resource &resource,
                                   void *queue = nullptr) {
    auto *q = queue ? static_cast<sycl::queue *>(queue)
                    : static_cast<sycl::queue *>(
                          resource.get_stream(StreamType::Memory));

    if (!q) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to get queue for download");
    }

    q->memcpy(pinned_dst, device_src, bytes).wait();
  }

  static void copy_from_host(void *pinned_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    std::memcpy(pinned_dst, host_src, bytes);
  }

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
    if (resource.type() != ResourceType::SYCL) {
      ARBD_Exception(ExceptionType::ValueError,
                     "SYCL UnifiedPolicy requires a SYCL resource, got {}",
                     resource.getTypeString());
    }

    auto *q = queue ? static_cast<sycl::queue *>(queue)
                    : static_cast<sycl::queue *>(
                          resource.get_stream(StreamType::Memory));

    if (!q) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to get queue for SYCL device {}", resource.id());
    }

    void *ptr = sycl::malloc_shared(bytes, *q);
    if (!ptr) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to allocate {} bytes of SYCL shared memory",
                     bytes);
    }
    return ptr;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    if (ptr && queue) {
      sycl::free(ptr, *static_cast<sycl::queue *>(queue));
    }
  }

  // SYCL USM prefetch/advise are no-ops to avoid deadlocks
  static void prefetch(void *ptr, size_t bytes, int device_id,
                       void *queue = nullptr) {
    // No-op: SYCL prefetch causes deadlocks in some implementations
  }

  static void mem_advise(void *ptr, size_t bytes, int advice, int device_id,
                         void *queue = nullptr) {
    // No-op: SYCL mem_advise causes deadlocks in some implementations
  }

  static void copy_from_host(void *unified_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    std::memcpy(unified_dst, host_src, bytes);
  }

  static void copy_to_host(void *host_dst, const void *unified_src,
                           size_t bytes, void *queue = nullptr,
                           bool sync = false) {
    std::memcpy(host_dst, unified_src, bytes);
  }

  static void copy_device_to_device(void *dst, const void *src, size_t bytes,
                                    void *queue = nullptr, bool sync = false) {
    // For SYCL unified memory, use regular memcpy since both src and dst are
    // accessible from host
    std::memcpy(dst, src, bytes);
  }
};

struct TexturePolicy {
  struct Texture {
    void *image{nullptr};
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
      tex->image = new sycl::image<3>(order, type, dims);
    } else if (tex->height > 0) {
      sycl::range<2> dims(width, height);
      tex->image = new sycl::image<2>(order, type, dims);
    } else {
      sycl::range<1> dims(width);
      tex->image = new sycl::image<1>(order, type, dims);
    }
    return tex;
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
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

  static void copy_from_buffer(void *texture_ptr, const void *buffer_ptr,
                               size_t bytes, const Resource &resource,
                               StreamType stream_type) {
    if (!texture_ptr || !buffer_ptr)
      return;

    Texture *tex = static_cast<Texture *>(texture_ptr);
    const bool is3D = tex->depth > 0;
    const bool is2D = (tex->depth == 0 && tex->height > 0);

    // Currently only support RGBA FP32
    if (tex->order != sycl::image_channel_order::rgba ||
        tex->type != sycl::image_channel_type::fp32) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "SYCL TexturePolicy currently supports only RGBA FP32");
      return;
    }

    auto *q = static_cast<sycl::queue *>(resource.get_stream(stream_type));
    if (!q) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Failed to get queue for texture upload");
    }

    size_t num_pixels = tex->width * tex->height * (is3D ? tex->depth : 1);
    size_t expected_bytes = num_pixels * sizeof(sycl::float4);
    if (bytes < expected_bytes) {
      ARBD_Exception(ExceptionType::RuntimeError,
                     "Insufficient bytes for image upload: have {}, need {}",
                     bytes, expected_bytes);
      return;
    }

    sycl::buffer<sycl::float4, 1> src_buf(
        reinterpret_cast<const sycl::float4 *>(buffer_ptr),
        sycl::range<1>(num_pixels), {sycl::property::buffer::use_host_ptr()});

    q->submit([&](sycl::handler &cgh) {
       auto accSrc = src_buf.template get_access<sycl::access::mode::read>(cgh);

       if (is3D) {
         sycl::accessor<sycl::float4, 3, sycl::access::mode::write,
                        sycl::access::target::image>
             accImg(*static_cast<sycl::image<3> *>(tex->image), cgh);
         cgh.parallel_for(
             sycl::range<3>(tex->width, tex->height, tex->depth),
             [=](sycl::id<3> idx) {
               size_t x = idx[0], y = idx[1], z = idx[2];
               size_t linear_idx =
                   (z * tex->height * tex->width + y * tex->width + x);
               accImg.write(sycl::int4(x, y, z, 0), accSrc[linear_idx]);
             });
       } else if (is2D) {
         sycl::accessor<sycl::float4, 2, sycl::access::mode::write,
                        sycl::access::target::image>
             accImg(*static_cast<sycl::image<2> *>(tex->image), cgh);
         cgh.parallel_for(sycl::range<2>(tex->width, tex->height),
                          [=](sycl::id<2> idx) {
                            size_t x = idx[0], y = idx[1];
                            size_t linear_idx = (y * tex->width + x);
                            accImg.write(sycl::int2(x, y), accSrc[linear_idx]);
                          });
       }
     }).wait_and_throw();
  }
};

} // namespace SYCL
} // namespace ARBD
#endif
