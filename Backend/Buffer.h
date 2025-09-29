#pragma once

#if defined(__METAL_VERSION__)
// Metal Shading Language uses the standard C99 'restrict'
#define __restrict__ restrict
#elif defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__) ||        \
    defined(_MSC_VER)
#define __restrict__ __restrict__
#else
// Fallback for other compilers: define it as nothing to ensure compilation
#define __restrict__
#endif

#ifndef __METAL_VERSION__
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#ifdef USE_CUDA
#include "CUDA/CUDABuffer.h"
#include "CUDA/CUDAManager.h"
#include <thrust/tuple.h>
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLBuffer.h"
#include "SYCL/SYCLManager.h"
#endif
#ifdef USE_METAL
#include "METAL/METALBuffer.h"
#include "METAL/METALManager.h"
#endif

#include "ARBDLogger.h"
#include "Events.h"
#include "Header.h"
#include "Resource.h"

namespace ARBD {

namespace CPU {
struct Policy {
  static void *allocate(const Resource &resource, size_t bytes,
                        void *queue = nullptr, bool sync = true) {
    // CPU allocation doesn't need queue or sync parameters
    (void)queue;
    (void)sync;
    return malloc(bytes);
  }

  static void deallocate(void *ptr, void *queue = nullptr, bool sync = true) {
    // CPU deallocation doesn't need queue or sync parameters
    (void)queue;
    (void)sync;
    free(ptr);
  }

  static void copy_to_host(void *host_dst, const void *device_src, size_t bytes,
                           void *queue = nullptr, bool sync = false) {
    // CPU copy doesn't need queue or sync parameters
    (void)queue;
    (void)sync;
    std::memcpy(host_dst, device_src, bytes);
  }

  static void copy_from_host(void *device_dst, const void *host_src,
                             size_t bytes, void *queue = nullptr,
                             bool sync = false) {
    // CPU copy doesn't need queue or sync parameters
    (void)queue;
    (void)sync;
    std::memcpy(device_dst, host_src, bytes);
  }

  static void copy_device_to_device(void *dst, const void *src, size_t bytes,
                                    void *queue = nullptr, bool sync = false) {
    // CPU copy doesn't need queue or sync parameters
    (void)queue;
    (void)sync;
    std::memcpy(dst, src, bytes);
  }
};
} // namespace CPU

#if defined(USE_CUDA)
using BackendPolicy = CUDA::Policy;
using PinnedPolicy = CUDA::PinnedPolicy;
using UnifiedPolicy = CUDA::UnifiedPolicy;
using TexturePolicy = CUDA::TexturePolicy;
#elif defined(USE_SYCL)
using BackendPolicy = SYCL::Policy;
using PinnedPolicy = SYCL::PinnedPolicy;
using UnifiedPolicy = SYCL::UnifiedPolicy;
using TexturePolicy = SYCL::TexturePolicy;
#elif defined(USE_METAL)
using BackendPolicy = METAL::Policy;
using PinnedPolicy = METAL::PinnedPolicy;
using UnifiedPolicy = METAL::UnifiedPolicy;
using TexturePolicy = METAL::TexturePolicy;
#else
#error "No backend selected. Please define USE_CUDA, USE_SYCL, or USE_METAL."
#endif

/**
 * @brief Buffer class with explicit resource management.
 *
 * This buffer implementation eliminates the global state dependency that causes
 * race conditions in multi-threaded, multi-GPU environments. All memory
 * operations are explicitly tied to a specific Resource, making the code
 * thread-safe and suitable for production deployment.
 *
 * @tparam T The element type.
 * @tparam Policy The memory management policy (CUDA, SYCL, or Metal).
 */
enum class BufferAccess : uint8_t { read_only, write_only, read_write };

template <typename T, typename Policy> class Buffer {
protected:
  Resource resource_{}; // The compute resource this buffer is allocated on
  size_t count_{0}; // total number of bytes managed by this buffer (assumed to
                    // be identical for host and device)
  T *device_ptr_{nullptr}; // Device memory pointer
  T *host_ptr_{nullptr};   // Host memory pointer
  void *stream_{nullptr};
  bool sync_{false}; // Sync flag
  StreamType stream_type_{StreamType::Memory};
  BufferAccess buffer_access_{BufferAccess::read_write};

public:
  /**
   * @brief Default constructor creates an empty buffer with no resource.
   * @todo Implement "get_best_available_resource" function.
   *
   * Note: Buffers created this way cannot allocate memory until a resource
   * is explicitly assigned via resize() or assignment.
   */
  Buffer() = default;
  explicit Buffer(size_t count)
      : count_(count), resource_(get_device_resource(0)) {
    if (count_ > 0) {
      stream_ =
          resource_.get_stream(stream_type_); // Use Memory stream by default
      allocate_on_resource(resource_, count_, stream_, sync_);
    }
  }
  explicit Buffer(size_t count, short device_id)
      : count_(count), resource_(get_device_resource(device_id)) {
    if (count_ > 0) {
      stream_ =
          resource_.get_stream(stream_type_); // Use Memory stream by default
      allocate_on_resource(resource_, count_, stream_, sync_);
    }
  }
  // Constructor with resource (uses Memory stream by default)
  explicit Buffer(size_t count, const Resource &resource)
      : count_(count), resource_(resource) {
    if (count_ > 0) {
      stream_ =
          resource_.get_stream(stream_type_); // Use Memory stream by default
      allocate_on_resource(resource_, count_, stream_, sync_);
    }
  }
  explicit Buffer(size_t count, short device_id, void *queue, bool sync = false)
      : count_(count), resource_(get_device_resource(device_id)),
        stream_(queue), sync_(sync) {
    if (count_ > 0) {
      allocate_on_resource(resource_, count_, stream_, sync_);
    }
  }
  // Constructor with resource and queue
  explicit Buffer(size_t count, const Resource &resource, void *queue,
                  bool sync = false)
      : count_(count), resource_(resource), stream_(queue), sync_(sync) {
    if (count_ > 0) {
      allocate_on_resource(resource_, count_, stream_, sync_);
    }
  }
  ~Buffer() { deallocate(); }

  // Type aliases for template compatibility
  using value_type = T;

  // Access methods
  BufferAccess get_access() const { return buffer_access_; }
  /**
   * @brief Copy constructor with explicit resource binding.
   *
   * @param other The source buffer to copy from
   * @param resource The target resource for the new buffer
   */
  Buffer(const Buffer &other, const Resource &resource)
      : count_(other.count_), resource_(resource) {
    if (count_ > 0) {
      stream_ =
          resource_.get_stream(stream_type_); // Use Memory stream by default
      allocate_on_resource(resource_, count_, stream_, sync_);
      copy_device_to_device(other, count_, stream_);
    }
  }

  /**
   * @brief Copy constructor (preserves source resource).
   */
  Buffer(const Buffer &other)
      : resource_(other.resource_), count_(other.count_) {
    if (count_ > 0) {
      stream_ =
          resource_.get_stream(stream_type_); // Use Memory stream by default
      allocate_on_resource(resource_, count_, stream_, sync_);
      copy_device_to_device(other, count_, stream_);
    }
  }

  /**
   * @brief Copy assignment operator.
   */
  Buffer &operator=(const Buffer &other) {
    if (this != &other) {
      deallocate();
      resource_ = other.resource_;
      count_ = other.count_;
      if (count_ > 0) {
        stream_ =
            resource_.get_stream(stream_type_); // Use Memory stream by default
        allocate_on_resource(resource_, count_, stream_, sync_);
        copy_device_to_device(other, count_, stream_);
      }
    }
    return *this;
  }
  // Move constructor
  Buffer(Buffer &&other) noexcept
      : resource_(other.resource_), count_(other.count_),
        device_ptr_(other.device_ptr_), stream_(other.stream_),
        sync_(other.sync_) {
    other.count_ = 0;
    other.device_ptr_ = nullptr;
    other.stream_ = nullptr;
    other.resource_ = Resource{};
    other.sync_ = false;
  }

  // Move assignment
  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      deallocate();
      resource_ = other.resource_;
      count_ = other.count_;
      device_ptr_ = other.device_ptr_;
      stream_ = other.stream_;
      sync_ = other.sync_;
      other.count_ = 0;
      other.device_ptr_ = nullptr;
      other.stream_ = nullptr;
      other.resource_ = Resource{};
    }
    return *this;
  }

  // Set queue for async operations
  void set_queue(void *queue) { stream_ = queue; }
  void *get_queue() const { return stream_; }

  void create(size_t count, const Resource &resource) {
    resource_ = resource;
    count_ = count;
    stream_ =
        resource_.get_stream(stream_type_); // Use Memory stream by default
    allocate_on_resource(resource_, count_, stream_, sync_);
  }

  /**
   * @brief Resizes the buffer and potentially changes the resource.
   *
   * @param count New number of elements
   * @param resource The target resource (optional, uses current resource if not
   * specified)
   */
  void resize(size_t count, const Resource &resource = Resource{}) {
    Resource target_resource = (resource == Resource{}) ? resource_ : resource;
    if (count == count_ && target_resource == resource_) {
      return; // No change needed
    }

    // Get the new queue for the target resource (use Memory stream)
    void *new_queue = target_resource.get_stream(stream_type_);

    // First, try to allocate the new buffer.
    T *new_ptr = nullptr;
    if (count > 0) {
      new_ptr = static_cast<T *>(Policy::allocate(
          target_resource, count * sizeof(T), new_queue, sync_));
      if (!new_ptr) {
        // Allocation failed. The original buffer is untouched.
        // You could throw an exception here to signal the failure.
        // For now, we'll just return, preserving the original buffer.
        return;
      }
    }

    // Copy existing data to new buffer before deallocating old buffer
    if (device_ptr_ && new_ptr && count_ > 0) {
      size_t copy_size = std::min(count_, count) * sizeof(T);

      // Use appropriate copy method based on policy type
      if constexpr (std::is_same_v<Policy, PinnedPolicy>) {
        // For pinned memory, use memcpy since it's host-accessible
        std::memcpy(new_ptr, device_ptr_, copy_size);
      } else {
        // For other policies, use copy_device_to_device if available
        Policy::copy_device_to_device(new_ptr, device_ptr_, copy_size,
                                      new_queue, sync_);
      }
    }

    if (device_ptr_) {
      void *dealloc_queue = stream_;
      if (!dealloc_queue && resource_.type() == ResourceType::SYCL) {
        dealloc_queue = resource_.get_queue(stream_type_);
      }
      Policy::deallocate(device_ptr_, dealloc_queue, sync_);
    }

    // Finally, update the buffer's state to point to the new memory.
    device_ptr_ = new_ptr;
    count_ = count;
    resource_ = target_resource;
    stream_ = new_queue; // Update queue for new resource
  }

  /**
   * @brief Returns the resource this buffer is allocated on.
   */
  const Resource &resource() const { return resource_; }

  /**
   * @brief Checks if the buffer has a valid resource (always true since CPU is
   * always available).
   */
  bool has_valid_resource() const { return true; }

  /**
   * @brief Returns the raw device pointer.
   */
  T *data() { return device_ptr_; }
  const T *data() const { return device_ptr_; }

  void clear() { deallocate(); }

  /**
   * @brief Returns device-qualified pointers for kernel use.
   */
  DEVICE_PTR(T) device_data() {
    return static_cast<DEVICE_PTR(T)>(device_ptr_);
  }

  const DEVICE_PTR(T) device_data() const {
    return static_cast<const DEVICE_PTR(T)>(device_ptr_);
  }

  DEVICE_PTR(T) deviceData() { // Alias
    return static_cast<DEVICE_PTR(T)>(device_ptr_);
  }

  CONSTANT_PTR(T) constant_data() const {
    return static_cast<CONSTANT_PTR(T)>(device_ptr_);
  }

  /**
   * @brief Returns the number of elements.
   */
  size_t size() const { return count_; }

  /**
   * @brief Returns the total size in bytes.
   */
  size_t bytes() const { return count_ * sizeof(T); }

  /**
   * @brief Checks if the buffer is empty.
   */
  bool empty() const { return count_ == 0; }

  /**
   * @brief Copy data to host (synchronous).
   */
  void copy_to_host(std::vector<T> &host_dst) const {
    host_dst.resize(count_);
    copy_to_host(host_dst.data(), count_, true);
  }

  /**
   * @brief Copy data to host with sync control.
   * @param host_dst Destination host pointer
   * @param num_elements Number of elements to copy
   * @param sync Whether to synchronize (default: true for backward
   * compatibility)
   */
  void copy_to_host(T *host_dst, size_t num_elements, bool sync = true) const {
    if (num_elements > count_) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Copy size exceeds buffer size");
    }
    if (!device_ptr_) {
      ARBD_Exception(ExceptionType::ValueError, "Cannot copy from null buffer");
    }
    // Ensure we have a valid stream for the operation
    void *active_stream = stream_;
    if (!active_stream) {
      active_stream = resource_.get_stream(stream_type_);
    }

    Policy::copy_to_host(host_dst, device_ptr_, num_elements * sizeof(T),
                         active_stream, sync);
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) &&               \
    !defined(__METAL_VERSION__)
    LOGTRACE("Copied {} bytes to host from {} ({}sync)",
             num_elements * sizeof(T), resource_.toString(), sync ? "" : "a");
#endif
  }

  /**
   * @brief Copy data to host asynchronously.
   * @param host_dst Destination host pointer
   * @param num_elements Number of elements to copy
   * @return void* Stream handle for synchronization
   */
  void *copy_to_host_async(T *host_dst, size_t num_elements) const {
    copy_to_host(host_dst, num_elements, false);
    // Return the active stream (may be different from stream_ if it was null)
    return stream_ ? stream_ : resource_.get_stream(stream_type_);
  }

  /**
   * @brief Copy data to host synchronously (explicit).
   */
  void copy_to_host_sync(T *host_dst, size_t num_elements) const {
    copy_to_host(host_dst, num_elements, true);
  }

  /**
   * @brief Copy data from host (synchronous with resize).
   */
  void copy_from_host(const std::vector<T> &host_src) {
    if (host_src.size() != count_) {
      resize(host_src.size());
    }
    copy_from_host(host_src.data(), host_src.size(), true);
  }

  /**
   * @brief Copy data from host with sync control.
   * @param host_src Source host pointer
   * @param num_elements Number of elements to copy
   * @param sync Whether to synchronize (default: true for backward
   * compatibility)
   */
  void copy_from_host(const T *host_src, size_t num_elements,
                      bool sync = true) {
    if (num_elements > count_) {
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) &&               \
    !defined(__METAL_VERSION__)
      ARBD_Exception(ExceptionType::ValueError,
                     "Copy size exceeds buffer size");
#else
      // Device code: clamp to safe size
      num_elements = (count_ < num_elements) ? count_ : num_elements;
      if (num_elements == 0)
        return;
#endif
    }
    if (!device_ptr_) {
      ARBD_Exception(ExceptionType::ValueError, "Cannot copy to null buffer");
    }
    if (!host_src) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Cannot copy from null host pointer");
    }
    // Ensure we have a valid stream for the operation
    void *active_stream = stream_;
    if (!active_stream) {
      active_stream = resource_.get_stream(stream_type_);
    }

    Policy::copy_from_host(device_ptr_, host_src, num_elements * sizeof(T),
                           active_stream, sync);
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) &&               \
    !defined(__METAL_VERSION__)
    LOGTRACE("Copied {} bytes from host to {} ({}sync)",
             num_elements * sizeof(T), resource_.toString(), sync ? "" : "a");
#endif
  }

  /**
   * @brief Copy data from host asynchronously.
   * @param host_src Source host pointer
   * @param num_elements Number of elements to copy
   * @return void* Stream handle for synchronization
   */
  void *copy_from_host_async(const T *host_src, size_t num_elements) {
    copy_from_host(host_src, num_elements, false);
    // Return the active stream (may be different from stream_ if it was null)
    return stream_ ? stream_ : resource_.get_stream(stream_type_);
  }

  /**
   * @brief Copy data from host synchronously (explicit).
   */
  void copy_from_host_sync(const T *host_src, size_t num_elements) {
    copy_from_host(host_src, num_elements, true);
  }

  /**
   * @brief Copy between device buffers with sync control.
   * @param src Source buffer
   * @param num_elements Number of elements to copy
   * @param sync Whether to synchronize (default: true for backward
   * compatibility)
   */
  void copy_device_to_device(const Buffer &src, size_t num_elements,
                             bool sync = true) {
    if (num_elements > count_ || num_elements > src.count_) {
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) &&               \
    !defined(__METAL_VERSION__)
      ARBD_Exception(ExceptionType::ValueError,
                     "Copy size exceeds buffer size");
#else
      // Device code: clamp to safe size
      num_elements = (count_ < num_elements) ? count_ : num_elements;
      num_elements = (src.count_ < num_elements) ? src.count_ : num_elements;
      if (num_elements == 0)
        return;
#endif
    }
    if (!device_ptr_ || !src.device_ptr_) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Cannot copy with null buffer(s)");
    }
    // Ensure we have a valid stream for the operation
    void *active_stream = stream_;
    if (!active_stream) {
      active_stream = resource_.get_stream(stream_type_);
    }

    bool use_sync = sync; // Use provided sync parameter
    Policy::copy_device_to_device(device_ptr_, src.device_ptr_,
                                  num_elements * sizeof(T), active_stream,
                                  use_sync);
#if !defined(__CUDA_ARCH__) && !defined(__SYCL_DEVICE_ONLY__) &&               \
    !defined(__METAL_VERSION__)
    LOGTRACE("Copied {} bytes device-to-device from {} to {} ({}sync)",
             num_elements * sizeof(T), src.resource_.toString(),
             resource_.toString(), use_sync ? "" : "a");
#endif
  }

  /**
   * @brief Copy between device buffers asynchronously.
   * @param src Source buffer
   * @param num_elements Number of elements to copy
   * @return void* Stream handle for synchronization
   */
  void *copy_device_to_device_async(const Buffer &src, size_t num_elements) {
    copy_device_to_device(src, num_elements, false);
    // Return the active stream (may be different from stream_ if it was null)
    return stream_ ? stream_ : resource_.get_stream(stream_type_);
  }

  /**
   * @brief Copy between device buffers synchronously.
   */
  void copy_device_to_device_sync(const Buffer &src, size_t num_elements) {
    copy_device_to_device(src, num_elements, true);
  }

  /**
   * @brief Copy data to host asynchronously and return an Event.
   * @param host_dst Destination host pointer
   * @param num_elements Number of elements to copy
   * @return Event for synchronization
   */
  Event copy_to_host_event(T *host_dst, size_t num_elements) const {
    copy_to_host(host_dst, num_elements, false);
    return create_event_from_stream();
  }

  /**
   * @brief Copy data from host asynchronously and return an Event.
   * @param host_src Source host pointer
   * @param num_elements Number of elements to copy
   * @return Event for synchronization
   */
  Event copy_from_host_event(const T *host_src, size_t num_elements) {
    copy_from_host(host_src, num_elements, false);
    return create_event_from_stream();
  }

  /**
   * @brief Copy between device buffers asynchronously and return an Event.
   * @param src Source buffer
   * @param num_elements Number of elements to copy
   * @return Event for synchronization
   */
  Event copy_device_to_device_event(const Buffer &src, size_t num_elements) {
    copy_device_to_device(src, num_elements, false);
    return create_event_from_stream();
  }

private:
  /**
   * @brief Create an Event from the current stream.
   */
  Event create_event_from_stream() const {
    if (!stream_) {
      return Event(nullptr, resource_);
    }

#ifdef USE_CUDA
    if (resource_.type() == ResourceType::CUDA) {
      cudaEvent_t event;
      CUDA_CHECK(cudaEventCreate(&event));
      CUDA_CHECK(cudaEventRecord(event, static_cast<cudaStream_t>(stream_)));
      return Event(event, resource_);
    }
#endif

#ifdef USE_SYCL
    if (resource_.type() == ResourceType::SYCL) {
      // For AdaptiveCpp, submit a simple single_task operation
      sycl::queue &q = *static_cast<sycl::queue *>(stream_);
      auto event = q.submit([&](sycl::handler &h) {
        h.single_task([]() {
          // Empty single task to create a valid event
        });
      });
      return Event(event, resource_);
    }
#endif

#ifdef USE_METAL
    if (resource_.type() == ResourceType::METAL) {
      // Create a Metal command buffer for the event
      auto &device = METAL::Manager::get_current_device();
      auto &queue = device.get_next_queue();
      void *cmd_buffer = queue.create_command_buffer();
      if (cmd_buffer) {
        // Create a Metal event wrapper around the command buffer
        METAL::Event metal_event(cmd_buffer);
        // Commit the command buffer to make it valid
        metal_event.commit();
        return Event(std::move(metal_event), resource_);
      }
      return Event(nullptr, resource_);
    }
#endif

    // Host or unsupported device
    return Event(nullptr, resource_);
  }

public:
#ifdef USE_METAL
  /**
   * @brief Bind buffer to Metal compute encoder.
   */
  void bind_to_encoder(MTL::ComputeCommandEncoder *encoder,
                       uint32_t index) const {
    auto *metal_buffer = METAL::Manager::get_metal_buffer_from_ptr(device_ptr_);
    if (!metal_buffer) {
      ARBD_Exception(ExceptionType::MetalRuntimeError,
                     "Failed to get Metal buffer for binding at index {}",
                     index);
    }
    LOGINFO("Binding Metal buffer {} to encoder at index {}",
            (void *)metal_buffer, index);
    encoder->setBuffer(metal_buffer, 0, index);
  }
#endif
  static Buffer create(size_t count, int device_id,
                       const Buffer *pool = nullptr) {
    Buffer new_buffer;
    new_buffer.resource_ =
        pool ? pool->resource() : get_device_resource(device_id);

    // Allocate memory directly
    new_buffer.device_ptr_ = static_cast<T *>(Policy::allocate(
        new_buffer.resource_, count * sizeof(T), nullptr, true));
    new_buffer.count_ = count;

    return new_buffer;
  }

private:
  /**
   * @brief Get the best available resource (prioritizes GPU devices over CPU)
   * @todo Implement this.
   */

  static Resource get_device_resource(short device_id) {
#ifdef USE_SYCL
    return Resource{ResourceType::SYCL, static_cast<idx_t>(device_id)};
#elif defined(USE_CUDA)
    return Resource{ResourceType::CUDA, static_cast<idx_t>(device_id)};
#elif defined(USE_METAL)
    return Resource{ResourceType::METAL, static_cast<idx_t>(device_id)};
#else
    return Resource{ResourceType::CPU, 0};
#endif
  };

  void allocate_on_resource(const Resource &resource, size_t count, void *queue,
                            bool sync) {
    count_ = count;
    if (count_ > 0) {
      // Use the resource-aware allocation method
      device_ptr_ = static_cast<T *>(
          Policy::allocate(resource, count_ * sizeof(T), queue, sync));
#ifdef HOST_GUARD
      if (!device_ptr_) {
        ARBD_Exception(ExceptionType::RuntimeError,
                       "Failed to allocate {} bytes on {}", count_ * sizeof(T),
                       resource.toString());
      }

      LOGTRACE("Allocated {} bytes on {}", count_ * sizeof(T),
               resource.toString());
#endif
    }
  }

  void deallocate() {
    if (device_ptr_) {
      void *dealloc_queue = stream_;
      if (!dealloc_queue && resource_.type() == ResourceType::SYCL) {
        dealloc_queue = resource_.get_queue(stream_type_);
      }
      Policy::deallocate(device_ptr_, dealloc_queue, sync_);
      device_ptr_ = nullptr;
#ifdef HOST_GUARD
      LOGTRACE("Deallocated buffer on {}", resource_.toString());
#endif
    }
    count_ = 0;
  }
};

template <typename T, typename Policy>
class PINBuffer : public Buffer<T, Policy> {
public:
  PINBuffer(size_t count, const Resource &resource, void *queue = nullptr,
            bool sync = true)
      : Buffer<T, Policy>(count, resource, queue, sync) {}
  void upload_to_device(const T *host_src, size_t num_elements) {
    Policy::upload_to_device(this->device_ptr_, host_src,
                             num_elements * sizeof(T), this->resource_,
                             this->stream_);
  }
  void download_from_device(T *host_dst, size_t num_elements) {
    Policy::download_from_device(host_dst, this->device_ptr_,
                                 num_elements * sizeof(T), this->resource_,
                                 this->stream_);
  }

  // Override copy_from_host to use pinned buffer's own methods
  void copy_from_host(const T *host_src, size_t num_elements) {
    upload_to_device(host_src, num_elements);
  }

  // Override copy_to_host to use pinned buffer's own methods
  void copy_to_host(T *host_dst, size_t num_elements) {
    download_from_device(host_dst, num_elements);
  }
  void copy_device_to_device(void *dst, const void *src, size_t num_elements,
                             void *queue = nullptr, bool sync = false) {
    std::memcpy(dst, src, num_elements * sizeof(T));
  }
};

template <typename T, typename Policy>
class USMBuffer : public Buffer<T, Policy> {
public:
  USMBuffer(size_t count, const Resource &resource, void *queue = nullptr,
            bool sync = true)
      : Buffer<T, Policy>(count, resource, queue, sync), capacity_(count),
        size_(count) {}

  // multi-device constructor with capacity
  USMBuffer(size_t count, size_t capacity,
            const std::vector<Resource> &resources, void *queue = nullptr,
            bool sync = true)
      : Buffer<T, Policy>(capacity, // Allocate capacity amount of memory
                          resources.empty() ? Resource{} : resources.front(),
                          queue, sync),
        devices_(resources), capacity_(capacity), size_(count) {
    // Override the Buffer's internal count to reflect actual size, not capacity
    this->count_ = count;
  }

  // Existing single-device helpers
  void prefetch(int device_id = -1, void *queue = nullptr) {
    Policy::prefetch(this->data(), this->bytes(), device_id, this->get_queue());
  }
  void mem_advise(int advice, int device_id = -1) {
    Policy::mem_advise(this->data(), this->bytes(), advice, device_id);
  }

  // multi-device helpers
  void set_devices(const std::vector<Resource> &resources) {
    devices_ = resources;
  }
  void prefetch_devices(void *queue = nullptr) {
    for (const auto &r : devices_)
      Policy::prefetch(this->data(), this->bytes(), int(r.id()), queue);
  }
  void advise_preferred_for_all(int advice) {
    for (const auto &r : devices_)
      Policy::mem_advise(this->data(), this->bytes(), advice, int(r.id()));
  }

  // Expandable features
  void reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
      // Reallocate with new capacity
      void *new_ptr = Policy::allocate(
          this->resource(), new_capacity * sizeof(T), this->get_queue(), true);
      if (new_ptr) {
        // Copy existing data
        if (this->device_ptr_) {
          Policy::copy_device_to_device(new_ptr, this->device_ptr_,
                                        size_ * sizeof(T), this->get_queue(),
                                        false);
          Policy::deallocate(this->device_ptr_, this->get_queue(), true);
        }
        // Update buffer state
        this->device_ptr_ = static_cast<T *>(new_ptr);
        capacity_ = new_capacity;
      }
    }
  }

  void resize(size_t new_size, int device_id = -1, void *queue = nullptr) {
    if (new_size > capacity_) {
      // Need to expand capacity
      size_t new_capacity =
          std::max(new_size, capacity_ * 2); // Double capacity strategy
      reserve(new_capacity);
    }
    size_ = new_size;
    // Update the base class count_ to match the new size
    this->count_ = new_size;

    // Apply memory advice if device specified
    if (device_id >= 0) {
      Policy::mem_advise(this->data(), this->size() * sizeof(T),
                         get_default_advice(), device_id);
    }
  }

  void expand(size_t additional_elements) {
    resize(size_ + additional_elements);
  }

  // Capacity management
  size_t capacity() const { return capacity_; }
  size_t available_space() const { return capacity_ - size_; }
  bool can_expand(size_t additional_elements) const {
    return (size_ + additional_elements) <= capacity_;
  }

  // Memory advice with offset support
  void advise_range(size_t offset_elements, size_t num_elements, int device_id,
                    int advice) {
    size_t byte_offset = offset_elements * sizeof(T);
    size_t byte_size = num_elements * sizeof(T);
    Policy::mem_advise(reinterpret_cast<char *>(this->data()) + byte_offset,
                       byte_size, advice, device_id);
  }

  void prefetch_range(size_t offset_elements, size_t num_elements,
                      int device_id, void *queue = nullptr) {
    size_t byte_offset = offset_elements * sizeof(T);
    size_t byte_size = num_elements * sizeof(T);
    Policy::prefetch(reinterpret_cast<char *>(this->data()) + byte_offset,
                     byte_size, device_id, queue);
  }

#ifdef USE_CUDA
  void set_preferred_location_all() {
    for (const auto &r : devices_)
      Policy::mem_advise(this->data(), this->bytes(),
                         cudaMemAdviseSetPreferredLocation, int(r.id()));
  }
  void set_accessed_by_all() {
    for (const auto &r : devices_)
      Policy::mem_advise(this->data(), this->bytes(),
                         cudaMemAdviseSetAccessedBy, int(r.id()));
  }
#endif

private:
  std::vector<Resource> devices_;
  size_t capacity_; // Total allocated capacity
  size_t size_;     // Currently used size

  int get_default_advice() {
#ifdef USE_CUDA
    return cudaMemAdviseSetPreferredLocation;
#elif defined(USE_SYCL)
    return 0; // SYCL doesn't have equivalent advice
#else
    return 0;
#endif
  }
};
// ============================================================================
// Convenient Aliases
// ============================================================================
struct TextureFormat {
#if defined(USE_CUDA)
  cudaChannelFormatDesc cuda_format;
#elif defined(USE_SYCL)
  sycl::image_channel_order sycl_order;
  sycl::image_channel_type sycl_type;
#elif defined(USE_METAL)
  MTL::PixelFormat metal_format;
#endif
};
template <typename T>

class TextureBuffer {
private:
  void *texture_obj_ptr_{nullptr};
  Resource resource_{};
  size_t width_{0}, height_{0}, depth_{0};
  StreamType stream_type_{StreamType::Memory};

public:
  TextureBuffer() = default;

  explicit TextureBuffer(const Resource &resource, size_t width, size_t height,
                         size_t depth, TextureFormat format)
      : resource_(resource), width_(width), height_(height), depth_(depth) {
    if (width_ > 0) {
#if defined(USE_CUDA)
      texture_obj_ptr_ =
          TexturePolicy::allocate(resource_, width_, height_, depth_,
                                  format.cuda_format, nullptr, false);
#elif defined(USE_SYCL)
      texture_obj_ptr_ =
          TexturePolicy::allocate(resource_, width_, height_, depth_,
                                  format.sycl_order, format.sycl_type);
#elif defined(USE_METAL)
      texture_obj_ptr_ = TexturePolicy::allocate(resource_, width_, height_,
                                                 depth_, format.metal_format);
#endif
    }
  }

  ~TextureBuffer() {
    if (texture_obj_ptr_) {
      TexturePolicy::deallocate(texture_obj_ptr_);
    }
  }

  // Move constructors/assignment (as before)
  TextureBuffer(TextureBuffer &&other) noexcept;
  TextureBuffer &operator=(TextureBuffer &&other) noexcept;

  // Disable copy
  TextureBuffer(const TextureBuffer &) = delete;
  TextureBuffer &operator=(const TextureBuffer &) = delete;

  void copy_from_buffer(const Buffer<T, TexturePolicy> &src_buffer) {
    if (!texture_obj_ptr_ || !src_buffer.data())
      return;
#if defined(USE_CUDA)
    TexturePolicy::copy_from_buffer(texture_obj_ptr_, src_buffer.data(), width_,
                                    height_, depth_, sizeof(T));
#elif defined(USE_SYCL)
    TexturePolicy::copy_from_buffer(texture_obj_ptr_, src_buffer.data(),
                                    src_buffer.bytes(), resource_);
#elif defined(USE_METAL)
    TexturePolicy::copy_from_buffer(texture_obj_ptr_, src_buffer.data(),
                                    src_buffer.bytes(), resource_);
#endif
  }

  void *get_native_handle() const { return texture_obj_ptr_; }

  // ... width(), height(), depth() methods ...
};
/**
 * @brief A convenient alias for device buffers using the active backend.
 *
 * This hides the policy template parameter since it's determined at compile
 * time by the backend selection (USE_CUDA, USE_SYCL, or USE_METAL).
 */
template <typename T> using DeviceBuffer = Buffer<T, BackendPolicy>;
template <typename T> using PinnedBuffer = PINBuffer<T, PinnedPolicy>;
template <typename T> using UnifiedBuffer = USMBuffer<T, UnifiedPolicy>;
template <typename T> using HostBuffer = Buffer<T, CPU::Policy>;

// Backend-specific aliases for explicit use cases
#ifdef USE_CUDA
template <typename T> using CudaBuffer = Buffer<T, CUDA::Policy>;
#endif

#ifdef USE_SYCL
template <typename T> using SyclBuffer = Buffer<T, SYCL::Policy>;
#endif

#ifdef USE_METAL
template <typename T> using MetalBuffer = Buffer<T, METAL::Policy>;
#endif

/**
 * @brief A convenient alias for the Buffer class using the active backend
 * policy.
 */

template <typename T> struct is_device_buffer : std::false_type {};

template <typename T>
struct is_device_buffer<DeviceBuffer<T>> : std::true_type {};

template <typename T>
constexpr bool is_device_buffer_v = is_device_buffer<std::decay_t<T>>::value;

template <typename T> struct is_pinned_buffer : std::false_type {};

template <typename T>
struct is_pinned_buffer<PinnedBuffer<T>> : std::true_type {};

template <typename T>
constexpr bool is_pinned_buffer_v = is_pinned_buffer<std::decay_t<T>>::value;

template <typename T> struct is_unified_buffer : std::false_type {};

template <typename T>
struct is_unified_buffer<UnifiedBuffer<T>> : std::true_type {};

template <typename T>
constexpr bool is_unified_buffer_v = is_unified_buffer<std::decay_t<T>>::value;

template <typename T> struct is_string : std::false_type {};

template <> struct is_string<std::string> : std::true_type {};

template <> struct is_string<const std::string> : std::true_type {};

template <> struct is_string<const char *> : std::true_type {};

template <> struct is_string<char *> : std::true_type {};

template <typename T>
constexpr bool is_string_v = is_string<std::decay_t<T>>::value;

// Helper function to get buffer pointers from tuples for legacy kernel launches
template <typename... Buffers, std::size_t... Is>
auto get_buffer_tuples_impl(const std::tuple<Buffers...> &buffer_tuple,
                            std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(buffer_tuple).device_data()...);
}

template <typename... Buffers>
auto get_buffer_tuples(const std::tuple<Buffers...> &buffer_tuple) {
  return get_buffer_tuples_impl(buffer_tuple,
                                std::make_index_sequence<sizeof...(Buffers)>{});
}

template <typename T> constexpr auto get_buffer_pointer(T &&arg) {
  if constexpr (is_device_buffer_v<std::decay_t<T>>) {
    auto ptr = arg.device_data();
    using ValueType = typename std::decay_t<T>::value_type;
    auto access = arg.get_access();

#if defined(USE_CUDA) || defined(USE_METAL)
    // Always return non-const pointer to avoid template deduction conflicts
    // The constness is handled at the kernel level
    return static_cast<ValueType *__restrict__>(ptr);
#elif defined(USE_SYCL)
    // Same logic for SYCL USM
    return static_cast<ValueType *__restrict__>(ptr);
#else
    // Fallback for other backends
    return ptr;
#endif
  } else {
    return std::forward<T>(arg);
  }
}

template <typename T> constexpr auto get_buffer_value(T &&arg) {
  return get_buffer_pointer(std::forward<T>(arg));
}
} // namespace ARBD

#endif // __METAL_VERSION__
