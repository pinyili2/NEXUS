// src/Backend/Resource.h
#pragma once

#include "ARBDException.h"
#include "Header.h"

#ifdef USE_CUDA
#include "CUDA/CUDAStreams.h"
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#include "SYCL/SYCLQueues.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

#ifdef HOST_GUARD
#include <memory>
#include <vector>
#endif

namespace ARBD {

/**
 * @brief Enumeration of supported compute resource types.
 */
enum class ResourceType : uint8_t {
  CPU = 0,  ///< CPU resource (default)
  CUDA = 1, ///< NVIDIA CUDA GPU
  SYCL = 2, ///< SYCL-compatible device
  METAL = 3 ///< Apple Metal GPU
};
// Compile-time default backend selection
#ifdef USE_SYCL
constexpr ResourceType DEFAULT_RESOURCE_TYPE = ResourceType::SYCL;
#elif defined(USE_CUDA)
constexpr ResourceType DEFAULT_RESOURCE_TYPE = ResourceType::CUDA;
#elif defined(USE_METAL)
constexpr ResourceType DEFAULT_RESOURCE_TYPE = ResourceType::METAL;
#else
constexpr ResourceType DEFAULT_RESOURCE_TYPE = ResourceType::CPU;
#endif
/**
 * @brief Stream type enumeration for different use cases
 *
 * Each StreamType maps to a dedicated stream ID for predictable performance:
 * - Compute operations always use stream 0
 * - Memory transfers always use stream 1
 * - Default operations use stream 2
 * - Optional async work uses stream 3
 */
enum class StreamType {
  Compute = 0, ///< Dedicated compute stream (stream 0)
  Memory = 1,  ///< Dedicated memory transfer stream (stream 1)
  Default = 2, ///< Default/synchronous stream (stream 2)
  Optional = 3 ///< Additional async stream (stream 3)
};

// namespace ARBD
/**
 * @brief A production-ready resource identifier that owns its streams/queues.
 *
 * This class provides explicit device management with owned stream pools,
 * eliminating Manager dependencies and enabling self-contained Resources
 * suitable for multi-GPU and future multi-node operations.
 */
class Resource {
private:
  ResourceType type_{DEFAULT_RESOURCE_TYPE};
  short id_{0};

// Stream pools owned by Resource (lazy initialization)
#ifdef USE_CUDA
  mutable std::shared_ptr<CUDA::InitStreams> streams_;
#elif defined(USE_SYCL)
  mutable std::shared_ptr<SYCL::InitQueues> streams_;
#endif

  // Device validation state
  mutable bool device_verified_{false};
  mutable bool device_available_{false};

public:
  /**
   * @brief Default constructor creates a resource using the compile-time
   * selected backend.
   */
  HOST DEVICE constexpr Resource() = default;

  HOST DEVICE constexpr Resource(short device_id) : id_(device_id) {}

  /**
   * @brief Construct a resource with specified type and optional ID.
   */
  HOST DEVICE constexpr Resource(ResourceType resource_type,
                                 short device_id = 0)
      : type_(resource_type), id_(device_id) {}

  // Destructor, copy, move operations
  ~Resource() = default;
  Resource(const Resource &other)
      : type_(other.type_), id_(other.id_),
        device_verified_(other.device_verified_),
        device_available_(other.device_available_),
        streams_(other.streams_) // Share stream pools
        {};
  Resource &operator=(const Resource &other) {
    if (this != &other) {
      type_ = other.type_;
      id_ = other.id_;
      device_verified_ = other.device_verified_;
      device_available_ = other.device_available_;
      streams_ = other.streams_; // Share stream pools
    }
    return *this;
  }
  Resource(Resource &&other) noexcept = default;
  Resource &operator=(Resource &&other) noexcept = default;

  // Factory methods with validation
  static Resource create_cuda_device(short device_id);
  static Resource create_sycl_device(short device_id);

  // Device context management
  void activate() const;
  void ensure_context() const;
  bool verify_device() const;

  // Stream management (Resource owns streams)
  void *get_stream(StreamType stream_type = StreamType::Compute) const {
    return get_stream_impl(stream_type);
  }

  void *get_stream(size_t stream_id,
                   StreamType stream_type = StreamType::Compute) const {
    return get_stream_impl(stream_id, stream_type);
  }
  void synchronize_streams() const;

  // Provide aliases for stream/queue terminology
  void *get_queue(StreamType stream_type = StreamType::Compute) const {
    return get_stream(stream_type);
  }
  void *get_queue(size_t stream_id,
                  StreamType stream_type = StreamType::Compute) const {
    return get_stream(stream_id, stream_type);
  }
  void synchronize_queues() const { synchronize_streams(); }

// Backend-specific resource access (for advanced users)
#ifdef USE_CUDA
  cudaStream_t get_cuda_stream(size_t id = 0) const {
    if (type_ != ResourceType::CUDA)
      throw std::runtime_error("Not a CUDA resource");
    return static_cast<cudaStream_t>(get_stream_impl(id));
  }
#endif

#ifdef USE_SYCL
  sycl::queue &get_sycl_queue(size_t id = 0) const {
    if (type_ != ResourceType::SYCL)
      throw std::runtime_error("Not a SYCL resource");
    return *static_cast<sycl::queue *>(get_stream_impl(id));
  }
#endif

  // Properties
  ResourceType type() const { return type_; }
  short id() const { return id_; }

  /**
   * @brief Get a human-readable string for the resource type.
   */
  HOST DEVICE constexpr const char *getTypeString() const {
    switch (type_) {
    case ResourceType::CPU:
      return "CPU";
    case ResourceType::CUDA:
      return "CUDA";
    case ResourceType::SYCL:
      return "SYCL";
    case ResourceType::METAL:
      return "METAL";
    default:
      return "Unknown";
    }
  }

  /**
   * @brief Get a string representation of this resource.
   */
  std::string toString() const {
    return std::string(getTypeString()) + "[" + std::to_string(id_) + "]";
  }

  /**
   * @brief Check if the resource supports asynchronous operations.
   */
  HOST DEVICE constexpr bool supports_async() const {
    return type_ != ResourceType::CPU;
  }

  /**
   * @brief Check if this resource represents a device (GPU).
   */
  HOST DEVICE constexpr bool is_device() const {
    return type_ == ResourceType::CUDA || type_ == ResourceType::SYCL ||
           type_ == ResourceType::METAL;
  }

  /**
   * @brief Check if this resource represents a host (CPU).
   */
  HOST DEVICE constexpr bool is_host() const {
    return type_ == ResourceType::CPU;
  }

  /**
   * @brief Equality comparison operator.
   */
  HOST DEVICE constexpr bool operator==(const Resource &other) const {
    return type_ == other.type_ && id_ == other.id_;
  }

  HOST DEVICE constexpr bool operator!=(const Resource &other) const {
    return !(*this == other);
  }

  HOST DEVICE constexpr bool operator<(const Resource &other) const {
    if (type_ != other.type_) {
      return static_cast<uint8_t>(type_) < static_cast<uint8_t>(other.type_);
    }
    return id_ < other.id_;
  }

  /**
   * @brief Check if this resource can access peer memory from another resource.
   */
  bool can_access_peer(const Resource &other) const;

  /**
   * @brief Validate that this resource exists and is accessible.
   */
  void validate() const;

private:
  void ensure_queues_initialized() const;

private:
  void *get_stream_impl(StreamType stream_type) const;
  void *get_stream_impl(size_t stream_id,
                        StreamType stream_type = StreamType::Compute) const;
};

/**
 * @brief A collection of resources. Stored on CPU only.
 */
#ifdef HOST_GUARD
struct ResourceCollection {
  std::vector<Resource> resources;
};
#endif

} // namespace ARBD
