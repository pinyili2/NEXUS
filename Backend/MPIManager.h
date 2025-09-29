// src/Backend/MPI/MPIManager.h
// src/Backend/MPI/MPIManager.h
#pragma once
#ifndef __METAL_VERSION__
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include <chrono>
#include <concepts>
#include <map>
#include <memory>
#include <mpi.h>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace ARBD::MPI {

/**
 * @brief MPI GPU-aware capability states
 */
enum class MPIAwareState {
  Unknown,   ///< Capability not determined
  Yes,       ///< GPU-aware MPI is available
  No,        ///< GPU-aware MPI is not available
  ForcedYes, ///< Forced to use GPU-aware MPI
  ForcedNo   ///< Forced to not use GPU-aware MPI
};

/**
 * @brief Communication protocols for MPI operations
 */
enum class CommunicationProtocol {
  CopyToHost, ///< Copy data to host before MPI operation
  DirectGPU   ///< Direct GPU-to-GPU communication (assumes GPU-aware MPI)
};

/**
 * @brief Concept for MPI-supported data types
 */
template <typename T>
concept MPISupported =
    std::is_arithmetic_v<T> &&
    (std::is_same_v<T, float> || std::is_same_v<T, double> ||
     std::is_same_v<T, int> || std::is_same_v<T, long> ||
     std::is_same_v<T, long long> || std::is_same_v<T, unsigned int> ||
     std::is_same_v<T, unsigned long> ||
     std::is_same_v<T, unsigned long long> || std::is_same_v<T, char> ||
     std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>);

/**
 * @brief MPI Manager for Multi-Backend GPU collective operations
 *
 * Designed for:
 * - Single node multi-GPU setups (for initial testing)
 * - Spatial decomposition with one patch per GPU
 * - Assumes GPU-aware MPI when using GPU resources
 * - Compatible with ARBD Resource system
 */
class Manager {
private:
  MPI_Comm comm_;
  int rank_; ///< Process rank (maps to GPU ID)
  int size_; ///< Total number of processes/GPUs
  bool initialized_ = false;
  bool assume_gpu_aware_ =
      true; ///< Assume GPU-aware MPI for supercomputer environments

  // GPU-aware MPI capability detection
  MPIAwareState cuda_aware_state_ = MPIAwareState::Unknown;
  MPIAwareState rocm_aware_state_ = MPIAwareState::Unknown;
  std::optional<MPIAwareState> forced_state_ = std::nullopt;

  // Communication protocol cache per resource type
  mutable std::unordered_map<ResourceType, CommunicationProtocol>
      protocol_cache_;

  // Staging buffers for host-based communication (fallback)
  mutable std::map<std::tuple<size_t, size_t, ResourceType>,
                   std::unique_ptr<PinnedBuffer<char>>>
      staging_buffers_;

  // Async operation tracking
  struct AsyncOperation {
    MPI_Request request;
    Event completion_event;
    std::unique_ptr<PinnedBuffer<char>> staging_buffer;
    ResourceType resource_type;
    std::chrono::high_resolution_clock::time_point start_time;
  };
  mutable std::vector<AsyncOperation> pending_operations_;

public:
  /**
   * @brief Get singleton instance
   */
  static Manager &instance() {
    static Manager instance_;
    return instance_;
  }

  /**
   * @brief Initialize MPI with thread support
   */
  void init(bool assume_gpu_aware = true);

  /**
   * @brief Finalize MPI and cleanup resources
   */
  void finalize();

  /**
   * @brief Check if MPI is initialized
   */
  bool is_initialized() const { return initialized_; }

  // ==================== Collective Operations for Spatial Decomposition
  // ====================

  /**
   * @brief All-reduce operation (e.g., for global force summation)
   */
  template <MPISupported T>
  Event allReduce(DeviceBuffer<T> &buffer, idx_t count,
                  const Resource &resource, MPI_Op op = MPI_SUM) {
    auto protocol = get_communication_protocol(resource);

    if (protocol == CommunicationProtocol::DirectGPU) {
      MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buffer.data(), count,
                              get_mpi_type<T>(), op, comm_));
    } else {
      T *host_buffer = get_staging_buffer<T>(count, resource);
      buffer.copy_to_host(host_buffer, count);
      MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, host_buffer, count,
                              get_mpi_type<T>(), op, comm_));
      buffer.copy_from_host(host_buffer, count);
    }

    return Event(nullptr, resource);
  }

  /**
   * @brief Broadcast operation (e.g., for global parameters)
   */
  template <MPISupported T>
  Event broadcast(DeviceBuffer<T> &buffer, idx_t count, int root,
                  const Resource &resource) {
    auto protocol = get_communication_protocol(resource);

    if (protocol == CommunicationProtocol::DirectGPU) {
      MPI_CHECK(
          MPI_Bcast(buffer.data(), count, get_mpi_type<T>(), root, comm_));
    } else {
      T *host_buffer = get_staging_buffer<T>(count, resource);

      if (rank_ == root) {
        buffer.copy_to_host(host_buffer, count);
      }

      MPI_CHECK(MPI_Bcast(host_buffer, count, get_mpi_type<T>(), root, comm_));

      if (rank_ != root) {
        buffer.copy_from_host(host_buffer, count);
      }
    }

    return Event(nullptr, resource);
  }

  /**
   * @brief All-gather operation (e.g., for collecting boundary data)
   */
  template <MPISupported T>
  Event allGather(DeviceBuffer<T> &send_buffer, DeviceBuffer<T> &recv_buffer,
                  idx_t count, const Resource &resource) {
    auto protocol = get_communication_protocol(resource);

    if (protocol == CommunicationProtocol::DirectGPU) {
      MPI_CHECK(MPI_Allgather(send_buffer.data(), count, get_mpi_type<T>(),
                              recv_buffer.data(), count, get_mpi_type<T>(),
                              comm_));
    } else {
      T *send_host = get_staging_buffer<T>(count, resource);
      T *recv_host = get_staging_buffer<T>(count * size_, resource, 1);

      send_buffer.copy_to_host(send_host, count);
      MPI_CHECK(MPI_Allgather(send_host, count, get_mpi_type<T>(), recv_host,
                              count, get_mpi_type<T>(), comm_));
      recv_buffer.copy_from_host(recv_host, count * size_);
    }

    return Event(nullptr, resource);
  }

  // ==================== Point-to-Point for Neighbor Communication
  // ====================

  /**
   * @brief Send boundary data to neighboring patches (async)
   */
  template <MPISupported T>
  Event send_boundary(DeviceBuffer<T> &buffer, idx_t count, int dest_rank,
                      int tag, const Resource &resource) {
    auto protocol = get_communication_protocol(resource);

    if (protocol == CommunicationProtocol::DirectGPU) {
      return perform_direct_send(buffer, count, dest_rank, tag, resource);
    } else {
      return perform_staged_send(buffer, count, dest_rank, tag, resource);
    }
  }

  /**
   * @brief Receive boundary data from neighboring patches (async)
   */
  template <MPISupported T>
  Event recv_boundary(DeviceBuffer<T> &buffer, idx_t count, int source_rank,
                      int tag, const Resource &resource) {
    auto protocol = get_communication_protocol(resource);

    if (protocol == CommunicationProtocol::DirectGPU) {
      return perform_direct_recv(buffer, count, source_rank, tag, resource);
    } else {
      return perform_staged_recv(buffer, count, source_rank, tag, resource);
    }
  }

  /**
   * @brief Exchange boundary data with all neighbors simultaneously
   */
  template <MPISupported T>
  std::vector<Event>
  exchange_boundaries(const std::vector<std::pair<DeviceBuffer<T> *, int>>
                          &send_list, // {buffer, dest_rank}
                      const std::vector<std::pair<DeviceBuffer<T> *, int>>
                          &recv_list, // {buffer, source_rank}
                      idx_t count, int base_tag, const Resource &resource) {

    std::vector<Event> events;

    // Launch all sends
    for (size_t i = 0; i < send_list.size(); ++i) {
      events.push_back(send_boundary(*send_list[i].first, count,
                                     send_list[i].second, base_tag + i,
                                     resource));
    }

    // Launch all receives
    for (size_t i = 0; i < recv_list.size(); ++i) {
      events.push_back(recv_boundary(*recv_list[i].first, count,
                                     recv_list[i].second, base_tag + 100 + i,
                                     resource));
    }

    return events;
  }

  // ==================== Utilities ====================

  /**
   * @brief Get process rank (corresponds to GPU ID in single-node setup)
   */
  int get_rank() const { return rank_; }

  /**
   * @brief Get total number of processes/GPUs
   */
  int get_size() const { return size_; }

  /**
   * @brief Barrier synchronization
   */
  void barrier() { MPI_Barrier(comm_); }

  /**
   * @brief Get communication protocol for given resource
   */
  CommunicationProtocol
  get_communication_protocol(const Resource &resource) const;

  /**
   * @brief Wait for all pending async operations
   */
  void wait_all();

  /**
   * @brief Test for completion of pending operations without blocking
   */
  void test_completions();

  /**
   * @brief Print MPI setup information
   */
  void print_info() const;

  /**
   * @brief Detect GPU-aware MPI capabilities
   */
  void detect_gpu_aware_capabilities(
      std::optional<MPIAwareState> forced_state = std::nullopt);

  /**
   * @brief Validate MPI communication with actual data transfer
   */
  bool validate_communication(const Resource &resource);

  /**
   * @brief Get CUDA-aware MPI state
   */
  MPIAwareState get_cuda_aware_state() const { return cuda_aware_state_; }

  /**
   * @brief Get ROCm-aware MPI state
   */
  MPIAwareState get_rocm_aware_state() const { return rocm_aware_state_; }

  /**
   * @brief Check if direct GPU communication should be used
   */
  bool is_direct_comm_capable(ResourceType resource_type) const;

private:
  Manager() = default;
  ~Manager() {
    if (initialized_)
      finalize();
  }

  // Disable copy/move for singleton
  Manager(const Manager &) = delete;
  Manager &operator=(const Manager &) = delete;

  /**
   * @brief Direct GPU send operation
   */
  template <MPISupported T>
  Event perform_direct_send(DeviceBuffer<T> &buffer, idx_t count, int dest,
                            int tag, const Resource &resource) {
    AsyncOperation async_op;
    async_op.resource_type = resource.type();
    async_op.start_time = std::chrono::high_resolution_clock::now();

    MPI_CHECK(MPI_Isend(buffer.data(), count, get_mpi_type<T>(), dest, tag,
                        comm_, &async_op.request));

    async_op.completion_event = Event(nullptr, resource);
    pending_operations_.push_back(std::move(async_op));
    return pending_operations_.back().completion_event;
  }

  /**
   * @brief Direct GPU receive operation
   */
  template <MPISupported T>
  Event perform_direct_recv(DeviceBuffer<T> &buffer, idx_t count, int source,
                            int tag, const Resource &resource) {
    AsyncOperation async_op;
    async_op.resource_type = resource.type();
    async_op.start_time = std::chrono::high_resolution_clock::now();

    MPI_CHECK(MPI_Irecv(buffer.data(), count, get_mpi_type<T>(), source, tag,
                        comm_, &async_op.request));

    async_op.completion_event = Event(nullptr, resource);
    pending_operations_.push_back(std::move(async_op));
    return pending_operations_.back().completion_event;
  }

  /**
   * @brief Staged send through host memory (fallback)
   */
  template <MPISupported T>
  Event perform_staged_send(DeviceBuffer<T> &buffer, idx_t count, int dest,
                            int tag, const Resource &resource) {
    AsyncOperation async_op;
    async_op.resource_type = resource.type();
    async_op.start_time = std::chrono::high_resolution_clock::now();

    auto staging_size = count * sizeof(T);
    async_op.staging_buffer =
        std::make_unique<PinnedBuffer<char>>(staging_size, resource);
    T *host_data = reinterpret_cast<T *>(async_op.staging_buffer->data());

    buffer.copy_to_host(host_data, count);

    MPI_CHECK(MPI_Isend(host_data, count, get_mpi_type<T>(), dest, tag, comm_,
                        &async_op.request));

    async_op.completion_event = Event(nullptr, resource);
    pending_operations_.push_back(std::move(async_op));
    return pending_operations_.back().completion_event;
  }

  /**
   * @brief Staged receive through host memory (fallback)
   */
  template <MPISupported T>
  Event perform_staged_recv(DeviceBuffer<T> &buffer, idx_t count, int source,
                            int tag, const Resource &resource) {
    AsyncOperation async_op;
    async_op.resource_type = resource.type();
    async_op.start_time = std::chrono::high_resolution_clock::now();

    auto staging_size = count * sizeof(T);
    async_op.staging_buffer =
        std::make_unique<PinnedBuffer<char>>(staging_size, resource);
    T *host_data = reinterpret_cast<T *>(async_op.staging_buffer->data());

    MPI_CHECK(MPI_Irecv(host_data, count, get_mpi_type<T>(), source, tag, comm_,
                        &async_op.request));

    async_op.completion_event = Event(nullptr, resource);
    pending_operations_.push_back(std::move(async_op));
    return pending_operations_.back().completion_event;
  }

  /**
   * @brief Get staging buffer for host-based communication
   */
  template <MPISupported T>
  T *get_staging_buffer(idx_t count, const Resource &resource,
                        int buffer_id = 0) const {
    auto key = std::make_tuple(sizeof(T), count + buffer_id, resource.type());
    auto it = staging_buffers_.find(key);

    if (it == staging_buffers_.end()) {
      auto buffer =
          std::make_unique<PinnedBuffer<char>>(count * sizeof(T), resource);
      auto ptr = buffer->data();
      staging_buffers_[key] = std::move(buffer);
      return reinterpret_cast<T *>(ptr);
    }

    return reinterpret_cast<T *>(it->second->data());
  }

  /**
   * @brief Get MPI datatype for template type
   */
  template <MPISupported T> constexpr MPI_Datatype get_mpi_type() const {
    if constexpr (std::is_same_v<T, float>)
      return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>)
      return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, int>)
      return MPI_INT;
    else if constexpr (std::is_same_v<T, long>)
      return MPI_LONG;
    else if constexpr (std::is_same_v<T, long long>)
      return MPI_LONG_LONG;
    else if constexpr (std::is_same_v<T, unsigned int>)
      return MPI_UNSIGNED;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return MPI_UNSIGNED_LONG;
    else if constexpr (std::is_same_v<T, unsigned long long>)
      return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::is_same_v<T, char>)
      return MPI_CHAR;
    else if constexpr (std::is_same_v<T, signed char>)
      return MPI_SIGNED_CHAR;
    else if constexpr (std::is_same_v<T, unsigned char>)
      return MPI_UNSIGNED_CHAR;
    else
      return MPI_BYTE;
  }

  /**
   * @brief MPI error checking helper
   */
  void MPI_CHECK(int result) const {
    if (result != MPI_SUCCESS) {
      char error_string[MPI_MAX_ERROR_STRING];
      int length;
      MPI_Error_string(result, error_string, &length);
      ARBD_Exception(ExceptionType::RuntimeError, "MPI error: {}",
                     std::string(error_string, length));
    }
  }
};

} // namespace ARBD::MPI
#endif
