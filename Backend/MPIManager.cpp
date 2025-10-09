// src/Backend/MPI/MPIManager.cpp
#include "Backend/MPIManager.h"
#include "ARBDLogger.h"
#include <algorithm>
#include <optional>

namespace ARBD::MPI {

void Manager::init(bool assume_gpu_aware) {
  if (initialized_) {
    LOGWARN("MPI Manager already initialized");
    return;
  }

  assume_gpu_aware_ = assume_gpu_aware;

  // Initialize MPI with thread support
  int provided_thread_level;
  MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE,
                            &provided_thread_level));

  if (provided_thread_level < MPI_THREAD_MULTIPLE) {
    LOGWARN("MPI does not support MPI_THREAD_MULTIPLE, got level: {}",
            provided_thread_level);
  }

  // Get basic MPI information
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size_));
  comm_ = MPI_COMM_WORLD;

  // Detect GPU-aware capabilities
  detect_gpu_aware_capabilities(std::nullopt);

  initialized_ = true;

  LOGINFO("MPI Manager initialized: rank {}/{} (GPU-aware: {})", rank_, size_,
          assume_gpu_aware_ ? "assumed" : "disabled");
}

void Manager::finalize() {
  if (!initialized_)
    return;

  // Wait for all pending operations
  wait_all();

  // Clear caches
  staging_buffers_.clear();
  protocol_cache_.clear();
  pending_operations_.clear();

  MPI_Finalize();
  initialized_ = false;

  LOGINFO("MPI Manager finalized");
}

CommunicationProtocol
Manager::get_communication_protocol(const Resource &resource) const {
  auto it = protocol_cache_.find(resource.type());
  if (it != protocol_cache_.end()) {
    return it->second;
  }

  CommunicationProtocol protocol;

  // Use the detected GPU-aware capabilities
  if (is_direct_comm_capable(resource.type())) {
    protocol = CommunicationProtocol::DirectGPU;
  } else {
    protocol = CommunicationProtocol::CopyToHost;
  }

  protocol_cache_[resource.type()] = protocol;
  return protocol;
}

void Manager::wait_all() {
  if (pending_operations_.empty())
    return;

  std::vector<MPI_Request> requests;
  requests.reserve(pending_operations_.size());

  for (const auto &op : pending_operations_) {
    requests.push_back(op.request);
  }

  // Wait for all operations to complete
  std::vector<MPI_Status> statuses(requests.size());
  MPI_CHECK(MPI_Waitall(requests.size(), requests.data(), statuses.data()));

  // Log timing information
  auto now = std::chrono::high_resolution_clock::now();
  for (const auto &op : pending_operations_) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        now - op.start_time)
                        .count();
    LOGDEBUG("MPI operation completed in {} μs", duration);
  }

  pending_operations_.clear();
}

void Manager::test_completions() {
  if (pending_operations_.empty())
    return;

  auto it = pending_operations_.begin();
  while (it != pending_operations_.end()) {
    int flag;
    MPI_Status status;
    MPI_CHECK(MPI_Test(&it->request, &flag, &status));

    if (flag) {
      // Operation completed
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::high_resolution_clock::now() - it->start_time)
              .count();
      LOGDEBUG("MPI async operation completed in {} μs", duration);

      it = pending_operations_.erase(it);
    } else {
      ++it;
    }
  }
}

void Manager::detect_gpu_aware_capabilities(
    std::optional<MPIAwareState> forced_state) {
  forced_state_ = forced_state;

  if (forced_state) {
    cuda_aware_state_ = *forced_state;
    rocm_aware_state_ = *forced_state;
    LOGINFO("MPI GPU-aware capabilities forced to: {}",
            (*forced_state == MPIAwareState::Yes) ? "Yes" : "No");
    return;
  }

  // Detect CUDA-aware MPI
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    cuda_aware_state_ = MPIAwareState::Yes;
  } else {
    cuda_aware_state_ = MPIAwareState::No;
  }
#else
  cuda_aware_state_ = MPIAwareState::Unknown;
#endif

  // Detect ROCm-aware MPI
#if defined(MPIX_ROCM_AWARE_SUPPORT)
  if (MPIX_Query_rocm_support() == 1) {
    rocm_aware_state_ = MPIAwareState::Yes;
  } else {
    rocm_aware_state_ = MPIAwareState::No;
  }
#else
  rocm_aware_state_ = MPIAwareState::Unknown;
#endif

  if (rank_ == 0) {
    LOGINFO("MPI GPU-aware capabilities detected:");
    LOGINFO("  CUDA-aware: {}",
            (cuda_aware_state_ == MPIAwareState::Yes)  ? "Yes"
            : (cuda_aware_state_ == MPIAwareState::No) ? "No"
                                                       : "Unknown");
    LOGINFO("  ROCm-aware: {}",
            (rocm_aware_state_ == MPIAwareState::Yes)  ? "Yes"
            : (rocm_aware_state_ == MPIAwareState::No) ? "No"
                                                       : "Unknown");
  }
}

bool Manager::validate_communication(const Resource &resource) {
  if (!initialized_) {
    LOGWARN("Cannot validate communication: MPI not initialized");
    return false;
  }

  const size_t test_size = 10000;

  try {
    // Create test buffers
    DeviceBuffer<float> send_buffer(test_size, 0);
    DeviceBuffer<float> recv_buffer(test_size, 0);

    // Initialize send buffer with rank-specific pattern
    std::vector<float> host_data(test_size);
    for (size_t i = 0; i < test_size; i++) {
      host_data[i] = static_cast<float>(rank_ * 1000 + i % 100);
    }
    send_buffer.copy_from_host(host_data);

    bool success = true;

    if (size_ >= 2) {
      // Ring communication test
      int next_rank = (rank_ + 1) % size_;
      int prev_rank = (rank_ - 1 + size_) % size_;

      MPI_Request send_req, recv_req;

      // Non-blocking send/receive using the detected protocol
      auto protocol = get_communication_protocol(resource);

      if (protocol == CommunicationProtocol::DirectGPU) {
        // Direct GPU communication
        MPI_CHECK(MPI_Isend(send_buffer.data(), test_size, MPI_FLOAT, next_rank,
                            0, comm_, &send_req));
        MPI_CHECK(MPI_Irecv(recv_buffer.data(), test_size, MPI_FLOAT, prev_rank,
                            0, comm_, &recv_req));
      } else {
        // Host-staged communication
        std::vector<float> send_host(test_size), recv_host(test_size);
        send_buffer.copy_to_host(send_host);

        MPI_CHECK(MPI_Isend(send_host.data(), test_size, MPI_FLOAT, next_rank,
                            0, comm_, &send_req));
        MPI_CHECK(MPI_Irecv(recv_host.data(), test_size, MPI_FLOAT, prev_rank,
                            0, comm_, &recv_req));

        MPI_CHECK(MPI_Wait(&send_req, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Wait(&recv_req, MPI_STATUS_IGNORE));

        recv_buffer.copy_from_host(recv_host);

        // Validate received data
        std::vector<float> received_data(test_size);
        recv_buffer.copy_to_host(received_data);

        for (size_t i = 0; i < test_size; i++) {
          float expected = static_cast<float>(prev_rank * 1000 + i % 100);
          if (std::abs(received_data[i] - expected) > 1e-6f) {
            success = false;
            break;
          }
        }

        return success;
      }

      // Wait for direct GPU operations
      MPI_CHECK(MPI_Wait(&send_req, MPI_STATUS_IGNORE));
      MPI_CHECK(MPI_Wait(&recv_req, MPI_STATUS_IGNORE));

      // Validate received data for direct GPU case
      std::vector<float> received_data(test_size);
      recv_buffer.copy_to_host(received_data);

      for (size_t i = 0; i < test_size; i++) {
        float expected = static_cast<float>(prev_rank * 1000 + i % 100);
        if (std::abs(received_data[i] - expected) > 1e-6f) {
          success = false;
          break;
        }
      }
    }

    // Synchronize all processes
    MPI_CHECK(MPI_Barrier(comm_));

    return success;
  } catch (const std::exception &e) {
    LOGWARN("Communication validation failed with exception: {}", e.what());
    return false;
  }
}

bool Manager::is_direct_comm_capable(ResourceType resource_type) const {
  switch (resource_type) {
  case ResourceType::CUDA:
  case ResourceType::SYCL:
    return (cuda_aware_state_ == MPIAwareState::Yes) ||
           (cuda_aware_state_ == MPIAwareState::ForcedYes);
  case ResourceType::CPU:
    return true; // CPU memory is always accessible
  default:
    return false;
  }
}

void Manager::print_info() const {
  if (rank_ == 0) {
    LOGINFO("=== MPI Communication Setup ===");
    LOGINFO("Total processes/GPUs: {}", size_);
    LOGINFO("GPU-aware MPI: {}", assume_gpu_aware_ ? "Enabled" : "Disabled");

    const char *cuda_state_str =
        (cuda_aware_state_ == MPIAwareState::Yes)         ? "Yes"
        : (cuda_aware_state_ == MPIAwareState::No)        ? "No"
        : (cuda_aware_state_ == MPIAwareState::ForcedYes) ? "Forced Yes"
        : (cuda_aware_state_ == MPIAwareState::ForcedNo)  ? "Forced No"
                                                          : "Unknown";

    const char *rocm_state_str =
        (rocm_aware_state_ == MPIAwareState::Yes)         ? "Yes"
        : (rocm_aware_state_ == MPIAwareState::No)        ? "No"
        : (rocm_aware_state_ == MPIAwareState::ForcedYes) ? "Forced Yes"
        : (rocm_aware_state_ == MPIAwareState::ForcedNo)  ? "Forced No"
                                                          : "Unknown";

    LOGINFO("CUDA-aware MPI: {}", cuda_state_str);
    LOGINFO("ROCm-aware MPI: {}", rocm_state_str);
    LOGINFO("Communication protocols:");
    LOGINFO("  - CUDA/SYCL: {}", is_direct_comm_capable(ResourceType::SYCL)
                                     ? "DirectGPU"
                                     : "CopyToHost");
    LOGINFO("  - CPU: DirectGPU");
    LOGINFO("===================================");
  }
}
} // namespace ARBD::MPI
