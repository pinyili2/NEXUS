#pragma once

#if defined(USE_CUDA) && defined(USE_NCCL)
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/CUDA/CUDAManager.h"
#include <concepts>
#include <cuda_runtime.h>
#include <nccl.h>
#include <type_traits>
#include <vector>

namespace ARBD {
namespace CUDA {

inline void check_nccl_error(ncclResult_t result, std::string_view file, int line) {
	if (result != ncclSuccess) {
		ARBD_Exception(ExceptionType::CUDARuntimeError,
					   "NCCL error at {}:{}: {}",
					   file,
					   line,
					   ncclGetErrorString(result));
	}
}

#define NCCL_CHECK(call) check_nccl_error(call, __FILE__, __LINE__)
/**
 * @brief Concept for arithmetic types supported by NCCL
 */
template<typename T>
concept NCCLSupported =
	std::is_arithmetic_v<T> &&
	(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int> ||
	 std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long> ||
	 std::is_same_v<T, char> || std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>);

/**
 * @brief NCCL-based multi-GPU communication manager
 *
 * This class provides a modern C++ interface for NCCL (NVIDIA Collective
 * Communications Library) operations, enabling efficient multi-GPU
 * communication patterns such as broadcast, reduce, allreduce, and
 * scatter/gather operations.
 *
 * Features:
 * - Type-safe collective operations using templates and concepts
 * - Automatic NCCL data type deduction
 * - Stream-based asynchronous operations
 * - RAII management of NCCL communicators
 * - Support for custom data types
 *
 * @example Basic Usage:
 * ```cpp
 * // Initialize NCCL for specific GPUs
 * std::vector<int> gpu_ids = {0, 1, 2, 3};
 * ARBD::CUDA::NCCLManager::init(gpu_ids);
 *
 * // Broadcast data from GPU 0 to all GPUs
 * std::vector<float*> device_ptrs = {ptr0, ptr1, ptr2, ptr3};
 * ARBD::CUDA::NCCLManager::broadcast<float>(0, device_ptrs, 1000);
 *
 * // All-reduce operation
 * ARBD::CUDA::NCCLManager::allreduce<float>(device_ptrs, 1000, ncclSum);
 *
 * // Cleanup
 * ARBD::CUDA::NCCLManager::finalize();
 * ```
 *
 * @note This class requires NCCL to be installed and USE_NCCL to be defined
 */
class NCCLManager {
  public:
	/**
	 * @brief Initialize NCCL communicators for specified GPUs
	 * @param gpu_ids Vector of GPU device IDs to include in communication group
	 */
	static void init(const int* gpu_ids, size_t count);

	// Convenience overload for std::vector
	static void init(const std::vector<int>& gpu_ids) {
		init(gpu_ids.data(), gpu_ids.size());
	}

	/**
	 * @brief Finalize and cleanup NCCL communicators
	 */
	static void finalize();

	/**
	 * @brief Check if NCCL is initialized
	 * @return True if NCCL communicators are initialized
	 */
	[[nodiscard]] static bool is_initialized() noexcept {
		return !comms_.empty();
	}

	/**
	 * @brief Get number of GPUs in the communication group
	 * @return Number of GPUs participating in NCCL operations
	 */
	[[nodiscard]] static size_t num_gpus() noexcept {
		return comms_.size();
	}

	/**
	 * @brief Broadcast data from root GPU to all other GPUs
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param root Root GPU rank (0-indexed within the group)
	 * @param device_ptrs Device pointers for each GPU (send and receive)
	 * @param count Number of elements to broadcast
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void
	broadcast(int root, T** device_ptrs, size_t num_gpus, size_t count, int stream_id = -1);

	// Convenience overload for std::vector
	template<NCCLSupported T>
	static void
	broadcast(int root, const std::vector<T*>& device_ptrs, size_t count, int stream_id = -1) {
		broadcast(root, const_cast<T**>(device_ptrs.data()), device_ptrs.size(), count, stream_id);
	}

	/**
	 * @brief All-reduce operation across all GPUs
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param device_ptrs Device pointers for each GPU (in-place operation)
	 * @param count Number of elements to reduce
	 * @param op Reduction operation (ncclSum, ncclProd, ncclMax, ncclMin)
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void allreduce(T** device_ptrs,
						  size_t num_gpus,
						  size_t count,
						  ncclRedOp_t op = ncclSum,
						  int stream_id = -1);

	// Convenience overload for std::vector
	template<NCCLSupported T>
	static void allreduce(const std::vector<T*>& device_ptrs,
						  size_t count,
						  ncclRedOp_t op = ncclSum,
						  int stream_id = -1) {
		allreduce(const_cast<T**>(device_ptrs.data()), device_ptrs.size(), count, op, stream_id);
	}

	/**
	 * @brief Reduce operation to root GPU
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param root Root GPU rank to receive the result
	 * @param send_ptrs Device pointers for send data on each GPU
	 * @param recv_ptrs Device pointers for receive data (only root needs valid
	 * pointer)
	 * @param count Number of elements to reduce
	 * @param op Reduction operation (ncclSum, ncclProd, ncclMax, ncclMin)
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void reduce(int root,
					   const T** send_ptrs,
					   T** recv_ptrs,
					   size_t num_gpus,
					   size_t count,
					   ncclRedOp_t op = ncclSum,
					   int stream_id = -1);

	/**
	 * @brief All-gather operation across all GPUs
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param send_ptrs Device pointers for send data on each GPU
	 * @param recv_ptrs Device pointers for receive data on each GPU
	 * @param count Number of elements per GPU to gather
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void allgather(const T** send_ptrs,
						  T** recv_ptrs,
						  size_t num_gpus,
						  size_t count,
						  int stream_id = -1);

	/**
	 * @brief Scatter data from root GPU to all GPUs
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param root Root GPU rank containing the data to scatter
	 * @param send_ptr Device pointer on root GPU (nullptr for non-root GPUs)
	 * @param recv_ptrs Device pointers for receive data on each GPU
	 * @param count Number of elements each GPU receives
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void scatter(int root,
						const T* send_ptr,
						T** recv_ptrs,
						size_t num_gpus,
						size_t count,
						int stream_id = -1);

	/**
	 * @brief Synchronize all NCCL operations on all GPUs
	 */
	static void synchronize();

	/**
	 * @brief Synchronize all GPUs in the system (replaces GPUManager::sync())
	 *
	 * This function provides comprehensive synchronization for multi-GPU systems.
	 * It works whether NCCL is initialized or not, making it suitable for both
	 * NCCL-based and basic multi-GPU applications.
	 *
	 * Behavior:
	 * - If NCCL is initialized: Synchronizes all GPUs in the NCCL communication
	 * group
	 * - If NCCL is not initialized: Falls back to GPUManager for basic sync
	 *
	 * @note This function replaces the old GPUManager::sync() for better
	 * separation of concerns between device management and communication
	 * coordination
	 *
	 * @example Usage:
	 * ```cpp
	 * // Launch kernels on multiple GPUs
	 * for (int i = 0; i < num_gpus; ++i) {
	 *     GPUManager::use(i);
	 *     my_kernel<<<blocks, threads>>>();
	 * }
	 *
	 * // Synchronize all GPUs
	 * NCCLManager::sync_all();
	 * ```
	 */
	static void sync_all();

	/**
	 * @brief Get the GPU rank for a given device ID
	 * @param device_id CUDA device ID
	 * @return NCCL rank (0-indexed within the group) or -1 if not found
	 */
	[[nodiscard]] static int get_rank(int device_id);

	/**
	 * @brief Legacy broadcast interface using std::vector
	 * @deprecated Use the modern span-based interface instead
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param root Root GPU rank
	 * @param send_d Vector of device pointers for send data
	 * @param recv_d Vector of device pointers for receive data
	 * @param size Number of elements to broadcast
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void nccl_broadcast(int root,
							   const std::vector<T*>& send_d,
							   const std::vector<T*>& recv_d,
							   size_t size,
							   int stream_id = -1) {
		broadcast<T>(root, const_cast<T**>(send_d.data()), send_d.size(), size, stream_id);
	}

	/**
	 * @brief Legacy reduce interface using std::vector
	 * @deprecated Use the modern span-based interface instead
	 * @tparam T Data type (must satisfy NCCLSupported concept)
	 * @param root Root GPU rank
	 * @param send_d Vector of device pointers for send data
	 * @param recv_d Vector of device pointers for receive data
	 * @param size Number of elements to reduce
	 * @param stream_id Stream ID to use (-1 for default stream)
	 */
	template<NCCLSupported T>
	static void nccl_reduce(int root,
							const std::vector<T*>& send_d,
							const std::vector<T*>& recv_d,
							size_t size,
							int stream_id = -1) {
		reduce<T>(root,
				  const_cast<const T**>(send_d.data()),
				  const_cast<T**>(recv_d.data()),
				  send_d.size(),
				  size,
				  ncclSum,
				  stream_id);
	}

  private:
	/**
	 * @brief Convert C++ type to NCCL data type
	 * @tparam T C++ data type
	 * @return Corresponding ncclDataType_t
	 */
	template<typename T>
	[[nodiscard]] static constexpr ncclDataType_t get_nccl_type();

	/**
	 * @brief Get stream for a specific GPU rank
	 * @param rank GPU rank
	 * @param stream_id Stream ID (-1 for default stream)
	 * @return CUDA stream handle
	 */
	[[nodiscard]] static cudaStream_t get_stream(int rank, int stream_id);

	static std::vector<ncclComm_t> comms_;
	static std::vector<int> gpu_ids_;
	static bool initialized_;
};

// Template implementations
template<NCCLSupported T>
void NCCLManager::broadcast(int root,
							T** device_ptrs,
							size_t num_gpus,
							size_t count,
							int stream_id) {
	if (!is_initialized()) {
		ARBD_Exception(ExceptionType::ValueError, "NCCL not initialized");
	}

	if (num_gpus != comms_.size()) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Device pointer count ({}) doesn't match GPU count ({})",
					   num_gpus,
					   comms_.size());
	}

	NCCL_CHECK(ncclGroupStart());
	for (size_t i = 0; i < comms_.size(); ++i) {
		cudaStream_t stream = get_stream(static_cast<int>(i), stream_id);
		NCCL_CHECK(ncclBroadcast(static_cast<const void*>(device_ptrs[i]),
								 static_cast<void*>(device_ptrs[i]),
								 count,
								 get_nccl_type<T>(),
								 root,
								 comms_[i],
								 stream));
	}
	NCCL_CHECK(ncclGroupEnd());
}

template<NCCLSupported T>
void NCCLManager::allreduce(T** device_ptrs,
							size_t num_gpus,
							size_t count,
							ncclRedOp_t op,
							int stream_id) {
	if (!is_initialized()) {
		ARBD_Exception(ExceptionType::ValueError, "NCCL not initialized");
	}

	if (device_ptrs.size() != comms_.size()) {
		ARBD_Exception(ExceptionType::ValueError,
					   "Device pointer count ({}) doesn't match GPU count ({})",
					   device_ptrs.size(),
					   comms_.size());
	}

	NCCL_CHECK(ncclGroupStart());
	for (size_t i = 0; i < comms_.size(); ++i) {
		cudaStream_t stream = get_stream(static_cast<int>(i), stream_id);
		NCCL_CHECK(ncclAllReduce(static_cast<const void*>(device_ptrs[i]),
								 static_cast<void*>(device_ptrs[i]),
								 count,
								 get_nccl_type<T>(),
								 op,
								 comms_[i],
								 stream));
	}
	NCCL_CHECK(ncclGroupEnd());
}

template<NCCLSupported T>
void NCCLManager::reduce(int root,
						 const T** send_ptrs,
						 T** recv_ptrs,
						 size_t num_gpus,
						 size_t count,
						 ncclRedOp_t op,
						 int stream_id) {
	if (!is_initialized()) {
		ARBD_Exception(ExceptionType::ValueError, "NCCL not initialized");
	}

	if (send_ptrs.size() != comms_.size() || recv_ptrs.size() != comms_.size()) {
		ARBD_Exception(ExceptionType::ValueError, "Pointer count doesn't match GPU count");
	}

	NCCL_CHECK(ncclGroupStart());
	for (size_t i = 0; i < comms_.size(); ++i) {
		cudaStream_t stream = get_stream(static_cast<int>(i), stream_id);
		NCCL_CHECK(ncclReduce(static_cast<const void*>(send_ptrs[i]),
							  static_cast<void*>(recv_ptrs[i]),
							  count,
							  get_nccl_type<T>(),
							  op,
							  root,
							  comms_[i],
							  stream));
	}
	NCCL_CHECK(ncclGroupEnd());
}

template<NCCLSupported T>
void NCCLManager::allgather(const T** send_ptrs,
							T** recv_ptrs,
							size_t num_gpus,
							size_t count,
							int stream_id) {
	if (!is_initialized()) {
		ARBD_Exception(ExceptionType::ValueError, "NCCL not initialized");
	}

	if (send_ptrs.size() != comms_.size() || recv_ptrs.size() != comms_.size()) {
		ARBD_Exception(ExceptionType::ValueError, "Pointer count doesn't match GPU count");
	}

	NCCL_CHECK(ncclGroupStart());
	for (size_t i = 0; i < comms_.size(); ++i) {
		cudaStream_t stream = get_stream(static_cast<int>(i), stream_id);
		NCCL_CHECK(ncclAllGather(static_cast<const void*>(send_ptrs[i]),
								 static_cast<void*>(recv_ptrs[i]),
								 count,
								 get_nccl_type<T>(),
								 comms_[i],
								 stream));
	}
	NCCL_CHECK(ncclGroupEnd());
}

template<NCCLSupported T>
void NCCLManager::scatter(int root,
						  const T* send_ptr,
						  T** recv_ptrs,
						  size_t num_gpus,
						  size_t count,
						  int stream_id) {
	if (!is_initialized()) {
		ARBD_Exception(ExceptionType::ValueError, "NCCL not initialized");
	}

	if (recv_ptrs.size() != comms_.size()) {
		ARBD_Exception(ExceptionType::ValueError, "Receive pointer count doesn't match GPU count");
	}

	// NCCL doesn't have a direct scatter operation, implement using broadcasts
	for (size_t target_rank = 0; target_rank < comms_.size(); ++target_rank) {
		const T* src =
			(root == static_cast<int>(target_rank)) ? send_ptr + target_rank * count : nullptr;

		NCCL_CHECK(ncclGroupStart());
		for (size_t i = 0; i < comms_.size(); ++i) {
			cudaStream_t stream = get_stream(static_cast<int>(i), stream_id);
			const void* send_data = (i == target_rank) ? static_cast<const void*>(src) : nullptr;
			void* recv_data = (i == target_rank) ? static_cast<void*>(recv_ptrs[i]) : nullptr;

			if (static_cast<int>(i) == root) {
				NCCL_CHECK(ncclBroadcast(send_data,
										 recv_data,
										 count,
										 get_nccl_type<T>(),
										 root,
										 comms_[i],
										 stream));
			} else if (static_cast<int>(i) == static_cast<int>(target_rank)) {
				NCCL_CHECK(ncclBroadcast(send_data,
										 recv_data,
										 count,
										 get_nccl_type<T>(),
										 root,
										 comms_[i],
										 stream));
			}
		}
		NCCL_CHECK(ncclGroupEnd());
	}
}

template<typename T>
constexpr ncclDataType_t NCCLManager::get_nccl_type() {
	if constexpr (std::is_same_v<T, float>)
		return ncclFloat;
	else if constexpr (std::is_same_v<T, double>)
		return ncclDouble;
	else if constexpr (std::is_same_v<T, int>)
		return ncclInt;
	else if constexpr (std::is_same_v<T, long long>)
		return ncclInt64;
	else if constexpr (std::is_same_v<T, unsigned long long>)
		return ncclUint64;
	else if constexpr (std::is_same_v<T, char>)
		return ncclChar;
	else if constexpr (std::is_same_v<T, signed char>)
		return ncclInt8;
	else if constexpr (std::is_same_v<T, unsigned char>)
		return ncclUint8;
	else
		static_assert(std::is_same_v<T, void>, "Unsupported NCCL data type");
}

} // namespace CUDA
} // namespace ARBD

#endif // USE_CUDA && USE_NCCL
