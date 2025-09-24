// src/Backend/MPI/MPIManager.h
#pragma once
#ifndef __METAL_VERSION__
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include <mpi.h>
#include <unordered_map>

namespace ARBD::MPI {

class Manager {
  private:
	MPI_Comm comm_;
	int rank_; //process ID
	int size_; //total number of devices
	bool initialized_ = false;

  public:
	static Manager& instance() {
		static Manager instance;
		return instance;
	}

	void init() {
		if (!initialized_) {
			int provided;
			MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
			MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
			MPI_Comm_size(MPI_COMM_WORLD, &size_);
			comm_ = MPI_COMM_WORLD;
			initialized_ = true;
			LOGINFO("MPI Manager initialized: rank {}/{}", rank_, size_);
		}
	}

	void finalize() {
		if (initialized_) {
			staging_buffers_.clear();
			MPI_Finalize();
			initialized_ = false;
		}
	}

	template<typename T>
	Event allReduce(DeviceBuffer<T>& buffer, idx_t count, const Resource& resource) {
		T* host_buffer = get_staging_buffer<T>(count, resource);

		// Download
		buffer.copy_to_host(host_buffer, count);

		// MPI collective
		MPI_Allreduce(MPI_IN_PLACE, host_buffer, count, get_mpi_type<T>(), MPI_SUM, comm_);

		// Upload
		buffer.copy_from_host(host_buffer, count);

		// Return a completed event
		return Event(nullptr, resource);
	}

	template<typename T>
	Event broadcast(DeviceBuffer<T>& buffer, idx_t count, int root, const Resource& resource) {
		T* host_buffer = get_staging_buffer<T>(count, resource);

		if (rank_ == root) {
			buffer.copy_to_host(host_buffer, count);
		}

		MPI_Bcast(host_buffer, count, get_mpi_type<T>(), root, comm_);

		if (rank_ != root) {
			buffer.copy_from_host(host_buffer, count);
		}

		return Event(nullptr, resource);
	}

	int get_rank() const {
		return rank_;
	}
	int get_size() const {
		return size_;
	}

  private:
	// Cache by element count and type for better type safety
	mutable std::unordered_map<std::pair<size_t, size_t>, std::unique_ptr<PinnedBuffer<char>>>
		staging_buffers_;

	template<typename T>
	T* get_staging_buffer(idx_t count, const Resource& resource) {
		auto key = std::make_pair(sizeof(T), count);
		auto it = staging_buffers_.find(key);

		if (it == staging_buffers_.end()) {
			auto buffer = std::make_unique<PinnedBuffer<char>>(count * sizeof(T), resource);
			auto ptr = buffer->data();
			staging_buffers_[key] = std::move(buffer);
			return reinterpret_cast<T*>(ptr);
		}

		return reinterpret_cast<T*>(it->second->data());
	}

	template<typename T>
	MPI_Datatype get_mpi_type() {
		if constexpr (std::is_same_v<T, float>)
			return MPI_FLOAT;
		else if constexpr (std::is_same_v<T, double>)
			return MPI_DOUBLE;
		else if constexpr (std::is_same_v<T, int>)
			return MPI_INT;
		else
			return MPI_BYTE;
	}
};

} // namespace ARBD::MPI
#endif
