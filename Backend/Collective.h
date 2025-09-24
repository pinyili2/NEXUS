// src/Backend/Collectives.h
#pragma once
#ifndef __METAL_VERSION__
#include "Buffer.h"
#include "Events.h"
#include "Resource.h"

#ifdef USE_NCCL
#include "CUDA/NCCLManager.h"
#endif
#include "MPIManager.h"

namespace ARBD {

class Collectives {
  public:
	enum class Backend { AUTO, MPI, NCCL, ONECCL };

  private:
	Resource resource_;
	Backend active_backend_;

	Backend select_backend(const Resource& res, Backend preferred) {
		if (preferred != Backend::AUTO) {
			return preferred;
		}

#ifdef USE_NCCL
		if (res.type == ResourceType::CUDA) {
			// Check if NCCLManager is initialized
			try {
				auto& nccl = NCCL::Manager::instance();
				if (nccl.is_initialized()) {
					return Backend::NCCL;
				}
			} catch (...) {
			}
		}
#endif

		// Default to MPI
		return Backend::MPI;
	}

  public:
	Collectives(const Resource& res, Backend preferred = Backend::AUTO)
		: resource_(res), active_backend_(select_backend(res, preferred)) {

		// Ensure MPI is initialized as fallback
		MPI::Manager::instance().init();

#ifdef USE_NCCL
		if (active_backend_ == Backend::NCCL) {
			NCCL::Manager::instance().init();
			LOGINFO("Using NCCL for collectives");
		} else
#endif
		{
			LOGINFO("Using MPI for collectives");
		}
	}
	// Performance comparison helper
	void benchmark_backends() {
		DeviceBuffer<float> test_buffer(1024 * 1024); // 1M floats

		// Test MPI
		auto start = std::chrono::high_resolution_clock::now();
		MPI::Manager::instance().allReduce(test_buffer, test_buffer.size(), resource_);
		auto mpi_time = std::chrono::high_resolution_clock::now() - start;

#ifdef USE_NCCL
		if (resource_.type == ResourceType::CUDA) {
			start = std::chrono::high_resolution_clock::now();
			NCCL::Manager::instance().allReduce(test_buffer, test_buffer.size(), resource_);
			auto nccl_time = std::chrono::high_resolution_clock::now() - start;

			LOGINFO("MPI: {}ms, NCCL: {}ms",
					std::chrono::duration_cast<std::chrono::milliseconds>(mpi_time).count(),
					std::chrono::duration_cast<std::chrono::milliseconds>(nccl_time).count());
		}
#endif
	};
	template<typename T>
	Event allReduce(DeviceBuffer<T>& buffer, idx_t count) {
		switch (active_backend_) {
#ifdef USE_NCCL
		case Backend::NCCL:
			return NCCL::Manager::instance().allReduce(buffer, count, resource_);
#endif
		default:
			return MPI::Manager::instance().allReduce(buffer, count, resource_);
		}
	}

	template<typename T>
	Event broadcast(DeviceBuffer<T>& buffer, idx_t count, int root) {
		switch (active_backend_) {
#ifdef USE_NCCL
		case Backend::NCCL:
			return NCCL::Manager::instance().broadcast(buffer, count, root, resource_);
#endif
		default:
			return MPI::Manager::instance().broadcast(buffer, count, root, resource_);
		}
	}
};

class DeviceMesh {
  private:
	struct DeviceNode {
		Resource resource;
		void* primary_stream;
		std::vector<void*> secondary_streams; // Multiple streams per device
		bool peer_access_enabled = false;

		void* get_queue(int device_id) const {
			return primary_stream;
		}

		void* get_secondary_stream(int device_id) const {
			if (device_id < secondary_streams.size()) {
				return secondary_streams[device_id];
			}
			return nullptr;
		}
	};

	std::vector<DeviceNode> nodes_;
	std::vector<std::vector<bool>> connectivity_matrix_; // P2P access
	std::vector<Resource> resources_;					 // Store the resources
	std::unordered_map<size_t, void*> queues_;			 // device_id -> queue mapping
	bool peer_access_enabled_ = false;

  public:
	DeviceMesh() = default;

	explicit DeviceMesh(std::vector<Resource> resources) : resources_(resources) {
		// Initialize device nodes
		for (const auto& res : resources_) {
			DeviceNode node;
			node.resource = res;

#ifdef USE_CUDA
			if (res.type == ResourceType::CUDA) {
				cudaStream_t queue;
				CUDA_CHECK(cudaSetDevice(res.id));
				CUDA_CHECK(cudaStreamCreate(&queue));
				node.primary_stream = queue;
				queues_[res.id] = queue;
			}
#endif

#ifdef USE_SYCL
			if (res.type == ResourceType::SYCL) {
				auto& device = SYCL::Manager::get_device(res.id);
				node.primary_stream = &device.get_queue(0);
				queues_[res.id] = node.primary_stream;
			}
#endif

#ifdef USE_METAL
			if (res.type == ResourceType::METAL) {
				auto& device = METAL::Manager::get_device(res.id);
				node.primary_stream = device.get_next_queue();
				queues_[res.id] = node.primary_stream;
			}
#endif

			nodes_.push_back(std::move(node));
		}

		// Initialize connectivity matrix
		connectivity_matrix_.resize(resources_.size());
		for (auto& row : connectivity_matrix_) {
			row.resize(resources_.size(), false);
		}

		enable_peer_access();
	}

	~DeviceMesh() {
#ifdef USE_CUDA
		for (const auto& res : resources_) {
			if (res.type == ResourceType::CUDA) {
				cudaSetDevice(res.id);
				auto it = queues_.find(res.id);
				if (it != queues_.end()) {
					cudaStreamDestroy(static_cast<cudaStream_t>(it->second));
				}
			}
		}
#endif
	}

	void enable_peer_access() {
#ifdef USE_CUDA
		for (size_t i = 0; i < resources_.size(); ++i) {
			if (resources_[i].type != ResourceType::CUDA)
				continue;

			cudaSetDevice(resources_[i].id);
			for (size_t j = 0; j < resources_.size(); ++j) {
				if (i != j && resources_[j].type == ResourceType::CUDA) {
					int can_access;
					cudaDeviceCanAccessPeer(&can_access, resources_[i].id, resources_[j].id);
					if (can_access) {
						cudaDeviceEnablePeerAccess(resources_[j].id, 0);
						connectivity_matrix_[i][j] = true;
						connectivity_matrix_[j][i] = true;
					}
				}
			}
		}
#endif
		peer_access_enabled_ = true;
	}

	// Getter methods
	size_t device_count() const {
		return nodes_.size();
	}

	const DeviceNode& operator[](size_t device_id) const {
		return nodes_[device_id];
	}

	bool can_access_peer(size_t device_id, size_t peer_id) const {
		if (device_id < connectivity_matrix_.size() &&
			peer_id < connectivity_matrix_[device_id].size()) {
			return connectivity_matrix_[device_id][peer_id];
		}
		return false;
	}

	void* get_queue(size_t device_id) const {
		auto it = queues_.find(device_id);
		return (it != queues_.end()) ? it->second : nullptr;
	}

	const std::vector<Resource>& get_resources() const {
		return resources_;
	}

	bool is_peer_access_enabled() const {
		return peer_access_enabled_;
	}
};
} // namespace ARBD
#endif