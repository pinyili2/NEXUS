#pragma once
#ifndef __METAL_VERSION__
#include "Resource.h"
#include <map>
#include <mutex>
#include <vector>

namespace ARBD {

template <typename T> class MemoryPool {
private:
  struct PoolBlock {
    void *ptr = nullptr;
    size_t size = 0;
    Resource resource;
    bool in_use = false;
  };

  std::vector<PoolBlock> blocks_;
  std::mutex mutex_;

public:
  // Allocate from pool or create new block
  void *allocate(size_t bytes, const Resource &resource) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try to find existing free block
    for (auto &block : blocks_) {
      if (!block.in_use && block.size >= bytes && block.resource == resource) {
        block.in_use = true;
        return block.ptr;
      }
    }

    // Allocate new block
    PoolBlock new_block;
    new_block.size = bytes;
    new_block.resource = resource;
    new_block.in_use = true;

#ifdef USE_CUDA
    if (resource.type() == ResourceType::CUDA) {
      cudaSetDevice(resource.id());
      cudaMalloc(&new_block.ptr, bytes);
    }
#endif

#ifdef USE_SYCL
    if (resource.type() == ResourceType::SYCL) {
      Resource device = Resource::create_sycl_device(resource.id());
      new_block.ptr = sycl::malloc_device(
          bytes, *static_cast<sycl::queue *>(device.get_queue(0)));
    }
#endif

    blocks_.push_back(new_block);
    return new_block.ptr;
  }

  void deallocate(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &block : blocks_) {
      if (block.ptr == ptr) {
        block.in_use = false;
        return;
      }
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto &block : blocks_) {
#ifdef USE_CUDA
      if (block.resource.type() == ResourceType::CUDA) {
        cudaFree(block.ptr);
      }
#endif

#ifdef USE_SYCL
      if (block.resource.type() == ResourceType::SYCL) {
        Resource device = Resource::create_sycl_device(block.resource.id());
        sycl::free(block.ptr, *static_cast<sycl::queue *>(device.get_queue()));
      }
#endif
    }

    blocks_.clear();
  }

  ~MemoryPool() { clear(); }
};

// Global memory pool for temporary allocations
inline MemoryPool<idx_t> &get_temp_pool() {
  static MemoryPool<idx_t> pool;
  return pool;
}

namespace MemoryAdvise {
#ifdef USE_CUDA
constexpr int READ_MOSTLY = cudaMemAdviseSetReadMostly;
constexpr int PREFERRED_LOCATION = cudaMemAdviseSetPreferredLocation;
constexpr int ACCESSED_BY = cudaMemAdviseSetAccessedBy;
#else
// SYCL or other backends
constexpr int READ_MOSTLY = 0;
constexpr int PREFERRED_LOCATION = 1;
constexpr int ACCESSED_BY = 2;
#endif
} // namespace MemoryAdvise

} // namespace ARBD
#endif
