#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Buffer.h"
#include "Events.h"
#include "Header.h"
#include "KernelConfig.h"
#include "Resource.h"
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef USE_SYCL
#include "SYCL/SYCLKernels.h"
#include "SYCL/SYCLManager.h"
#endif

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include "CUDA/KernelHelper.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/tuple.h>
#endif

#ifdef USE_METAL
#include "METAL/METALKernels.h"
#endif

namespace ARBD {

template <typename Functor, typename... Args>
Event launch_kernel(const Resource &resource, const KernelConfig &config,
                    Functor kernel_functor, Args &&...args) {
  // Auto-configure the kernel if grid_size is not set (default 0,0,0)
  KernelConfig local_config = config;
  local_config.validate_block_size(resource);

  // Standardize problem size across backends:
  // If problem_size is not specified, derive it from grid_size * block_size
  if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0 ||
      local_config.problem_size.z == 0) {
    kerneldim3 new_problem{};
    new_problem.x = std::max<idx_t>(1, local_config.grid_size.x *
                                           local_config.block_size.x);
    new_problem.y = std::max<idx_t>(1, local_config.grid_size.y *
                                           local_config.block_size.y);
    new_problem.z = std::max<idx_t>(1, local_config.grid_size.z *
                                           local_config.block_size.z);
    local_config.problem_size = new_problem;
  }

#ifdef USE_CUDA
  if (resource.type() == ResourceType::CUDA) {
    return launch_cuda_kernel(resource, local_config, kernel_functor,
                              get_buffer_pointer(std::forward<Args>(args))...);
  }
#endif

#ifdef USE_SYCL
  if (resource.type() == ResourceType::SYCL) {
    return launch_sycl_kernel(resource, local_config, kernel_functor,
                              get_buffer_pointer(std::forward<Args>(args))...);
  }
#endif

#ifdef USE_METAL
  if (resource.type() == ResourceType::METAL) {
    throw_value_error(
        "Metal backend requires a kernel name (string), not a functor. "
        "Please use launch_metal_kernel with a kernel name.");
  }
#endif

  // CPU fallback
  return launch_cpu_kernel(resource, local_config, kernel_functor,
                           get_buffer_pointer(std::forward<Args>(args))...);
}

// ============================================================================
// Kernel Launcher with WorkItem Support (for shared memory/reductions)
// ============================================================================

/**
 * @brief Launch kernel with WorkItem support for shared memory and barriers
 *
 * Use this for kernels that need:
 * - Shared/local memory access
 * - Work-group synchronization (barriers)
 * - Reduction operations
 *
 * The functor must have signature:
 *   DEVICE void operator()(size_t i, WorkItem& item, Args...)
 *
 * @param resource The compute resource
 * @param config Kernel configuration (must set shared_memory > 0 if needed)
 * @param kernel_functor The kernel functor
 * @param args Kernel arguments
 * @return Event for synchronization
 */
template <typename Functor, typename... Args>
Event launch_kernel_with_workitem(const Resource &resource,
                                  const KernelConfig &config,
                                  Functor kernel_functor, Args &&...args) {
  KernelConfig local_config = config;
  local_config.validate_block_size(resource);

  // Standardize problem size
  if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0 ||
      local_config.problem_size.z == 0) {
    kerneldim3 new_problem{};
    new_problem.x = std::max<idx_t>(1, local_config.grid_size.x *
                                           local_config.block_size.x);
    new_problem.y = std::max<idx_t>(1, local_config.grid_size.y *
                                           local_config.block_size.y);
    new_problem.z = std::max<idx_t>(1, local_config.grid_size.z *
                                           local_config.block_size.z);
    local_config.problem_size = new_problem;
  }

#ifdef USE_CUDA
  if (resource.type() == ResourceType::CUDA) {
    return launch_cuda_kernel_with_workitem(
        resource, local_config, kernel_functor,
        get_buffer_pointer(std::forward<Args>(args))...);
  }
#endif

#ifdef USE_SYCL
  if (resource.type() == ResourceType::SYCL) {
    // Default to int for shared memory type - can be specialized for other
    // types
    return launch_sycl_kernel_with_workitem<int>(
        resource, local_config, kernel_functor,
        get_buffer_pointer(std::forward<Args>(args))...);
  }
#endif

  // CPU fallback (WorkItem still works, just no shared memory)
  throw_value_error("launch_kernel_with_workitem requires GPU backend");
}

// ============================================================================
// Dimensional Kernel Launchers (1D, 2D, 3D)
// ============================================================================

/**
 * @brief 1D kernel launcher with auto-configuration
 */
template <typename Functor, typename... Args>
Event launch_kernel_1d(const Resource &resource, idx_t thread_count,
                       KernelConfig config, Functor kernel_functor,
                       Args... args) {
  config = KernelConfig::for_1d(thread_count, resource);
  return launch_kernel(resource, config, kernel_functor, args...);
}

/**
 * @brief 2D kernel launcher with auto-configuration
 */
template <typename Functor, typename... Args>
Event launch_kernel_2d(const Resource &resource, idx_t width, idx_t height,
                       KernelConfig config, Functor kernel_functor,
                       Args... args) {
  config = KernelConfig::for_2d(width, height, resource);
  return launch_kernel(resource, config, kernel_functor, args...);
}

/**
 * @brief 3D kernel launcher with auto-configuration
 */
template <typename Functor, typename... Args>
Event launch_kernel_3d(const Resource &resource, idx_t width, idx_t height,
                       idx_t depth, KernelConfig config, Functor kernel_functor,
                       Args... args) {
  config = KernelConfig::for_3d(width, height, depth, resource);
  return launch_kernel(resource, config, kernel_functor, args...);
}

// ============================================================================
// CPU Kernel Launcher (Host-only)
// ============================================================================

/**
 * @brief CPU kernel launcher - streamlined interface
 */
template <typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource &resource, const KernelConfig &config,
                        Functor kernel_functor, Args... args) {

  config.dependencies.wait_all();

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  idx_t thread_count = config.grid_size.x * config.block_size.x *
                       config.grid_size.y * config.block_size.y *
                       config.grid_size.z * config.block_size.z;
  std::vector<std::thread> threads;
  idx_t chunk_size = (thread_count + num_threads - 1) / num_threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    threads.emplace_back([=]() mutable {
      idx_t start = t * chunk_size;
      idx_t end = std::min(start + chunk_size, thread_count);
      for (idx_t i = start; i < end; ++i) {
        kernel_functor(i, args...);
      }
    });
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  return Event(nullptr, resource);
}

// ============================================================================
// Result Wrapper for Kernel Calls
// ============================================================================

template <typename T> struct KernelResult {
  T result;
  Event completion_event;

  KernelResult(T &&res, Event &&event)
      : result(std::forward<T>(res)), completion_event(std::move(event)) {}

  void wait() { completion_event.wait(); }

  bool is_ready() const { return completion_event.is_complete(); }

  T get() {
    wait();
    return std::move(result);
  }
};

class KernelGraph {
private:
  struct KernelNode {
    std::function<Event()> launcher;
    std::vector<size_t> dependencies;
    size_t node_id;
    std::string name;
    Event completion_event;
    bool executed{false};
  };

#ifdef USE_CUDA
  cudaGraph_t cuda_graph_{nullptr};
  cudaGraphExec_t cuda_graph_instance_{nullptr};
  bool is_recorded_{false};
#endif

#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
  using command_graph = sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::modifiable>;
  using executable_graph = sycl::ext::oneapi::experimental::command_graph<
      sycl::ext::oneapi::experimental::graph_state::executable>;
  command_graph *sycl_graph_{nullptr};
  executable_graph *sycl_exec_graph_{nullptr};
  bool sycl_graph_recorded_{false};
#endif

  std::vector<KernelNode> nodes_;
  const Resource &resource_;

public:
  explicit KernelGraph(const Resource &resource) : resource_(resource) {
#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
    if (resource.type() == ResourceType::SYCL) {
      sycl::queue &q = *static_cast<sycl::queue *>(resource.get_stream());
      sycl_graph_ = new command_graph(q.get_context(), q.get_device());
    }
#endif
  }

  ~KernelGraph() {
#ifdef USE_CUDA
    if (cuda_graph_instance_)
      cudaGraphExecDestroy(cuda_graph_instance_);
    if (cuda_graph_)
      cudaGraphDestroy(cuda_graph_);
#endif
#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
    if (sycl_exec_graph_)
      delete sycl_exec_graph_;
    if (sycl_graph_)
      delete sycl_graph_;
#endif
  }

  // Add kernel with direct arguments (zero overhead)
  template <typename Functor, typename... Args>
  size_t add_kernel(const std::string &name, idx_t thread_count,
                    Functor kernel_functor, const KernelConfig &base_config,
                    Args... args) {

    size_t node_id = nodes_.size();

    // Zero-overhead launcher - capture by value, no std::forward
    auto launcher = [=, this]() -> Event {
      KernelConfig config = base_config;
      config.sync = false;
      return launch_kernel(resource_, config, kernel_functor, args...);
    };

    nodes_.emplace_back(
        KernelNode{launcher, {}, node_id, name, Event{}, false});

    return node_id;
  }

  void add_dependency(size_t dependent, size_t dependency) {
    if (dependent < nodes_.size() && dependency < nodes_.size()) {
      nodes_[dependent].dependencies.push_back(dependency);
    }
  }

  EventList execute() {
#ifdef USE_CUDA
    if (resource_.type() == ResourceType::CUDA && !nodes_.empty()) {
      if (!is_recorded_) {
        record_cuda_graph();
      }

      cudaStream_t stream = static_cast<cudaStream_t>(resource_.get_stream());
      CUDA_CHECK(cudaGraphLaunch(cuda_graph_instance_, stream));

      cudaEvent_t completion_event;
      CUDA_CHECK(
          cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(completion_event, stream));

      EventList result;
      result.add(Event(completion_event, resource_));
      return result;
    }
#endif

#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
    if (resource_.type() == ResourceType::SYCL && !nodes_.empty() &&
        sycl_exec_graph_) {
      if (!sycl_graph_recorded_) {
        record_sycl_graph();
      }

      sycl::queue &q = *static_cast<sycl::queue *>(resource_.get_stream());
      auto sycl_event = q.submit(
          [&](sycl::handler &h) { h.ext_oneapi_graph(*sycl_exec_graph_); });

      EventList result;
      result.add(Event(sycl_event, resource_));
      return result;
    }
#endif

    return execute_topologically();
  }

private:
#ifdef USE_CUDA
  void record_cuda_graph() {
    cudaStream_t stream = static_cast<cudaStream_t>(resource_.get_stream());

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    execute_topologically();
    CUDA_CHECK(cudaStreamEndCapture(stream, &cuda_graph_));
    CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_instance_, cuda_graph_, nullptr,
                                    nullptr, 0));
    is_recorded_ = true;
  }
#endif

#if defined(USE_SYCL) && defined(USE_SYCL_ICPX)
  void record_sycl_graph() {
    // Record SYCL graph nodes
    for (auto &node : nodes_) {
      sycl_graph_->add([&](sycl::handler &h) {
        // Add kernel to graph
        node.launcher();
      });
    }

    sycl_exec_graph_ = new executable_graph(sycl_graph_->finalize());
    sycl_graph_recorded_ = true;
  }
#endif

  EventList execute_topologically() {
    EventList all_events;
    std::vector<bool> visited(nodes_.size(), false);

    std::function<void(idx_t)> execute_node;
    execute_node = [&](idx_t node_id) {
      if (visited[node_id] || nodes_[node_id].executed)
        return;

      auto &node = nodes_[node_id];

      for (idx_t dep_id : node.dependencies) {
        execute_node(dep_id);
      }

      for (idx_t dep_id : node.dependencies) {
        nodes_[dep_id].completion_event.wait();
      }

      node.completion_event = node.launcher();
      node.executed = true;
      visited[node_id] = true;

      all_events.add(node.completion_event);
    };

    for (idx_t i = 0; i < nodes_.size(); ++i) {
      execute_node(i);
    }

    return all_events;
  }
};

/**
 * @brief Kernel Pipeline
 * A pipeline of kernels that are executed in order.
 * The pipeline is executed in order, and the output of each kernel is used as
 * the input to the next kernel.
 * @param resource The resource to use for the pipeline.
 * @param stream_id The stream ID to use for the pipeline.
 */
class KernelPipeline {
private:
  const Resource &resource_;
  void *dedicated_queue_;
  EventList pipeline_events_;

public:
  explicit KernelPipeline(const Resource &resource, int stream_id = 0)
      : resource_(resource) {
    dedicated_queue_ = (stream_id == 0) ? resource.get_stream()
                                        : resource.get_stream(stream_id);
  }

  template <typename Functor, typename... Args>
  KernelPipeline &then(idx_t thread_count, Functor kernel_functor,
                       const KernelConfig &base_config, Args... args) {

    KernelConfig config = base_config;
    config.explicit_queue = dedicated_queue_;
    config.dependencies = pipeline_events_;
    config.sync = false;

    Event completion =
        launch_kernel(resource_, config, kernel_functor, args...);

    pipeline_events_.clear();
    pipeline_events_.add(completion);

    return *this;
  }

  void synchronize() { pipeline_events_.wait_all(); }

  EventList get_events() const { return pipeline_events_; }
};

} // namespace ARBD
