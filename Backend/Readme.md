- Folder for Backend Object.
If Tesela has its own machine it would go here (?)
Open for Kokos, Vulken...etc implementaitons

- Works:
SYCL: Buffer, Events, Resource, Backend.
- For metal, guard headers with #ifndef __metal_version__ if don't want to be seen from xcode compiler!!

```cpp
// Complete flow: Resource → Stream → Kernel → Event

namespace ARBD {

// ============================================================================
// 1. Resource.h manages stream access
// ============================================================================

class Resource {
public:
    // Primary stream access method
    void* get_stream(StreamType stream_type = StreamType::Compute) const {
        #ifdef USE_CUDA
        if (type == ResourceType::CUDA) {
            auto& device = CUDA::Manager::get_device(id);
            return device.get_next_stream();  // Round-robin stream selection
        }
        #endif

        #ifdef USE_SYCL
        if (type == ResourceType::SYCL) {
            auto& device = SYCL::Manager::get_device(id);
            return &device.get_next_queue();  // Round-robin queue selection
        }
        #endif

        return nullptr;
    }

    // Specific stream by ID
    void* get_stream(size_t stream_id, StreamType stream_type = StreamType::Compute) const {
        #ifdef USE_CUDA
        if (type == ResourceType::CUDA) {
            auto& device = CUDA::Manager::get_device(id);
            return device.get_stream(stream_id);  // Specific stream
        }
        #endif

        #ifdef USE_SYCL
        if (type == ResourceType::SYCL) {
            auto& device = SYCL::Manager::get_device(id);
            return &device.get_queue(stream_id);  // Specific queue
        }
        #endif

        return nullptr;
    }
};

// ============================================================================
// 2. Backend Managers control stream pools
// ============================================================================

namespace CUDA {
class Manager {
public:
    static constexpr size_t NUM_STREAMS = 8;

    class Device {
    private:
        std::array<cudaStream_t, NUM_STREAMS> streams_;
        std::atomic<size_t> next_stream_index_{0};

    public:
        cudaStream_t get_next_stream() {
            size_t index = next_stream_index_.fetch_add(1) % NUM_STREAMS;
            return streams_[index];
        }

        cudaStream_t get_stream(size_t stream_id) {
            return streams_[stream_id % NUM_STREAMS];
        }
    };
};
}

namespace SYCL {
class Manager {
public:
    static constexpr size_t NUM_QUEUES = 8;

    class Device {
    private:
        std::array<sycl::queue, NUM_QUEUES> queues_;
        std::atomic<size_t> next_queue_index_{0};

    public:
        sycl::queue& get_next_queue() {
            size_t index = next_queue_index_.fetch_add(1) % NUM_QUEUES;
            return queues_[index];
        }

        sycl::queue& get_queue(size_t queue_id) {
            return queues_[queue_id % NUM_QUEUES];
        }
    };
};
}

// ============================================================================
// 3. Kernel launches use Resource streams and produce Events
// ============================================================================

template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_kernel(const Resource& resource,
                   idx_t thread_count,
                   const KernelConfig& config,
                   const InputTuple& inputs,
                   const OutputTuple& outputs,
                   Functor&& kernel_func,
                   Args&&... args) {

    // Step 1: Resource provides stream
    void* stream = config.explicit_queue;
    if (!stream) {
        stream = resource.get_stream(config.stream_id);  // ← Resource manages streams
    }

    // Step 2: Launch kernel on stream
    #ifdef USE_CUDA
    if (resource.type == ResourceType::CUDA) {
        return launch_cuda_kernel_impl(resource, thread_count, stream,
                                      config, inputs, outputs,
                                      std::forward<Functor>(kernel_func),
                                      std::forward<Args>(args)...);
    }
    #endif

    // Step 3: Kernel produces Event for completion tracking
    return Event{};  // ← Kernel produces Event
}

// ============================================================================
// 4. Complete workflow example
// ============================================================================

void example_complete_workflow() {
    // 1. Create resource (points to device 0)
    Resource cuda_resource{ResourceType::CUDA, 0};

    // 2. Resource provides streams
    void* stream_0 = cuda_resource.get_stream(0);  // Specific stream
    void* stream_1 = cuda_resource.get_stream(1);  // Different stream
    void* auto_stream = cuda_resource.get_stream(); // Auto round-robin

    // 3. Launch kernels that produce events
    KernelConfig config_A;
    config_A.explicit_queue = stream_0;  // Use specific stream

    Event event_A = launch_kernel(cuda_resource, 1000, config_A,
                                 inputs_A, outputs_A, kernel_A);

    KernelConfig config_B;
    config_B.explicit_queue = stream_1;  // Different stream = parallel execution
    config_B.dependencies.add(event_A);  // But depends on kernel A

    Event event_B = launch_kernel(cuda_resource, 1000, config_B,
                                 inputs_B, outputs_B, kernel_B);

    // 4. Events control execution flow
    event_B.wait();  // Wait for both kernels to complete
}

// ============================================================================
// 5. Stream lifecycle is managed by Resource backends
// ============================================================================

// Resource.h provides the interface:
// - get_stream() → stream/queue pointer
// - synchronize_streams() → sync all streams

// Backend managers provide the implementation:
// - CUDAManager: manages cudaStream_t pools per device
// - SYCLManager: manages sycl::queue pools per device
// - MetalManager: manages MTL::CommandQueue pools per device

// Kernels use streams and produce events:
// - Stream determines WHERE kernel executes
// - Event determines WHEN kernel completes
// - EventList coordinates DEPENDENCIES across streams/devices

// ============================================================================
// 5. Multi-device coordination (TO BE IMPLEMENTED)
// ============================================================================

void example_multi_device() {
    // Multiple resources = multiple devices
    Resource cuda_0{ResourceType::CUDA, 0};
    Resource cuda_1{ResourceType::CUDA, 1};
    Resource sycl_cpu{ResourceType::SYCL, 0};

    // Each resource manages its own streams
    EventList device_events;

    // GPU 0: Launch kernel on its streams
    Event gpu0_event = launch_kernel(cuda_0, 1000, {},
                                    inputs_gpu0, outputs_gpu0, kernel_compute);
    device_events.add(gpu0_event);

    // GPU 1: Launch kernel on its streams
    Event gpu1_event = launch_kernel(cuda_1, 1000, {},
                                    inputs_gpu1, outputs_gpu1, kernel_compute);
    device_events.add(gpu1_event);

    // CPU: Launch kernel (no streams, but same interface)
    Event cpu_event = launch_kernel(sycl_cpu, 1000, {},
                                   inputs_cpu, outputs_cpu, kernel_compute);
    device_events.add(cpu_event);

    // Events coordinate across all devices
    device_events.wait_all();  // Wait for all devices to complete

    // Final reduction kernel depends on all devices
    KernelConfig reduction_config;
    reduction_config.dependencies = device_events;

    Event final_event = launch_kernel(cuda_0, 100, reduction_config,
                                     all_outputs, final_result,
                                     reduction_kernel);

    final_event.wait();
}
} // namespace ARBD
```
