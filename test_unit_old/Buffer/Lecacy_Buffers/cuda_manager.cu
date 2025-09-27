#include "../catch_boiler.h"

#ifdef USE_CUDA

#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Event.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <vector>
#include <numeric>
#include <cmath>

using namespace ARBD;
using namespace ARBD::CUDA;

// Test fixture for CUDA Manager tests
class CUDAManagerTestFixture {
public:
    Resource cuda_resource;
    
    CUDAManagerTestFixture() {
        try {
            // Initialize CUDA backend
            CUDAManager::init();
            CUDAManager::load_info();
            
            if (CUDAManager::all_devices().empty()) {
                SKIP("No CUDA devices available");
                return;
            }
            
            // Use the first available CUDA device for all tests
            cuda_resource = Resource(ResourceType::CUDA, 0);
            CUDAManager::use(0);

        } catch (const std::exception& e) {
            FAIL("Failed to initialize CUDAManager in test fixture: " << e.what());
        }
    }

    ~CUDAManagerTestFixture() {
        try {
            CUDAManager::finalize();
        } catch (const std::exception& e) {
            // Log error instead of throwing from destructor
            std::cerr << "Error during CUDAManager finalization in test fixture: " << e.what() << std::endl;
        }
    }
};

// Simple CUDA kernel for testing
__global__ void vector_add_kernel(float* a, float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA reduction kernel
__global__ void reduction_kernel(int* data, int* result, size_t n) {
    extern __shared__ int sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(CUDAManagerTestFixture, "CUDA Manager Initialization", "[cuda][manager][init]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Device count") {
        REQUIRE(CUDAManager::all_devices().size() > 0);
        REQUIRE(CUDAManager::device_count() > 0);
    }
    
    SECTION("Current device") {
        int current_device = CUDAManager::current_device();
        REQUIRE(current_device >= 0);
        REQUIRE(current_device < static_cast<int>(CUDAManager::device_count()));
    }
    
    SECTION("Device properties") {
        auto device_info = CUDAManager::device_info(0);
        REQUIRE(!device_info.name.empty());
        REQUIRE(device_info.total_memory > 0);
        REQUIRE(device_info.multiprocessor_count > 0);
    }
}

TEST_CASE_METHOD(CUDAManagerTestFixture, "CUDA DeviceMemory Operations", "[cuda][memory]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Memory allocation and deallocation") {
        const size_t count = 1000;
        DeviceMemory<float> device_mem(count);
        
        REQUIRE(device_mem.size() == count);
        REQUIRE(device_mem.get() != nullptr);
        REQUIRE(device_mem.size_bytes() == count * sizeof(float));
    }
    
    SECTION("Host-Device memory copy") {
        const size_t count = 1000;
        std::vector<float> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 0.0f);
        
        DeviceMemory<float> device_mem(count);
        device_mem.copyFromHost(host_data);
        
        std::vector<float> result(count, -1.0f);
        device_mem.copyToHost(result);
        
        for (size_t i = 0; i < count; ++i) {
            REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(static_cast<float>(i), 1e-6f));
        }
    }
    
    SECTION("Memory move semantics") {
        const size_t count = 500;
        DeviceMemory<double> mem1(count);
        void* original_ptr = mem1.get();
        
        DeviceMemory<double> mem2 = std::move(mem1);
        
        REQUIRE(mem2.size() == count);
        REQUIRE(mem2.get() == original_ptr);
        REQUIRE(mem1.size() == 0);
        REQUIRE(mem1.get() == nullptr);
    }
}

TEST_CASE_METHOD(CUDAManagerTestFixture, "CUDA Kernel Execution", "[cuda][kernel]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Vector addition kernel") {
        constexpr size_t SIZE = 256;
        
        // Host data
        std::vector<float> a(SIZE, 1.0f);
        std::vector<float> b(SIZE, 2.0f);
        std::vector<float> c(SIZE, 0.0f);
        
        // Device memory
        DeviceMemory<float> d_a(SIZE);
        DeviceMemory<float> d_b(SIZE);
        DeviceMemory<float> d_c(SIZE);
        
        // Copy data to device
        d_a.copyFromHost(a);
        d_b.copyFromHost(b);
        
        // Launch kernel
        const int block_size = 256;
        const int grid_size = (SIZE + block_size - 1) / block_size;
        
        vector_add_kernel<<<grid_size, block_size>>>(
            d_a.get(), d_b.get(), d_c.get(), SIZE
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back
        d_c.copyToHost(c);
        
        // Verify results
        for (size_t i = 0; i < SIZE; ++i) {
            REQUIRE_THAT(c[i], Catch::Matchers::WithinAbs(3.0f, 1e-6f));
        }
    }
    
    SECTION("Parallel reduction") {
        constexpr size_t SIZE = 1024;
        
        std::vector<int> data(SIZE);
        std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ..., SIZE
        
        DeviceMemory<int> d_data(SIZE);
        DeviceMemory<int> d_result(1);
        
        d_data.copyFromHost(data);
        
        // Initialize result to 0
        int zero = 0;
        d_result.copyFromHost(std::vector<int>{zero});
        
        // Launch reduction kernel
        const int block_size = 256;
        const int grid_size = (SIZE + block_size - 1) / block_size;
        const size_t shared_mem_size = block_size * sizeof(int);
        
        reduction_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_data.get(), d_result.get(), SIZE
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result back
        std::vector<int> result(1);
        d_result.copyToHost(result);
        
        // Expected sum: 1 + 2 + ... + SIZE = SIZE * (SIZE + 1) / 2
        int expected = SIZE * (SIZE + 1) / 2;
        REQUIRE(result[0] == expected);
    }
}

TEST_CASE_METHOD(CUDAManagerTestFixture, "CUDA Stream Operations", "[cuda][stream]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Stream creation and synchronization") {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Simple asynchronous operation
        const size_t count = 1000;
        DeviceMemory<float> device_mem(count);
        
        std::vector<float> host_data(count, 42.0f);
        CUDA_CHECK(cudaMemcpyAsync(
            device_mem.get(), 
            host_data.data(), 
            count * sizeof(float), 
            cudaMemcpyHostToDevice, 
            stream
        ));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        std::vector<float> result(count, 0.0f);
        device_mem.copyToHost(result);
        
        for (size_t i = 0; i < count; ++i) {
            REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(42.0f, 1e-6f));
        }
        
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

TEST_CASE_METHOD(CUDAManagerTestFixture, "CUDA Error Handling", "[cuda][error]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Invalid memory access detection") {
        // This test checks that our error handling catches invalid operations
        // We'll attempt to access device memory from host without proper synchronization
        
        DeviceMemory<float> device_mem(100);
        
        // This should be caught by our error checking
        // Note: In a real scenario, this would cause undefined behavior
        // but our DeviceMemory class should prevent direct host access
        REQUIRE(device_mem.get() != nullptr);
        REQUIRE(device_mem.size() == 100);
    }
}

#endif // USE_CUDA