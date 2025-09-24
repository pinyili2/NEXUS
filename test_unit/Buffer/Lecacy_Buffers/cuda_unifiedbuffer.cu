#include "../catch_boiler.h"

#ifdef USE_CUDA

#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Backend/CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

using namespace ARBD;
using namespace ARBD::CUDA;

// Simple CUDA kernels for UnifiedBuffer testing
__global__ void buffer_fill_kernel(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void buffer_transform_kernel(float* data, float multiplier, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= multiplier;
    }
}

__global__ void buffer_copy_kernel(const float* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void buffer_sum_reduce_kernel(const float* data, float* result, size_t n) {
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
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

TEST_CASE("UnifiedBuffer CUDA Basic Allocation", "[UnifiedBuffer][CUDA]") {
    try {
        // Initialize CUDA backend
        CUDAManager::init();
        CUDAManager::load_info();
        
        if (CUDAManager::all_devices().empty()) {
            SKIP("No CUDA devices available");
        }
        
        Resource cuda_resource{ResourceType::CUDA, 0};
        
        SECTION("Basic allocation and deallocation") {
            UnifiedBuffer<float> buffer(1000, cuda_resource);
            
            REQUIRE(buffer.size() == 1000);
            REQUIRE(!buffer.empty());
            REQUIRE(buffer.primary_location().type == ResourceType::CUDA);
            
            // Check that pointer is valid
            float* ptr = buffer.get_ptr(cuda_resource);
            REQUIRE(ptr != nullptr);
        }
        
        SECTION("Zero-sized buffer") {
            UnifiedBuffer<int> buffer(0, cuda_resource);
            
            REQUIRE(buffer.size() == 0);
            REQUIRE(buffer.empty());
        }
        
        SECTION("Different data types") {
            UnifiedBuffer<double> double_buffer(500, cuda_resource);
            UnifiedBuffer<int> int_buffer(1000, cuda_resource);
            UnifiedBuffer<char> char_buffer(2000, cuda_resource);
            
            REQUIRE(double_buffer.size() == 500);
            REQUIRE(int_buffer.size() == 1000);
            REQUIRE(char_buffer.size() == 2000);
            
            REQUIRE(double_buffer.get_ptr(cuda_resource) != nullptr);
            REQUIRE(int_buffer.get_ptr(cuda_resource) != nullptr);
            REQUIRE(char_buffer.get_ptr(cuda_resource) != nullptr);
        }
        
        CUDAManager::finalize();
    } catch (const Exception& e) {
        FAIL("CUDA test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer CUDA Data Transfer", "[UnifiedBuffer][CUDA]") {
    try {
        // Initialize CUDA backend
        CUDAManager::init();
        CUDAManager::load_info();
        
        if (CUDAManager::all_devices().empty()) {
            SKIP("No CUDA devices available");
        }
        
        Resource cuda_resource{ResourceType::CUDA, 0};
        Resource host_resource{}; // Default host resource
        
        SECTION("Host to device transfer") {
            const size_t size = 1000;
            std::vector<float> host_data(size);
            std::iota(host_data.begin(), host_data.end(), 1.0f);
            
            UnifiedBuffer<float> buffer(size, host_resource);
            
            // Fill buffer with host data
            float* host_ptr = buffer.get_ptr(host_resource);
            std::copy(host_data.begin(), host_data.end(), host_ptr);
            
            // Transfer to device
            buffer.ensure_location(cuda_resource);
            float* device_ptr = buffer.get_ptr(cuda_resource);
            REQUIRE(device_ptr != nullptr);
            
            // Transfer back to host and verify
            buffer.ensure_location(host_resource);
            host_ptr = buffer.get_ptr(host_resource);
            
            for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(host_ptr[i], Catch::Matchers::WithinAbs(static_cast<float>(i + 1), 1e-6f));
            }
        }
        
        SECTION("Device computation") {
            const size_t size = 1000;
            UnifiedBuffer<float> buffer(size, cuda_resource);
            
            // Fill buffer on device
            float* device_ptr = buffer.get_ptr(cuda_resource);
            const int block_size = 256;
            const int grid_size = (size + block_size - 1) / block_size;
            
            buffer_fill_kernel<<<grid_size, block_size>>>(device_ptr, 5.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Transform data on device
            buffer_transform_kernel<<<grid_size, block_size>>>(device_ptr, 2.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Transfer to host and verify
            buffer.ensure_location(host_resource);
            float* host_ptr = buffer.get_ptr(host_resource);
            
            for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(host_ptr[i], Catch::Matchers::WithinAbs(10.0f, 1e-6f));
            }
        }
        
        CUDAManager::finalize();
    } catch (const Exception& e) {
        FAIL("CUDA data transfer test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer CUDA Move Semantics", "[UnifiedBuffer][CUDA]") {
    try {
        // Initialize CUDA backend
        CUDAManager::init();
        CUDAManager::load_info();
        
        if (CUDAManager::all_devices().empty()) {
            SKIP("No CUDA devices available");
        }
        
        Resource cuda_resource{ResourceType::CUDA, 0};
        
        SECTION("Move constructor") {
            UnifiedBuffer<float> buffer1(1000, cuda_resource);
            float* original_ptr = buffer1.get_ptr(cuda_resource);
            
            UnifiedBuffer<float> buffer2(std::move(buffer1));
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(cuda_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
            REQUIRE(buffer1.empty());
        }
        
        SECTION("Move assignment") {
            UnifiedBuffer<float> buffer1(1000, cuda_resource);
            UnifiedBuffer<float> buffer2(500, cuda_resource);
            
            float* original_ptr = buffer1.get_ptr(cuda_resource);
            
            buffer2 = std::move(buffer1);
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(cuda_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
        }
        
        CUDAManager::finalize();
    } catch (const Exception& e) {
        FAIL("CUDA move semantics test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer CUDA Existing Data Constructor", "[UnifiedBuffer][CUDA]") {
    try {
        // Initialize CUDA backend
        CUDAManager::init();
        CUDAManager::load_info();
        
        if (CUDAManager::all_devices().empty()) {
            SKIP("No CUDA devices available");
        }
        
        Resource host_resource{}; // Default host resource
        Resource cuda_resource{ResourceType::CUDA, 0};
        
        SECTION("Construct from existing host data") {
            const size_t size = 500;
            std::vector<double> host_data(size, 3.14159);
            
            // Create buffer from existing data
            UnifiedBuffer<double> buffer(host_data.data(), size, host_resource);
            
            REQUIRE(buffer.size() == size);
            REQUIRE(buffer.primary_location().type == ResourceType::Host);
            
            // Verify data integrity
            double* ptr = buffer.get_ptr(host_resource);
            for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(ptr[i], Catch::Matchers::WithinAbs(3.14159, 1e-10));
            }
            
            // Transfer to device and back
            buffer.ensure_location(cuda_resource);
            buffer.ensure_location(host_resource);
            
            ptr = buffer.get_ptr(host_resource);
            for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(ptr[i], Catch::Matchers::WithinAbs(3.14159, 1e-10));
            }
        }
        
        CUDAManager::finalize();
    } catch (const Exception& e) {
        FAIL("CUDA existing data constructor test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer CUDA Multi-location Operations", "[UnifiedBuffer][CUDA]") {
    try {
        // Initialize CUDA backend
        CUDAManager::init();
        CUDAManager::load_info();
        
        if (CUDAManager::all_devices().empty()) {
            SKIP("No CUDA devices available");
        }
        
        Resource host_resource{};
        Resource cuda_resource{ResourceType::CUDA, 0};
        
        SECTION("Multi-location data consistency") {
            const size_t size = 1000;
            UnifiedBuffer<float> buffer(size, host_resource);
            
            // Initialize on host
            float* host_ptr = buffer.get_ptr(host_resource);
            for (size_t i = 0; i < size; ++i) {
                host_ptr[i] = static_cast<float>(i * 2);
            }
            
            // Ensure data is available on device
            buffer.ensure_location(cuda_resource);
            float* device_ptr = buffer.get_ptr(cuda_resource);
            REQUIRE(device_ptr != nullptr);
            
            // Modify data on device
            const int block_size = 256;
            const int grid_size = (size + block_size - 1) / block_size;
            buffer_transform_kernel<<<grid_size, block_size>>>(device_ptr, 0.5f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Ensure modified data is available on host
            buffer.ensure_location(host_resource);
            host_ptr = buffer.get_ptr(host_resource);
            
            // Verify data was modified correctly
            for (size_t i = 0; i < size; ++i) {
                float expected = static_cast<float>(i * 2) * 0.5f;
                REQUIRE_THAT(host_ptr[i], Catch::Matchers::WithinAbs(expected, 1e-6f));
            }
        }
        
        SECTION("Buffer copy operations") {
            const size_t size = 500;
            
            UnifiedBuffer<float> src_buffer(size, cuda_resource);
            UnifiedBuffer<float> dst_buffer(size, cuda_resource);
            
            // Fill source buffer
            float* src_ptr = src_buffer.get_ptr(cuda_resource);
            const int block_size = 256;
            const int grid_size = (size + block_size - 1) / block_size;
            
            buffer_fill_kernel<<<grid_size, block_size>>>(src_ptr, 42.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy to destination buffer using kernel
            float* dst_ptr = dst_buffer.get_ptr(cuda_resource);
            buffer_copy_kernel<<<grid_size, block_size>>>(src_ptr, dst_ptr, size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Verify copy on host
            dst_buffer.ensure_location(host_resource);
            float* host_ptr = dst_buffer.get_ptr(host_resource);
            
            for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(host_ptr[i], Catch::Matchers::WithinAbs(42.0f, 1e-6f));
            }
        }
        
        CUDAManager::finalize();
    } catch (const Exception& e) {
        FAIL("CUDA multi-location test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer CUDA Performance Operations", "[UnifiedBuffer][CUDA][performance]") {
    try {
        // Initialize CUDA backend
        CUDAManager::init();
        CUDAManager::load_info();
        
        if (CUDAManager::all_devices().empty()) {
            SKIP("No CUDA devices available");
        }
        
        Resource cuda_resource{ResourceType::CUDA, 0};
        Resource host_resource{};
        
        SECTION("Large buffer operations") {
            const size_t large_size = 100000;
            UnifiedBuffer<float> buffer(large_size, cuda_resource);
            
            // Fill with known pattern
            float* device_ptr = buffer.get_ptr(cuda_resource);
            const int block_size = 256;
            const int grid_size = (large_size + block_size - 1) / block_size;
            
            buffer_fill_kernel<<<grid_size, block_size>>>(device_ptr, 1.0f, large_size);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Perform reduction to verify all elements
            UnifiedBuffer<float> result_buffer(1, cuda_resource);
            float* result_ptr = result_buffer.get_ptr(cuda_resource);
            
            // Initialize result to zero
            CUDA_CHECK(cudaMemset(result_ptr, 0, sizeof(float)));
            
            const size_t shared_mem_size = block_size * sizeof(float);
            buffer_sum_reduce_kernel<<<grid_size, block_size, shared_mem_size>>>(
                device_ptr, result_ptr, large_size
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Transfer result to host and verify
            result_buffer.ensure_location(host_resource);
            float* host_result = result_buffer.get_ptr(host_resource);
            
            float expected_sum = static_cast<float>(large_size);
            REQUIRE_THAT(*host_result, Catch::Matchers::WithinAbs(expected_sum, large_size * 1e-6f));
        }
        
        SECTION("Memory efficiency test") {
            // Test that multiple buffers can coexist
            const size_t buffer_size = 10000;
            std::vector<std::unique_ptr<UnifiedBuffer<float>>> buffers;
            
            for (int i = 0; i < 5; ++i) {
                buffers.push_back(std::make_unique<UnifiedBuffer<float>>(buffer_size, cuda_resource));
                
                // Fill each buffer with different values
                float* ptr = buffers.back()->get_ptr(cuda_resource);
                const int block_size = 256;
                const int grid_size = (buffer_size + block_size - 1) / block_size;
                
                buffer_fill_kernel<<<grid_size, block_size>>>(ptr, static_cast<float>(i + 1), buffer_size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            
            // Verify each buffer has correct values
            for (size_t i = 0; i < buffers.size(); ++i) {
                buffers[i]->ensure_location(host_resource);
                float* host_ptr = buffers[i]->get_ptr(host_resource);
                
                float expected_value = static_cast<float>(i + 1);
                for (size_t j = 0; j < buffer_size; ++j) {
                    REQUIRE_THAT(host_ptr[j], Catch::Matchers::WithinAbs(expected_value, 1e-6f));
                }
            }
        }
        
        CUDAManager::finalize();
    } catch (const Exception& e) {
        FAIL("CUDA performance test failed with ARBD exception: " << e.what());
    }
}

#endif // USE_CUDA