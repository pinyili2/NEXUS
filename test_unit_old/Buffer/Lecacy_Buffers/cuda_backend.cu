#include "../catch_boiler.h"

#ifdef USE_CUDA

#include "Backend/CUDA/CUDAManager.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Backend/Events.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

using namespace ARBD;
using namespace ARBD::CUDA;

// Test fixture for CUDA Backend tests
class CUDATestFixture {
public:
    Resource cuda_resource;
    
    CUDATestFixture() {
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

    ~CUDATestFixture() {
        try {
            CUDAManager::finalize();
        } catch (const std::exception& e) {
            // Log error instead of throwing from destructor
            std::cerr << "Error during CUDAManager finalization in test fixture: " << e.what() << std::endl;
        }
    }
};

// Simple test class for proxy testing
class TestMath {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    float multiply(float a, float b) {
        return a * b;
    }
};

// Simple CUDA kernels for testing
__global__ void simple_add_kernel(float* result, float a, float b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = a + b;
    }
}

__global__ void vector_scale_kernel(float* data, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(CUDATestFixture, "CUDA Resource Creation and Properties", "[cuda][resource]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }

    SECTION("Resource type is correct") {
        CHECK(cuda_resource.type == ResourceType::CUDA);
        CHECK(cuda_resource.id == 0);
    }

    SECTION("Resource type string") {
        CHECK(std::string(cuda_resource.getTypeString()) == "CUDA");
    }

    SECTION("Resource is device") {
        CHECK(cuda_resource.is_device());
        CHECK_FALSE(cuda_resource.is_host());
    }
}

TEST_CASE_METHOD(CUDATestFixture, "CUDA DeviceBuffer Basic Operations", "[cuda][buffer]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }

    SECTION("Buffer allocation") {
        const size_t count = 1000;
        DeviceBuffer<float> buffer(count, cuda_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.get_resource() == cuda_resource);
        CHECK(buffer.data() != nullptr);
    }
    
    SECTION("Buffer move semantics") {
        const size_t count = 500;
        DeviceBuffer<double> buffer1(count, cuda_resource);
        auto* original_ptr = buffer1.data();
        
        DeviceBuffer<double> buffer2 = std::move(buffer1);
        
        CHECK(buffer2.size() == count);
        CHECK(buffer2.data() == original_ptr);
        CHECK(buffer1.size() == 0);
        CHECK(buffer1.empty());
    }
    
    SECTION("Zero-sized buffer") {
        DeviceBuffer<int> empty_buffer(0, cuda_resource);
        
        CHECK(empty_buffer.size() == 0);
        CHECK(empty_buffer.empty());
        CHECK(empty_buffer.get_resource() == cuda_resource);
    }
    
    SECTION("Different data types") {
        DeviceBuffer<float> float_buffer(100, cuda_resource);
        DeviceBuffer<double> double_buffer(200, cuda_resource);
        DeviceBuffer<int> int_buffer(300, cuda_resource);
        DeviceBuffer<char> char_buffer(400, cuda_resource);
        
        CHECK(float_buffer.size() == 100);
        CHECK(double_buffer.size() == 200);
        CHECK(int_buffer.size() == 300);
        CHECK(char_buffer.size() == 400);
        
        CHECK(float_buffer.data() != nullptr);
        CHECK(double_buffer.data() != nullptr);
        CHECK(int_buffer.data() != nullptr);
        CHECK(char_buffer.data() != nullptr);
    }
}

TEST_CASE_METHOD(CUDATestFixture, "CUDA Buffer Data Transfer", "[cuda][buffer][transfer]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Host to device transfer") {
        const size_t count = 1000;
        std::vector<float> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1.0f);
        
        DeviceBuffer<float> device_buffer(count, cuda_resource);
        device_buffer.copy_from_host(host_data.data());
        
        std::vector<float> result(count, 0.0f);
        device_buffer.copy_to_host(result.data());
        
        for (size_t i = 0; i < count; ++i) {
            REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(static_cast<float>(i + 1), 1e-6f));
        }
    }
    
    SECTION("Device to device transfer") {
        const size_t count = 500;
        std::vector<double> host_data(count, 3.14159);
        
        DeviceBuffer<double> buffer1(count, cuda_resource);
        DeviceBuffer<double> buffer2(count, cuda_resource);
        
        buffer1.copy_from_host(host_data.data());
        buffer2.copy_from(buffer1);
        
        std::vector<double> result(count, 0.0);
        buffer2.copy_to_host(result.data());
        
        for (size_t i = 0; i < count; ++i) {
            REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(3.14159, 1e-10));
        }
    }
}

TEST_CASE_METHOD(CUDATestFixture, "CUDA Kernel Execution with DeviceBuffer", "[cuda][kernel][buffer]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Simple kernel execution") {
        DeviceBuffer<float> result_buffer(1, cuda_resource);
        
        float a = 5.0f, b = 3.0f;
        simple_add_kernel<<<1, 1>>>(result_buffer.data(), a, b);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<float> result(1);
        result_buffer.copy_to_host(result.data());
        
        REQUIRE_THAT(result[0], Catch::Matchers::WithinAbs(8.0f, 1e-6f));
    }
    
    SECTION("Vector processing kernel") {
        const size_t count = 1000;
        std::vector<float> host_data(count, 2.0f);
        
        DeviceBuffer<float> device_buffer(count, cuda_resource);
        device_buffer.copy_from_host(host_data.data());
        
        const float scale = 3.5f;
        const int block_size = 256;
        const int grid_size = (count + block_size - 1) / block_size;
        
        vector_scale_kernel<<<grid_size, block_size>>>(
            device_buffer.data(), scale, count
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<float> result(count);
        device_buffer.copy_to_host(result.data());
        
        const float expected = 2.0f * 3.5f;
        for (size_t i = 0; i < count; ++i) {
            REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(expected, 1e-5f));
        }
    }
}

TEST_CASE_METHOD(CUDATestFixture, "CUDA Event Operations", "[cuda][events]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Event creation and timing") {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        const size_t count = 10000;
        DeviceBuffer<float> buffer(count, cuda_resource);
        std::vector<float> host_data(count, 1.0f);
        
        CUDA_CHECK(cudaEventRecord(start));
        
        buffer.copy_from_host(host_data.data());
        
        const int block_size = 256;
        const int grid_size = (count + block_size - 1) / block_size;
        vector_scale_kernel<<<grid_size, block_size>>>(
            buffer.data(), 2.0f, count
        );
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        
        // Just check that timing worked (elapsed time should be positive)
        REQUIRE(elapsed_time >= 0.0f);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

TEST_CASE_METHOD(CUDATestFixture, "CUDA Memory Allocation Patterns", "[cuda][memory][patterns]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Multiple simultaneous allocations") {
        const size_t count = 1000;
        
        std::vector<std::unique_ptr<DeviceBuffer<float>>> buffers;
        for (int i = 0; i < 10; ++i) {
            buffers.push_back(std::make_unique<DeviceBuffer<float>>(count, cuda_resource));
        }
        
        // Verify all allocations succeeded
        for (const auto& buffer : buffers) {
            REQUIRE(buffer->size() == count);
            REQUIRE(buffer->data() != nullptr);
        }
    }
    
    SECTION("Large allocation") {
        // Try to allocate a reasonably large buffer
        const size_t large_count = 1024 * 1024; // 1M floats = 4MB
        
        try {
            DeviceBuffer<float> large_buffer(large_count, cuda_resource);
            REQUIRE(large_buffer.size() == large_count);
            REQUIRE(large_buffer.data() != nullptr);
            
            // Test that we can actually use the memory
            std::vector<float> test_data(1000, 42.0f);
            large_buffer.copy_from_host(test_data.data(), 1000);
            
            std::vector<float> result(1000);
            large_buffer.copy_to_host(result.data(), 1000);
            
            for (size_t i = 0; i < 1000; ++i) {
                REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(42.0f, 1e-6f));
            }
        } catch (const std::exception& e) {
            // If allocation fails due to insufficient memory, that's also acceptable
            WARN("Large allocation failed (may be due to insufficient GPU memory): " << e.what());
        }
    }
}

TEST_CASE_METHOD(CUDATestFixture, "CUDA Backend Integration", "[cuda][integration]") {
    if (CUDAManager::all_devices().empty()) {
        SKIP("No CUDA devices available");
    }
    
    SECTION("Resource and buffer integration") {
        Resource host_resource; // Default host resource
        
        const size_t count = 500;
        std::vector<float> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1.0f);
        
        // Create device buffer
        DeviceBuffer<float> device_buffer(count, cuda_resource);
        device_buffer.copy_from_host(host_data.data());
        
        // Verify data integrity
        std::vector<float> result(count);
        device_buffer.copy_to_host(result.data());
        
        for (size_t i = 0; i < count; ++i) {
            REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(static_cast<float>(i + 1), 1e-6f));
        }
    }
}

#endif // USE_CUDA