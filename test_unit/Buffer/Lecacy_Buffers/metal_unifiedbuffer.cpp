#ifdef USE_METAL
#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Backend/METAL/METALManager.h"
#include <vector>
#include <numeric>
using namespace ARBD;

TEST_CASE("Metal Utility Functions", "[utilities]") {
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 80;
    
    SECTION("Buffer copy utility") {
        DeviceBuffer<int> source(count, metal_resource);
        DeviceBuffer<int> dest(count, metal_resource);
        
        // Fill source with data
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 100);
        source.copy_from_host(host_data.data(), count);
        
        CHECK_NOTHROW(copy_buffer(source, dest, metal_resource));
        
        // Verify copy worked
        std::vector<int> result(count);
        dest.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == host_data[i]);
        }
    }
    
    SECTION("Buffer fill utility") {
        DeviceBuffer<float> buffer(count, metal_resource);
        const float fill_value = 2.71828f; // e
        
        CHECK_NOTHROW(fill_buffer(buffer, fill_value, metal_resource));
        
        std::vector<float> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (float val : result) {
            CHECK(val == Approx(fill_value));
        }
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal Memory Operations", "[memory]") {
    // Initialize Metal manager and select devices
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Memory allocation and deallocation") {
        const size_t size = 1024 * sizeof(float);
        
        void* ptr = MemoryOps::allocate(size, metal_resource);
        CHECK(ptr != nullptr);
        
        CHECK_NOTHROW(MemoryOps::deallocate(ptr, metal_resource));
    }
    
    SECTION("Host to device memory copy") {
        const size_t count = 100;
        const size_t size = count * sizeof(int);
        
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1);
        
        void* device_ptr = MemoryOps::allocate(size, metal_resource);
        CHECK(device_ptr != nullptr);
        
        CHECK_NOTHROW(
            MemoryOps::copy_from_host(device_ptr, host_data.data(), size, metal_resource)
        );
        
        std::vector<int> result(count);
        CHECK_NOTHROW(
            MemoryOps::copy_to_host(result.data(), device_ptr, size, metal_resource)
        );
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == host_data[i]);
        }
        
        MemoryOps::deallocate(device_ptr, metal_resource);
    }
    
    SECTION("Large memory allocation") {
        const size_t large_size = 1024 * 1024 * sizeof(float); // 4MB
        
        void* large_ptr = MemoryOps::allocate(large_size, metal_resource);
        CHECK(large_ptr != nullptr);
        
        MemoryOps::deallocate(large_ptr, metal_resource);
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal GenericAllocator", "[allocator]") {
    // Initialize Metal manager and select devices
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Basic allocation") {
        const size_t count = 150;
        auto buffer = GenericAllocator<float>::allocate(count, metal_resource);
        
        CHECK(buffer.size() == count);
        CHECK(buffer.get_resource().type == ResourceType::METAL);
    }
    
    SECTION("Zero-initialized allocation") {
        const size_t count = 100;
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, metal_resource);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (int val : result) {
            CHECK(val == 0);
        }
    }
    
    SECTION("Sequence allocation") {
        const size_t count = 50;
        auto buffer = GenericAllocator<int>::allocate_sequence(count, metal_resource, 10, 2);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == 10 + static_cast<int>(i) * 2);
        }
    }
    
    SECTION("Custom initialization") {
        const size_t count = 75;
        auto init_func = [](size_t i) { return static_cast<float>(i * i); };
        auto buffer = GenericAllocator<float>::allocate_initialized(count, metal_resource, init_func);
        
        std::vector<float> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = static_cast<float>(i * i);
            CHECK(result[i] == Approx(expected));
        }
    }
}

TEST_CASE("Metal MultiRef Operations", "[multiref]") {
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 75;
    SECTION("MultiRef with multiple buffers") {
        DeviceBuffer<int> buffer1(count, metal_resource);
        DeviceBuffer<int> buffer2(count, metal_resource);
        DeviceBuffer<int> buffer3(count, metal_resource);
        
        auto multi_ref = make_multi_ref(buffer1, buffer2, buffer3);
        
        CHECK(multi_ref.size() == 3);
        
        EventList deps;
        auto ptrs = multi_ref.get_write_access(deps);
        
        // Verify we got valid pointers
        CHECK(std::get<0>(ptrs) != nullptr);
        CHECK(std::get<1>(ptrs) != nullptr);
        CHECK(std::get<2>(ptrs) != nullptr);
    }
    
    SECTION("MultiRef event completion") {
        DeviceBuffer<float> buffer1(count, metal_resource);
        DeviceBuffer<float> buffer2(count, metal_resource);
        
        auto multi_ref = make_multi_ref(buffer1, buffer2);
        
        Event dummy_event;
        CHECK_NOTHROW(multi_ref.complete_event_state(dummy_event));
    }
}

TEST_CASE("Metal DeviceBuffer Basic Operations", "[buffer]") {
    // Initialize Metal manager and select devices
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Empty buffer creation") {
        DeviceBuffer<int> buffer;
        CHECK(buffer.empty());
        CHECK(buffer.size() == 0);
    }
    
    SECTION("Buffer allocation") {
        const size_t count = 1000;
        DeviceBuffer<float> buffer(count, metal_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.get_resource().type == ResourceType::METAL);
        CHECK(buffer.data() != nullptr);
    }
    
    SECTION("Buffer move semantics") {
        const size_t count = 500;
        DeviceBuffer<double> buffer1(count, metal_resource);
        auto* original_ptr = buffer1.data();
        
        DeviceBuffer<double> buffer2 = std::move(buffer1);
        
        CHECK(buffer2.size() == count);
        CHECK(buffer2.data() == original_ptr);
        CHECK(buffer1.size() == 0);  // Moved-from buffer should be empty
        CHECK(buffer1.data() == nullptr);
    }
    
    SECTION("Different data types") {
        const size_t count = 100;
        
        DeviceBuffer<int> int_buffer(count, metal_resource);
        DeviceBuffer<float> float_buffer(count, metal_resource);
        DeviceBuffer<double> double_buffer(count, metal_resource);
        
        CHECK(int_buffer.size() == count);
        CHECK(float_buffer.size() == count);
        CHECK(double_buffer.size() == count);
        
        CHECK(int_buffer.data() != nullptr);
        CHECK(float_buffer.data() != nullptr);
        CHECK(double_buffer.data() != nullptr);
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal DeviceBuffer Data Transfer", "[buffer][transfer]") {
    // Initialize Metal manager and select devices
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 100;
    
    SECTION("Host to device copy") {
        DeviceBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1); // 1, 2, 3, ...
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), count));
    }
    
    SECTION("Device to host copy") {
        DeviceBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_input(count);
        std::vector<int> host_output(count);
        std::iota(host_input.begin(), host_input.end(), 10); // 10, 11, 12, ...
        
        buffer.copy_from_host(host_input.data(), count);
        CHECK_NOTHROW(buffer.copy_to_host(host_output.data(), count));
        
        // Verify data integrity
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_output[i] == host_input[i]);
        }
    }
    
    SECTION("Partial copy operations") {
        DeviceBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_data(count / 2);
        std::iota(host_data.begin(), host_data.end(), 100);
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), count / 2));
        
        std::vector<int> readback(count / 2);
        CHECK_NOTHROW(buffer.copy_to_host(readback.data(), count / 2));
        
        for (size_t i = 0; i < count / 2; ++i) {
            CHECK(readback[i] == host_data[i]);
        }
    }
    
    SECTION("Float data transfer") {
        DeviceBuffer<float> buffer(count, metal_resource);
        std::vector<float> host_data(count);
        for (size_t i = 0; i < count; ++i) {
            host_data[i] = static_cast<float>(i) * 0.5f + 1.0f;
        }
        
        buffer.copy_from_host(host_data.data(), count);
        
        std::vector<float> readback(count);
        buffer.copy_to_host(readback.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(readback[i] == Approx(host_data[i]));
        }
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal UnifiedBuffer Operations", "[unified_buffer]") {
    // Initialize Metal manager and select devices
    ARBD::METAL::METALManager::load_info();
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 200;
    
    SECTION("Unified buffer creation") {
        UnifiedBuffer<float> buffer(count, metal_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.primary_location().type == ResourceType::METAL);
    }
    
    SECTION("Memory migration") {
        UnifiedBuffer<int> buffer(count, metal_resource);
        
        // Ensure data is available at Metal device
        buffer.ensure_available_at(metal_resource);
        auto* ptr = buffer.get_ptr(metal_resource);
        CHECK(ptr != nullptr);
        
        auto locations = buffer.available_locations();
        CHECK(locations.size() >= 1);
        CHECK(locations[0].type == ResourceType::METAL);
    }
    
    SECTION("Data copy operations") {
        UnifiedBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 42);
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), metal_resource, count));
        
        std::vector<int> readback(count);
        CHECK_NOTHROW(buffer.copy_to_host(readback.data(), metal_resource, count));
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(readback[i] == host_data[i]);
        }
    }
    
    SECTION("Multiple resource management") {
        UnifiedBuffer<float> buffer(count, metal_resource);
        
        // Fill with initial data
        std::vector<float> initial_data(count, 3.14f);
        buffer.copy_from_host(initial_data.data(), metal_resource, count);
        
        // Test synchronization
        CHECK_NOTHROW(buffer.synchronize_all());
        
        // Verify data is still correct after sync
        std::vector<float> synced_data(count);
        buffer.copy_to_host(synced_data.data(), metal_resource, count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(synced_data[i] == Approx(3.14f));
        }
    }
    
    // Cleanup
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("UnifiedBuffer Metal Basic Allocation", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        
        SECTION("Basic allocation and deallocation") {
            ARBD::UnifiedBuffer<float> buffer(1000, metal_resource);
            
            REQUIRE(buffer.size() == 1000);
            REQUIRE(!buffer.empty());
            REQUIRE(buffer.primary_location().type == ARBD::ResourceType::METAL);
            
            // Check that pointer is valid
            float* ptr = buffer.get_ptr(metal_resource);
            REQUIRE(ptr != nullptr);
        }
        
        SECTION("Zero-sized buffer") {
            ARBD::UnifiedBuffer<int> buffer(0, metal_resource);
            
            REQUIRE(buffer.size() == 0);
            REQUIRE(buffer.empty());
        }
        
        SECTION("Different data types") {
            ARBD::UnifiedBuffer<double> double_buffer(500, metal_resource);
            ARBD::UnifiedBuffer<int> int_buffer(1000, metal_resource);
            ARBD::UnifiedBuffer<char> char_buffer(2000, metal_resource);
            
            REQUIRE(double_buffer.size() == 500);
            REQUIRE(int_buffer.size() == 1000);
            REQUIRE(char_buffer.size() == 2000);
            
            REQUIRE(double_buffer.get_ptr(metal_resource) != nullptr);
            REQUIRE(int_buffer.get_ptr(metal_resource) != nullptr);
            REQUIRE(char_buffer.get_ptr(metal_resource) != nullptr);
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Data Migration", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        ARBD::Resource host_resource{}; // Default host resource
        
        SECTION("Ensure availability at different resources") {
            ARBD::UnifiedBuffer<float> buffer(100, metal_resource);
            
            // Should be available at primary location
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == ARBD::ResourceType::METAL);
            
            // Metal buffers should only have 1 location
            REQUIRE(buffer.get_ptr(metal_resource) != nullptr);
        }
        
        SECTION("Metal buffer location management") {
            ARBD::UnifiedBuffer<int> buffer(50, metal_resource);
            
            // Metal buffer should only have 1 location
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == ARBD::ResourceType::METAL);
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal data migration test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Unified Memory", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        auto& device = ARBD::METAL::METALManager::get_current_device();
        
        SECTION("Test unified memory access") {
            ARBD::UnifiedBuffer<float> buffer(1000, metal_resource);
            float* ptr = buffer.get_ptr(metal_resource);
            REQUIRE(ptr != nullptr);
            
            // On Metal with unified memory, we should be able to write from host
            if (device.has_unified_memory()) {
                // Fill with test data
                for (size_t i = 0; i < 1000; ++i) {
                    ptr[i] = static_cast<float>(i * 0.5f);
                }
                
                // Verify data
                REQUIRE(ptr[0] == 0.0f);
                REQUIRE(ptr[500] == 250.0f);
                REQUIRE(ptr[999] == 499.5f);
            }
        }
        
        SECTION("Device properties verification") {
            INFO("Device name: " << device.name());
            INFO("Has unified memory: " << device.has_unified_memory());
            INFO("Is low power: " << device.is_low_power());
            INFO("Max threads per group: " << device.max_threads_per_group());
            
            // Basic sanity checks
            REQUIRE(!device.name().empty());
            REQUIRE(device.max_threads_per_group() > 0);
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal unified memory test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Move Semantics", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        ARBD::Resource metal_resource{ARBD::ResourceType::METAL, 0};
        
        SECTION("Move constructor") {
            ARBD::UnifiedBuffer<float> buffer1(1000, metal_resource);
            float* original_ptr = buffer1.get_ptr(metal_resource);
            
            ARBD::UnifiedBuffer<float> buffer2(std::move(buffer1));
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(metal_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
            REQUIRE(buffer1.empty());
        }
        
        SECTION("Move assignment") {
            ARBD::UnifiedBuffer<float> buffer1(1000, metal_resource);
            ARBD::UnifiedBuffer<float> buffer2(500, metal_resource);
            
            float* original_ptr = buffer1.get_ptr(metal_resource);
            
            buffer2 = std::move(buffer1);
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(metal_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal move semantics test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer Metal Raw Allocation Functions", "[UnifiedBuffer][Metal]") {
    try {
        // Initialize Metal backend
        ARBD::METAL::METALManager::init();
        ARBD::METAL::METALManager::load_info();
        
        if (ARBD::METAL::METALManager::devices().empty()) {
            SKIP("No Metal devices available");
        }
        
        SECTION("Test raw allocation/deallocation functions") {
            // Test the raw allocation functions we added
            size_t test_size = 1024; // 1KB
            void* ptr = ARBD::METAL::METALManager::allocate_raw(test_size);
            
            REQUIRE(ptr != nullptr);
            
            // Try to write to the memory (should work with unified memory)
            auto& device = ARBD::METAL::METALManager::get_current_device();
            if (device.has_unified_memory()) {
                char* char_ptr = static_cast<char*>(ptr);
                char_ptr[0] = 'A';
                char_ptr[test_size - 1] = 'Z';
                
                REQUIRE(char_ptr[0] == 'A');
                REQUIRE(char_ptr[test_size - 1] == 'Z');
            }
            
            // Clean up
            ARBD::METAL::METALManager::deallocate_raw(ptr);
        }
        
        SECTION("Test multiple allocations") {
            std::vector<void*> ptrs;
            
            // Allocate multiple buffers
            for (int i = 0; i < 10; ++i) {
                void* ptr = ARBD::METAL::METALManager::allocate_raw(100 * (i + 1));
                REQUIRE(ptr != nullptr);
                ptrs.push_back(ptr);
            }
            
            // All pointers should be different
            for (size_t i = 0; i < ptrs.size(); ++i) {
                for (size_t j = i + 1; j < ptrs.size(); ++j) {
                    REQUIRE(ptrs[i] != ptrs[j]);
                }
            }
            
            // Clean up all allocations
            for (void* ptr : ptrs) {
                ARBD::METAL::METALManager::deallocate_raw(ptr);
            }
        }
        
        ARBD::METAL::METALManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("Metal raw allocation test failed with ARBD exception: " << e.what());
    }
}

#endif // USE_METAL 