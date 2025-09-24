#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    std::cout << "=== Testing Queue with Properties ===" << std::endl;
    
    try {
        // Get the first available device
        auto platforms = sycl::platform::get_platforms();
        sycl::device test_device;
        bool device_found = false;
        
        for (const auto& platform : platforms) {
            auto devices = platform.get_devices();
            if (!devices.empty()) {
                test_device = devices[0];
                device_found = true;
                std::cout << "Using device: " << test_device.get_info<sycl::info::device::name>() << std::endl;
                break;
            }
        }
        
        if (!device_found) {
            std::cout << "No devices found!" << std::endl;
            return 1;
        }
        
        // Test 1: Basic queue with device (this should work)
        std::cout << "\nTest 1: Basic queue with device..." << std::endl;
        try {
            sycl::queue q1(test_device);
            std::cout << "SUCCESS: Basic queue created" << std::endl;
            
            // Try get_device() - should work for single-device queue
            auto dev = q1.get_device();
            std::cout << "SUCCESS: get_device() works - " << dev.get_info<sycl::info::device::name>() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        // Test 2: Queue with device and properties (single device approach)
        std::cout << "\nTest 2: Queue with device and properties (single device)..." << std::endl;
        try {
            sycl::property_list props{sycl::property::queue::enable_profiling{}};
            sycl::queue q2(test_device, props);
            std::cout << "SUCCESS: Queue with properties created" << std::endl;
            
            // Try get_device() - should work for single-device queue
            auto dev = q2.get_device();
            std::cout << "SUCCESS: get_device() works - " << dev.get_info<sycl::info::device::name>() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        // Test 3: Multi-device queue (this should fail get_device())
        std::cout << "\nTest 3: Multi-device queue..." << std::endl;
        try {
            sycl::property_list props{sycl::property::queue::enable_profiling{}};
            sycl::queue q3(std::vector<sycl::device>{test_device}, props);
            std::cout << "SUCCESS: Multi-device queue created" << std::endl;
            
            // Try get_device() - should fail for multi-device queue
            try {
                auto dev = q3.get_device();
                std::cout << "UNEXPECTED: get_device() worked on multi-device queue - " << dev.get_info<sycl::info::device::name>() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "EXPECTED: get_device() failed on multi-device queue - " << e.what() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        // Test 4: Memory allocation test (the real issue)
        std::cout << "\nTest 4: Memory allocation with single-device queue..." << std::endl;
        try {
            sycl::property_list props{sycl::property::queue::enable_profiling{}};
            sycl::queue q4(test_device, props);  // Single device approach
            
            // Try malloc_device - this should work
            int* device_ptr = sycl::malloc_device<int>(10, q4);
            if (device_ptr) {
                std::cout << "SUCCESS: malloc_device works with single-device queue" << std::endl;
                sycl::free(device_ptr, q4);
            } else {
                std::cout << "FAILED: malloc_device returned nullptr" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        std::cout << "\nAll tests completed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
} 