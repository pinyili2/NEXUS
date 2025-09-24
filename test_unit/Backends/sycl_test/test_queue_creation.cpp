#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    std::cout << "=== SYCL Queue Creation Test ===" << std::endl;
    
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
        
        // Test 1: Basic queue with device
        std::cout << "\nTest 1: Basic queue creation with device..." << std::endl;
        try {
            sycl::queue q1(test_device);
            std::cout << "SUCCESS: Basic queue created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        // Test 2: Default queue
        std::cout << "\nTest 2: Default queue creation..." << std::endl;
        try {
            sycl::queue q2;
            std::cout << "SUCCESS: Default queue created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        // Test 3: Queue with properties
        std::cout << "\nTest 3: Queue with properties..." << std::endl;
        try {
            sycl::property_list props{sycl::property::queue::enable_profiling{}};
            sycl::queue q3(test_device, props);
            std::cout << "SUCCESS: Queue with properties created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        // Test 4: Try selectors (if available)
        std::cout << "\nTest 4: CPU selector..." << std::endl;
        try {
            sycl::queue q4(sycl::cpu_selector_v);
            std::cout << "SUCCESS: CPU selector queue created" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
} 