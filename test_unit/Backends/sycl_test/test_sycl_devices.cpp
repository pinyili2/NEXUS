#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== SYCL Device Discovery Test ===" << std::endl;
    
    try {
        // Get all platforms
        auto platforms = sycl::platform::get_platforms();
        std::cout << "Found " << platforms.size() << " SYCL platforms:" << std::endl;
        
        int total_devices = 0;
        for (size_t p = 0; p < platforms.size(); ++p) {
            const auto& platform = platforms[p];
            std::cout << "\nPlatform " << p << ":" << std::endl;
            std::cout << "  Name: " << platform.get_info<sycl::info::platform::name>() << std::endl;
            std::cout << "  Vendor: " << platform.get_info<sycl::info::platform::vendor>() << std::endl;
            std::cout << "  Version: " << platform.get_info<sycl::info::platform::version>() << std::endl;
            
            // Get all devices for this platform
            auto devices = platform.get_devices();
            std::cout << "  Devices: " << devices.size() << std::endl;
            
            for (size_t d = 0; d < devices.size(); ++d) {
                const auto& device = devices[d];
                auto device_type = device.get_info<sycl::info::device::device_type>();
                
                std::string type_str;
                switch (device_type) {
                    case sycl::info::device_type::cpu:
                        type_str = "CPU";
                        break;
                    case sycl::info::device_type::gpu:
                        type_str = "GPU";
                        break;
                    case sycl::info::device_type::accelerator:
                        type_str = "Accelerator";
                        break;
                    default:
                        type_str = "Unknown";
                        break;
                }
                
                std::cout << "    Device " << d << " (" << type_str << "):" << std::endl;
                std::cout << "      Name: " << device.get_info<sycl::info::device::name>() << std::endl;
                std::cout << "      Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
                
                // Test if we can create a queue with this device
                try {
                    sycl::queue test_queue(device);
                    std::cout << "      Queue creation: SUCCESS" << std::endl;
                    
                    // Simple test kernel
                    int test_data = 42;
                    int* device_ptr = sycl::malloc_device<int>(1, test_queue);
                    test_queue.memcpy(device_ptr, &test_data, sizeof(int));
                    
                    test_queue.submit([=](sycl::handler& h) {
                        h.single_task([=]() {
                            device_ptr[0] *= 2;
                        });
                    }).wait();
                    
                    int result = 0;
                    test_queue.memcpy(&result, device_ptr, sizeof(int)).wait();
                    sycl::free(device_ptr, test_queue);
                    
                    if (result == 84) {
                        std::cout << "      Kernel execution: SUCCESS" << std::endl;
                    } else {
                        std::cout << "      Kernel execution: FAILED (got " << result << ", expected 84)" << std::endl;
                    }
                    
                } catch (const std::exception& e) {
                    std::cout << "      Queue/kernel test: FAILED - " << e.what() << std::endl;
                }
                
                total_devices++;
            }
        }
        
        std::cout << "\nTotal devices found: " << total_devices << std::endl;
        
        if (total_devices == 0) {
            std::cout << "ERROR: No SYCL devices found!" << std::endl;
            return 1;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
} 