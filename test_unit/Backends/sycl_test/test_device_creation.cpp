#include <sycl/sycl.hpp>
#include <iostream>
#include <array>

// Simplified Queue class for testing
class TestQueue {
private:
    std::optional<sycl::queue> queue_;
public:
    TestQueue() = default;
    
    explicit TestQueue(const sycl::device& dev) {
        try {
            queue_ = sycl::queue(dev);
            std::cout << "Queue created successfully for device: " 
                      << dev.get_info<sycl::info::device::name>() << std::endl;
        } catch (const sycl::exception& e) {
            std::cout << "Queue creation failed: " << e.what() << std::endl;
            throw;
        }
    }
};

// Simplified Device class for testing
class TestDevice {
public:
    static constexpr size_t NUM_QUEUES = 4;
    
private:
    unsigned int id_;
    sycl::device device_;
    std::array<TestQueue, NUM_QUEUES> queues_;
    
public:
    explicit TestDevice(const sycl::device& dev, unsigned int id) 
        : id_(id), device_(dev) {
        
        std::cout << "Creating device " << id_ << " with " << NUM_QUEUES << " queues..." << std::endl;
        
        // Create queues - this is where the error might occur
        try {
            for (size_t i = 0; i < queues_.size(); ++i) {
                std::cout << "Creating queue " << i << "..." << std::endl;
                queues_[i] = TestQueue(device_);
                std::cout << "Queue " << i << " created successfully" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Error during queue creation: " << e.what() << std::endl;
            throw;
        }
        
        std::cout << "Device " << id_ << " initialized successfully" << std::endl;
    }
};

int main() {
    std::cout << "=== SYCL Device Creation Test ===" << std::endl;
    
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
        
        // Test device creation with multiple queues
        std::cout << "\nTesting device creation with multiple queues..." << std::endl;
        TestDevice test_dev(test_device, 0);
        
        std::cout << "SUCCESS: Device created with all queues" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
} 