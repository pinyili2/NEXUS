#include <sycl/sycl.hpp>
#include <iostream>
#include <array>
#include <optional>

// Simplified versions of ARBD classes for testing
class TestDeviceMemory {
private:
    std::optional<sycl::queue> queue_;
    int* ptr_{nullptr};
    size_t size_{0};

public:
    TestDeviceMemory() = default;
    
    explicit TestDeviceMemory(sycl::queue& q, size_t count) : queue_(q), size_(count) {
        if (count > 0) {
            try {
                ptr_ = sycl::malloc_device<int>(count, *queue_);
                if (!ptr_) {
                    std::cout << "FAILED: malloc_device returned nullptr" << std::endl;
                }
            } catch (const sycl::exception& e) {
                std::cout << "SYCL error during malloc_device: " << e.what() << std::endl;
                throw;
            }
        }
    }

    ~TestDeviceMemory() {
        if (ptr_ && queue_.has_value()) {
            sycl::free(ptr_, *queue_);
        }
    }

    TestDeviceMemory(const TestDeviceMemory&) = delete;
    TestDeviceMemory& operator=(const TestDeviceMemory&) = delete;

    TestDeviceMemory(TestDeviceMemory&& other) noexcept 
        : queue_(std::move(other.queue_))
        , ptr_(std::exchange(other.ptr_, nullptr))
        , size_(std::exchange(other.size_, 0)) {}
    
    TestDeviceMemory& operator=(TestDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_ && queue_.has_value()) sycl::free(ptr_, *queue_);
            queue_ = std::move(other.queue_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    int* get() noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
};

class TestQueue {
private:
    std::optional<sycl::queue> queue_;
public:
    TestQueue() = default; // Default constructor - queue remains uninitialized
    
    explicit TestQueue(const sycl::device& dev) {
        try {
            queue_ = sycl::queue(dev); // Single device queue
            std::cout << "Queue created successfully for device: " 
                      << dev.get_info<sycl::info::device::name>() << std::endl;
        } catch (const sycl::exception& e) {
            std::cout << "Queue creation failed: " << e.what() << std::endl;
            throw;
        }
    }
    
    TestQueue(TestQueue&& other) noexcept : queue_(std::move(other.queue_)) {}
    
    TestQueue& operator=(TestQueue&& other) noexcept {
        if (this != &other) {
            queue_ = std::move(other.queue_);
        }
        return *this;
    }
    
    sycl::queue& get() {
        if (!queue_.has_value()) {
            throw std::runtime_error("Queue not initialized");
        }
        return *queue_;
    }
};

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
        
        // Create queues - this is the critical test
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
    
    TestQueue& get_queue(size_t queue_id) {
        return queues_[queue_id % NUM_QUEUES];
    }
};

int main() {
    std::cout << "=== ARBD-style SYCL Test ===" << std::endl;
    
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
        
        // Test device creation with multiple queues (this was failing before)
        std::cout << "\nTesting device creation with multiple queues..." << std::endl;
        TestDevice test_dev(test_device, 0);
        
        // Test device memory allocation (this was failing before)
        std::cout << "\nTesting device memory allocation..." << std::endl;
        auto& queue = test_dev.get_queue(0);
        TestDeviceMemory device_mem(queue.get(), 100);
        
        if (device_mem.get() != nullptr) {
            std::cout << "SUCCESS: Device memory allocated - " << device_mem.size() << " elements" << std::endl;
        } else {
            std::cout << "FAILED: Device memory allocation returned nullptr" << std::endl;
            return 1;
        }
        
        std::cout << "\nSUCCESS: All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
} 