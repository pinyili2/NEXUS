#ifdef USE_METAL

#include "../catch_boiler.h"
#include <vector>
#include <string>
#include <numeric>
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"

using namespace ARBD;

// This test case is a placeholder for when Metal is not supported.
// We need at least one test case in the file.
TEST_CASE("Metal Support Check", "[init]") {

    // Use the Manager to check for available devices. This is a more
    // thorough check that ensures the manager can initialize and find devices.
    REQUIRE_NOTHROW(METAL::Manager::init());
    REQUIRE(METAL::Manager::all_device_size() > 0);
    REQUIRE_NOTHROW(METAL::Manager::finalize());
}

TEST_CASE("Metal Manager Initialization", "[init][manager]") {
    SECTION("Device Discovery and Properties") {
        METAL::Manager::init();
        auto& devices = METAL::Manager::all_devices();
        REQUIRE_FALSE(devices.empty());

        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            // After sorting, IDs might not be sequential from 0
            // So we just check that the name is not empty
            REQUIRE_FALSE(device.name().empty());
        }
        METAL::Manager::finalize();
    }
}

TEST_CASE("Metal Manager Device Properties", "[metal][manager][properties]") {
    METAL::Manager::load_info();
    auto& devices = METAL::Manager::devices();
    REQUIRE_FALSE(devices.empty());

    for (const auto& device : devices) {
        SECTION("Device Properties for " + device.name()) {
            LOGINFO("Device {}: {}", device.id(), device.name().c_str());
            REQUIRE_NOTHROW(device.metal_device());
            REQUIRE(device.max_threads_per_group() > 0);
        }
    }
    METAL::Manager::finalize();
}

TEST_CASE("Metal Manager Device Selection", "[metal][manager]") {
    METAL::Manager::init();
    SECTION("Select first device") {
        const auto& all_devices = METAL::Manager::all_devices();
        if (all_devices.empty()) {
            WARN("No Metal devices found, skipping test.");
            METAL::Manager::finalize();
            return;
        }
        
        unsigned int first_device_id = all_devices[0].id();
        std::vector<unsigned int> dev_ids = {first_device_id};
        REQUIRE_NOTHROW(METAL::Manager::select_devices(dev_ids));
        
        auto& selected_devices = METAL::Manager::devices();
        REQUIRE(selected_devices.size() == 1);
        REQUIRE(selected_devices[0].id() == first_device_id);
    }
    METAL::Manager::finalize();
}

TEST_CASE("Metal Queue Operations", "[metal][queue]") {
    // Setup: Initialize the manager
    METAL::Manager::load_info();

    // Iterate by index to get non-const device objects, which is safer
    // than using const_cast.
    for (size_t i = 0; i < METAL::Manager::devices().size(); ++i) {
        auto& device = METAL::Manager::devices()[i];

        SECTION("Queue Operations on device " + std::to_string(device.id())) {

            // Get a queue from the device
            auto& queue = const_cast<METAL::Manager::Device&>(device).get_next_queue();
            REQUIRE(queue.is_available());

            // Create a command buffer using the queue
            void* cmd_buffer_ptr = nullptr;
            REQUIRE_NOTHROW(cmd_buffer_ptr = queue.create_command_buffer());
            REQUIRE(cmd_buffer_ptr != nullptr);

            // The modern C++ way: Wrap the raw command buffer pointer in an Event object.
            // The Event's destructor will automatically call release() on the command buffer,
            // preventing memory leaks. This is the core of RAII.
            METAL::Event event(cmd_buffer_ptr);

            // Commit the command buffer via the Event object
            REQUIRE_NOTHROW(event.commit());

            // Wait for the (empty) command buffer to complete
            REQUIRE_NOTHROW(event.wait());

            // Check that it's now complete
            REQUIRE(event.is_complete() == true);
        }
    }

    // Teardown: Finalize the manager
    METAL::Manager::finalize();
}

TEST_CASE("Metal Synchronization", "[metal][sync]") {
    METAL::Manager::load_info();
    const auto& devices = METAL::Manager::devices();
    if (devices.empty()) {
        WARN("No Metal devices found. Skipping test.");
        METAL::Manager::finalize();
        return;
    }

    for (const auto& device : METAL::Manager::devices()) {
        SECTION("Synchronization for device " + std::to_string(device.id())) {
            INFO("Device ID: " << device.id());
            REQUIRE_NOTHROW(device.synchronize_all_queues());
        }
    }
    METAL::Manager::finalize();
}

TEST_CASE("Metal Memory Allocation and Transfer", "[metal][memory]") {
    METAL::Manager::load_info();
    if (METAL::Manager::devices().empty()) {
        WARN("No Metal devices found. Skipping test.");
        METAL::Manager::finalize();
        return;
    }
    auto& device = METAL::Manager::get_current_device();
    
    SECTION("DeviceMemory construction and destruction") {
        METAL::DeviceMemory<float> device_mem(device.metal_device(), 10);
        REQUIRE(device_mem.size() == 10);
    }

    SECTION("DeviceMemory move semantics") {
        METAL::DeviceMemory<int> mem1(device.metal_device(), 100);
        REQUIRE(mem1.size() == 100);
        
        METAL::DeviceMemory<int> mem2 = std::move(mem1);
        REQUIRE(mem1.size() == 0);
        REQUIRE(mem1.get() == nullptr);
        REQUIRE(mem2.size() == 100);
        REQUIRE(mem2.get() != nullptr);
    }

    SECTION("Data transfer Host to Device and back") {
        std::vector<float> host_data(256);
        std::iota(host_data.begin(), host_data.end(), 0.0f);

        METAL::DeviceMemory<float> device_mem(device.metal_device(), host_data.size());
        REQUIRE_NOTHROW(device_mem.copyFromHost(host_data));

        std::vector<float> host_result(host_data.size(), 0.0f);
        REQUIRE_NOTHROW(device_mem.copyToHost(host_result));

        REQUIRE(host_data == host_result);
    }
    METAL::Manager::finalize();
}

TEST_CASE("Metal Resource Creation and Properties", "[resource]") {
	Resource metal_resource(ResourceType::METAL, 0);

	SECTION("Resource type is correct") {
		CHECK(metal_resource.type == ResourceType::METAL);
		CHECK(metal_resource.id == 0);
	}

	SECTION("Resource type string") {
		CHECK(std::string(metal_resource.getTypeString()) == "METAL");
	}

	SECTION("Resource is device") {
		CHECK(metal_resource.is_device());
		CHECK_FALSE(metal_resource.is_host());
	}

	SECTION("Memory space is device") {
		CHECK(std::string(metal_resource.getMemorySpace()) == "device");
	}

	SECTION("Resource toString") {
		std::string expected = "METAL[0]";
		CHECK(metal_resource.toString() == expected);
	}
}

TEST_CASE("Metal Device Discovery", "[device]") {
	METAL::Manager::init();

	INFO("Number of available Metal devices: " << METAL::Manager::all_device_size());

	for (const auto& device : METAL::Manager::all_devices()) {
		INFO("Device " << device.id() << ": " << device.name()
					   << " (Low Power: " << device.is_low_power()
					   << ", Unified Memory: " << device.has_unified_memory() << ")");
	}

	// Select all available devices
	std::vector<unsigned int> device_ids;
	for (const auto& device : METAL::Manager::all_devices()) {
		device_ids.push_back(device.id());
	}

	REQUIRE_NOTHROW(METAL::Manager::select_devices(device_ids));

	CHECK(METAL::Manager::devices().size() > 0);
}

#endif // USE_METAL 
 