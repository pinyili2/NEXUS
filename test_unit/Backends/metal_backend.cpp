#include "../catch_boiler.h"
#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLLibrary.hpp"
#ifdef USE_METAL

#include "ARBDException.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"
#include <numeric>
#include <string>
#include <vector>
using Catch::Approx;

TEST_CASE("Manager Basic Initialization", "[Manager][Backend]") {
	SECTION("Initialize and discover devices") {
		REQUIRE_NOTHROW(ARBD::METAL::Manager::init());

		// Check that we found some devices
		const auto& all_devices = ARBD::METAL::Manager::all_devices();
		REQUIRE(!all_devices.empty());
		LOGDEBUG("Found {} Metal devices", all_devices.size());

		// Check that at least one device is usable
		REQUIRE_NOTHROW(ARBD::METAL::Manager::load_info());
		const auto& devices = ARBD::METAL::Manager::devices();
		REQUIRE(!devices.empty());
	}

	SECTION("Device properties validation") {
		ARBD::METAL::Manager::init();
		ARBD::METAL::Manager::load_info();

		const auto& devices = ARBD::METAL::Manager::devices();
		for (size_t i = 0; i < devices.size(); ++i) {
			const auto& device = devices[i];

			INFO("Testing device " << i << ": " << device.name());

			// Basic property checks
			REQUIRE(!device.name().empty());
			REQUIRE(device.id() >= 0);

			// Metal-specific properties
			REQUIRE(device.metal_device() != nullptr);

			// Log device characteristics
			LOGDEBUG("Device {}: {}", i, device.name());
			LOGDEBUG("  Low power: {}", device.is_low_power());
			LOGDEBUG("  Removable: {}", device.is_removable());
			LOGDEBUG("  Unified memory: {}", device.has_unified_memory());
			LOGDEBUG("  Max threads per group: {}", device.max_threads_per_group());
			LOGDEBUG("  Recommended max working set: {:.1f} MB",
					static_cast<float>(device.recommended_max_working_set_size()) /
						(1024.0f * 1024.0f));
		}
	}
}

TEST_CASE("Manager Device Selection and Usage", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();

	const auto& devices = ARBD::METAL::Manager::devices();
	REQUIRE(!devices.empty());

	SECTION("Device selection") {
		// Test using device 0
		REQUIRE_NOTHROW(ARBD::METAL::Manager::use(0));
		REQUIRE(ARBD::METAL::Manager::current() == 0);

		// Test current device access
		REQUIRE_NOTHROW(ARBD::METAL::Manager::get_current_device());
		const auto& current_device = ARBD::METAL::Manager::get_current_device();
		REQUIRE(current_device.id() == devices[0].id());

		// Test cycling through devices if multiple available
		if (devices.size() > 1) {
			REQUIRE_NOTHROW(ARBD::METAL::Manager::use(1));
			REQUIRE(ARBD::METAL::Manager::current() == 1);

			// Test wraparound
			int wrapped_id = static_cast<int>(devices.size());
			REQUIRE_NOTHROW(ARBD::METAL::Manager::use(wrapped_id));
			REQUIRE(ARBD::METAL::Manager::current() == 0); // Should wrap to 0
		}
	}

	SECTION("Device selection by IDs") {
		// Create a vector with device IDs
		std::vector<unsigned int> device_ids;
		for (const auto& device : devices) {
			device_ids.push_back(device.id());
		}

		REQUIRE_NOTHROW(ARBD::METAL::Manager::select_devices(device_ids));

		// Verify devices are still accessible
		REQUIRE_NOTHROW(ARBD::METAL::Manager::load_info());
		const auto& selected_devices = ARBD::METAL::Manager::devices();
		REQUIRE(selected_devices.size() == device_ids.size());
	}

	SECTION("Power preference settings") {
		const auto& all_devices = ARBD::METAL::Manager::all_devices();

		// Test low power preference
		ARBD::METAL::Manager::prefer_low_power(true);
		REQUIRE_NOTHROW(ARBD::METAL::Manager::load_info());

		// Check if filtering worked (might not have low power devices)
		const auto& low_power_devices = ARBD::METAL::Manager::devices();
		for (const auto& device : low_power_devices) {
			// If we have devices, they should be low power when preference is set
			if (!low_power_devices.empty()) {
				bool found_low_power = false;
				for (const auto& all_device : all_devices) {
					if (all_device.id() == device.id() && all_device.is_low_power()) {
						found_low_power = true;
						break;
					}
				}
				// If no low power devices exist, it should fall back to all devices
				REQUIRE((found_low_power || low_power_devices.size() == all_devices.size()));
			}
		}

		// Reset to high performance
		ARBD::METAL::Manager::prefer_low_power(false);
		REQUIRE_NOTHROW(ARBD::METAL::Manager::load_info());
		const auto& high_perf_devices = ARBD::METAL::Manager::devices();
		REQUIRE(!high_perf_devices.empty());
	}
}

TEST_CASE("Manager Command Queue Management", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();
	ARBD::METAL::Manager::use(0);

	auto& device = ARBD::METAL::Manager::get_current_device();

	SECTION("Command queue creation") {
		// Test creating command queues
		auto& queue1 = device.get_next_queue();
		REQUIRE(queue1.get() != nullptr);

		auto& queue2 = device.get_next_queue();
		REQUIRE(queue2.get() != nullptr);

		// Should be different queue objects (different addresses)
		REQUIRE(&queue1 != &queue2);

		// The underlying Metal queues might be the same or different depending on implementation
		LOGDEBUG("Queue 1 address: {}, Queue 2 address: {}",
				static_cast<void*>(&queue1),
				static_cast<void*>(&queue2));
		LOGDEBUG("Queue 1 MTL queue: {}, Queue 2 MTL queue: {}", queue1.get(), queue2.get());
	}

	SECTION("Command buffer operations") {
		auto& queue = device.get_next_queue();
		REQUIRE(queue.get() != nullptr);

		// Create command buffer
		auto* cmd_buffer = queue.create_command_buffer();
		REQUIRE(cmd_buffer != nullptr);

		// Command buffer should be in uncommitted state initially
		// We can't test much without actual compute shaders
		LOGDEBUG("Created command buffer: {}", cmd_buffer);

		// Queue will be cleaned up automatically when it goes out of scope
	}
}

TEST_CASE("Manager Memory Management", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();
	ARBD::METAL::Manager::use(0);

	SECTION("Raw memory allocation") {
		constexpr size_t test_size = 1024 * sizeof(float);

		void* ptr = nullptr;
		REQUIRE_NOTHROW(ptr = ARBD::METAL::Manager::allocate_raw(test_size));
		REQUIRE(ptr != nullptr);

		// Memory should be accessible (Metal uses unified memory)
		float* float_ptr = static_cast<float*>(ptr);

		// Test writing to memory
		REQUIRE_NOTHROW([&]() {
			for (size_t i = 0; i < test_size / sizeof(float); ++i) {
				float_ptr[i] = static_cast<float>(i) * 3.14f;
			}
		}());

		// Test reading from memory
		for (size_t i = 0; i < test_size / sizeof(float); ++i) {
			REQUIRE(float_ptr[i] == Approx(static_cast<float>(i) * 3.14f));
		}

		// Test getting Metal buffer from pointer
		void* metal_buffer = ARBD::METAL::Manager::get_metal_buffer_from_ptr(ptr);
		REQUIRE(metal_buffer != nullptr);

		REQUIRE_NOTHROW(ARBD::METAL::Manager::deallocate_raw(ptr));
	}

	SECTION("Multiple allocations") {
		constexpr size_t num_allocs = 10;
		constexpr size_t alloc_size = 1024;

		std::vector<void*> pointers;

		// Allocate multiple buffers
		for (size_t i = 0; i < num_allocs; ++i) {
			void* ptr = nullptr;
			REQUIRE_NOTHROW(ptr = ARBD::METAL::Manager::allocate_raw(alloc_size));
			REQUIRE(ptr != nullptr);
			pointers.push_back(ptr);
		}

		// All pointers should be different
		for (size_t i = 0; i < num_allocs; ++i) {
			for (size_t j = i + 1; j < num_allocs; ++j) {
				REQUIRE(pointers[i] != pointers[j]);
			}
		}

		// Deallocate all
		for (void* ptr : pointers) {
			REQUIRE_NOTHROW(ARBD::METAL::Manager::deallocate_raw(ptr));
		}
	}

	SECTION("Large allocation") {
		// Try allocating a reasonably large buffer (1MB)
		size_t large_size = 1024 * 1024; // 1MB

		void* ptr = nullptr;
		REQUIRE_NOTHROW(ptr = ARBD::METAL::Manager::allocate_raw(large_size));
		REQUIRE(ptr != nullptr);

		REQUIRE_NOTHROW(ARBD::METAL::Manager::deallocate_raw(ptr));

		LOGDEBUG("Successfully allocated and freed {:.1f} MB buffer",
				static_cast<float>(large_size) / (1024.0f * 1024.0f));
	}
}

TEST_CASE("Manager Library and Function Management", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();

	SECTION("Library access") {
		auto* library = ARBD::METAL::Manager::get_library();

		if (library != nullptr) {
			LOGDEBUG("Metal library loaded successfully");

			// Test function preloading
			REQUIRE_NOTHROW(ARBD::METAL::Manager::preload_all_functions());

			// Try to get function names (this might fail if no functions are available)
			auto* function_names = library->functionNames();
			if (function_names && function_names->count() > 0) {
				LOGDEBUG("Found {} functions in Metal library", function_names->count());

				// Test getting a function (if any exist)
				for (NS::UInteger i = 0; i < std::min(function_names->count(), NS::UInteger(5));
					 ++i) {
					auto* name = static_cast<NS::String*>(function_names->object(i));
					if (name) {
						std::string func_name = name->utf8String();
						LOGDEBUG("Function {}: {}", i, func_name);

						// Test getting function
						auto* function = ARBD::METAL::Manager::get_function(func_name);
						if (function) {
							LOGDEBUG("  Successfully retrieved function: {}", func_name);
						}
					}
				}
			} else {
				LOGDEBUG("No functions found in Metal library (library might be empty)");
			}
		} else {
			LOGDEBUG("No Metal library available - this is normal without compiled shaders");
		}
	}

	SECTION("Pipeline state management") {
		auto* library = ARBD::METAL::Manager::get_library();

		if (library != nullptr) {
			auto* function_names = library->functionNames();
			if (function_names && function_names->count() > 0) {
				// Try to create a compute pipeline state for the first function
				auto* first_name = static_cast<NS::String*>(function_names->object(0));
				if (first_name) {
					std::string func_name = first_name->utf8String();

					// This might fail if the function isn't a compute kernel
					try {
						auto* pipeline_state =
							ARBD::METAL::Manager::get_compute_pipeline_state(func_name);
						if (pipeline_state) {
							LOGDEBUG("Successfully created pipeline state for: {}", func_name);

							// Test pipeline properties
							REQUIRE(pipeline_state->maxTotalThreadsPerThreadgroup() > 0);
							LOGDEBUG("  Max threads per threadgroup: {}",
									pipeline_state->maxTotalThreadsPerThreadgroup());
						}
					} catch (const ARBD::Exception& e) {
						LOGDEBUG("Expected failure creating pipeline for {}: {}",
								func_name,
								e.what());
					}
				}
			}
		}
	}
}

TEST_CASE("Manager Device Properties", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();

	const auto& all_devices = ARBD::METAL::Manager::all_devices();

	SECTION("Device property access") {
		for (size_t i = 0; i < all_devices.size(); ++i) {
			const auto& device = all_devices[i];

			LOGDEBUG("Device {} properties:", i);
			LOGDEBUG("  Name: {}", device.name());
			LOGDEBUG("  Low power: {}", device.is_low_power());
			LOGDEBUG("  Removable: {}", device.is_removable());
			LOGDEBUG("  Unified memory: {}", device.has_unified_memory());
			LOGDEBUG("  Max threads per group: {}", device.max_threads_per_group());

			// All Metal devices should support compute
			REQUIRE(device.supports_compute());

			// All modern Apple devices have unified memory
			// (This may not be true for external GPUs, but it's a reasonable assumption)
			if (!device.is_removable()) {
				REQUIRE(device.has_unified_memory());
			}
		}
	}
}

TEST_CASE("Manager Event System", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();
	ARBD::METAL::Manager::use(0);

	auto& device = ARBD::METAL::Manager::get_current_device();

	SECTION("Basic event operations") {
		auto& queue = device.get_next_queue();
		REQUIRE(queue.is_available());

		auto* cmd_buffer = queue.create_command_buffer();
		REQUIRE(cmd_buffer != nullptr);

		// Create Metal event wrapper
		ARBD::METAL::Event event(cmd_buffer);

		// Event should be valid but not complete initially
		REQUIRE_FALSE(event.is_complete());

		// Commit the command buffer
		event.commit();

		// Wait for completion
		REQUIRE_NOTHROW(event.wait());

		// Should be complete now
		REQUIRE(event.is_complete());
	}

	SECTION("Event timing") {
		auto& queue = device.get_next_queue();
		REQUIRE(queue.is_available());

		auto* cmd_buffer = queue.create_command_buffer();
		REQUIRE(cmd_buffer != nullptr);

		ARBD::METAL::Event event(cmd_buffer);

		event.commit();
		event.wait();

		// Try to get execution time (might not be available without actual work)
		auto duration = event.get_execution_time();
		REQUIRE(duration.count() >= 0);
	}
}

TEST_CASE("Manager Error Handling", "[Manager][Backend]") {
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();

	SECTION("Invalid device access") {
		const auto& devices = ARBD::METAL::Manager::devices();
		int invalid_id = static_cast<int>(devices.size());

		REQUIRE_THROWS_AS(ARBD::METAL::Manager::use(invalid_id), ARBD::Exception);
	}

	SECTION("Invalid function access") {
		auto* library = ARBD::METAL::Manager::get_library();

		if (library != nullptr) {
			REQUIRE_THROWS_AS(ARBD::METAL::Manager::get_function("nonexistent_function"),
							  ARBD::Exception);
		}
	}

	SECTION("Null pointer deallocation") {
		// Should handle null pointer gracefully
		REQUIRE_NOTHROW(ARBD::METAL::Manager::deallocate_raw(nullptr));
	}

	SECTION("Double deallocation") {
		void* ptr = ARBD::METAL::Manager::allocate_raw(1024);
		REQUIRE(ptr != nullptr);

		REQUIRE_NOTHROW(ARBD::METAL::Manager::deallocate_raw(ptr));

		// Second deallocation should be logged as warning but not crash
		REQUIRE_NOTHROW(ARBD::METAL::Manager::deallocate_raw(ptr));
	}
}

TEST_CASE("Manager Finalization", "[Manager][Backend]") {
	// Initialize first
	ARBD::METAL::Manager::init();
	ARBD::METAL::Manager::load_info();

	SECTION("Clean finalization") {
		REQUIRE_NOTHROW(ARBD::METAL::Manager::finalize());

		// After finalization, devices should be empty
		const auto& devices = ARBD::METAL::Manager::devices();
		REQUIRE(devices.empty());
	}
}

#else
TEST_CASE("METAL Backend Not Available", "[Manager][Backend]") {
	SKIP("METAL backend not compiled in or not on Apple platform");
}
#endif // USE_METAL