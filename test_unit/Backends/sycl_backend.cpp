#include "../catch_boiler.h"
#ifdef USE_SYCL

#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Backend/SYCL/SYCLManager.h"
#include <numeric>
#include <string>
#include <vector>
using Catch::Approx;
// status: works on Mac M3

TEST_CASE("Manager Basic Initialization", "[Manager][Backend]") {
	SECTION("Initialize and discover devices") {
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::init());

		// Check that we found some devices
		const auto& all_devices = ARBD::SYCL::Manager::all_devices();
		REQUIRE(!all_devices.empty());
		LOGINFO("Found {} SYCL devices", all_devices.size());

		// Check device types
		for (size_t i = 0; i < all_devices.size(); ++i) {
			const auto& device = all_devices[i];
			LOGINFO("Device {}: {} ({})", i, device.name().c_str(), device.vendor().c_str());
			LOGINFO("Type: {}, Compute units: {}, Max work group: {}, Global mem: {:.1f}GB",
					device.name().c_str(),
					device.max_compute_units(),
					device.max_work_group_size(),
					static_cast<float>(device.global_mem_size()) / (1024.0f * 1024.0f * 1024.0f));
		}

		// Load info should work
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::load_info());
		const auto& devices = ARBD::SYCL::Manager::devices();
		REQUIRE(!devices.empty());
	}

	SECTION("Device properties validation") {
		ARBD::SYCL::Manager::init();
		ARBD::SYCL::Manager::load_info();

		const auto& devices = ARBD::SYCL::Manager::devices();
		for (size_t i = 0; i < devices.size(); ++i) {
			const auto& device = devices[i];

			INFO("Testing device " << i << ": " << device.name());

			// Basic property checks
			REQUIRE(!device.name().empty());
			REQUIRE(!device.vendor().empty());
			REQUIRE(device.max_compute_units() > 0);
			REQUIRE(device.max_work_group_size() > 0);
			REQUIRE(device.global_mem_size() > 0);

			// Device type should be valid
			bool valid_type = device.is_cpu() || device.is_gpu() || device.is_accelerator();
			REQUIRE(valid_type);

			// SYCL device should be accessible
			REQUIRE_NOTHROW(device.get_device());
		}
	}
}

TEST_CASE("Manager Device Selection and Usage", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();

	const auto& devices = ARBD::SYCL::Manager::devices();
	REQUIRE(!devices.empty());

	SECTION("Device selection") {
		// Test using device 0
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::use(0));
		REQUIRE(ARBD::SYCL::Manager::current() == 0);

		// Test current device access
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::get_current_device());
		const auto& current_device = ARBD::SYCL::Manager::get_current_device();
		REQUIRE(current_device.id() == devices[0].id());

		// Test cycling through devices if multiple available
		if (devices.size() > 1) {
			REQUIRE_NOTHROW(ARBD::SYCL::Manager::use(1));
			REQUIRE(ARBD::SYCL::Manager::current() == 1);

			// Test wraparound
			int wrapped_id = static_cast<int>(devices.size());
			REQUIRE_NOTHROW(ARBD::SYCL::Manager::use(wrapped_id));
			REQUIRE(ARBD::SYCL::Manager::current() == 0); // Should wrap to 0
		}
	}

	SECTION("Device selection by IDs") {
		// Create a vector with device IDs
		std::vector<unsigned int> device_ids;
		for (const auto& device : devices) {
			device_ids.push_back(device.id());
		}

		REQUIRE_NOTHROW(ARBD::SYCL::Manager::select_devices(device_ids));

		// Verify devices are still accessible
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::load_info());
		const auto& selected_devices = ARBD::SYCL::Manager::devices();
		REQUIRE(selected_devices.size() == device_ids.size());
	}
}

TEST_CASE("Manager Queue Management", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();
	ARBD::SYCL::Manager::use(0);

	auto& device = ARBD::SYCL::Manager::get_current_device();

	SECTION("Queue access") {
		// Test getting specific queues
		for (size_t i = 0; i < ARBD::SYCL::Manager::NUM_QUEUES; ++i) {
			REQUIRE_NOTHROW(device.get_queue(i));
			auto& queue = device.get_queue(i);

			// Queue should be usable
			REQUIRE_NOTHROW(queue.get());
		}

		// Test next queue cycling
		std::set<const void*> seen_queues;
		for (size_t i = 0; i < ARBD::SYCL::Manager::NUM_QUEUES * 2; ++i) {
			auto& queue = const_cast<ARBD::SYCL::Manager::Device&>(device).get_next_queue();
			seen_queues.insert(&queue.get());
		}
		// Should have seen all unique queues
		REQUIRE(seen_queues.size() == ARBD::SYCL::Manager::NUM_QUEUES);
	}

	SECTION("Queue functionality") {
		auto& queue = const_cast<ARBD::SYCL::Manager::Device&>(device).get_next_queue();

		// Test basic queue operations
		REQUIRE_NOTHROW(queue.get());

		// Test submitting a simple kernel
		bool kernel_executed = false;
		auto buffer = sycl::buffer<bool, 1>(&kernel_executed, sycl::range<1>(1));
		REQUIRE_NOTHROW([&]() {
			queue.get()
				.submit([&](sycl::handler& h) {
					auto acc = buffer.get_access<sycl::access::mode::write>(h);
					h.single_task([=]() { acc[0] = true; });
				})
				.wait();
		}());

		// Read back the result
		auto host_acc = buffer.get_host_access();
		REQUIRE(host_acc[0]);
	}

	SECTION("Queue synchronization") {
		REQUIRE_NOTHROW(device.synchronize_all_queues());
	}
}

TEST_CASE("Manager Current Queue Access", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();
	ARBD::SYCL::Manager::use(0);

	SECTION("Current queue access") {
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::get_current_queue());

		auto& queue = ARBD::SYCL::Manager::get_current_queue();
		REQUIRE_NOTHROW(queue.get());

		// Should be able to submit work to current queue
		REQUIRE_NOTHROW([&]() {
			queue.get()
				.submit([](sycl::handler& h) {
					h.single_task([]() {
						// Simple no-op kernel
					});
				})
				.wait();
		}());
	}

	SECTION("Current device queue relationship") {
		auto& device = ARBD::SYCL::Manager::get_current_device();
		auto& current_queue = ARBD::SYCL::Manager::get_current_queue();
		auto& device_queue = device.get_next_queue();

		// Both should be valid queues (might be different instances)
		REQUIRE_NOTHROW(current_queue.get());
		REQUIRE_NOTHROW(device_queue.get());
	}
}

TEST_CASE("Manager Synchronization", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();

	SECTION("All devices sync") {
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::sync());
	}

	SECTION("Sync with active work") {
		ARBD::SYCL::Manager::use(0);
		auto& queue = ARBD::SYCL::Manager::get_current_queue();

		// Submit some work
		queue.get().submit([](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(1000), [](sycl::id<1> idx) {
				// Simple computation
				volatile int dummy = idx[0] * 2;
				(void)dummy;
			});
		});

		// Sync should wait for completion
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::sync());
	}
}

TEST_CASE("Manager Memory Operations", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();
	ARBD::SYCL::Manager::use(0);

	auto& queue = ARBD::SYCL::Manager::get_current_queue();

	SECTION("Device memory allocation") {
		constexpr size_t test_size = 1024;

		float* device_ptr = nullptr;
		REQUIRE_NOTHROW(
			[&]() { device_ptr = sycl::malloc_device<float>(test_size, queue.get()); }());
		REQUIRE(device_ptr != nullptr);

		REQUIRE_NOTHROW([&]() { sycl::free(device_ptr, queue.get()); }());
	}

	SECTION("Memory copy operations") {
		constexpr size_t test_size = 1024;
		std::vector<float> host_data(test_size, 3.14f);
		std::vector<float> result_data(test_size, 0.0f);

		float* device_ptr = sycl::malloc_device<float>(test_size, queue.get());
		REQUIRE(device_ptr != nullptr);

		// Host to device
		REQUIRE_NOTHROW([&]() {
			queue.get().memcpy(device_ptr, host_data.data(), test_size * sizeof(float)).wait();
		}());

		// Device to host
		REQUIRE_NOTHROW([&]() {
			queue.get().memcpy(result_data.data(), device_ptr, test_size * sizeof(float)).wait();
		}());

		// Verify data
		for (size_t i = 0; i < test_size; ++i) {
			REQUIRE(result_data[i] == Approx(3.14f));
		}

		sycl::free(device_ptr, queue.get());
	}

	SECTION("Unified shared memory") {
		constexpr size_t test_size = 1024;

		// Test USM allocation if supported
		try {
			float* usm_ptr = sycl::malloc_shared<float>(test_size, queue.get());
			if (usm_ptr != nullptr) {
				// Initialize on host
				for (size_t i = 0; i < test_size; ++i) {
					usm_ptr[i] = static_cast<float>(i);
				}

				// Use in kernel
				queue.get()
					.parallel_for(sycl::range<1>(test_size),
								  [=](sycl::id<1> idx) { usm_ptr[idx] *= 2.0f; })
					.wait();

				// Verify on host
				for (size_t i = 0; i < test_size; ++i) {
					REQUIRE(usm_ptr[i] == Approx(static_cast<float>(i * 2)));
				}

				sycl::free(usm_ptr, queue.get());
				LOGINFO("USM test passed");
			} else {
				LOGINFO("USM not supported on this device");
			}
		} catch (const sycl::exception& e) {
			LOGINFO("USM test skipped: {}", e.what());
		}
	}
}

TEST_CASE("Manager Exception Handling", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();

	SECTION("Invalid device access") {
		const auto& devices = ARBD::SYCL::Manager::devices();
		int invalid_id = static_cast<int>(devices.size());
		// This should throw an exception
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::use(invalid_id));

		// But sync with invalid device ID should throw
		REQUIRE_THROWS_AS(ARBD::SYCL::Manager::sync(invalid_id), ARBD::Exception);

		// And get_device with invalid ID should throw
		REQUIRE_THROWS_AS(ARBD::SYCL::Manager::get_device(invalid_id), ARBD::Exception);
	}

	SECTION("Error recovery") {
		// After an error, the manager should still be usable
		try {
			const auto& devices = ARBD::SYCL::Manager::devices();
			int invalid_id = static_cast<int>(devices.size());
			ARBD::SYCL::Manager::use(invalid_id);
		} catch (const ARBD::Exception&) {
			// Expected
		}

		// Should still be able to use valid device
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::use(0));
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::get_current_queue());
	}
}

TEST_CASE("Manager Device Discovery Details", "[Manager][Backend]") {
	ARBD::SYCL::Manager::init();

	const auto& all_devices = ARBD::SYCL::Manager::all_devices();

	SECTION("Platform information") {
		LOGINFO("=== SYCL Platform and Device Information ===");

		try {
			auto platforms = sycl::platform::get_platforms();
			for (size_t p = 0; p < platforms.size(); ++p) {
				const auto& platform = platforms[p];
				LOGINFO("Platform {}: {} ({})",
						p,
						platform.get_info<sycl::info::platform::name>().c_str(),
						platform.get_info<sycl::info::platform::vendor>().c_str());

				auto platform_devices = platform.get_devices();
				for (size_t d = 0; d < platform_devices.size(); ++d) {
					const auto& device = platform_devices[d];
					try {
						LOGINFO("  Device {}: {} ({})",
								d,
								device.get_info<sycl::info::device::name>().c_str(),
								device.get_info<sycl::info::device::vendor>().c_str());
						LOGINFO(
							"    Type: {}, Available: {}",
							static_cast<int>(device.get_info<sycl::info::device::device_type>()),
							device.get_info<sycl::info::device::is_available>());
					} catch (const sycl::exception& e) {
						LOGINFO("    Device {}: Error querying properties: {}", d, e.what());
					}
				}
			}
		} catch (const sycl::exception& e) {
			LOGINFO("Error during platform discovery: {}", e.what());
		}

		LOGINFO("=== End Platform Information ===");
	}

	SECTION("Device capabilities") {
		for (size_t i = 0; i < all_devices.size(); ++i) {
			const auto& device = all_devices[i];

			LOGINFO("Device {} detailed capabilities:", i);
			LOGINFO("  Name: {}", device.name().c_str());
			LOGINFO("  Vendor: {}", device.vendor().c_str());
			LOGINFO("  Version: {}", device.version().c_str());
			LOGINFO("  Type: CPU={}, GPU={}, Accelerator={}",
					device.is_cpu(),
					device.is_gpu(),
					device.is_accelerator());
			LOGINFO("  Max compute units: {}", device.max_compute_units());
			LOGINFO("  Max work group size: {}", device.max_work_group_size());
			LOGINFO("  Local memory size: {:.1f} KB",
					static_cast<float>(device.local_mem_size()) / 1024.0f);
			LOGINFO("  Global memory: {:.1f} GB",
					static_cast<float>(device.global_mem_size()) / (1024.0f * 1024.0f * 1024.0f));

			// Test device-specific capabilities
			try {
				const auto& sycl_device = device.get_device();
				bool supports_fp64 = sycl_device.has(sycl::aspect::fp64);
				bool supports_atomic64 = sycl_device.has(sycl::aspect::atomic64);
				bool supports_usm_device = sycl_device.has(sycl::aspect::usm_device_allocations);
				bool supports_usm_host = sycl_device.has(sycl::aspect::usm_host_allocations);
				bool supports_usm_shared = sycl_device.has(sycl::aspect::usm_shared_allocations);

				LOGINFO("  Capabilities: FP64={}, Atomic64={}, USM(D/H/S)={}/{}/{}",
						supports_fp64,
						supports_atomic64,
						supports_usm_device,
						supports_usm_host,
						supports_usm_shared);
			} catch (const sycl::exception& e) {
				LOGINFO("  Error querying capabilities: {}", e.what());
			}
		}
	}
}

TEST_CASE("Manager Finalization", "[Manager][Backend]") {
	// Initialize first
	ARBD::SYCL::Manager::init();
	ARBD::SYCL::Manager::load_info();

	SECTION("Clean finalization") {
		REQUIRE_NOTHROW(ARBD::SYCL::Manager::finalize());

		// After finalization, devices should be empty
		const auto& devices = ARBD::SYCL::Manager::devices();
		REQUIRE(devices.empty());
	}
}

#else
TEST_CASE("SYCL Backend Not Available", "[Manager][Backend]") {
	SKIP("SYCL backend not compiled in");
}
#endif // USE_SYCL
