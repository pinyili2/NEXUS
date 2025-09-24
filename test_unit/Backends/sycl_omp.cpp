#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include <cmath>
#include <string>

#ifdef USE_SYCL

using namespace ARBD::SYCL;

// Helper function for floating point comparison
bool within_tolerance(float a, float b, float tolerance = 1e-6f) {
	return std::fabs(a - b) <= tolerance;
}

TEST_CASE("SYCL OpenMP Backend Device Discovery", "[SYCL][OpenMP]") {
	SECTION("OpenMP Platform Detection") {
		// Set environment variable to prefer OpenMP
		setenv("ONEAPI_DEVICE_SELECTOR", "omp:*", 1);

		// Initialize SYCL manager
		Manager::init();
		Manager::load_info();

		// Check that we have devices
		REQUIRE(Manager::all_device_size() > 0);

		// Check that we have selected devices
		const auto& selected_devices = Manager::devices();
		REQUIRE(selected_devices.size() > 0);

		// Verify that we're using OpenMP backend
		bool found_openmp_device = false;
		for (const auto& device : selected_devices) {
			std::string device_name = device.name();
			if (device_name.find("OpenMP") != std::string::npos ||
				device_name.find("AdaptiveCpp") != std::string::npos) {
				found_openmp_device = true;
				INFO("Found OpenMP device: " << device_name);
				break;
			}
		}

		// This should be true on macOS with AdaptiveCpp
		if (found_openmp_device) {
			SUCCEED("OpenMP backend detected successfully");
		} else {
			WARN("OpenMP backend not detected, but continuing with available devices");
		}
	}
}

TEST_CASE("SYCL OpenMP Queue Operations", "[SYCL][OpenMP]") {
	SECTION("Queue Creation and Synchronization") {
		auto& queue = Manager::get_current_queue();

		// Test queue synchronization
		queue.synchronize();
		SUCCEED("Queue synchronization successful");
	}

	SECTION("Simple Kernel Execution") {
		auto& queue = Manager::get_current_queue();

		// Test simple kernel execution
		int data[10] = {0};
		sycl::buffer<int, 1> buf(data, sycl::range<1>(10));

		queue.submit([&](sycl::handler& h) {
			auto acc = buf.get_access<sycl::access::mode::write>(h);
			h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) { acc[idx] = idx[0] * 2; });
		});

		queue.synchronize();

		// Verify results
		auto host_acc = buf.get_host_access();
		for (int i = 0; i < 10; ++i) {
			REQUIRE(host_acc[i] == i * 2);
		}
	}
}

TEST_CASE("SYCL OpenMP Memory Operations", "[SYCL][OpenMP]") {
	SECTION("Basic Memory Copy Operations") {
		auto& queue = Manager::get_current_queue();

		// Test memory allocation and copy operations using USM
		constexpr size_t SIZE = 1000;
		std::vector<float> host_data(SIZE);
		std::iota(host_data.begin(), host_data.end(), 0.0f);

		ARBD::DeviceBuffer<float> device_mem(SIZE);
		device_mem.copy_from_host(host_data);

		std::vector<float> result(SIZE, -1.0f); // Initialize with different value
		device_mem.copy_to_host(result);

		// Verify the copy operation
		for (size_t i = 0; i < SIZE; ++i) {
			REQUIRE(within_tolerance(result[i], host_data[i]));
		}
	}
}

TEST_CASE("SYCL OpenMP Parallel For", "[SYCL][OpenMP]") {
	SECTION("Vector Addition Kernel") {
		auto& queue = Manager::get_current_queue();

		// Vector addition test using USM and in-order queue
		constexpr size_t SIZE = 1000;
		std::vector<float> a(SIZE, 1.0f);
		std::vector<float> b(SIZE, 2.0f);
		std::vector<float> c(SIZE, 0.0f);

		// Use ARBD DeviceBuffer for automatic USM management
		ARBD::DeviceBuffer<float> d_a(SIZE);
		ARBD::DeviceBuffer<float> d_b(SIZE);
		ARBD::DeviceBuffer<float> d_c(SIZE);

		d_a.copy_from_host(a.data(), SIZE);
		d_b.copy_from_host(b.data(), SIZE);

		// Get raw pointers for kernel
		float* ptr_a = d_a.data();
		float* ptr_b = d_b.data();
		float* ptr_c = d_c.data();

		// Submit parallel kernel using USM pointers
		auto event = queue.submit([=](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> idx) {
				size_t i = idx[0];
				ptr_c[i] = ptr_a[i] + ptr_b[i];
			});
		});

		event.wait(); // Wait for kernel completion
		d_c.copy_to_host(c);

		// Verify results
		for (size_t i = 0; i < SIZE; ++i) {
			REQUIRE(within_tolerance(c[i], 3.0f));
		}
	}
}

TEST_CASE("SYCL OpenMP Reduction", "[SYCL][OpenMP]") {
	SECTION("Parallel Sum Reduction") {
		auto& queue = Manager::get_current_queue();

		// Parallel reduction test
		constexpr size_t SIZE = 1000;
		std::vector<int> data(SIZE);
		std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ..., SIZE

		ARBD::DeviceBuffer<int> d_data(SIZE);
		ARBD::DeviceBuffer<int> d_result(1);

		d_data.copy_from_host(data);

		int* ptr_data = d_data.data();
		int* ptr_result = d_result.data();

		// Initialize result to 0
		int zero = 0;
		queue.get().memcpy(ptr_result, &zero, sizeof(int)).wait();

		// Parallel sum reduction using SYCL2020 reduction
		auto event = queue.submit([=](sycl::handler& h) {
			auto sum_reduction = sycl::reduction(ptr_result, sycl::plus<int>());
			h.parallel_for(sycl::range<1>(SIZE), sum_reduction, [=](sycl::id<1> idx, auto& sum) {
				sum += ptr_data[idx[0]];
			});
		});

		event.wait();

		std::vector<int> result(1);
		d_result.copy_to_host(result);

		// Verify result: sum of 1 to SIZE = SIZE * (SIZE + 1) / 2
		int expected = SIZE * (SIZE + 1) / 2;
		REQUIRE(result[0] == expected);
	}
}

#else // USE_SYCL

TEST_CASE("SYCL OpenMP Backend", "[SYCL][OpenMP]") {
	SECTION("SYCL Not Enabled") {
		SKIP("SYCL support not enabled, skipping OpenMP SYCL tests");
	}
}

#endif // USE_SYCL
