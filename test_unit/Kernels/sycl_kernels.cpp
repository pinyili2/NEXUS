#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>
using namespace ARBD;

#ifdef USE_SYCL

#include "Backend/SYCL/SYCLManager.h"

// ============================================================================
// Test Fixture for SYCL Kernel Tests
// ============================================================================
struct SYCLKernelTestFixture {
	Resource sycl_resource;
	bool sycl_available = false;

	SYCLKernelTestFixture() {
		try {
			SYCL::Manager::init();
			SYCL::Manager::load_info();

			if (SYCL::Manager::all_devices().empty()) {
				WARN("No SYCL devices found. Skipping SYCL kernel tests.");
				return;
			}

			sycl_resource = Resource(ResourceType::SYCL, 0);
			SYCL::Manager::use(0);
			sycl_available = true;

		} catch (const std::exception& e) {
			FAIL("Failed to initialize Manager for kernel tests: " << e.what());
		}
	}

	~SYCLKernelTestFixture() {
		try {
			if (sycl_available) {
				SYCL::Manager::finalize();
			}
		} catch (const std::exception& e) {
			std::cerr << "Error during Manager finalization: " << e.what() << std::endl;
		}
	}
};

// ============================================================================
// Basic SYCL Backend Tests
// ============================================================================

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Buffer Operations", "[sycl][kernels]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}

	SECTION("Basic buffer creation and data transfer") {
		const size_t n = 1000;

		// Create device buffer
		DeviceBuffer<float> buffer(n);
		REQUIRE(buffer.size() == n);
		REQUIRE(!buffer.empty());

		// Test data transfer
		std::vector<float> host_data(n);
		std::iota(host_data.begin(), host_data.end(), 1.0f);

		// Copy to device
		buffer.copy_from_host(host_data);

		// Copy back from device
		std::vector<float> result(n);
		buffer.copy_to_host(result);

		// Verify data integrity
		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == host_data[i]);
		}
	}
}

TEST_CASE_METHOD(SYCLKernelTestFixture,
				 "SYCL Direct Backend Functions",
				 "[sycl][kernels][backend]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}

	SECTION("SYCL kernel launch function") {
		const size_t n = 256;

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> output_buf(n);

		// Initialize input
		std::vector<float> input_data(n, 2.0f);
		input_buf.copy_from_host(input_data);

		// Simple scaling kernel
		auto scale_kernel = [](size_t i, const float* input, float* output) {
			output[i] = input[i] * 3.0f;
		};

		KernelConfig config;
		config.block_size = {64, 1, 1};
		config.sync = true;

		// Use the streamlined interface with individual buffer arguments
		auto event = launch_sycl_kernel(sycl_resource, config, scale_kernel, input_buf, output_buf);
		event.wait();

		// Verify results
		std::vector<float> result(n);
		output_buf.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			printf("result[%zu] = %f\n", i, result[i]);
			REQUIRE(std::abs(result[i] - 6.0f) < 1e-6f);
		}
	}
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Utility Functors", "[sycl][kernels][utils]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Configuration Testing", "[sycl][kernels][config]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}

	SECTION("Different block sizes") {
		const size_t n = 1024;
		std::vector<size_t> block_sizes = {32, 64, 128, 256};

		DeviceBuffer<int> input_buf(n);

		// Initialize with test pattern
		std::vector<int> input_data(n);
		std::iota(input_data.begin(), input_data.end(), 1);
		input_buf.copy_from_host(input_data);

		auto double_kernel = [](size_t i, const int* input, int* output) {
			output[i] = input[i] * 2;
		};

		for (size_t block_size : block_sizes) {
			DeviceBuffer<int> output_buf(n);

			KernelConfig config;
			config.block_size = {block_size, 1, 1};
			config.sync = true;

			auto event =
				launch_sycl_kernel(sycl_resource, config, double_kernel, input_buf, output_buf);
			event.wait();

			// Verify results are correct regardless of block size
			std::vector<int> result(n);
			output_buf.copy_to_host(result);

			for (size_t i = 0; i < n; ++i) {
				REQUIRE(result[i] == input_data[i] * 2);
			}
		}
	}

	SECTION("Async vs sync execution") {
		const size_t n = 512;

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> sync_output(n);
		DeviceBuffer<float> async_output(n);

		// Initialize input
		std::vector<float> input_data(n, 1.5f);
		input_buf.copy_from_host(input_data);

		auto add_kernel = [](size_t i, const float* input, float* output) {
			output[i] = input[i] + 10.0f;
		};

		// Synchronous execution
		{
			KernelConfig config;
			config.sync = true;
			config.block_size = {64, 1, 1};

			auto event =
				launch_sycl_kernel(sycl_resource, config, add_kernel, input_buf, sync_output);
			event.wait();
		}

		// Asynchronous execution
		{
			KernelConfig config;
			config.sync = false;
			config.block_size = {64, 1, 1};

			auto event =
				launch_sycl_kernel(sycl_resource, config, add_kernel, input_buf, async_output);
			event.wait();
		}

		// Both should produce identical results
		std::vector<float> sync_result(n), async_result(n);
		sync_output.copy_to_host(sync_result);
		async_output.copy_to_host(async_result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(std::abs(sync_result[i] - 11.5f) < 1e-6f);
			REQUIRE(std::abs(async_result[i] - 11.5f) < 1e-6f);
			REQUIRE(sync_result[i] == async_result[i]);
		}
	}
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Matrix Operations", "[sycl][kernels][matrix]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}

	SECTION("Matrix transpose") {
		const size_t rows = 8;
		const size_t cols = 6;
		const size_t total = rows * cols;

		// Create input matrix
		std::vector<float> input_host(total);
		std::iota(input_host.begin(), input_host.end(), 1.0f);

		// Calculate expected transpose
		std::vector<float> expected(total);
		for (size_t r = 0; r < rows; ++r) {
			for (size_t c = 0; c < cols; ++c) {
				expected[c * rows + r] = input_host[r * cols + c];
			}
		}

		// Create device buffers
		DeviceBuffer<float> input_buf(total);
		DeviceBuffer<float> output_buf(total);

		input_buf.copy_from_host(input_host);

		// Transpose kernel
		auto transpose_kernel = [rows, cols](size_t idx, const float* input, float* output) {
			size_t row = idx / cols;
			size_t col = idx % cols;

			if (row < rows && col < cols) {
				size_t input_idx = row * cols + col;
				size_t output_idx = col * rows + row;
				output[output_idx] = input[input_idx];
			}
		};

		KernelConfig config;
		config.block_size = {16, 1, 1};

		auto event =
			launch_sycl_kernel(sycl_resource, config, transpose_kernel, input_buf, output_buf);
		event.wait();

		// Verify results
		std::vector<float> result(total);
		output_buf.copy_to_host(result);

		for (size_t i = 0; i < total; ++i) {
			REQUIRE(result[i] == expected[i]);
		}
	}
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Error Handling", "[sycl][kernels][errors]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}

	SECTION("Zero-sized buffer handling") {
		DeviceBuffer<int> empty_buf(0);
		REQUIRE(empty_buf.size() == 0);
		REQUIRE(empty_buf.empty());

		// Ensure it doesn't crash
		REQUIRE_NOTHROW(empty_buf.data());
	}

	SECTION("Large configuration values") {
		const size_t n = 100;
		DeviceBuffer<float> input_buffer(n);
		DeviceBuffer<float> output_buffer(n);

		// Initialize buffer
		std::vector<float> data(n, 1.0f);
		input_buffer.copy_from_host(data);

		auto simple_kernel = [](size_t i, const float* input, float* output) {
			output[i] = input[i] + 1.0f;
		};

		// Test with very large block size (should be handled internally)
		std::vector<kerneldim3> oversized_blocks = {
			{65536, 1, 1},	  // Way too large in X
			{1024, 1024, 64}, // At the theoretical limit
			{2048, 1, 1},	  // Moderately too large
			{1, 1, 1024}	  // Too large in Z dimension
		};

		for (const auto& block_config : oversized_blocks) {
			KernelConfig config;
			config.block_size = block_config;

			// None of these should crash
			auto event = launch_sycl_kernel(sycl_resource,
											config,
											simple_kernel,
											input_buffer,
											output_buffer);
			REQUIRE_NOTHROW(event);

			// Wait for kernel completion
			event.wait();

			// Verify computation correctness
			std::vector<float> result(n);
			output_buffer.copy_to_host(result);

			for (size_t i = 0; i < n; ++i) {
				REQUIRE(std::abs(result[i] - 2.0f) < 1e-6f);
			}

			// Reset input buffer for next iteration
			input_buffer.copy_from_host(data);
		}
	}
}

TEST_CASE_METHOD(SYCLKernelTestFixture,
				 "SYCL Performance Characteristics",
				 "[sycl][kernels][performance]") {
	if (!sycl_available) {
		SKIP("SYCL not available");
	}

	SECTION("Throughput test with different data sizes") {
		std::vector<size_t> sizes = {1000, 10000};

		for (size_t n : sizes) {
			DeviceBuffer<float> input_buf(n);
			DeviceBuffer<float> output_buf(n);

			// Initialize with test data
			std::vector<float> input_data(n);
			std::iota(input_data.begin(), input_data.end(), 1.0f);
			input_buf.copy_from_host(input_data);

			auto scale_kernel = [](size_t i, const float* input, float* output) {
				output[i] = input[i] * 2.0f + 1.0f;
			};

			auto inputs = std::tie(input_buf);
			auto outputs = std::tie(output_buf);

			KernelConfig config;
			config.block_size = {256, 1, 1};
			config.sync = true;

			auto start = std::chrono::high_resolution_clock::now();
			auto event =
				launch_sycl_kernel(sycl_resource, config, scale_kernel, input_buf, output_buf);
			event.wait();
			auto end = std::chrono::high_resolution_clock::now();

			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

			// Verify correctness for first few elements
			std::vector<float> result(n);
			output_buf.copy_to_host(result);

			for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
				float expected = input_data[i] * 2.0f + 1.0f;
				REQUIRE(std::abs(result[i] - expected) < 1e-6f);
			}

			// Report performance (informational)
			double throughput = static_cast<double>(n) / duration.count();
			LOGINFO("Size: {}, Duration: {} μs, Throughput: {:.2f} Melem/s",
					n,
					duration.count(),
					throughput);
		}
	}
}

#endif // USE_SYCL
