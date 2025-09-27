#include "cuda_kernels.cuh"
#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>

using namespace ARBD;
using namespace Catch;
using Catch::Approx;

#ifdef USE_CUDA

class CUDAKernelTestFixture {
  private:
	Resource CUDA_resource;
	bool CUDA_available = false;

  public:
	CUDAKernelTestFixture() {
		try {
			CUDA::Manager::init();
			CUDA::Manager::load_info();

			if (CUDA::Manager::devices().empty()) {
				SKIP("No CUDA devices available. Skipping CUDA kernel tests.");
				return;
			}

			CUDA_resource = Resource(ResourceType::CUDA, 0);
			CUDA::Manager::use(0);
			CUDA_available = true;

		} catch (const std::exception& e) {
			FAIL("Failed to initialize Manager for kernel tests: " << e.what());
		}
	}

	~CUDAKernelTestFixture() {
		if (CUDA_available) {
			try {
				CUDA::Manager::finalize();
			} catch (const std::exception& e) {
				std::cerr << "Error during Manager finalization: " << e.what() << std::endl;
			}
		}
	}

	bool is_cuda_available() const {
		return CUDA_available;
	}

	const Resource& get_cuda_resource() const {
		return CUDA_resource;
	}
};

TEST_CASE_METHOD(CUDAKernelTestFixture, "CUDA Buffer Operations", "[CUDA][buffer]") {
	if (!is_cuda_available()) {
		SKIP("CUDA not available");
	}

	SECTION("Basic buffer creation and data transfer") {
		const size_t n = 1000;

		DeviceBuffer<float> buffer(n);
		REQUIRE(buffer.size() == n);
		REQUIRE(!buffer.empty());

		std::vector<float> host_data(n);
		std::iota(host_data.begin(), host_data.end(), 1.0f);

		buffer.copy_from_host(host_data);

		std::vector<float> result(n);
		buffer.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == host_data[i]);
		}
	}
}

TEST_CASE_METHOD(CUDAKernelTestFixture, "CUDA Kernel Launch", "[CUDA][kernels][launch]") {
	if (!is_cuda_available()) {
		SKIP("CUDA not available");
	}

	SECTION("Simple scaling kernel") {
		const size_t n = 256;

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> output_buf(n);

		std::vector<float> input_data(n, 2.0f);
		input_buf.copy_from_host(input_data);

		// Simple scaling kernel - Lambda signature: (size_t, const float*, float*)
		ScaleKernel scale_kernel;

		// Launch kernel with streamlined call
		KernelConfig config;
		Event completion =
			launch_kernel(get_cuda_resource(), n, config, input_buf, output_buf, scale_kernel);

		// Wait for completion and verify results
		completion.wait(); // Temporarily commented out to test kernel launch
		// cudaDeviceSynchronize(); // Use direct CUDA sync instead

		std::vector<float> result(n);
		output_buf.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == Approx(6.0f)); // 2.0 * 3.0 = 6.0
		}
	}

	SECTION("Kernel with scalar parameter") {
		const size_t n = 64;

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> output_buf(n);

		std::vector<float> input_data(n, 5.0f);
		input_buf.copy_from_host(input_data);

		float multiplier = 2.5f;

		// Kernel with extra scalar parameter
		MultiplyKernel multiply_kernel;

		KernelConfig config;
		Event completion = launch_kernel(get_cuda_resource(),
										 n,
										 config,
										 input_buf,
										 output_buf,
										 multiply_kernel,
										 multiplier);

		completion.wait();

		std::vector<float> result(n);
		output_buf.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == Approx(12.5f)); // 5.0 * 2.5 = 12.5
		}
	}
}

TEST_CASE_METHOD(CUDAKernelTestFixture, "CUDA Kernel Performance", "[CUDA][kernels][performance]") {
	if (!is_cuda_available()) {
		SKIP("CUDA not available");
	}

	SECTION("Large array processing") {
		const size_t n = 1024 * 1024; // 1M elements

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> output_buf(n);

		std::vector<float> input_data(n);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dist(0.0f, 100.0f);

		for (size_t i = 0; i < n; ++i) {
			input_data[i] = dist(gen);
		}

		input_buf.copy_from_host(input_data);

		SquareKernel square_kernel;

		KernelConfig config;
		config.block_size = {512, 1, 1}; // Larger block size for performance

		auto start = std::chrono::high_resolution_clock::now();
		Event completion =
			launch_kernel(get_cuda_resource(), n, config, input_buf, output_buf, square_kernel);
		completion.wait();
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		std::vector<float> result(n);
		output_buf.copy_to_host(result);

		// Verify a subset of results
		for (size_t i = 0; i < std::min(n, size_t(100)); ++i) {
			float expected = input_data[i] * input_data[i];
			REQUIRE(result[i] == Approx(expected).epsilon(1e-5f));
		}

		INFO("Processed " << n << " elements in " << duration.count() << " microseconds");
		REQUIRE(duration.count() < 10000); // Should complete in under 10ms
	}
}

TEST_CASE_METHOD(CUDAKernelTestFixture, "Simple CUDA Kernel Test", "[CUDA][kernels][simple]") {
	if (!is_cuda_available()) {
		SKIP("CUDA not available");
	}

	SECTION("Direct kernel launch test") {
		const size_t n = 256;

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> output_buf(n);

		std::vector<float> input_data(n, 2.0f);
		input_buf.copy_from_host(input_data);

		// Simple scaling kernel
		ScaleKernel scale_kernel;

		// Create simple input/output tuples
		auto inputs = std::make_tuple(std::ref(input_buf));
		auto outputs = std::make_tuple(std::ref(output_buf));

		// Launch kernel with simple configuration
		KernelConfig config;
		config.block_size.x = 256;
		config.grid_size.x = 1;

		Event completion =
			launch_kernel(get_cuda_resource(), n, config, inputs, outputs, scale_kernel);

		// Wait for completion
		completion.wait();

		// Verify results
		std::vector<float> result(n);
		output_buf.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == Approx(6.0f)); // 2.0 * 3.0 = 6.0
		}
	}
}

#endif
