#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/Vector3.h"

#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

using namespace ARBD;

// Include the Catch2 test runner
DEF_RUN_TRIAL

// Test constants
constexpr size_t VECTOR_BUFFER_SIZE = 1024;
constexpr size_t GRID_SIZE_X = 16;
constexpr size_t GRID_SIZE_Y = 16;
constexpr size_t GRID_SIZE_Z = 16;

// ============================================================================
// Backend Initialization Fixture
// ============================================================================

struct BackendInitFixture {
	BackendInitFixture() {
		try {
#ifdef USE_CUDA
			CUDA::Manager::init();
			CUDA::Manager::load_info();
			if (!CUDA::Manager::devices().empty()) {
				CUDA::Manager::use(0);
				LOGDEBUG("Initialized CUDA with {} device(s)", CUDA::Manager::devices().size());
			}
#endif

#ifdef USE_SYCL
			SYCL::Manager::init();
			SYCL::Manager::load_info();
			if (!SYCL::Manager::devices().empty()) {
				SYCL::Manager::use(0);
				LOGDEBUG("Initialized SYCL with {} device(s)", SYCL::Manager::devices().size());
			}
#endif

#ifdef USE_METAL
			METAL::Manager::init();
			METAL::Manager::load_info();
			if (!METAL::Manager::devices().empty()) {
				METAL::Manager::use(0);
				LOGDEBUG("Initialized Metal with {} device(s)", METAL::Manager::devices().size());
			}
#endif
		} catch (const std::exception& e) {
			LOGWARN("Backend initialization failed: {}", e.what());
		}
	}

	~BackendInitFixture() {
		try {
#ifdef USE_CUDA
			CUDA::Manager::finalize();
#endif
#ifdef USE_SYCL
			SYCL::Manager::finalize();
#endif
#ifdef USE_METAL
			METAL::Manager::finalize();
#endif
		} catch (const std::exception& e) {
			LOGWARN("Backend finalization failed: {}", e.what());
		}
	}
};

// ============================================================================
// Vector3_t Buffer Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Vector3_t Buffer Creation", "[Generic][vector3][creation]") {

	SECTION("Basic Vector3_t<float> buffer creation") {
		Resource resource = Resource::Local();

		DeviceBuffer<Vector3_t<float>> vec_buffer(VECTOR_BUFFER_SIZE, resource);

		REQUIRE(vec_buffer.size() == VECTOR_BUFFER_SIZE);
		REQUIRE_FALSE(vec_buffer.empty());
		REQUIRE(vec_buffer.data() != nullptr);
		REQUIRE(vec_buffer.bytes() == VECTOR_BUFFER_SIZE * sizeof(Vector3_t<float>));
	}

	SECTION("Vector3_t<double> buffer creation") {
		Resource resource = Resource::Local();

		DeviceBuffer<Vector3_t<double>> vec_buffer(512);

		REQUIRE(vec_buffer.size() == 512);
		REQUIRE(vec_buffer.bytes() == 512 * sizeof(Vector3_t<double>));
	}

	SECTION("Vector3_t<int> buffer creation") {
		Resource resource = Resource::Local();

		DeviceBuffer<Vector3_t<int>> vec_buffer(256);

		REQUIRE(vec_buffer.size() == 256);
		REQUIRE(vec_buffer.bytes() == 256 * sizeof(Vector3_t<int>));
	}

	SECTION("Empty buffer creation") {
		Resource resource = Resource::Local();

		DeviceBuffer<Vector3_t<float>> empty_buffer(0);

		REQUIRE(empty_buffer.size() == 0);
		REQUIRE(empty_buffer.empty());
		REQUIRE(empty_buffer.bytes() == 0);
	}
}

TEST_CASE_METHOD(BackendInitFixture,
				 "Vector3_t Buffer Host-Device Transfer",
				 "[Generic][vector3][transfer]") {
	// Skip buffer transfer tests on macOS when using SYCL due to unified memory architecture
	// skip_if_sycl_on_macos();

	SECTION("Vector3_t<float> host to device to host") {
		Resource resource = Resource::Local();
		DeviceBuffer<Vector3_t<float>> vec_buffer(VECTOR_BUFFER_SIZE);

		// Create host data with specific pattern
		std::vector<Vector3_t<float>> host_data(VECTOR_BUFFER_SIZE);
		for (size_t i = 0; i < VECTOR_BUFFER_SIZE; ++i) {
			host_data[i] = Vector3_t<float>(static_cast<float>(i),
											static_cast<float>(i * 2.5f),
											static_cast<float>(i * -1.5f));
		}

		// Transfer to device
		vec_buffer.copy_from_host(host_data.data(), VECTOR_BUFFER_SIZE);

		// Transfer back from device
		std::vector<Vector3_t<float>> result_data(VECTOR_BUFFER_SIZE);
		vec_buffer.copy_to_host(result_data);

		// Verify data integrity
		for (size_t i = 0; i < VECTOR_BUFFER_SIZE; ++i) {
			REQUIRE(std::abs(result_data[i].x - host_data[i].x) < 1e-6f);
			REQUIRE(std::abs(result_data[i].y - host_data[i].y) < 1e-6f);
			REQUIRE(std::abs(result_data[i].z - host_data[i].z) < 1e-6f);
		}
	}

	SECTION("Partial transfer operations") {
		Resource resource = Resource::Local();
		DeviceBuffer<Vector3_t<float>> vec_buffer(VECTOR_BUFFER_SIZE);

		// Test partial copy to device
		const size_t partial_size = VECTOR_BUFFER_SIZE / 2;
		std::vector<Vector3_t<float>> host_data(partial_size);

		for (size_t i = 0; i < partial_size; ++i) {
			host_data[i] = Vector3_t<float>(i * 10.0f, i * 20.0f, i * 30.0f);
		}

		vec_buffer.copy_from_host(host_data.data(), partial_size);

		// Test partial copy from device
		std::vector<Vector3_t<float>> result_data(partial_size);
		vec_buffer.copy_to_host(result_data.data(), partial_size);

		// Verify partial data
		for (size_t i = 0; i < partial_size; ++i) {
			REQUIRE(std::abs(result_data[i].x - host_data[i].x) < 1e-6f);
			REQUIRE(std::abs(result_data[i].y - host_data[i].y) < 1e-6f);
			REQUIRE(std::abs(result_data[i].z - host_data[i].z) < 1e-6f);
		}
	}
}

TEST_CASE_METHOD(BackendInitFixture,
				 "Vector3_t Buffer Move Semantics",
				 "[Generic][vector3][move]") {

	SECTION("Move constructor") {
		Resource resource = Resource::Local();
		DeviceBuffer<Vector3_t<float>> original_buffer(VECTOR_BUFFER_SIZE);

		// Store original pointer for verification
		Vector3_t<float>* original_ptr = original_buffer.data();

		// Move construct
		DeviceBuffer<Vector3_t<float>> moved_buffer = std::move(original_buffer);

		// Verify move semantics
		REQUIRE(moved_buffer.size() == VECTOR_BUFFER_SIZE);
		REQUIRE(moved_buffer.data() == original_ptr);
		REQUIRE(original_buffer.size() == 0);
		REQUIRE(original_buffer.empty());
		REQUIRE(original_buffer.data() == nullptr);
	}

	SECTION("Move assignment") {
		Resource resource = Resource::Local();
		DeviceBuffer<Vector3_t<float>> buffer1(VECTOR_BUFFER_SIZE);
		DeviceBuffer<Vector3_t<float>> buffer2(128); // Different size

		Vector3_t<float>* original_ptr = buffer1.data();

		// Move assign
		buffer2 = std::move(buffer1);

		// Verify move assignment
		REQUIRE(buffer2.size() == VECTOR_BUFFER_SIZE);
		REQUIRE(buffer2.data() == original_ptr);
		REQUIRE(buffer1.size() == 0);
		REQUIRE(buffer1.empty());
		REQUIRE(buffer1.data() == nullptr);
	}
}

// ============================================================================
// Memory Alignment Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Memory Alignment", "[Generic][alignment]") {

	SECTION("Vector3_t alignment requirements") {
		Resource resource = Resource::Local();

		DeviceBuffer<Vector3_t<float>> float_buffer(100);
		DeviceBuffer<Vector3_t<double>> double_buffer(100);

		// Check pointer alignment for Vector3_t types
		uintptr_t float_ptr = reinterpret_cast<uintptr_t>(float_buffer.data());
		uintptr_t double_ptr = reinterpret_cast<uintptr_t>(double_buffer.data());

		// Vector3_t should maintain proper alignment for SIMD operations
		// Note: Exact alignment requirements depend on backend implementation
		REQUIRE(float_ptr % sizeof(float) == 0);
		REQUIRE(double_ptr % sizeof(double) == 0);
	}

	SECTION("Large buffer alignment") {
		Resource resource = Resource::Local();

		const size_t large_size = 1024 * 1024; // 1M elements
		DeviceBuffer<Vector3_t<float>> large_buffer(large_size);

		uintptr_t ptr = reinterpret_cast<uintptr_t>(large_buffer.data());

		// Should be properly aligned for efficient memory access
		REQUIRE(ptr % sizeof(Vector3_t<float>) == 0);
		REQUIRE(large_buffer.size() == large_size);
		REQUIRE(large_buffer.bytes() == large_size * sizeof(Vector3_t<float>));
	}
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Performance Tests", "[Generic][performance]") {

	SECTION("Large Vector3_t buffer allocation") {
		Resource resource = Resource::Local();

		const size_t large_size = 1024 * 1024; // 1M Vector3 elements

		auto start = std::chrono::high_resolution_clock::now();
		DeviceBuffer<Vector3_t<float>> large_buffer(large_size);
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		REQUIRE(large_buffer.size() == large_size);

		// Log performance info (actual thresholds depend on hardware)
		INFO("Large buffer allocation took " << duration.count() << " ms");
		REQUIRE(duration.count() < 5000); // Should complete within 5 seconds
	}

	SECTION("Multiple buffer allocation stress test") {
		Resource resource = Resource::Local();

		std::vector<std::unique_ptr<DeviceBuffer<Vector3_t<float>>>> buffers;
		const size_t num_buffers = 10;
		const size_t buffer_size = 1024;

		// Allocate multiple buffers
		for (size_t i = 0; i < num_buffers; ++i) {
			buffers.push_back(std::make_unique<DeviceBuffer<Vector3_t<float>>>(buffer_size));
			REQUIRE(buffers.back()->size() == buffer_size);
		}

		// All buffers should be properly allocated
		REQUIRE(buffers.size() == num_buffers);

		// Test that all buffers have unique pointers
		for (size_t i = 0; i < num_buffers; ++i) {
			for (size_t j = i + 1; j < num_buffers; ++j) {
				REQUIRE(buffers[i]->data() != buffers[j]->data());
			}
		}
	}
}
