#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Types/Vector3.h"

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
using namespace ARBD::CUDA;
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
using namespace ARBD::SYCL;
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
using namespace ARBD::METAL;
#endif

#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace ARBD;

// Include the Catch2 test runner
DEF_RUN_TRIAL

// Test constants
constexpr size_t SMALL_BUFFER_SIZE = 64;
constexpr size_t MEDIUM_BUFFER_SIZE = 1024;
constexpr size_t LARGE_BUFFER_SIZE = 8192;
constexpr size_t HUGE_BUFFER_SIZE = 65536;

// Helper function to skip buffer transfer tests on macOS with SYCL
inline void skip_if_sycl_on_macos() {
#ifdef USE_SYCL
#ifdef __APPLE__
	SKIP("Skipping buffer transfer tests on macOS - unified memory architecture");
#endif
#endif
}

// ============================================================================
// Backend Initialization Fixture
// ============================================================================

struct BackendInitFixture {
	BackendInitFixture() {
		try {
			Manager::init();
			Manager::load_info();
			if (!Manager::devices().empty()) {
				Manager::use(0);
				std::cout << "Initialized Device with " << Manager::devices().size() << " device(s)"
						  << std::endl;
			} else {
				std::cout << "Warning: No devices found" << std::endl;
			}

		} catch (const std::exception& e) {
			std::cerr << "Warning: Backend initialization failed: " << e.what() << std::endl;
		}
	}

	~BackendInitFixture() {
		try {
			Manager::finalize();
		} catch (const std::exception& e) {
			std::cerr << "Warning: Backend finalization failed: " << e.what() << std::endl;
		}
	}
};

// ============================================================================
// Helper Functions
// ============================================================================

template<typename T>
std::vector<T> generate_test_data(size_t size) {
	std::vector<T> data(size);
	std::random_device rd;
	std::mt19937 gen(rd());

	if constexpr (std::is_integral_v<T>) {
		std::uniform_int_distribution<T> dis(1, 1000);
		for (auto& val : data) {
			val = dis(gen);
		}
	} else if constexpr (std::is_floating_point_v<T>) {
		std::uniform_real_distribution<T> dis(0.0, 1000.0);
		for (auto& val : data) {
			val = dis(gen);
		}
	} else {
		// For complex types like Vector3, generate random components
		std::uniform_real_distribution<float> dis(0.0, 1000.0);
		for (auto& val : data) {
			if constexpr (std::is_same_v<T, Vector3_t<float>>) {
				val.x = dis(gen);
				val.y = dis(gen);
				val.z = dis(gen);
			} else if constexpr (std::is_same_v<T, Vector3_t<double>>) {
				val.x = static_cast<double>(dis(gen));
				val.y = static_cast<double>(dis(gen));
				val.z = static_cast<double>(dis(gen));
			}
		}
	}

	return data;
}

template<typename T>
bool vectors_equal(const std::vector<T>& a, const std::vector<T>& b, double tolerance = 1e-6) {
	if (a.size() != b.size())
		return false;

	for (size_t i = 0; i < a.size(); ++i) {
		if constexpr (std::is_floating_point_v<T>) {
			if (std::abs(a[i] - b[i]) > tolerance)
				return false;
		} else if constexpr (std::is_same_v<T, Vector3_t<float>> ||
							 std::is_same_v<T, Vector3_t<double>>) {
			if (std::abs(a[i].x - b[i].x) > tolerance || std::abs(a[i].y - b[i].y) > tolerance ||
				std::abs(a[i].z - b[i].z) > tolerance)
				return false;
		} else {
			if (a[i] != b[i])
				return false;
		}
	}
	return true;
}

// ============================================================================
// Basic Buffer Creation Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Basic Creation", "[buffer_test][creation]") {

	SECTION("Default constructor creates empty buffer") {
		DeviceBuffer<int> buffer;
		REQUIRE(buffer.empty());
		REQUIRE(buffer.size() == 0);
		REQUIRE(buffer.bytes() == 0);
		REQUIRE(buffer.data() == nullptr);
	}

	SECTION("Constructor with size allocates buffer") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		REQUIRE_FALSE(buffer.empty());
		REQUIRE(buffer.size() == MEDIUM_BUFFER_SIZE);
		REQUIRE(buffer.bytes() == MEDIUM_BUFFER_SIZE * sizeof(int));
		REQUIRE(buffer.data() != nullptr);
	}

	SECTION("Different data types") {
		// Test int buffer
		DeviceBuffer<int> int_buffer(SMALL_BUFFER_SIZE);
		REQUIRE(int_buffer.size() == SMALL_BUFFER_SIZE);
		REQUIRE(int_buffer.bytes() == SMALL_BUFFER_SIZE * sizeof(int));

		// Test float buffer
		DeviceBuffer<float> float_buffer(SMALL_BUFFER_SIZE);
		REQUIRE(float_buffer.size() == SMALL_BUFFER_SIZE);
		REQUIRE(float_buffer.bytes() == SMALL_BUFFER_SIZE * sizeof(float));

		// Test double buffer
		DeviceBuffer<double> double_buffer(SMALL_BUFFER_SIZE);
		REQUIRE(double_buffer.size() == SMALL_BUFFER_SIZE);
		REQUIRE(double_buffer.bytes() == SMALL_BUFFER_SIZE * sizeof(double));

		// Test Vector3 buffer
		DeviceBuffer<Vector3_t<float>> vec_buffer(SMALL_BUFFER_SIZE);
		REQUIRE(vec_buffer.size() == SMALL_BUFFER_SIZE);
		REQUIRE(vec_buffer.bytes() == SMALL_BUFFER_SIZE * sizeof(Vector3_t<float>));
	}

	SECTION("Large buffer allocation") {
		DeviceBuffer<int> large_buffer(LARGE_BUFFER_SIZE);
		REQUIRE(large_buffer.size() == LARGE_BUFFER_SIZE);
		REQUIRE(large_buffer.bytes() == LARGE_BUFFER_SIZE * sizeof(int));
		REQUIRE(large_buffer.data() != nullptr);
	}

	SECTION("Huge buffer allocation") {
		DeviceBuffer<int> huge_buffer(HUGE_BUFFER_SIZE);
		REQUIRE(huge_buffer.size() == HUGE_BUFFER_SIZE);
		REQUIRE(huge_buffer.bytes() == HUGE_BUFFER_SIZE * sizeof(int));
		REQUIRE(huge_buffer.data() != nullptr);
	}
}

// ============================================================================
// Buffer Resize Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Resize", "[buffer_test][resize]") {
	SECTION("Resize to smaller size") {
		DeviceBuffer<int> buffer(LARGE_BUFFER_SIZE);
		REQUIRE(buffer.size() == LARGE_BUFFER_SIZE);

		buffer.resize(SMALL_BUFFER_SIZE);
		REQUIRE(buffer.size() == SMALL_BUFFER_SIZE);
		REQUIRE(buffer.data() != nullptr);
	}

	SECTION("Resize to larger size") {
		DeviceBuffer<int> buffer(SMALL_BUFFER_SIZE);
		REQUIRE(buffer.size() == SMALL_BUFFER_SIZE);

		buffer.resize(LARGE_BUFFER_SIZE);
		REQUIRE(buffer.size() == LARGE_BUFFER_SIZE);
		REQUIRE(buffer.data() != nullptr);
	}

	SECTION("Resize to same size") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		void* original_ptr = buffer.data();

		buffer.resize(MEDIUM_BUFFER_SIZE);
		REQUIRE(buffer.size() == MEDIUM_BUFFER_SIZE);
		REQUIRE(buffer.data() != nullptr);
		// Note: The pointer might change even for same size due to Metal's memory management
	}

	SECTION("Resize to zero") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		REQUIRE_FALSE(buffer.empty());

		buffer.resize(0);
		REQUIRE(buffer.empty());
		REQUIRE(buffer.size() == 0);
		REQUIRE(buffer.bytes() == 0);
	}
}

// ============================================================================
// Copy From Host Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Copy From Host", "[buffer_test][copy_from_host]") {

	SECTION("Copy int data from host") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data));

		// Verify by copying back
		std::vector<int> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy float data from host") {
		DeviceBuffer<float> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<float>(MEDIUM_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data));

		std::vector<float> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy Vector3 data from host") {
		DeviceBuffer<Vector3_t<float>> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<Vector3_t<float>>(MEDIUM_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data));

		std::vector<Vector3_t<float>> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy partial data from host") {
		DeviceBuffer<int> buffer(LARGE_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data.data(), MEDIUM_BUFFER_SIZE));

		std::vector<int> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data.data(), MEDIUM_BUFFER_SIZE));
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy with automatic resize") {
		DeviceBuffer<int> buffer(SMALL_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(LARGE_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data));
		REQUIRE(buffer.size() == LARGE_BUFFER_SIZE);

		std::vector<int> result_data(LARGE_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy large dataset") {
		DeviceBuffer<int> buffer(HUGE_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(HUGE_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data));

		std::vector<int> result_data(HUGE_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(vectors_equal(host_data, result_data));
	}
}

// ============================================================================
// Copy To Host Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Copy To Host", "[buffer_test][copy_to_host]") {

	SECTION("Copy int data to host") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		std::vector<int> result_data;
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(result_data.size() == MEDIUM_BUFFER_SIZE);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy float data to host") {
		DeviceBuffer<float> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<float>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		std::vector<float> result_data;
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(result_data.size() == MEDIUM_BUFFER_SIZE);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy Vector3 data to host") {
		DeviceBuffer<Vector3_t<float>> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<Vector3_t<float>>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		std::vector<Vector3_t<float>> result_data;
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(result_data.size() == MEDIUM_BUFFER_SIZE);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy partial data to host") {
		DeviceBuffer<int> buffer(LARGE_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(LARGE_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		std::vector<int> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data.data(), MEDIUM_BUFFER_SIZE));
		REQUIRE(vectors_equal(
			std::vector<int>(host_data.begin(), host_data.begin() + MEDIUM_BUFFER_SIZE),
			result_data));
	}

	SECTION("Copy to pre-allocated host buffer") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		std::vector<int> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data.data(), MEDIUM_BUFFER_SIZE));
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy large dataset to host") {
		DeviceBuffer<int> buffer(HUGE_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(HUGE_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		std::vector<int> result_data;
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(result_data.size() == HUGE_BUFFER_SIZE);
		REQUIRE(vectors_equal(host_data, result_data));
	}
}

// ============================================================================
// Device to Device Copy Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture,
				 "Buffer Device to Device Copy",
				 "[buffer_test][device_to_device]") {

	SECTION("Copy int data device to device") {
		DeviceBuffer<int> src_buffer(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<int> dst_buffer(MEDIUM_BUFFER_SIZE);

		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		src_buffer.copy_from_host(host_data);

		REQUIRE_NOTHROW(dst_buffer.copy_device_to_device(src_buffer, MEDIUM_BUFFER_SIZE));

		std::vector<int> result_data;
		dst_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy float data device to device") {
		DeviceBuffer<float> src_buffer(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<float> dst_buffer(MEDIUM_BUFFER_SIZE);

		auto host_data = generate_test_data<float>(MEDIUM_BUFFER_SIZE);
		src_buffer.copy_from_host(host_data);

		REQUIRE_NOTHROW(dst_buffer.copy_device_to_device(src_buffer, MEDIUM_BUFFER_SIZE));

		std::vector<float> result_data;
		dst_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy Vector3 data device to device") {
		DeviceBuffer<Vector3_t<float>> src_buffer(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<Vector3_t<float>> dst_buffer(MEDIUM_BUFFER_SIZE);

		auto host_data = generate_test_data<Vector3_t<float>>(MEDIUM_BUFFER_SIZE);
		src_buffer.copy_from_host(host_data);

		REQUIRE_NOTHROW(dst_buffer.copy_device_to_device(src_buffer, MEDIUM_BUFFER_SIZE));

		std::vector<Vector3_t<float>> result_data;
		dst_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy partial data device to device") {
		DeviceBuffer<int> src_buffer(LARGE_BUFFER_SIZE);
		DeviceBuffer<int> dst_buffer(MEDIUM_BUFFER_SIZE);

		auto host_data = generate_test_data<int>(LARGE_BUFFER_SIZE);
		src_buffer.copy_from_host(host_data);

		REQUIRE_NOTHROW(dst_buffer.copy_device_to_device(src_buffer, MEDIUM_BUFFER_SIZE));

		std::vector<int> result_data;
		dst_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(
			std::vector<int>(host_data.begin(), host_data.begin() + MEDIUM_BUFFER_SIZE),
			result_data));
	}

	SECTION("Copy large dataset device to device") {
		DeviceBuffer<int> src_buffer(HUGE_BUFFER_SIZE);
		DeviceBuffer<int> dst_buffer(HUGE_BUFFER_SIZE);

		auto host_data = generate_test_data<int>(HUGE_BUFFER_SIZE);
		src_buffer.copy_from_host(host_data);

		REQUIRE_NOTHROW(dst_buffer.copy_device_to_device(src_buffer, HUGE_BUFFER_SIZE));

		std::vector<int> result_data;
		dst_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Self copy (buffer copying to itself)") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		// Create a copy of the original data
		std::vector<int> original_data = host_data;

		// Modify the host data to ensure we're not just reading the same data
		for (auto& val : host_data) {
			val *= 2;
		}

		// Copy the modified data to device
		buffer.copy_from_host(host_data);

		// Now copy device to device (self copy)
		REQUIRE_NOTHROW(buffer.copy_device_to_device(buffer, MEDIUM_BUFFER_SIZE));

		std::vector<int> result_data;
		buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}
}

// ============================================================================
// Copy Constructor and Assignment Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture,
				 "Buffer Copy Constructor and Assignment",
				 "[buffer_test][copy_constructor]") {

	SECTION("Copy constructor") {
		DeviceBuffer<int> original_buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		original_buffer.copy_from_host(host_data);

		DeviceBuffer<int> copied_buffer(original_buffer);
		REQUIRE(copied_buffer.size() == original_buffer.size());
		REQUIRE(copied_buffer.bytes() == original_buffer.bytes());
		REQUIRE(copied_buffer.data() != original_buffer.data()); // Different pointers

		std::vector<int> result_data;
		copied_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy assignment operator") {
		DeviceBuffer<int> original_buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		original_buffer.copy_from_host(host_data);

		DeviceBuffer<int> assigned_buffer(SMALL_BUFFER_SIZE);
		assigned_buffer = original_buffer;
		REQUIRE(assigned_buffer.size() == original_buffer.size());
		REQUIRE(assigned_buffer.bytes() == original_buffer.bytes());
		REQUIRE(assigned_buffer.data() != original_buffer.data()); // Different pointers

		std::vector<int> result_data;
		assigned_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Copy assignment with different sizes") {
		DeviceBuffer<int> small_buffer(SMALL_BUFFER_SIZE);
		auto small_data = generate_test_data<int>(SMALL_BUFFER_SIZE);
		small_buffer.copy_from_host(small_data);

		DeviceBuffer<int> large_buffer(LARGE_BUFFER_SIZE);
		auto large_data = generate_test_data<int>(LARGE_BUFFER_SIZE);
		large_buffer.copy_from_host(large_data);

		// Copy small to large
		large_buffer = small_buffer;
		REQUIRE(large_buffer.size() == SMALL_BUFFER_SIZE);

		std::vector<int> result_data;
		large_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(small_data, result_data));

		// Copy large to small
		small_buffer = large_buffer;
		REQUIRE(small_buffer.size() == SMALL_BUFFER_SIZE);

		result_data.clear();
		small_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(small_data, result_data));
	}

	SECTION("Self assignment") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		void* original_ptr = buffer.data();
		size_t original_size = buffer.size();

		buffer = buffer; // Self assignment

		REQUIRE(buffer.size() == original_size);
		REQUIRE(buffer.data() != nullptr);

		std::vector<int> result_data;
		buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}
}

// ============================================================================
// Move Constructor and Assignment Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture,
				 "Buffer Move Constructor and Assignment",
				 "[buffer_test][move]") {

	SECTION("Move constructor") {
		DeviceBuffer<int> original_buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		original_buffer.copy_from_host(host_data);

		void* original_ptr = original_buffer.data();
		size_t original_size = original_buffer.size();

		DeviceBuffer<int> moved_buffer(std::move(original_buffer));

		// Original should be empty
		REQUIRE(original_buffer.empty());
		REQUIRE(original_buffer.data() == nullptr);
		REQUIRE(original_buffer.size() == 0);

		// Moved should have the data
		REQUIRE(moved_buffer.size() == original_size);
		REQUIRE(moved_buffer.data() != nullptr);

		std::vector<int> result_data;
		moved_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Move assignment operator") {
		DeviceBuffer<int> original_buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		original_buffer.copy_from_host(host_data);

		DeviceBuffer<int> assigned_buffer(SMALL_BUFFER_SIZE);
		assigned_buffer = std::move(original_buffer);

		// Original should be empty
		REQUIRE(original_buffer.empty());
		REQUIRE(original_buffer.data() == nullptr);

		// Assigned should have the data
		REQUIRE(assigned_buffer.size() == MEDIUM_BUFFER_SIZE);
		REQUIRE(assigned_buffer.data() != nullptr);

		std::vector<int> result_data;
		assigned_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Move assignment with different sizes") {
		DeviceBuffer<int> small_buffer(SMALL_BUFFER_SIZE);
		auto small_data = generate_test_data<int>(SMALL_BUFFER_SIZE);
		small_buffer.copy_from_host(small_data);

		DeviceBuffer<int> large_buffer(LARGE_BUFFER_SIZE);
		auto large_data = generate_test_data<int>(LARGE_BUFFER_SIZE);
		large_buffer.copy_from_host(large_data);

		// Move small to large
		large_buffer = std::move(small_buffer);
		REQUIRE(large_buffer.size() == SMALL_BUFFER_SIZE);
		REQUIRE(small_buffer.empty());

		std::vector<int> result_data;
		large_buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(small_data, result_data));
	}
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture,
				 "Buffer Edge Cases and Error Handling",
				 "[buffer_test][edge_cases]") {

	SECTION("Copy operations with mismatched sizes") {
		DeviceBuffer<int> buffer(SMALL_BUFFER_SIZE);
		auto large_data = generate_test_data<int>(LARGE_BUFFER_SIZE);

		// This should resize the buffer automatically
		REQUIRE_NOTHROW(buffer.copy_from_host(large_data));
		REQUIRE(buffer.size() == LARGE_BUFFER_SIZE);

		std::vector<int> result_data;
		buffer.copy_to_host(result_data);
		REQUIRE(vectors_equal(large_data, result_data));
	}

	SECTION("Copy operations with partial data") {
		DeviceBuffer<int> buffer(LARGE_BUFFER_SIZE);
		auto partial_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(partial_data.data(), MEDIUM_BUFFER_SIZE));

		std::vector<int> result_data(MEDIUM_BUFFER_SIZE);
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data.data(), MEDIUM_BUFFER_SIZE));
		REQUIRE(vectors_equal(partial_data, result_data));
	}

	SECTION("Multiple resize operations") {
		DeviceBuffer<int> buffer(SMALL_BUFFER_SIZE);

		for (size_t size :
			 {MEDIUM_BUFFER_SIZE, LARGE_BUFFER_SIZE, HUGE_BUFFER_SIZE, SMALL_BUFFER_SIZE}) {
			buffer.resize(size);
			REQUIRE(buffer.size() == size);
			REQUIRE(buffer.data() != nullptr);
		}
	}

	SECTION("Buffer with complex data types") {
		DeviceBuffer<Vector3_t<double>> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<Vector3_t<double>>(MEDIUM_BUFFER_SIZE);

		REQUIRE_NOTHROW(buffer.copy_from_host(host_data));

		std::vector<Vector3_t<double>> result_data;
		REQUIRE_NOTHROW(buffer.copy_to_host(result_data));
		REQUIRE(vectors_equal(host_data, result_data));
	}
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Performance", "[buffer_test][performance]") {

	SECTION("Large data transfer performance") {
		const size_t test_size = 1000000; // 1M elements

		DeviceBuffer<int> buffer(test_size);
		auto host_data = generate_test_data<int>(test_size);

		auto start = std::chrono::high_resolution_clock::now();
		buffer.copy_from_host(host_data);
		auto end = std::chrono::high_resolution_clock::now();

		auto copy_to_device_time =
			std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGINFO("Copy {} elements to device took {} ms", test_size, copy_to_device_time.count());

		std::vector<int> result_data;
		start = std::chrono::high_resolution_clock::now();
		buffer.copy_to_host(result_data);
		end = std::chrono::high_resolution_clock::now();

		auto copy_from_device_time =
			std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGINFO("Copy {} elements from device took {} ms",
				test_size,
				copy_from_device_time.count());

		REQUIRE(vectors_equal(host_data, result_data));
	}

	SECTION("Device to device copy performance") {
		const size_t test_size = 500000; // 500K elements

		DeviceBuffer<int> src_buffer(test_size);
		// Create destination buffer with the same resource as source and sync=true
		DeviceBuffer<int> dst_buffer(test_size, src_buffer.resource(), nullptr, true);

		auto host_data = generate_test_data<int>(test_size);
		src_buffer.copy_from_host(host_data);

		// Ensure source buffer is ready by performing a small operation
		// This helps ensure the device is ready for operations
		DeviceBuffer<int> test_buffer(10, src_buffer.resource());
		test_buffer.copy_from_host(std::vector<int>(10, 1));

		auto start = std::chrono::high_resolution_clock::now();
		dst_buffer.copy_device_to_device(src_buffer, test_size);
		auto end = std::chrono::high_resolution_clock::now();

		auto copy_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		LOGINFO("Device to device copy of {} elements took {} ms", test_size, copy_time.count());

		// The copy_to_host method already forces synchronization, so we don't need additional sync
		// The copy_from_host and copy_to_host methods use sync=true internally
		std::vector<int> result_data;
		dst_buffer.copy_to_host(result_data);

		// Debug: Check sizes and first few elements
		LOGINFO("Expected size: {}, Result size: {}", host_data.size(), result_data.size());
		if (result_data.size() > 0) {
			LOGINFO("First 5 expected: {}, {}, {}, {}, {}",
					host_data[0],
					host_data[1],
					host_data[2],
					host_data[3],
					host_data[4]);
			LOGINFO("First 5 result: {}, {}, {}, {}, {}",
					result_data[0],
					result_data[1],
					result_data[2],
					result_data[3],
					result_data[4]);
		}

		// Check if the data is actually different or just a timing issue
		if (result_data.size() == host_data.size()) {
			size_t mismatch_count = 0;
			size_t zero_count = 0;
			for (size_t i = 0; i < std::min(result_data.size(), host_data.size()); ++i) {
				if (result_data[i] != host_data[i]) {
					mismatch_count++;
					if (result_data[i] == 0) {
						zero_count++;
					}
					if (mismatch_count <= 10) { // Log first 10 mismatches
						LOGINFO("Mismatch at index {}: expected {}, got {}",
								i,
								host_data[i],
								result_data[i]);
					}
				}
			}
			LOGINFO("Total mismatches: {} out of {}, Zero values: {}",
					mismatch_count,
					host_data.size(),
					zero_count);

			// If we have too many mismatches, the copy operation failed
			if (mismatch_count > test_size * 0.1) { // More than 10% mismatches
				FAIL("Device-to-device copy failed with too many mismatches: "
					 << mismatch_count << " out of " << test_size);
			}
		}

		REQUIRE(vectors_equal(host_data, result_data));
	}
}

// ============================================================================
// Metal-Specific Tests
// ============================================================================

#ifdef USE_METAL
TEST_CASE_METHOD(BackendInitFixture,
				 "Metal Buffer Metal-Specific Features",
				 "[metal][buffer_test][metal]") {

	SECTION("Buffer binding to compute encoder") {
		DeviceBuffer<int> buffer(MEDIUM_BUFFER_SIZE);
		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		buffer.copy_from_host(host_data);

		// Get the current device
		const auto& device = Manager::get_current_device();
		REQUIRE(device.metal_device() != nullptr);

		// Create command buffer using the queue
		auto& queue = const_cast<Manager::Device&>(device).get_next_queue();
		auto command_buffer = queue.create_command_buffer();
		REQUIRE(command_buffer != nullptr);

		// Create compute command encoder
		auto command_buffer_ptr = reinterpret_cast<MTL::CommandBuffer*>(command_buffer);
		auto compute_encoder = command_buffer_ptr->computeCommandEncoder();
		REQUIRE(compute_encoder != nullptr);

		// Test binding the buffer to the encoder
		REQUIRE_NOTHROW(buffer.bind_to_encoder(compute_encoder, 0));

		// Clean up
		compute_encoder->endEncoding();
	}

	SECTION("Multiple buffer bindings") {
		DeviceBuffer<int> buffer1(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<float> buffer2(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<Vector3_t<float>> buffer3(MEDIUM_BUFFER_SIZE);

		auto host_data1 = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
		auto host_data2 = generate_test_data<float>(MEDIUM_BUFFER_SIZE);
		auto host_data3 = generate_test_data<Vector3_t<float>>(MEDIUM_BUFFER_SIZE);

		buffer1.copy_from_host(host_data1);
		buffer2.copy_from_host(host_data2);
		buffer3.copy_from_host(host_data3);

		const auto& device = Manager::get_current_device();
		auto& queue = const_cast<Manager::Device&>(device).get_next_queue();
		auto command_buffer = queue.create_command_buffer();
		REQUIRE(command_buffer != nullptr);

		// Create compute command encoder
		auto command_buffer_ptr = reinterpret_cast<MTL::CommandBuffer*>(command_buffer);
		auto compute_encoder = command_buffer_ptr->computeCommandEncoder();
		REQUIRE(compute_encoder != nullptr);

		// Bind multiple buffers to different indices
		REQUIRE_NOTHROW(buffer1.bind_to_encoder(compute_encoder, 0));
		REQUIRE_NOTHROW(buffer2.bind_to_encoder(compute_encoder, 1));
		REQUIRE_NOTHROW(buffer3.bind_to_encoder(compute_encoder, 2));

		compute_encoder->endEncoding();
	}
}
#endif

// ============================================================================
// Stress Tests trace trap!!!!
// ============================================================================

TEST_CASE_METHOD(BackendInitFixture, "Buffer Stress Tests", "[buffer_test][stress]") {

	SECTION("Multiple buffer creation and destruction") {
		const int num_buffers = 100;
		std::vector<std::unique_ptr<DeviceBuffer<int>>> buffers;

		for (int i = 0; i < num_buffers; ++i) {
			buffers.push_back(std::make_unique<DeviceBuffer<int>>(MEDIUM_BUFFER_SIZE));
			auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);
			buffers.back()->copy_from_host(host_data);
		}

		// Verify all buffers work correctly
		for (auto& buffer : buffers) {
			std::vector<int> result_data;
			buffer->copy_to_host(result_data);
			REQUIRE(result_data.size() == MEDIUM_BUFFER_SIZE);
			REQUIRE(buffer->size() == MEDIUM_BUFFER_SIZE);
		}
	}

	SECTION("Rapid resize operations") {
		DeviceBuffer<int> buffer(SMALL_BUFFER_SIZE);

		for (int i = 0; i < 50; ++i) {
			size_t new_size = SMALL_BUFFER_SIZE + (i * 100);
			buffer.resize(new_size);
			REQUIRE(buffer.size() == new_size);
			REQUIRE(buffer.data() != nullptr);
		}
	}

	SECTION("Concurrent copy operations") {
		DeviceBuffer<int> buffer1(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<int> buffer2(MEDIUM_BUFFER_SIZE);
		DeviceBuffer<int> buffer3(MEDIUM_BUFFER_SIZE);

		auto host_data = generate_test_data<int>(MEDIUM_BUFFER_SIZE);

		// Perform multiple copy operations
		buffer1.copy_from_host(host_data);
		buffer2.copy_from_host(host_data);
		buffer3.copy_from_host(host_data);

		std::vector<int> result1, result2, result3;
		buffer1.copy_to_host(result1);
		buffer2.copy_to_host(result2);
		buffer3.copy_to_host(result3);

		REQUIRE(vectors_equal(host_data, result1));
		REQUIRE(vectors_equal(host_data, result2));
		REQUIRE(vectors_equal(host_data, result3));
	}
}
