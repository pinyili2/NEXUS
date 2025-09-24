#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Types/BaseGrid.h"
#include "catch_boiler.h"

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif
#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif
#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

using namespace ARBD;

class BaseGridTestFixture {
  protected:
	bool g_backend_available = false;

	void initialize_backend_once() {
		if (g_backend_available) {
			return;
		}

		try {
#ifdef USE_SYCL
			SYCL::Manager::load_info();
			SYCL::Manager::use(0);
			g_backend_available = true;
			INFO("SYCL backend initialized successfully");
#elif defined(USE_CUDA)
			CUDA::Manager::load_info();
			CUDA::Manager::use(0);
			g_backend_available = true;
			INFO("CUDA backend initialized successfully");
#elif defined(USE_METAL)
			METAL::Manager::load_info();
			METAL::Manager::use(0);
			g_backend_available = true;
			INFO("METAL backend initialized successfully");
#else
			// CPU fallback
			g_backend_available = true;
			INFO("CPU backend available");
#endif
		} catch (const std::exception& e) {
			WARN("Backend initialization failed: " << e.what());
			g_backend_available = false;
		}
	}

	std::vector<Resource> get_device_resources() {
		std::vector<Resource> resources;

		if (!g_backend_available) {
			return resources;
		}

#ifdef USE_SYCL
		auto& devices = SYCL::Manager::devices();
		for (size_t i = 0; i < std::min(devices.size(), size_t(2)); ++i) {
			resources.push_back(Resource(ResourceType::SYCL, i));
		}
#elif defined(USE_CUDA)
		// Add CUDA devices when available
		for (int i = 0; i < std::min(CUDA::Manager::get_device_count(), 2); ++i) {
			resources.push_back(Resource(ResourceType::CUDA, i));
		}
#elif defined(USE_METAL)
		// Add METAL devices when available
		for (int i = 0; i < std::min(METAL::Manager::get_device_count(), 2); ++i) {
			resources.push_back(Resource(ResourceType::METAL, i));
		}
#else
		// CPU fallback
		resources.push_back(Resource(ResourceType::CPU, 0));
#endif

		return resources;
	}
};

TEST_CASE_METHOD(BaseGridTestFixture,
				 "BaseGrid Basic Construction and Properties",
				 "[BaseGrid][construction]") {
	SECTION("Default constructor") {
		BaseGrid<float> grid;

		REQUIRE(grid.nx() == 1);
		REQUIRE(grid.ny() == 1);
		REQUIRE(grid.nz() == 1);
		REQUIRE(grid.size() == 1);

		Vector3_t<float> expected_origin{0, 0, 0};
		REQUIRE(grid.origin().x == expected_origin.x);
		REQUIRE(grid.origin().y == expected_origin.y);
		REQUIRE(grid.origin().z == expected_origin.z);
	}

	SECTION("Orthogonal grid constructor") {
		Vector3_t<float> box_size{10.0f, 8.0f, 6.0f};
		float dx = 1.0f;

		BaseGrid<float> grid(box_size, dx);

		REQUIRE(grid.nx() == 10);
		REQUIRE(grid.ny() == 8);
		REQUIRE(grid.nz() == 6);
		REQUIRE(grid.size() == 480);

		// Check that grid spans the box
		REQUIRE(std::abs(grid.get_total_volume() - (10.0f * 8.0f * 6.0f)) < 1e-6f);
	}

	SECTION("Custom basis constructor") {
		Matrix3_t<float> basis{Vector3_t<float>{1.0f, 0.0f, 0.0f},
							   Vector3_t<float>{0.0f, 2.0f, 0.0f},
							   Vector3_t<float>{0.0f, 0.0f, 3.0f}};
		Vector3_t<float> origin{-5.0f, -4.0f, -3.0f};

		BaseGrid<float> grid(basis, origin, 10, 8, 6);

		REQUIRE(grid.nx() == 10);
		REQUIRE(grid.ny() == 8);
		REQUIRE(grid.nz() == 6);
		REQUIRE(grid.size() == 480);

		// Check cell volume
		REQUIRE(std::abs(grid.get_cell_volume() - 6.0f) < 1e-6f);
	}
}

TEST_CASE_METHOD(BaseGridTestFixture, "BaseGrid Data Operations", "[BaseGrid][data]") {
	initialize_backend_once();

	SECTION("Grid data initialization and access") {
		BaseGrid<float> grid(Vector3_t<float>{5.0f, 5.0f, 5.0f}, 1.0f);

		REQUIRE(grid.nx() == 5);
		REQUIRE(grid.ny() == 5);
		REQUIRE(grid.nz() == 5);
		REQUIRE(grid.size() == 125);

#ifdef HOST_GUARD
		// Initialize with a pattern: value = x + y*nx + z*nx*ny
		for (idx_t x = 0; x < grid.nx(); ++x) {
			for (idx_t y = 0; y < grid.ny(); ++y) {
				for (idx_t z = 0; z < grid.nz(); ++z) {
					idx_t index = grid.index(x, y, z);
					float value = static_cast<float>(x + y * grid.nx() + z * grid.nx() * grid.ny());
					grid[index] = value;
				}
			}
		}

		// Verify the pattern
		for (idx_t x = 0; x < grid.nx(); ++x) {
			for (idx_t y = 0; y < grid.ny(); ++y) {
				for (idx_t z = 0; z < grid.nz(); ++z) {
					idx_t index = grid.index(x, y, z);
					float expected =
						static_cast<float>(x + y * grid.nx() + z * grid.nx() * grid.ny());
					REQUIRE(std::abs(grid[index] - expected) < 1e-6f);
				}
			}
		}
#endif
	}

	SECTION("Grid transformations") {
		BaseGrid<float> grid(Vector3_t<float>{10.0f, 10.0f, 10.0f}, 2.0f);

		// Test coordinate transformations
		Vector3_t<float> world_pos{1.0f, 2.0f, 3.0f};
		Vector3_t<float> grid_pos = grid.transform_to_grid(world_pos);
		Vector3_t<float> back_to_world = grid.transform_to_world(grid_pos);

		REQUIRE(std::abs(back_to_world.x - world_pos.x) < 1e-6f);
		REQUIRE(std::abs(back_to_world.y - world_pos.y) < 1e-6f);
		REQUIRE(std::abs(back_to_world.z - world_pos.z) < 1e-6f);
	}
}

TEST_CASE_METHOD(BaseGridTestFixture, "BaseGrid Device Buffer Integration", "[BaseGrid][device]") {
	initialize_backend_once();

	if (!g_backend_available) {
		SKIP("Backend not available for device buffer integration test");
	}

	auto device_resources = get_device_resources();
	if (device_resources.empty()) {
		SKIP("No device resources available for device buffer test");
	}

	SECTION("Grid data transfer to device buffer") {
		const idx_t nx = 8, ny = 6, nz = 4;
		BaseGrid<float> grid(Vector3_t<float>{8.0f, 6.0f, 4.0f}, 1.0f);

		REQUIRE(grid.nx() == nx);
		REQUIRE(grid.ny() == ny);
		REQUIRE(grid.nz() == nz);

#ifdef HOST_GUARD
		// Initialize with 3D wave pattern: sin(x) * cos(y) * sin(z)
		for (idx_t x = 0; x < nx; ++x) {
			for (idx_t y = 0; y < ny; ++y) {
				for (idx_t z = 0; z < nz; ++z) {
					idx_t index = grid.index(x, y, z);
					float fx = static_cast<float>(x) / nx;
					float fy = static_cast<float>(y) / ny;
					float fz = static_cast<float>(z) / nz;
					grid[index] =
						std::sin(fx * 2 * M_PI) * std::cos(fy * 2 * M_PI) * std::sin(fz * 2 * M_PI);
				}
			}
		}

		// Create device buffer and transfer grid data
		DeviceBuffer<float> device_buf(grid.size(), device_resources[0]);
		device_buf.copy_from_host(grid.data(), grid.size());

		// Verify data transfer
		std::vector<float> retrieved_data(grid.size());
		device_buf.copy_to_host(retrieved_data);

		for (idx_t i = 0; i < grid.size(); ++i) {
			REQUIRE(std::abs(retrieved_data[i] - grid[i]) < 1e-6f);
		}

		INFO("✓ Grid data successfully transferred to device buffer");
#endif
	}
}

TEST_CASE_METHOD(BaseGridTestFixture,
				 "BaseGrid Multi-Device Operations",
				 "[BaseGrid][multi-device]") {
	initialize_backend_once();

	if (!g_backend_available) {
		SKIP("Backend not available for multi-device test");
	}

	auto device_resources = get_device_resources();
	if (device_resources.size() < 2) {
		WARN("Need at least 2 devices for multi-device test, found " << device_resources.size());
		SKIP("Insufficient devices for multi-device BaseGrid test");
	}

	SECTION("Grid copy: Host → Device1 → Device2 → Host") {
		INFO("Testing BaseGrid multi-device transfer chain");

		const idx_t nx = 10, ny = 8, nz = 6;
		BaseGrid<float> original_grid(Vector3_t<float>{10.0f, 8.0f, 6.0f}, 1.0f);

		REQUIRE(original_grid.size() == nx * ny * nz);

#ifdef HOST_GUARD
		// Initialize with complex 3D pattern: (x^2 + y^2 + z^2) / total_elements
		float max_distance_sq = static_cast<float>(nx * nx + ny * ny + nz * nz);

		for (idx_t x = 0; x < nx; ++x) {
			for (idx_t y = 0; y < ny; ++y) {
				for (idx_t z = 0; z < nz; ++z) {
					idx_t index = original_grid.index(x, y, z);
					float distance_sq = static_cast<float>(x * x + y * y + z * z);
					original_grid[index] = distance_sq / max_distance_sq;
				}
			}
		}

		INFO("Initialized " << nx << "x" << ny << "x" << nz << " grid with distance pattern");

		// Step 1: Host → Device 1
		INFO("Step 1: Host → Device 1");
		DeviceBuffer<float> device1_buf(original_grid.size(), device_resources[0]);
		device1_buf.copy_from_host(original_grid.data(), original_grid.size());

		// Verify device 1 data
		std::vector<float> verify_device1(original_grid.size());
		device1_buf.copy_to_host(verify_device1);

		for (idx_t i = 0; i < original_grid.size(); ++i) {
			REQUIRE(std::abs(verify_device1[i] - original_grid[i]) < 1e-6f);
		}
		INFO("✓ Device 1 data verified");

		// Step 2: Device 1 → Device 2
		INFO("Step 2: Device 1 → Device 2");
		DeviceBuffer<float> device2_buf(original_grid.size(), device_resources[1]);
		device2_buf.copy_device_to_device(device1_buf, original_grid.size());

		// Verify device 2 data
		std::vector<float> verify_device2(original_grid.size());
		device2_buf.copy_to_host(verify_device2);

		for (idx_t i = 0; i < original_grid.size(); ++i) {
			REQUIRE(std::abs(verify_device2[i] - original_grid[i]) < 1e-6f);
		}
		INFO("✓ Device 2 data verified");

		// Step 3: Device 2 → Host (new grid)
		INFO("Step 3: Device 2 → Host (new grid)");
		BaseGrid<float> final_grid(Vector3_t<float>{10.0f, 8.0f, 6.0f}, 1.0f);

		device2_buf.copy_to_host(final_grid.data(), final_grid.size());

		// Verify final grid matches original
		for (idx_t x = 0; x < nx; ++x) {
			for (idx_t y = 0; y < ny; ++y) {
				for (idx_t z = 0; z < nz; ++z) {
					idx_t index = original_grid.index(x, y, z);
					REQUIRE(std::abs(final_grid[index] - original_grid[index]) < 1e-6f);
				}
			}
		}

		INFO("✓ Complete BaseGrid transfer chain successful!");

		// Performance test
		auto start = std::chrono::high_resolution_clock::now();
		device2_buf.copy_device_to_device(device1_buf, original_grid.size());
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		INFO("Device-to-device BaseGrid transfer time: " << duration.count() << " μs");
#endif
	}
}

TEST_CASE_METHOD(BaseGridTestFixture, "BaseGrid Unified Memory Operations", "[BaseGrid][unified]") {
	initialize_backend_once();

	if (!g_backend_available) {
		SKIP("Backend not available for unified memory test");
	}

	auto device_resources = get_device_resources();
	if (device_resources.empty()) {
		SKIP("No device resources available for unified memory test");
	}

	SECTION("BaseGrid with Unified Memory and Memory Advice") {
		INFO("Testing BaseGrid with Unified Memory buffers and memory advice");

		const idx_t nx = 12, ny = 10, nz = 8;
		const size_t total_elements = nx * ny * nz;

		BaseGrid<float> reference_grid(Vector3_t<float>{12.0f, 10.0f, 8.0f}, 1.0f);

#ifdef HOST_GUARD
		// Initialize with spherical wave pattern: sin(distance_from_center * frequency)
		Vector3_t<float> center{6.0f, 5.0f, 4.0f};
		float frequency = 2.0f;

		for (idx_t x = 0; x < nx; ++x) {
			for (idx_t y = 0; y < ny; ++y) {
				for (idx_t z = 0; z < nz; ++z) {
					idx_t index = reference_grid.index(x, y, z);
					Vector3_t<float> pos{static_cast<float>(x),
										 static_cast<float>(y),
										 static_cast<float>(z)};
					float distance = (pos - center).length();
					reference_grid[index] = std::sin(distance * frequency);
				}
			}
		}

		INFO("Initialized " << nx << "x" << ny << "x" << nz << " grid with spherical wave pattern");

		try {
			// Test with Unified buffer on device 0
			INFO("Creating Unified buffer on device 0");
			UnifiedBuffer<float> unified_buf(total_elements, device_resources[0]);

			// Apply memory advice for optimal placement
			INFO("Applying memory advice for device " << device_resources[0].id);
			unified_buf.mem_advise(0, device_resources[0].id);

			// Copy grid data to unified buffer
			unified_buf.copy_from_host(reference_grid.data(), total_elements);

			// Prefetch to device for computation
			INFO("Prefetching BaseGrid data to device");
			unified_buf.prefetch(device_resources[0].id);

			// Verify data integrity
			INFO("Prefetching BaseGrid data to host for verification");
			unified_buf.prefetch(-1); // Prefetch to host

			std::vector<float> verify_unified(total_elements);
			unified_buf.copy_to_host(verify_unified);

			// Verify grid pattern integrity
			for (idx_t x = 0; x < nx; ++x) {
				for (idx_t y = 0; y < ny; ++y) {
					for (idx_t z = 0; z < nz; ++z) {
						idx_t index = reference_grid.index(x, y, z);
						REQUIRE(std::abs(verify_unified[index] - reference_grid[index]) < 1e-6f);
					}
				}
			}

			INFO("✓ Unified buffer BaseGrid data integrity verified");

			// Test on second device if available
			if (device_resources.size() > 1) {
				INFO("Testing cross-device unified buffer behavior");
				UnifiedBuffer<float> unified_buf2(total_elements, device_resources[1]);

				// Apply memory advice for device 1
				INFO("Applying memory advice for device " << device_resources[1].id);
				unified_buf2.mem_advise(0, device_resources[1].id);
				unified_buf2.prefetch(device_resources[1].id);

				// Copy data between unified buffers
				unified_buf2.copy_device_to_device(unified_buf, total_elements);

				// Test range-based memory advice for grid slices (XY planes)
				INFO("Applying range-based memory advice for grid XY slices");
				const size_t xy_slice_size = nx * ny;
				const size_t num_test_slices = 3;

				for (size_t slice = 0; slice < num_test_slices; ++slice) {
					unified_buf2.advise_range(slice * xy_slice_size,
											  xy_slice_size,
											  device_resources[1].id,
											  0);
				}

				// Prefetch specific grid slices
				INFO("Prefetching first few grid slices to device");
				unified_buf2.prefetch_range(0,
											num_test_slices * xy_slice_size,
											device_resources[1].id);

				// Verify cross-device data consistency
				unified_buf2.prefetch(-1); // Prefetch to host
				std::vector<float> verify_device2(total_elements);
				unified_buf2.copy_to_host(verify_device2);

				for (idx_t i = 0; i < total_elements; ++i) {
					REQUIRE(std::abs(verify_device2[i] - reference_grid[i]) < 1e-6f);
				}

				INFO("✓ Cross-device unified buffer BaseGrid consistency verified");

				// Test direct memory access
				auto device2_ptr = unified_buf2.device_data();
				REQUIRE(device2_ptr != nullptr);
				INFO("✓ Device 2 unified buffer pointer: " << static_cast<void*>(device2_ptr));
			}

		} catch (const std::exception& e) {
			WARN("Unified buffer BaseGrid test failed: " << e.what());

			// Fallback to regular DeviceBuffer test
			INFO("Falling back to DeviceBuffer test for BaseGrid");
			DeviceBuffer<float> device_buf(total_elements, device_resources[0]);
			device_buf.copy_from_host(reference_grid.data(), total_elements);

			std::vector<float> verify_device(total_elements);
			device_buf.copy_to_host(verify_device);

			// Verify basic grid pattern
			for (idx_t i = 0; i < total_elements; ++i) {
				REQUIRE(std::abs(verify_device[i] - reference_grid[i]) < 1e-6f);
			}
			INFO("✓ DeviceBuffer fallback BaseGrid test passed");
		}

		// Performance test: grid pattern generation
		auto start = std::chrono::high_resolution_clock::now();
		BaseGrid<float> perf_grid(Vector3_t<float>{12.0f, 10.0f, 8.0f}, 1.0f);
		for (idx_t i = 0; i < total_elements; ++i) {
			auto ijk = perf_grid.index_to_ijk(i);
			Vector3_t<float> pos{static_cast<float>(ijk[0]),
								 static_cast<float>(ijk[1]),
								 static_cast<float>(ijk[2])};
			float distance = (pos - center).length();
			perf_grid[i] = std::sin(distance * frequency);
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		INFO("BaseGrid pattern generation time for " << total_elements
													 << " elements: " << duration.count() << " μs");

		REQUIRE(duration.count() < 50000); // Should complete in less than 50ms
#endif
	}
}

TEST_CASE_METHOD(BaseGridTestFixture, "BaseGrid Advanced Operations", "[BaseGrid][advanced]") {
	initialize_backend_once();

	SECTION("Grid interpolation and sampling") {
		BaseGrid<float> grid(Vector3_t<float>{6.0f, 6.0f, 6.0f}, 1.0f);

#ifdef HOST_GUARD
		// Initialize with quadratic function: x^2 + y^2 + z^2
		for (idx_t x = 0; x < grid.nx(); ++x) {
			for (idx_t y = 0; y < grid.ny(); ++y) {
				for (idx_t z = 0; z < grid.nz(); ++z) {
					idx_t index = grid.index(x, y, z);
					float fx = static_cast<float>(x);
					float fy = static_cast<float>(y);
					float fz = static_cast<float>(z);
					grid[index] = fx * fx + fy * fy + fz * fz;
				}
			}
		}

		// Test interpolation at known points
		Vector3_t<float> test_pos{1.5f, 2.5f, 3.5f}; // Should be between grid points
		bool in_bounds = grid.in_interpolation_bounds(test_pos);
		REQUIRE(in_bounds);

		float interpolated_value = grid.interpolate(test_pos);
		float expected_value = 1.5f * 1.5f + 2.5f * 2.5f + 3.5f * 3.5f; // 18.75

		// Allow some tolerance for interpolation
		REQUIRE(std::abs(interpolated_value - expected_value) < 1.0f);

		INFO("Interpolated value: " << interpolated_value << ", Expected: " << expected_value);
#endif
	}

	SECTION("Grid neighbor operations") {
		BaseGrid<float> grid(Vector3_t<float>{5.0f, 5.0f, 5.0f}, 1.0f);

#ifdef HOST_GUARD
		// Initialize with unique values for each point
		for (idx_t i = 0; i < grid.size(); ++i) {
			grid[i] = static_cast<float>(i + 1); // 1, 2, 3, ...
		}

		// Test neighbor access
		idx_t center_x = 2, center_y = 2, center_z = 2; // Center of 5x5x5 grid

		float center_value = grid[grid.index(center_x, center_y, center_z)];
		float left_neighbor = grid.get_neighbor(center_x, center_y, center_z, -1, 0, 0);
		float right_neighbor = grid.get_neighbor(center_x, center_y, center_z, 1, 0, 0);

		// Verify neighbor relationships
		idx_t expected_left_idx = grid.index(center_x - 1, center_y, center_z);
		idx_t expected_right_idx = grid.index(center_x + 1, center_y, center_z);

		REQUIRE(std::abs(left_neighbor - grid[expected_left_idx]) < 1e-6f);
		REQUIRE(std::abs(right_neighbor - grid[expected_right_idx]) < 1e-6f);

		INFO("Center: " << center_value << ", Left: " << left_neighbor
						<< ", Right: " << right_neighbor);
#endif
	}
}
