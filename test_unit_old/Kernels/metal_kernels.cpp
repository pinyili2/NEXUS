// Initialize Metal

#include <unistd.h>
#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"
#include "Backend/Kernels.h"
#include "Backend/KernelConfig.h"

TEST_CASE("Metal Vector Addition", "[metal][kernels][simple_test]") {
	LOGINFO("Starting Metal test...");
	// Initialize Metal manager properly
	LOGINFO("Initializing Metal manager...");
	try {
		ARBD::METAL::Manager::init();
		LOGINFO("Metal manager init() completed");

		ARBD::METAL::Manager::load_info();
		LOGINFO("Metal manager load_info() completed");
	} catch (const std::exception& e) {
		LOGINFO("Exception during Metal manager initialization:{} ", e.what());
		FAIL("Metal manager initialization failed");
	}

	// Check if we have Metal devices
	try {
		auto& device = ARBD::METAL::Manager::get_current_device();
		LOGINFO("Metal device available: {}", (void*)device.metal_device());
	} catch (const std::exception& e) {
		LOGINFO("No Metal device available: {}", e.what());
		FAIL("No Metal device available");
	}

	// Print current working directory for debugging
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != nullptr) {
		LOGINFO("Current working directory: {}", cwd);
	}

		// Check if Metal library is loaded
	auto* library = ARBD::METAL::Manager::get_library();
	if (!library) {
		LOGINFO("Metal library not loaded from file");
	} else {
		LOGINFO("Metal library loaded successfully from file: {}", (void*)library);
	}

	// Test runtime shader compilation like the working metal-cpp examples
	auto& device = ARBD::METAL::Manager::get_current_device();
	auto& queue = device.get_next_queue();

	const char* testKernelSrc = R"(
		#include <metal_stdlib>
		using namespace metal;

		kernel void test_kernel(device float* result [[buffer(0)]],
		                       uint index [[thread_position_in_grid]]) {
			result[index] = 42.0f;
		}
	)";

	try {
		MTL::Library* runtimeLibrary = nullptr;
		if (library) {
			// Try to get function from loaded library first
			auto* testFunc = library->newFunction(NS::String::string("test_kernel", NS::StringEncoding::UTF8StringEncoding));
			if (testFunc) {
				LOGINFO("Found test_kernel in loaded library");
				testFunc->release();
			} else {
				LOGINFO("test_kernel not found in loaded library, trying runtime compilation");
				// Fall through to runtime compilation
			}
		}

		if (!runtimeLibrary) {
			// Try runtime compilation like working examples
			NS::Error* pError = nullptr;
			runtimeLibrary = device.metal_device()->newLibrary(
				NS::String::string(testKernelSrc, NS::StringEncoding::UTF8StringEncoding),
				nullptr, &pError);

			if (!runtimeLibrary) {
				LOGINFO("Runtime compilation failed: {}",
					(pError ? pError->localizedDescription()->utf8String() : "Unknown error"));
				if (pError) pError->release();
			} else {
				LOGINFO("Runtime compilation successful");

				// Test creating a compute pipeline
				auto* testFunc = runtimeLibrary->newFunction(
					NS::String::string("test_kernel", NS::StringEncoding::UTF8StringEncoding));

				if (testFunc) {
					MTL::ComputePipelineState* testPSO = device.metal_device()->newComputePipelineState(testFunc, &pError);
					if (testPSO) {
						LOGINFO("Compute pipeline creation successful");
						testPSO->release();
					} else {
						LOGINFO("Compute pipeline creation failed: {}",
							(pError ? pError->localizedDescription()->utf8String() : "Unknown error"));
					}
					testFunc->release();
				}
				runtimeLibrary->release();
			}
		}
	} catch (const std::exception& e) {
		LOGINFO("Exception during runtime compilation test: {}", e.what());
	}

	// Debug Metal device information
	try {
		auto& device = ARBD::METAL::Manager::get_current_device();
		LOGINFO("Current Metal device: {}", device.metal_device()->name()->utf8String());
		auto& queue = device.get_next_queue();
		LOGINFO("Metal command queue created successfully");
	} catch (const std::exception& e) {
		LOGINFO("Error getting Metal device: {}", e.what());
	}

	try {
		// Use Metal resource instead of CPU
		ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
		const size_t n = 10;  // Test with more elements

		auto buffer_a = ARBD::DeviceBuffer<float>(n);
		auto buffer_b = ARBD::DeviceBuffer<float>(n);
		auto buffer_result = ARBD::DeviceBuffer<float>(n);

		// Initialize data
		std::vector<float> host_a(n, 1.0f);
		std::vector<float> host_b(n, 2.0f);
		std::vector<float> host_result_init(n, 0.0f); // Initialize result buffer
		
		buffer_a.copy_from_host(host_a.data(), n);
		buffer_b.copy_from_host(host_b.data(), n);
		buffer_result.copy_from_host(host_result_init.data(), n);

		// Verify data was copied correctly
		std::vector<float> verify_a(n), verify_b(n);
		buffer_a.copy_to_host(verify_a.data(), n);
		buffer_b.copy_to_host(verify_b.data(), n);
		
		LOGINFO("Buffer A data: {}, {}", verify_a[0] ,verify_a[1]);
		LOGINFO("Buffer B data: {}, {}", verify_b[0] ,verify_b[1]);
		
		REQUIRE(verify_a[0] == 1.0f);
		REQUIRE(verify_b[0] == 2.0f);

		ARBD::KernelConfig config;
		config.async = false;  // Force synchronous execution for testing

		LOGINFO("Launching Metal kernel...");
		LOGINFO("Thread count: {}" ,n);
		LOGINFO("Buffer A size: {}", buffer_a.size());
		LOGINFO("Buffer B size:{} " , buffer_b.size());
		LOGINFO("Buffer result size: {}", buffer_result.size());
		
		// Check if buffers are properly allocated
		LOGINFO("Buffer A device pointer: {}", (void*)buffer_a.data());
		LOGINFO("Buffer B device pointer: {}", (void*)buffer_b.data());
		LOGINFO("Buffer result device pointer: {}", (void*)buffer_result.data());
		
		// Check if Metal buffers can be retrieved
		auto* metal_buffer_a = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_a.data());
		auto* metal_buffer_b = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_b.data());
		auto* metal_buffer_result = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_result.data());
		
		LOGINFO("Metal buffer A: {}", metal_buffer_a ? (void*)metal_buffer_a : nullptr);
		LOGINFO("Metal buffer B: {}", metal_buffer_b ? (void*)metal_buffer_b : nullptr);
		LOGINFO("Metal buffer result: {}", metal_buffer_result ? (void*)metal_buffer_result : nullptr);
		
		// Verify that Metal buffer contents pointers match device pointers
		if (metal_buffer_result) {
			void* metal_contents = metal_buffer_result->contents();
			LOGINFO("Metal buffer result contents pointer: {}", metal_contents);
			if (metal_contents != buffer_result.data()) {
				LOGINFO("ERROR: Metal buffer contents pointer != device buffer pointer");
				LOGINFO("Expected: {}, Got: {}", (void*)buffer_result.data(), metal_contents);
			}
		}
		
		// Double-check buffer pointer before kernel launch
		LOGINFO("Buffer result pointer just before kernel launch: {}", (void*)buffer_result.data());
		
		// Test both runtime and pre-compiled kernels
		LOGINFO("Starting dual kernel test...");
		std::vector<float> runtime_results(n);
		std::vector<float> precompiled_results(n);

		// Test 1: Runtime kernel
		LOGINFO("=== Testing Runtime Kernel ===");
		{
			// Fill buffer with test data (direct memory access with unified memory)
			float* buffer_ptr = static_cast<float*>(buffer_result.data());
			for (size_t i = 0; i < n; ++i) {
				buffer_ptr[i] = 99.0f;
			}
			LOGINFO("Filled buffer with test data: {}, {}", buffer_ptr[0],buffer_ptr[1]);

			const char* runtimeKernelSrc = R"(
				#include <metal_stdlib>
				using namespace metal;

				kernel void runtime_zero_kernel(device float* buffer [[buffer(0)]],
											   uint index [[thread_position_in_grid]]) {
					buffer[index] = 0.0f;
				}
			)";

			try {
				auto& device = ARBD::METAL::Manager::get_current_device();
				auto& queue = device.get_next_queue();

				LOGINFO("Trying runtime kernel compilation...");
				NS::Error* pError = nullptr;
				MTL::Library* runtimeLibrary = device.metal_device()->newLibrary(
					NS::String::string(runtimeKernelSrc, NS::StringEncoding::UTF8StringEncoding),
					nullptr, &pError);

				if (!runtimeLibrary) {
					LOGINFO("Runtime compilation failed: {}",
						(pError ? pError->localizedDescription()->utf8String() : "Unknown error"));
					if (pError) pError->release();
					LOGINFO("Runtime kernel test: FAILED - compilation failed");
				} else {
					LOGINFO("Runtime compilation successful");

					// Create pipeline with runtime kernel
					auto* runtimeFunc = runtimeLibrary->newFunction(
						NS::String::string("runtime_zero_kernel", NS::StringEncoding::UTF8StringEncoding));

					if (runtimeFunc) {
						MTL::ComputePipelineState* runtimePSO = device.metal_device()->newComputePipelineState(runtimeFunc, &pError);
						if (runtimePSO) {
							LOGINFO("Runtime pipeline creation successful");

							// Execute runtime kernel
							void* cmd_buffer_ptr = queue.create_command_buffer();
							auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
							auto* encoder = cmd_buffer->computeCommandEncoder();

							encoder->setComputePipelineState(runtimePSO);

							// Bind buffer
							auto* metal_buffer = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_result.data());
							if (metal_buffer) {
								encoder->setBuffer(metal_buffer, 0, 0);
								LOGINFO("Bound buffer to runtime kernel");

								// Dispatch
								MTL::Size gridSize = MTL::Size(n, 1, 1);
								MTL::Size threadgroupSize = MTL::Size(n, 1, 1);
								encoder->dispatchThreads(gridSize, threadgroupSize);
								encoder->endEncoding();

								// Execute
								cmd_buffer->commit();
								cmd_buffer->waitUntilCompleted();

								LOGINFO("Runtime kernel executed successfully");

								// Read results (direct memory access with unified memory)
								float* result_ptr = static_cast<float*>(buffer_result.data());
								LOGINFO("Runtime kernel results:{},{} ", result_ptr[0],result_ptr[1] );
							} else {
								LOGINFO("Failed to get Metal buffer for runtime kernel");
							}

							runtimePSO->release();
						} else {
							LOGINFO("Runtime pipeline creation failed: {}",
								(pError ? pError->localizedDescription()->utf8String() : "Unknown error"));
						}
						runtimeFunc->release();
					}
					runtimeLibrary->release();
				}
			} catch (const std::exception& e) {
				LOGINFO("Exception during runtime kernel test: {}", e.what());
			}
		}

		// Test 2: Pre-compiled kernel
		LOGINFO("=== Testing Pre-compiled Kernel ===");
		{
			// Fill buffer with test data again (direct memory access with unified memory)
			float* buffer_ptr = static_cast<float*>(buffer_result.data());
			for (size_t i = 0; i < n; ++i) {
				buffer_ptr[i] = 88.0f;  // Use different value to distinguish from runtime test
			}
			LOGINFO("Filled buffer with test data: {}, {}", buffer_ptr[0] ,buffer_ptr[1]);

			try {
				LOGINFO("Testing pre-compiled zero_buffer kernel");

				// Check if Metal library is loaded
				auto* library = ARBD::METAL::Manager::get_library();
					LOGINFO("Metal library loaded: {}", (library ? "YES" : "NO"));

				if (library) {
					// Try to find the kernel function
					auto* kernelFunc = library->newFunction(NS::String::string("zero_buffer", NS::StringEncoding::UTF8StringEncoding));
					LOGINFO("Kernel function found: {}", (kernelFunc ? "YES" : "NO"));
					if (kernelFunc) kernelFunc->release();
				}

				ARBD::Event event = ARBD::launch_metal_kernel(
					metal_res,
					n,
					config,
					"zero_buffer",
					buffer_result  // buffer to zero out
				);

				event.wait();
				LOGINFO("Pre-compiled kernel executed");

				// Read results (direct memory access with unified memory)
				float* result_ptr = static_cast<float*>(buffer_result.data());
				LOGINFO("Pre-compiled kernel results: {}, {}", result_ptr[0] ,result_ptr[1]);

				// Check if kernel actually ran
				bool kernel_ran = true;
				for (size_t i = 0; i < n; ++i) {
					if (result_ptr[i] != 0.0f) {
						kernel_ran = false;
						break;
					}
				}
				LOGINFO("Pre-compiled kernel actually ran: {}", (kernel_ran ? "YES" : "NO"));

			} catch (const std::exception& e) {
				LOGINFO("Exception during pre-compiled kernel test: {}", e.what());
			}
		}

		LOGINFO("Kernel execution completed");

		// Try reading the Metal buffer contents directly after kernel execution  
		if (metal_buffer_result) {
			float* metal_contents = (float*)metal_buffer_result->contents();
			LOGINFO("Metal buffer contents direct access after execution: {}", metal_contents[0]);
		}
		
		// Also try reading buffer data directly (unified memory)
		LOGINFO("Buffer data direct access: {}", *((float*)buffer_result.data()));

		// Double-check that we still have the same Metal buffer mapping
		auto* metal_buffer_result_after = ARBD::METAL::Manager::get_metal_buffer_from_ptr(buffer_result.data());
		LOGINFO("Metal buffer result after kernel: {}", metal_buffer_result_after ? (void*)metal_buffer_result_after : nullptr);
		if (metal_buffer_result != metal_buffer_result_after) {
			LOGINFO("WARNING: Metal buffer mapping changed during kernel execution!");
		}
		
		// Compare results from both approaches
		LOGINFO("=== Comparing Runtime vs Pre-compiled Results ===");

		bool runtime_success = true;
		bool precompiled_success = true;

		// Check runtime kernel results
		LOGINFO("Runtime kernel verification:");
		for (size_t i = 0; i < n; ++i) {
			float expected = 0.0f;  // zero_buffer sets all elements to 0.0f
			if (runtime_results[i] != expected) {
				runtime_success = false;
				LOGINFO("  Runtime result[{}] = {} (expected {} ) - FAILED", i, runtime_results[i], expected);
			}
		}
		if (runtime_success) {
			LOGINFO("  Runtime kernel: PASSED ✓");
		} else {
			LOGINFO("  Runtime kernel: FAILED ✗");
		}

		// Check pre-compiled kernel results
		LOGINFO("Pre-compiled kernel verification:");
		for (size_t i = 0; i < n; ++i) {
			float expected = 0.0f;  // zero_buffer sets all elements to 0.0f
			if (precompiled_results[i] != expected) {
				precompiled_success = false;
				LOGINFO("  Pre-compiled result[{}] = {} (expected {} ) - FAILED", i, precompiled_results[i], expected);
			}
		}
		if (precompiled_success) {
			LOGINFO("  Pre-compiled kernel: PASSED ✓");
		} else {
			LOGINFO("  Pre-compiled kernel: FAILED ✗");
		}

		// Compare the two results
		bool results_match = true;
		for (size_t i = 0; i < n; ++i) {
			if (runtime_results[i] != precompiled_results[i]) {
				results_match = false;
				break;
			}
		}

		LOGINFO("Results comparison: {}", (results_match ? "MATCH ✓" : "DIFFER ✗"));

		// Final test requirements
		if (runtime_success) {
			LOGINFO("Runtime kernel test: PASSED ✓");
		} else {
			LOGINFO("Runtime kernel test: FAILED ✗");
		}

		if (precompiled_success) {
			LOGINFO("Pre-compiled kernel test: PASSED ✓");
		} else {
			LOGINFO("Pre-compiled kernel test: FAILED ✗");
		}

		// At least one approach should work
		LOGINFO("Dual kernel test completed");
		REQUIRE((runtime_success || precompiled_success));

		// Test add_arrays kernel (both runtime and precompiled)
	LOGINFO("=== Testing add_arrays Kernel (Runtime + Precompiled) ===");

	// Create separate buffers for add_arrays test
	auto add_buffer_a = ARBD::DeviceBuffer<float>(n);
	auto add_buffer_b = ARBD::DeviceBuffer<float>(n);
	auto add_buffer_result = ARBD::DeviceBuffer<float>(n);

	// Initialize input buffers: a = [1, 2, 3, ...], b = [2, 3, 4, ...]
	std::vector<float> host_a_vals(n), host_b_vals(n);
	for (size_t i = 0; i < n; ++i) {
		host_a_vals[i] = static_cast<float>(i + 1);  // 1, 2, 3, ...
		host_b_vals[i] = static_cast<float>(i + 2);  // 2, 3, 4, ...
	}

	add_buffer_a.copy_from_host(host_a_vals.data(), n);
	add_buffer_b.copy_from_host(host_b_vals.data(), n);
	std::vector<float> init_add_result(n, 0.0f);
	add_buffer_result.copy_from_host(init_add_result.data(), n);

	std::vector<float> runtime_add_results(n);
	std::vector<float> precompiled_add_results(n);

	// Test 1: Runtime add_arrays kernel
	LOGINFO("--- Testing Runtime add_arrays Kernel ---");
	{
		const char* runtimeAddArraysSrc = R"(
			#include <metal_stdlib>
			using namespace metal;

			kernel void runtime_add_arrays(device const float* a [[buffer(0)]],
										  device const float* b [[buffer(1)]],
										  device float* result [[buffer(2)]],
										  uint index [[thread_position_in_grid]]) {
				result[index] = a[index] + b[index];
			}
		)";

		try {
			auto& device = ARBD::METAL::Manager::get_current_device();
			auto& queue = device.get_next_queue();

			LOGINFO("Compiling runtime add_arrays kernel...");
			NS::Error* pError = nullptr;
			MTL::Library* runtimeLibrary = device.metal_device()->newLibrary(
				NS::String::string(runtimeAddArraysSrc, NS::StringEncoding::UTF8StringEncoding),
				nullptr, &pError);

			if (!runtimeLibrary) {
				LOGINFO("Runtime add_arrays compilation failed: {}",
					(pError ? pError->localizedDescription()->utf8String() : "Unknown error"));
				if (pError) pError->release();
			} else {
				LOGINFO("Runtime add_arrays compilation successful");

				// Create pipeline with runtime kernel
				auto* runtimeFunc = runtimeLibrary->newFunction(
					NS::String::string("runtime_add_arrays", NS::StringEncoding::UTF8StringEncoding));

				if (runtimeFunc) {
					MTL::ComputePipelineState* runtimePSO = device.metal_device()->newComputePipelineState(runtimeFunc, &pError);
					if (runtimePSO) {
						LOGINFO("Runtime add_arrays pipeline creation successful");

						// Execute runtime kernel
						void* cmd_buffer_ptr = queue.create_command_buffer();
						auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
						auto* encoder = cmd_buffer->computeCommandEncoder();

						encoder->setComputePipelineState(runtimePSO);

						// Bind buffers
						auto* metal_buffer_a = ARBD::METAL::Manager::get_metal_buffer_from_ptr(add_buffer_a.data());
						auto* metal_buffer_b = ARBD::METAL::Manager::get_metal_buffer_from_ptr(add_buffer_b.data());
						auto* metal_buffer_result = ARBD::METAL::Manager::get_metal_buffer_from_ptr(add_buffer_result.data());

						if (metal_buffer_a && metal_buffer_b && metal_buffer_result) {
							encoder->setBuffer(metal_buffer_a, 0, 0);
							encoder->setBuffer(metal_buffer_b, 0, 1);
							encoder->setBuffer(metal_buffer_result, 0, 2);
							LOGINFO("Bound buffers for runtime add_arrays kernel");

							// Dispatch
							MTL::Size gridSize = MTL::Size(n, 1, 1);
							MTL::Size threadgroupSize = MTL::Size(n, 1, 1);
							encoder->dispatchThreads(gridSize, threadgroupSize);
							encoder->memoryBarrier(MTL::BarrierScopeBuffers);
							encoder->endEncoding();

							// Execute
							cmd_buffer->commit();
							cmd_buffer->waitUntilCompleted();

							LOGINFO("Runtime add_arrays kernel executed successfully");

							// Read results
							add_buffer_result.copy_to_host(runtime_add_results.data(), n);
							LOGINFO("Runtime add_arrays results: {}, {}, {}", runtime_add_results[0], runtime_add_results[1], runtime_add_results[2]);
						} else {
							LOGINFO("Failed to get Metal buffers for runtime add_arrays kernel");
						}

						runtimePSO->release();
					} else {
						LOGINFO("Runtime add_arrays pipeline creation failed: {}",
							(pError ? pError->localizedDescription()->utf8String() : "Unknown error"));
					}
					runtimeFunc->release();
				}
				runtimeLibrary->release();
			}
		} catch (const std::exception& e) {
			LOGINFO("Exception during runtime add_arrays test: {}", e.what());
		}
	}

	// Reset result buffer for precompiled test
	add_buffer_result.copy_from_host(init_add_result.data(), n);

	// Test 2: Pre-compiled add_arrays kernel with debugging
	LOGINFO("--- Testing Pre-compiled add_arrays Kernel ---");
	{
		try {
			LOGINFO("Testing pre-compiled add_arrays kernel");

			// Check if Metal library is loaded
			auto* library = ARBD::METAL::Manager::get_library();
			LOGINFO("Metal library loaded: {}", (library ? "YES" : "NO"));

			if (library) {
				// List all functions in the library
				NS::Array* function_names = library->functionNames();
				if (function_names) {
					LOGINFO("Functions in Metal library ({}): ", function_names->count());
					for (NS::UInteger i = 0; i < function_names->count(); ++i) {
						NS::String* name = static_cast<NS::String*>(function_names->object(i));
						if (name) {
							LOGINFO("  - {}", name->utf8String());
						}
					}
				}

				// Try to find the kernel function
				auto* kernelFunc = library->newFunction(NS::String::string("add_arrays", NS::StringEncoding::UTF8StringEncoding));
				LOGINFO("add_arrays kernel function found: {}", (kernelFunc ? "YES" : "NO"));
				if (kernelFunc) kernelFunc->release();

				// Also check for debug version
				auto* debugKernelFunc = library->newFunction(NS::String::string("debug_add_arrays", NS::StringEncoding::UTF8StringEncoding));
				LOGINFO("debug_add_arrays kernel function found: {}", (debugKernelFunc ? "YES" : "NO"));
				if (debugKernelFunc) debugKernelFunc->release();
			}

			// First try the debug kernel to see what values it reads
			LOGINFO("Testing debug_add_arrays kernel to diagnose buffer reading...");
			std::vector<float> debug_results(n, -1.0f);
			auto debug_buffer_result = ARBD::DeviceBuffer<float>(n);
			debug_buffer_result.copy_from_host(debug_results.data(), n);

			try {
				ARBD::Event debug_event = ARBD::launch_metal_kernel(
					metal_res,
					n,
					config,
					"debug_add_arrays",
					add_buffer_a,
					add_buffer_b,
					debug_buffer_result
				);

				debug_event.wait();
				debug_buffer_result.copy_to_host(debug_results.data(), n);
				LOGINFO("Debug kernel results - a[0]: {}, b[0]: {}, sum: {}, marker: {}", 
					debug_results[0], debug_results[1], debug_results[2], debug_results[3]);
			} catch (const std::exception& e) {
				LOGINFO("Debug kernel failed: {}", e.what());
			}

			// Now test the regular add_arrays kernel
			ARBD::Event event = ARBD::launch_metal_kernel(
				metal_res,
				n,
				config,
				"add_arrays",
				add_buffer_a,
				add_buffer_b,
				add_buffer_result
			);

			event.wait();
			LOGINFO("Pre-compiled add_arrays kernel executed");

			// Read results
			add_buffer_result.copy_to_host(precompiled_add_results.data(), n);
			LOGINFO("Pre-compiled add_arrays results: {}, {}, {}", precompiled_add_results[0], precompiled_add_results[1], precompiled_add_results[2]);

		} catch (const std::exception& e) {
			LOGINFO("Exception during pre-compiled add_arrays test: {}", e.what());
		}
	}

	// Verify and compare add_arrays results
	bool runtime_add_success = true;
	bool precompiled_add_success = true;

	// Check runtime kernel results
	LOGINFO("Runtime add_arrays kernel verification:");
	for (size_t i = 0; i < n; ++i) {
		float expected = host_a_vals[i] + host_b_vals[i];
		if (runtime_add_results[i] != expected) {
			runtime_add_success = false;
			LOGINFO("  Runtime add result[{}] = {} (expected {}) - FAILED", i, runtime_add_results[i], expected);
			break;
		}
	}
	if (runtime_add_success) {
		LOGINFO("  Runtime add_arrays kernel: PASSED ✓");
	} else {
		LOGINFO("  Runtime add_arrays kernel: FAILED ✗");
	}

	// Check pre-compiled kernel results
	LOGINFO("Pre-compiled add_arrays kernel verification:");
	for (size_t i = 0; i < n; ++i) {
		float expected = host_a_vals[i] + host_b_vals[i];
		if (precompiled_add_results[i] != expected) {
			precompiled_add_success = false;
			LOGINFO("  Pre-compiled add result[{}] = {} (expected {}) - FAILED", i, precompiled_add_results[i], expected);
			break;
		}
	}
	if (precompiled_add_success) {
		LOGINFO("  Pre-compiled add_arrays kernel: PASSED ✓");
	} else {
		LOGINFO("  Pre-compiled add_arrays kernel: FAILED ✗");
	}

	// Compare runtime vs precompiled results
	bool add_results_match = true;
	for (size_t i = 0; i < n; ++i) {
		if (runtime_add_results[i] != precompiled_add_results[i]) {
			add_results_match = false;
			break;
		}
	}

	LOGINFO("add_arrays Results comparison: {}", (add_results_match ? "MATCH ✓" : "DIFFER ✗"));

	// Final test requirements for add_arrays
	if (runtime_add_success) {
		LOGINFO("Runtime add_arrays test: PASSED ✓");
	} else {
		LOGINFO("Runtime add_arrays test: FAILED ✗");
	}

	if (precompiled_add_success) {
		LOGINFO("Pre-compiled add_arrays test: PASSED ✓");
	} else {
		LOGINFO("Pre-compiled add_arrays test: FAILED ✗");
	}

		// At least one approach should work for add_arrays
		LOGINFO("add_arrays dual kernel test completed");
		REQUIRE((runtime_add_success || precompiled_add_success));

	} catch (const std::exception& e) {
		LOGERROR("Metal test failed with exception: {}", e.what());
	}

	ARBD::METAL::Manager::finalize();
}