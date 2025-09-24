#pragma once
#ifdef USE_METAL
#include "../Buffer.h"
#include "../Events.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "Header.h"
#include "METALManager.h"
#include "Metal/Metal.hpp"

namespace ARBD {

struct MetalGridConfig {
	MTL::Size grid_size;
	MTL::Size threadgroup_size;
};

inline MetalGridConfig calculate_metal_grid_config(idx_t thread_count,
												   const KernelConfig& config,
												   MTL::ComputePipelineState* pipeline) {
	MetalGridConfig result;

	// For Metal, grid_size from config represents number of threadgroups (CUDA-style)
	// but Metal's dispatch needs total thread count, so we need to convert
	if (config.grid_size.x > 0) {
		// Convert from number of threadgroups to total threads
		// config.grid_size.x is number of threadgroups, config.block_size.x is threads per group
		NS::UInteger total_threads_x = config.grid_size.x * config.block_size.x;
		NS::UInteger total_threads_y = (config.grid_size.y > 0) ? config.grid_size.y * config.block_size.y : 1;
		NS::UInteger total_threads_z = (config.grid_size.z > 0) ? config.grid_size.z * config.block_size.z : 1;
		
		result.grid_size = MTL::Size::Make(total_threads_x, total_threads_y, total_threads_z);
	} else {
		// Fall back to 1D grid with total thread count
		result.grid_size = MTL::Size::Make(thread_count, 1, 1);
	}

	// Use config.block_size if specified, otherwise calculate optimal size
	if (config.block_size.x > 0 || config.block_size.y > 0 || config.block_size.z > 0) {
		// Use the explicit block dimensions from config
		result.threadgroup_size = MTL::Size::Make(
			std::max((NS::UInteger)1, (NS::UInteger)config.block_size.x),
			std::max((NS::UInteger)1, (NS::UInteger)config.block_size.y),
			std::max((NS::UInteger)1, (NS::UInteger)config.block_size.z)
		);
	} else {
		// Calculate optimal 1D threadgroup size
		NS::UInteger max_threads = pipeline->maxTotalThreadsPerThreadgroup();
		NS::UInteger execution_width = pipeline->threadExecutionWidth();
		
		NS::UInteger desired_threads = 256; // Default reasonable size
		NS::UInteger final_threads = std::max(execution_width, std::min(desired_threads, max_threads));
		
		// The threadgroup size cannot be larger than the total number of threads in the grid
		final_threads = std::min((NS::UInteger)thread_count, final_threads);
		
		result.threadgroup_size = MTL::Size::Make(final_threads, 1, 1);
	}

	return result;
}
/**
 * @example
 * Example: Launching a Metal kernel for vector operations
 *
 * @code
 * #include "Backend/Kernels.h"
 * #include "Math/Vector3.h"
 * using namespace ARBD;
 *
 * // Prepare Metal resource and buffers
 * Resource metal_res(ResourceType::METAL, 0);
 * constexpr idx_t n = 16;
 * std::vector<Vector3_t<float>> host_a(n), host_b(n), host_out(n);
 * for (idx_t i = 0; i < n; ++i) {
 *     host_a[i] = Vector3_t<float>(float(i), float(i+1), float(i+2));
 *     host_b[i] = Vector3_t<float>(float(2*i), float(2*i+1), float(2*i+2));
 * }
 * DeviceBuffer<Vector3_t<float>> buf_a(n), buf_b(n), buf_out(n);
 * buf_a.copy_from_host(host_a.data(), n);
 * buf_b.copy_from_host(host_b.data(), n);
 *
 * KernelConfig config;
 * config.async = false;
 * config.grid_size = {n, 1, 1};
 *
 * // Launch the Metal kernel by name
 * Event event = launch_metal_kernel(
 *     metal_res,
 *     n,
 *     std::make_tuple(buf_a, buf_b),
 *     std::forward_as_tuple(buf_out),
 *     config,
 *     "vector_operations_kernel"
 * );
 * event.wait();
 * buf_out.copy_to_host(host_out.data(), n);
 * @endcode
 *
 * Example: Launching a Metal kernel for matrix elementwise multiplication
 *
 * @code
 * #include "Backend/Kernels.h"
 * #include "Math/Matrix3.h"
 * using namespace ARBD;
 *
 * Resource metal_res(ResourceType::METAL, 0);
 * constexpr idx_t n = 4;
 * std::vector<Matrix3_t<float>> host_a(n), host_b(n), host_out(n);
 * for (idx_t i = 0; i < n; ++i) {
 *     Matrix3_t<float> m1, m2;
 *     m1.ex().x = float(i + 1); m1.ex().y = float(i + 2); m1.ex().z = float(i + 3);
 *     m1.ey().x = float(i + 4); m1.ey().y = float(i + 5); m1.ey().z = float(i + 6);
 *     m1.ez().x = float(i + 7); m1.ez().y = float(i + 8); m1.ez().z = float(i + 9);
 *     m2.ex().x = float(2 * (i + 1)); m2.ex().y = float(2 * (i + 2)); m2.ex().z = float(2 * (i +
 * 3)); m2.ey().x = float(2 * (i + 4)); m2.ey().y = float(2 * (i + 5)); m2.ey().z = float(2 * (i +
 * 6)); m2.ez().x = float(2 * (i + 7)); m2.ez().y = float(2 * (i + 8)); m2.ez().z = float(2 * (i +
 * 9)); host_a[i] = m1; host_b[i] = m2;
 * }
 * DeviceBuffer<Matrix3_t<float>> buf_a(n), buf_b(n), buf_out(n);
 * buf_a.copy_from_host(host_a.data(), n);
 * buf_b.copy_from_host(host_b.data(), n);
 *
 * KernelConfig config;
 * config.async = false;
 * config.grid_size = {n, 1, 1};
 *
 * Event event = launch_metal_kernel(
 *     metal_res,
 *     n,
 *     std::make_tuple(buf_a, buf_b),
 *     std::forward_as_tuple(buf_out),
 *     config,
 *     "matrix3_mult_kernel"
 * );
 * event.wait();
 * buf_out.copy_to_host(host_out.data(), n);
 * @endcode
 */

template<typename... Args>
Event launch_metal_kernel_impl(const Resource& resource,
							 idx_t thread_count,
							 const KernelConfig& config,
							 const std::string& kernel_name,
							 Args&&... args) {
	// Get Metal components
	auto* pipeline = METAL::Manager::get_compute_pipeline_state(kernel_name);
	if (!pipeline) {
		throw_value_error("Failed to get compute pipeline state for kernel: {}", kernel_name);
	}
	LOGINFO("Got compute pipeline state for kernel: {}", kernel_name);

	auto& device = METAL::Manager::get_current_device();
	LOGINFO("Got Metal device: {}", (void*)device.metal_device());

	auto& queue = device.get_next_queue();
	LOGINFO("Got Metal command queue: {}", (void*)queue.get());

	// Create command buffer and encoder
	void* cmd_buffer_ptr = queue.create_command_buffer();
	auto* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
	LOGINFO("Created Metal command buffer: {}", (void*)cmd_buffer);

	auto* encoder = cmd_buffer->computeCommandEncoder();
	LOGINFO("Created Metal compute command encoder: {}", (void*)encoder);

	if (!encoder) {
		throw_value_error("Failed to create Metal compute command encoder!");
	}

	encoder->setComputePipelineState(pipeline);
	LOGINFO("Set compute pipeline state on encoder");

	// Configure grid and threadgroup sizes first
	auto grid_config = calculate_metal_grid_config(thread_count, config, pipeline);

	// Bind arguments directly to encoder
	uint32_t buffer_index = 0;

	// Automatically bind grid dimensions as first parameters (following CUDA tradition)
	uint32_t grid_width = static_cast<uint32_t>(grid_config.grid_size.width);
	uint32_t grid_height = static_cast<uint32_t>(grid_config.grid_size.height);
	uint32_t grid_depth = static_cast<uint32_t>(grid_config.grid_size.depth);
	
	encoder->setBytes(&grid_width, sizeof(uint32_t), buffer_index++);
	encoder->setBytes(&grid_height, sizeof(uint32_t), buffer_index++);
	encoder->setBytes(&grid_depth, sizeof(uint32_t), buffer_index++);
	LOGINFO("Auto-bound grid dimensions: width={}, height={}, depth={}", grid_width, grid_height, grid_depth);

	// Bind each user argument individually
	auto bind_arg = [&](auto&& arg) {
		using ArgType = std::decay_t<decltype(arg)>;
		if constexpr (is_device_buffer_v<ArgType>) {
			// Get the Metal buffer from the device buffer
			auto* metal_buffer = METAL::Manager::get_metal_buffer_from_ptr(arg.data());
			if (metal_buffer) {
				// Calculate the offset from the start of the MTL::Buffer
				size_t offset = reinterpret_cast<uintptr_t>(arg.data()) -
								reinterpret_cast<uintptr_t>(metal_buffer->contents());

				encoder->setBuffer(metal_buffer, offset, buffer_index++);
				LOGINFO("Bound device buffer to encoder at index {} with offset {}",
						buffer_index - 1,
						offset);
			} else {
				throw_value_error("Failed to get Metal buffer for device buffer");
			}
		} else if constexpr (std::is_arithmetic_v<ArgType> || std::is_trivial_v<ArgType>) {
			encoder->setBytes(&arg, sizeof(ArgType), buffer_index++);
			LOGINFO("Bound scalar argument to encoder at index {}", buffer_index - 1);
		}
	};

	// Apply binding to all user arguments
	(bind_arg(std::forward<Args>(args)), ...);

	// Debug pipeline characteristics
	LOGINFO("Pipeline {} maxTotalThreadsPerThreadgroup: {}",
			kernel_name,
			pipeline->maxTotalThreadsPerThreadgroup());
	LOGINFO(
		"Pipeline {} threadExecutionWidth: {}", kernel_name, pipeline->threadExecutionWidth());

	LOGINFO("Dispatching Metal kernel: {} with grid size ({}, {}, {}) and threadgroup size ({}, "
			"{}, {})",
			kernel_name,
			grid_config.grid_size.width,
			grid_config.grid_size.height,
			grid_config.grid_size.depth,
			grid_config.threadgroup_size.width,
			grid_config.threadgroup_size.height,
			grid_config.threadgroup_size.depth);
	encoder->dispatchThreads(grid_config.grid_size, grid_config.threadgroup_size);
	encoder->memoryBarrier(MTL::BarrierScopeBuffers);
	encoder->endEncoding();
	LOGINFO("Metal kernel dispatch and encoding completed for: {}", kernel_name);

	// Check for any dispatch errors
	if (cmd_buffer->error()) {
		LOGERROR("Command buffer error after encoding: {}",
				 cmd_buffer->error()->localizedDescription()->utf8String());
	}
	LOGINFO("Config async setting: {}", config.async);

	// Create and return event
	ARBD::METAL::Event metal_event(cmd_buffer_ptr);
	if (!config.async) {
		LOGINFO("Committing Metal command buffer for kernel: {}", kernel_name);
		metal_event.commit();
		LOGINFO("Waiting for Metal command buffer completion for kernel: {}", kernel_name);
		metal_event.wait();
		LOGINFO("Metal command buffer completed for kernel: {}", kernel_name);

		// Check for command buffer errors
		MTL::CommandBuffer* pCmdBuffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
		auto status = pCmdBuffer->status();
		LOGINFO("Command buffer status: {}", (int)status);
		if (status == MTL::CommandBufferStatusError) {
			auto* error = pCmdBuffer->error();
			if (error) {
				LOGERROR("Metal command buffer error: {}",
						 error->localizedDescription()->utf8String());
			}
		} else if (status != MTL::CommandBufferStatusCompleted) {
			LOGWARN("Metal command buffer did not complete successfully. Status: {}", (int)status);
		}
	} else {
		metal_event.commit();
	}

	return Event(std::move(metal_event), resource);
}

/**
 * @brief Buffer-based Metal kernel launcher - new streamlined interface
 * Takes individual buffer arguments directly instead of tuples
 */
template<typename... Args>
Event launch_metal_kernel(const Resource& resource,
						  idx_t thread_count,
						  const KernelConfig& config,
						  const std::string& kernel_name,
						  Args&&... args) {
	// Wait for dependencies
	config.dependencies.wait_all();

	auto event = launch_metal_kernel_impl(
		resource, thread_count, config, kernel_name, std::forward<Args>(args)...);

	if (!config.async) {
		event.wait(); // Ensure the main kernel command buffer is finished

		auto& device = METAL::Manager::get_current_device();
		auto& queue = device.get_next_queue();
		void* sync_cmd_buffer_ptr = queue.create_command_buffer();
		auto* sync_cmd_buffer = static_cast<MTL::CommandBuffer*>(sync_cmd_buffer_ptr);
		auto* sync_encoder = sync_cmd_buffer->blitCommandEncoder();
		if (sync_encoder) {
			// With MTLStorageModeShared, explicit synchronization is not needed.
			// The command buffer wait provides sufficient synchronization.
			sync_encoder->endEncoding();
			sync_cmd_buffer->commit();
			sync_cmd_buffer->waitUntilCompleted();
			sync_cmd_buffer->release();
			LOGINFO("Added memory synchronization barrier");
		}
	}

	return event;
}

template<typename InputBuffer, typename OutputBuffer, typename... Args>
std::enable_if_t<is_device_buffer_v<OutputBuffer> && !is_device_buffer_v<InputBuffer> &&
					 !is_string_v<InputBuffer>,
				 Event>
launch_metal_kernel(const Resource& resource,
					idx_t thread_count,
					const InputBuffer& input_buffer,
					const OutputBuffer& output_buffer,
					const KernelConfig& config,
					const std::string& kernel_name,
					Args&&... args) {
	auto input = std::make_tuple(std::ref(input_buffer));
	auto output = std::make_tuple(std::ref(output_buffer), std::ref(thread_count));
	return launch_metal_kernel(resource,
							   thread_count,
							   input,
							   output,
							   config,
							   kernel_name,
							   std::forward<Args>(args)...);
}

} // namespace ARBD
#endif
