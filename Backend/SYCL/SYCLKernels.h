#pragma once
#include "Header.h"
#ifdef USE_SYCL
#include "../Buffer.h"
#include "../Events.h"
#include "../KernelConfig.h"
#include "../Resource.h"
#include "Header.h"
#include "SYCLManager.h"
#include <sycl/sycl.hpp>

namespace ARBD {

/**
 * @brief Streamlined SYCL kernel launcher - new structure matching CUDA
 * Eliminates tuple overhead for better multi-GPU performance
 */
template<typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 const KernelConfig& config,
						 Functor kernel_func,
						 Args... args) {
	if (config.dim == 1) {
		return launch_sycl_kernel_1d(resource, config, kernel_func, args...);
	} else if (config.dim == 2) {
		return launch_sycl_kernel_2d(resource, config, kernel_func, args...);
	} else if (config.dim == 3) {
		return launch_sycl_kernel_3d(resource, config, kernel_func, args...);
	} else {
		throw_value_error("Invalid dimension for SYCL kernel launch");
	}
}

template<typename Functor, typename... Args>
Event launch_sycl_kernel_1d(const Resource& resource,
							const KernelConfig& config,
							Functor kernel_func,
							Args... args) {

	auto* queue_wrapper_ptr = static_cast<ARBD::SYCL::Queue*>(resource.get_stream_type());
	sycl::queue& queue = queue_wrapper_ptr->get();
	idx_t local_range_sycl = std::min(config.block_size.x, config.problem_size.x);
	idx_t global_range_sycl =
		((config.problem_size.x + local_range_sycl - 1) / local_range_sycl) * local_range_sycl;

	std::cout << "DEBUG: 1D SYCL kernel - problem_size.x=" << config.problem_size.x
			  << ", local_range=" << local_range_sycl << ", global_range=" << global_range_sycl
			  << std::endl;
	// Ensure global range is divisible by local range to avoid non-uniform work-groups
	sycl::range<1> local_range(static_cast<size_t>(local_range_sycl));
	sycl::range<1> global_range(static_cast<size_t>(global_range_sycl));
	sycl::nd_range<1> execution_range(global_range, local_range);

	// Submit kernel with dependency handling
	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// Handle dependencies
		if (!config.dependencies.empty()) {
			h.depends_on(config.dependencies.get_sycl_events());
		}

		// Extract device-copyable parts before capture
		idx_t problem_size_x = config.problem_size.x;

		// Launch kernel with streamlined interface - capture raw pointers directly
		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			idx_t gx = static_cast<idx_t>(item.get_global_id(0));

			if (gx < problem_size_x) {
				size_t i = gx;
				kernel_func(i, args...);
			}
		});
	});

	std::cout << "DEBUG: SYCL kernel submitted successfully" << std::endl;

	// Sync if requested
	if (config.sync) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}

template<typename Functor, typename... Args>
Event launch_sycl_kernel_2d(const Resource& resource,
							const KernelConfig& config,
							Functor kernel_func,
							Args... args) {
	// Ensure standardized problem/grid/block are available
	KernelConfig local_config = config;
	if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0) {
		kerneldim3 new_problem{};
		new_problem.x = std::max<idx_t>(1, local_config.grid_size.x * local_config.block_size.x);
		new_problem.y = std::max<idx_t>(1, local_config.grid_size.y * local_config.block_size.y);
		local_config.problem_size = new_problem;
	}

	// Get queue from config or resource
	auto* queue_wrapper_ptr = static_cast<ARBD::SYCL::Queue*>(resource.get_stream_type());
	sycl::queue& queue = queue_wrapper_ptr->get();

	// Ensure global range is divisible by local range to avoid non-uniform work-groups
	// SYCL 2D: (y, x) mapping to maintain x as fastest varying dimension
	sycl::range<2> local_range(local_config.block_size.y, local_config.block_size.x);
	sycl::range<2> global_range(((local_config.problem_size.y + local_config.block_size.y - 1) /
								 local_config.block_size.y) *
									local_config.block_size.y,
								((local_config.problem_size.x + local_config.block_size.x - 1) /
								 local_config.block_size.x) *
									local_config.block_size.x);
	sycl::nd_range<2> execution_range(global_range, local_range);
	idx_t problem_size_x = local_config.problem_size.x;
	idx_t problem_size_y = local_config.problem_size.y;
	// Submit kernel with dependency handling
	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// Handle dependencies
		if (!config.dependencies.empty()) {
			h.depends_on(config.dependencies.get_sycl_events());
		}

		h.parallel_for(execution_range, [=](sycl::nd_item<2> item) {
			idx_t gx = static_cast<idx_t>(item.get_global_id(1)); // x is fastest (dim 1)
			idx_t gy = static_cast<idx_t>(item.get_global_id(0)); // y is slower (dim 0)

			// Bounds checking for 2D - only execute if within problem size
			if (gx < problem_size_x && gy < problem_size_y) {
				// Calculate linear index from 2D coordinates using problem size
				idx_t i = gy * problem_size_x + gx;
				kernel_func(i, args...); // Only pass linear index, not coordinates
			}
		});
	});

	// Sync if requested
	if (config.sync) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}

template<typename Functor, typename... Args>
Event launch_sycl_kernel_3d(const Resource& resource,
							const KernelConfig& config,
							Functor kernel_func,
							Args... args) {
	// Ensure standardized problem/grid/block are available
	KernelConfig local_config = config;
	if (local_config.problem_size.x == 0 || local_config.problem_size.y == 0 ||
		local_config.problem_size.z == 0) {
		kerneldim3 new_problem{};
		new_problem.x = std::max<idx_t>(1, local_config.grid_size.x * local_config.block_size.x);
		new_problem.y = std::max<idx_t>(1, local_config.grid_size.y * local_config.block_size.y);
		new_problem.z = std::max<idx_t>(1, local_config.grid_size.z * local_config.block_size.z);
		local_config.problem_size = new_problem;
	}

	// Get queue from config or resource
	auto* queue_wrapper_ptr = static_cast<ARBD::SYCL::Queue*>(resource.get_stream_type());
	sycl::queue& queue = queue_wrapper_ptr->get();

	// Ensure global range is divisible by local range to avoid non-uniform work-groups
	// SYCL 3D: (z, y, x) mapping where x is fastest varying (dim 2), z is slowest (dim 0)
	sycl::range<3> local_range(local_config.block_size.z,
							   local_config.block_size.y,
							   local_config.block_size.x);
	sycl::range<3> global_range(((local_config.problem_size.z + local_config.block_size.z - 1) /
								 local_config.block_size.z) *
									local_config.block_size.z,
								((local_config.problem_size.y + local_config.block_size.y - 1) /
								 local_config.block_size.y) *
									local_config.block_size.y,
								((local_config.problem_size.x + local_config.block_size.x - 1) /
								 local_config.block_size.x) *
									local_config.block_size.x);
	sycl::nd_range<3> execution_range(global_range, local_range);

	// Submit kernel with dependency handling
	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// Handle dependencies
		if (!config.dependencies.empty()) {
			h.depends_on(config.dependencies.get_sycl_events());
		}

		// Extract device-copyable parts before capture
		idx_t problem_size_x = local_config.problem_size.x;
		idx_t problem_size_y = local_config.problem_size.y;
		idx_t problem_size_z = local_config.problem_size.z;

		// Launch kernel with streamlined interface - capture raw pointers directly
		h.parallel_for(execution_range, [=](sycl::nd_item<3> item) {
			// Extract coordinates: SYCL maps z->0, y->1, x->2 to keep x fastest varying
			idx_t gx = static_cast<idx_t>(item.get_global_id(2)); // x is fastest (dim 2)
			idx_t gy = static_cast<idx_t>(item.get_global_id(1)); // y is middle (dim 1)
			idx_t gz = static_cast<idx_t>(item.get_global_id(0)); // z is slowest (dim 0)

			// Bounds checking for 3D - only execute if within problem size
			if (gx < problem_size_x && gy < problem_size_y && gz < problem_size_z) {
				// Calculate linear index from 3D coordinates using problem size
				idx_t i = (gz * problem_size_y + gy) * problem_size_x + gx;
				kernel_func(i, args...); // Only pass linear index, not coordinates
			}
		});
	});

	// Sync if requested
	if (config.sync) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}
} // namespace ARBD
#endif
