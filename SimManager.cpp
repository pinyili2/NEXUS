#include "SimManager.h"

namespace ARBD {

SimManager::SimManager(SimSystem& sys, const ResourceCollection& resources)
	: sys_(sys), resources_(resources) { // Remove patch_manager initialization for now
	LOGINFO("SimManager: Constructing simulation manager");
}

void SimManager::init() {
	wkf_timer_start(&timer0_);
	LOGINFO("SimManager: Initializing simulation");

	// Initialize domain decomposition
	sys_.decompose_system();

	// Initialize random number generators
	initialize_rngs();

	// Initialize output writers
	initialize_output_writers();

	// Initialize IMD if requested (placeholder for now)
	// if (sys_.get_config().imd_enabled) {
	//     initialize_imd(sys_.get_config().imd_port);
	// }

	// Load restart or initialize particles
	// Configuration handles I/O, SimManager handles the data transfer
	load_initial_conditions();

	LOGINFO("SimManager: Initialization completed");
}

void SimManager::run() {
	LOGINFO("SimManager: Starting simulation loop");

	// Get configuration parameters
	const size_t numSteps = sys_.get_num_steps();
	const size_t outputPeriod = static_cast<size_t>(sys_.get_output_period());
	const size_t outputEnergyPeriod = static_cast<size_t>(sys_.get_energy_output_period());

	for (size_t step = 1; step <= numSteps; ++step) {
		// ===== FORCE CALCULATION PHASE =====
		execute_force_calculation(step);

		// ===== INTEGRATION PHASE =====
		execute_integration(step);

		// ===== MULTI-RESOURCE SYNCHRONIZATION =====
		if (resources_.get_count() > 1) { // Assume ResourceCollection has get_count() method
			synchronize_multi_resource();
		}

		// ===== OUTPUT PHASE =====
		handle_output(step);

		// IMD handling
		if (imd_on_ && clientsock_) {
			handle_imd_commands();
		}

		// Progress reporting
		if (step % 1000 == 0) {
			report_progress(step, numSteps);
		}
	}

	// ===== FINALIZATION =====
	wkf_timer_stop(&timer0_);

	const float elapsed = wkf_timer_time(&timer0_);
	report_performance(elapsed, numSteps);

	// Final restart - SimManager handles I/O
	write_final_restart();

	// Cleanup IMD
	if (imd_on_ && clientsock_) {
		// imd_disconnect(clientsock_);
	}

	LOGINFO("SimManager: Simulation completed");
}

//================================================================================
// Implementation of private methods
//================================================================================

void SimManager::initialize_output_writers() {
	const auto& config = sys_.get_config();

	switch (config.output_format) {
	case OutputFormat::DCD:
		dcd_writer_ = std::make_unique<DcdWriter>(config.output_name + ".dcd");
		LOGINFO("SimManager: Initialized DCD writer for '{}'", config.output_name);
		break;
	case OutputFormat::PDB:
		// traj_writer_ = std::make_unique<PdbWriter>(config.output_name + ".pdb");
		LOGINFO("SimManager: PDB writer not yet implemented");
		break;
	case OutputFormat::HDF5:
		// traj_writer_ = std::make_unique<Hdf5Writer>(config.output_name + ".h5");
		LOGINFO("SimManager: HDF5 writer not yet implemented");
		break;
	}
}

void SimManager::initialize_imd(int port) {
	// Placeholder for IMD initialization
	LOGINFO("SimManager: IMD initialization (port {}) not yet implemented", port);
}

void SimManager::initialize_rngs() {
	// Initialize random number generators for each resource
	for (const auto& resource : resources_) {
		// rngs_[&resource] = std::make_unique<RandomState>(/* seed based on resource ID */);
		LOGINFO("SimManager: Initialized RNG for resource");
	}
}

void SimManager::execute_force_calculation(size_t step) {
	// Implementation would depend on your kernel pipeline system
	// This is a placeholder showing the structure

	for (auto& [resource, rng] : rngs_) {
		// Create pipeline for this resource
		// KernelPipeline pipeline(resource);

		// Get buffers for this resource's patch
		// auto& buffers = patch_manager_.get_patch(resource)->get_buffers();

		// 1. Update neighbor list if needed
		// 2. Clear forces
		// 3. Compute pairwise forces
		// 4. Compute bonded forces if present
		if (sys_.has_bonds()) {
			// Compute bond forces
		}

		// 5. Apply external forces (electric field, grids)
		if (sys_.has_external_forces()) {
			// Compute external forces
		}

		// Synchronize this resource's pipeline
		// pipeline.synchronize();
	}
}

void SimManager::execute_integration(size_t step) {
	const Algorithm algorithm = sys_.get_algorithm();
	const float timestep = sys_.get_timestep();
	const float temperature = sys_.get_temperature();

	for (auto& [resource, rng] : rngs_) {
		// Get buffers for this resource's patch
		// auto& buffers = patch_manager_.get_patch(resource)->get_buffers();

		// Select integrator based on algorithm
		switch (algorithm) {
		case Algorithm::Langevin:
			// BAOAB integrator
			// launch_baoab_kernel(buffers, timestep, temperature, rng, step == 1);
			break;

		case Algorithm::Brownian:
			// Brownian dynamics integrator
			// launch_brownian_kernel(buffers, timestep, temperature, rng);
			break;

		case Algorithm::DPD:
			// DPD integrator
			// launch_dpd_kernel(buffers, timestep, temperature, rng);
			break;
		}
	}
}

void SimManager::synchronize_multi_resource() {
	// Exchange halos between patches
	// patch_manager_.exchange_halos();
}

void SimManager::handle_output(size_t step) {
	const size_t outputPeriod = static_cast<size_t>(sys_.get_output_period());
	const size_t outputEnergyPeriod = static_cast<size_t>(sys_.get_energy_output_period());

	// Energy calculation
	if (step % outputEnergyPeriod == 0) {
		wkf_timer_start(&timerE_);
		output_energy(step);
		wkf_timer_stop(&timerE_);
	}

	// Trajectory output
	if (step % outputPeriod == 0) {
		wkf_timer_start(&timerS_);

		if (dcd_writer_) {
			write_dcd_frame(step);
		}

		// Write restart files periodically
		// if (step % restart_period == 0) {
		//     sys_.write_restart();
		// }

		wkf_timer_stop(&timerS_);
	}
}

void SimManager::write_dcd_frame(size_t step) {
	// Implementation depends on your data structures
	// This is a placeholder
	LOGINFO("SimManager: Writing DCD frame for step {}", step);
}

void SimManager::output_energy(size_t step) {
	// Implementation depends on your energy calculation system
	// This is a placeholder
	LOGINFO("SimManager: Computing energy for step {}", step);
}

void SimManager::report_progress(size_t current_step, size_t total_steps) {
	const float progress =
		static_cast<float>(current_step) / static_cast<float>(total_steps) * 100.0f;
	const float elapsed = wkf_timer_time(&timer0_);
	LOGINFO("SimManager: Step {}/{} ({:.1f}%), Elapsed: {:.2f}s",
			current_step,
			total_steps,
			progress,
			elapsed);
}

void SimManager::report_performance(float elapsed_time, size_t total_steps) {
	const float steps_per_second = static_cast<float>(total_steps) / elapsed_time;
	const float io_time = wkf_timer_time(&timerS_);
	const float energy_time = wkf_timer_time(&timerE_);

	LOGINFO("SimManager: Performance Summary:");
	LOGINFO("  Total time: {:.2f}s", elapsed_time);
	LOGINFO("  Steps/second: {:.2f}", steps_per_second);
	LOGINFO("  I/O time: {:.2f}s ({:.1f}%)", io_time, io_time / elapsed_time * 100);
	LOGINFO("  Energy time: {:.2f}s ({:.1f}%)", energy_time, energy_time / elapsed_time * 100);
}

void SimManager::handle_imd_commands() {
	// Placeholder for IMD command handling
}

void SimManager::load_initial_conditions() {
	// SimManager handles I/O operations that Configuration parses
	const auto& config = sys_.get_config();

	// Load or generate initial particle data
	std::vector<Vector3> positions;
	std::vector<int> types;

	// TODO: Load from restart file if specified
	// if (!config.restart_file.empty()) {
	//     load_restart_data(config.restart_file, positions, types);
	// } else {
	// Generate initial positions and types based on configuration
	generate_initial_particles(positions, types);
	// }

	// Pass to SimSystem for GPU-compatible initialization
	sys_.initialize_particles(positions, types);

	LOGINFO("SimManager: Loaded {} particles", positions.size());
}

void SimManager::generate_initial_particles(std::vector<Vector3>& positions,
											std::vector<int>& types) {
	// Placeholder particle generation
	// In real implementation, this would read from input files or generate based on config
	const Vector3 box_size = sys_.get_box_size();
	const size_t num_particles = 1000; // Should come from configuration

	positions.reserve(num_particles);
	types.reserve(num_particles);

	// Simple random placement
	for (size_t i = 0; i < num_particles; ++i) {
		positions.emplace_back(box_size.x * (float)rand() / RAND_MAX,
							   box_size.y * (float)rand() / RAND_MAX,
							   box_size.z * (float)rand() / RAND_MAX);
		types.push_back(0); // All type 0 for now
	}
}

void SimManager::write_final_restart() {
	// Get current particle positions from SimSystem
	auto positions = sys_.get_particle_positions();

	// Write restart file - Configuration class would handle the actual file format
	const auto& config = sys_.get_config();
	std::string restart_filename = config.output_name + "_final.restart";

	LOGINFO("SimManager: Writing final restart to '{}'", restart_filename);
	// TODO: Implement actual restart file writing
	// write_restart_file(restart_filename, positions, types);
}

} // namespace ARBD
