#pragma once
/**
 * @file SimSystem.h
 * @author Pin-Yi Li <pinyili2@illinois.edu>
 * @brief Simulation system class. Stores the system configuration and objects that won't change
 * during the simulation.
 * @version 2.0
 * @date 2025-09-09
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifdef HOST_GUARD
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Configuration.h"
#include "Types/IndexList.h"
#include "Types/Types.h"
#include <array>
#include <iostream> // For logging placeholders
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ARBD {
class SimSystem;
class Decomposer {
  public:
	virtual ~Decomposer() = default;
	DecomposerType get_type() const {
		return type_;
	}
	/**
	 * @brief The core function that performs the decomposition.
	 * @param sys The global system containing particle data and configuration.
	 * @param resources The collection of hardware resources (e.g., GPUs) to distribute to.
	 */
	virtual void decompose(SimSystem& sys, const ResourceCollection& resources) = 0;

	virtual const std::string get_name() {
		switch (type_) {
		case DecomposerType::Cell:
			return "CellDecomposer";
		case DecomposerType::RecursiveBisection:
			return "RecursiveBisectionDecomposer";
		case DecomposerType::Geometric:
			return "GeometricDecomposer";
		default:
			throw ARBD::Exception(ARBD::ExceptionType::ValueError, "Unsupported decomposer type.");
		}
	};

  private:
	DecomposerType type_;
};

class CellDecomposer : public Decomposer {
  public:
	void decompose(SimSystem& sys, const ResourceCollection& resources) override;

  private:
	DecomposerType type_ = DecomposerType::Cell;
};
//================================================================================
// Simulation System - Global State and Configuration Management
//================================================================================
class SimSystem {
  public:
	/**
	 * @brief Construct simulation system from configuration
	 * @param conf Configuration manager with validated parameters
	 * @param resources Available computational resources
	 */
	SimSystem(const SimConf& conf, const ResourceCollection& resources)
		: config_(conf.get_config()), resources_(resources) {
		LOGINFO("SimSystem: Initializing from configuration");

		// Copy configuration to GPU-compatible members
		copy_config_to_gpu_members();

		// Create boundary conditions from configuration
		boundary_conditions_ = conf.create_boundary_conditions();

		// Create the chosen decomposer instance (Factory Pattern)
		// create_decomposer();
		decomposer_ = std::make_unique<CellDecomposer>();
		// Initialize global system objects
		initialize_system_objects();

		LOGINFO("SimSystem: Using decomposer '{}'", decomposer_->get_name());
	}

	//================================================================================
	// System Management Methods
	//================================================================================

	/**
	 * @brief Triggers the domain decomposition process
	 * Typically called once at initialization, but can be used for rebalancing
	 */
	void decompose_system() {
		if (!decomposer_) {
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"No decomposer has been set");
		}
		LOGINFO("SimSystem: Starting domain decomposition");
		decomposer_->decompose(*this, resources_);
		LOGINFO("SimSystem: Domain decomposition completed");
	}

	/**
	 * @brief Rebalances the system after a change in the number of resources
	 */
	void rebalance_system() {
		LOGINFO("SimSystem: Rebalancing system");
		decompose_system(); // For now, just re-decompose
	}

	/**
	 * @brief
	 *
	 *@param particle_data Host particle data loaded by Configuration / SimManager
	 */
	void initialize_particles(const std::vector<Vector3>& positions, const std::vector<int>& types);

	/**
	 * @brief Set particle positions (GPU-compatible)
	 * @param positions New particle positions
	 */
	void set_particle_positions(const std::vector<Vector3>& positions);
	void build_neighbor_list();

	/**
	 * @brief Get particle positions (GPU-compatible)
	 * @return Current particle positions
	 */
	std::vector<Vector3> get_particle_positions() const;

	//================================================================================
	// Configuration and State Accessors
	//================================================================================

	/**
	 * @brief Get temperature at a specific position (GPU-compatible)
	 * @param position Optional position for grid-based temperature
	 * @return Temperature value at the given position
	 */
	float get_temperature(Vector3 position = {0, 0, 0}) const {
		if (temperature_format_ == 0) { // Value format
			return temperature_value_;
		} else if (temperature_grid_) {
			return temperature_grid_->get_value(position);
		}
		return temperature_value_; // Fallback
	}

	/**
	 * @brief Get cutoff distance for interactions (GPU-compatible)
	 */
	float get_cutoff() const {
		return cutoff_;
	}

	/**
	 * @brief Get boundary conditions
	 */
	const BoundaryConditions& get_boundary_conditions() const {
		return boundary_conditions_;
	}

	/**
	 * @brief Get timestep (GPU-compatible)
	 */
	float get_timestep() const {
		return timestep_;
	}

	/**
	 * @brief Get number of simulation steps
	 */
	int get_num_steps() const {
		return num_steps_;
	}

	/**
	 * @brief Get box dimensions (GPU-compatible)
	 */
	Vector3 get_box_size() const {
		return box_size_;
	}

	/**
	 * @brief Get complete configuration
	 */
	const Configuration& get_config() const {
		return config_;
	}

	/**
	 * @brief Get available computational resources
	 */
	const ResourceCollection& get_resources() const {
		return resources_;
	}

	//================================================================================
	// System State Queries
	//================================================================================

	/**
	 * @brief Check if system has bonded interactions (GPU-compatible)
	 */
	bool has_bonds() const {
		return has_bonds_;
	}

	/**
	 * @brief Check if system has external forces (GPU-compatible)
	 */
	bool has_external_forces() const {
		return has_external_forces_;
	}

	/**
	 * @brief Check if system has reactions (GPU-compatible)
	 */
	bool has_reactions() const {
		return has_reactions_;
	}

	/**
	 * @brief Get number of particles in the system (GPU-compatible)
	 */
	size_t get_num_particles() const {
		return num_particles_;
	}

	//================================================================================
	// System Object Accessors (for SimManager)
	//================================================================================

	/**
	 * @brief Get electric field vector (if any)
	 */
	Vector3 get_electric_field() const {
		return electric_field_;
	}

	/**
	 * @brief Get force grid (if any)
	 */
	const BaseGrid<Vector3>* get_force_grid() const {
		return force_grid_;
	}

	/**
	 * @brief Get bond list for bonded force calculations
	 */
	const int* get_bond_list() const {
		return bond_list_;
	}

	/**
	 * @brief Get bond list size
	 */
	size_t get_bond_list_size() const {
		return bond_list_size_;
	}

	/**
	 * @brief Get reservoirs for grand canonical simulations
	 */
	const std::vector<Reservoir>& get_reservoirs() const {
		return config_.reservoirs;
	}

  private:
	//================================================================================
	// Member Variables
	//================================================================================

	// Configuration management (host-only)
	Configuration config_;
	BoundaryConditions boundary_conditions_;

	// Resources and decomposition (host-only)
	ResourceCollection resources_;
	std::unique_ptr<Decomposer> decomposer_;

	// GPU-compatible system parameters (can be copied to device)
	float temperature_value_{298.15f};
	int temperature_format_{0}; // 0 = value, 1 = grid
	float cutoff_{50.0f};
	float timestep_{1e-5f};
	int num_steps_{1000};
	Vector3 box_size_{5000.0f, 5000.0f, 5000.0f};
	// System state (GPU-compatible)
	size_t num_particles_{0};
	bool has_bonds_{false};
	bool has_external_forces_{false};
	bool has_reactions_{false};

	// Global system objects (GPU-compatible pointers when needed)
	Vector3 electric_field_{0, 0, 0};
	BaseGrid<float>* temperature_grid_{nullptr}; // Device pointer
	BaseGrid<Vector3>* force_grid_{nullptr};	 // Device pointer
	int* bond_list_{nullptr};					 // Device pointer
	size_t bond_list_size_{0};

	//================================================================================
	// Private Methods
	//================================================================================

	/**
	 * @brief Create appropriate decomposer based on configuration

	void create_decomposer() {
		switch (config_.decomposer) {
		case DecomposerType::Cell:
			decomposer_ = std::make_unique<CellDecomposer>();
			break;
		case DecomposerType::RecursiveBisection:
			throw Exception(ExceptionType::NotImplementedError,
							SourceLocation(),
							"RecursiveBisectionDecomposer not implemented");
			break;
		case DecomposerType::Geometric:
			throw Exception(ExceptionType::NotImplementedError,
							SourceLocation(),
							"GeometricDecomposer not implemented");
			break;
		default:
			throw Exception(ExceptionType::ValueError,
							SourceLocation(),
							"Unsupported decomposer type");
		}
	}
		*/

	/**
	 * @brief Copy configuration parameters to GPU-compatible members
	 */
	void copy_config_to_gpu_members() {
		temperature_value_ = config_.temperature.value;
		temperature_format_ = static_cast<int>(config_.temperature.format);
		cutoff_ = config_.cutoff.value;
		timestep_ = config_.steps.timestep;
		num_steps_ = config_.steps.steps;
		box_size_ = Vector3(config_.box_lengths[0], config_.box_lengths[1], config_.box_lengths[2]);
		has_reactions_ = config_.has_reaction;
	}

	/**
	 * @brief Initialize system objects based on configuration
	 */
	void initialize_system_objects() {
		// Initialize force grids, electric fields, bonds, etc. based on config
		// This would be implemented based on your specific system requirements
		LOGINFO("SimSystem: Initializing system objects");

		// Placeholder implementations
		has_bonds_ = false;			  // Set based on actual bond data
		has_external_forces_ = false; // Set based on grid/field configuration

		// Initialize GPU grids if needed
		if (config_.temperature.format == Temperature::Format::Grid && config_.temperature.grid) {
			// Copy temperature grid to device
			// temperature_grid_ = copy_grid_to_device(config_.temperature.grid.get());
		}
	}
};

} // namespace ARBD

#endif // HOST_GUARD
