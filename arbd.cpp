
#ifndef HOST_GUARD
#define HOST_GUARD
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "ARBDException.h"
#include "Backend/Resource.h"
#include "Configuration.h"
#include "SimManager.h"
#include "SimSystem.h"

#include "SignalManager.h"
#include <cstdio>	// For printf
#include <cstring>	// For strcmp
#include <iostream> // For std::cout, std::endl (modern C++)
#include <string>	// For std::string (modern C++)

// Define this if not provided by CMake/build system for version info
#ifndef VERSION
#define VERSION "Development Build - June 2025"
#endif

// Consider moving constants to a dedicated configuration header or class
const unsigned int kDefaultIMDPort = 71992;
const unsigned int kDefaultNodes = 1;
const unsigned int kDefaultGpus = 0;
unsigned int gpus[] = {kDefaultGpus};

struct ProgramOptions {
	std::string configFile;
	std::string outputFile;
	std::vector<int> gpuIds;
	int numGpus = 0;
	int numNodes = 1;
};

bool parse_basic_args(int argc, char* argv[], ProgramOptions& opts) {
	if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
		printf("Usage: %s [OPTIONS] CONFIGFILE OUTPUT [SEED]\n", argv[0]);
		printf("\n");
		printf("  -h, --help         Display this help and exit\n");
		printf("  --info             Output basic CPU and CUDA information (stubbed) and exit\n");
		printf("  --version          Output version information and exit\n");
		printf("  -i, --imd=         IMD port (defaults to %u)\n", kDefaultIMDPort);
		printf("  -g, --gpus=        Number of GPUs to use (defaults to %u)\n", kDefaultGpus);
		printf("  -gid, --gpu_ids=   List of GPU IDs to use (e.g., --gid 0 1 2 3)\n");
		printf("  -n, --nodes=       Number of nodes to use (defaults to %u)\n", kDefaultNodes);
		return false; // Indicates help was shown, program should exit
	} else if (argc == 2 && (strcmp(argv[1], "--version") == 0)) {
		printf("%s %s\n", argv[0], VERSION);
		return false; // Indicates version was shown, program should exit
	} else if (argc == 2 && (strcmp(argv[1], "--info") == 0)) {
		printf("Use the main program to see detailed resource information.\n");
		printf("Example: %s --help\n", argv[0]);
		return false;	   // Indicates info was shown, program should exit
	} else if (argc < 3) { // Expecting at least program_name, config, output
		printf("%s: missing arguments (expected CONFIGFILE OUTPUT)\n", argv[0]);
		printf("Try '%s --help' for more information.\n", argv[0]);
		return false; // Indicates error, program should exit
	}

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gpus") == 0) {
			if (i + 1 < argc) {
				opts.numGpus = atoi(argv[i + 1]);
				++i; // Skip next argument
			}
		} else if (strcmp(argv[i], "-gid") == 0 || strcmp(argv[i], "--gpu_ids") == 0) {
			// Parse GPU IDs until we hit another flag or end of args
			++i;
			while (i < argc && argv[i][0] != '-') {
				opts.gpuIds.push_back(atoi(argv[i]));
				++i;
			}
			--i; // Back up one since we'll increment in the loop
		} else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--nodes") == 0) {
			if (i + 1 < argc) {
				opts.numNodes = atoi(argv[i + 1]);
				++i; // Skip next argument
			}
		}
	}

	// Find config and output files (last two non-flag arguments)
	int fileArgs = 0;
	for (int i = argc - 1; i >= 1 && fileArgs < 2; --i) {
		if (argv[i][0] != '-') {
			if (fileArgs == 0) {
				opts.outputFile = argv[i];
			} else if (fileArgs == 1) {
				opts.configFile = argv[i];
			}
			++fileArgs;
		}
	}

	if (opts.configFile.empty() || opts.outputFile.empty()) {
		printf("%s: missing arguments (expected CONFIGFILE OUTPUT)\n", argv[0]);
		printf("Try '%s --help' for more information.\n", argv[0]);
		return false;
	}

	return true;
}

int main(int argc, char* argv[]) {
	// MPI Initialization (kept as is, conditional)

	ARBD::SignalManager::manage_segfault();

	ProgramOptions options;
	if (!parse_basic_args(argc, argv, options)) {
		return (argc < 3 &&
				!(argc == 2 &&
				  (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 ||
				   strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "--info") == 0)))
				   ? 1
				   : 0;
	}

	// Print a startup message
	// Use std::cout for modern C++
	std::cout << "--- Atomic Resolution Brownian Dynamics (ARBD) ---" << std::endl;
	std::cout << "Version: " << VERSION << std::endl;
	std::cout << "Config File: " << options.configFile << std::endl;
	std::cout << "Output Target: " << options.outputFile << std::endl;

	std::cout << "Initializing Simulation Manager..." << std::endl;

	ARBD::ResourceCollection resource_collection;

#ifdef USE_CUDA
	std::cout << "ARBD compiled with CUDA support." << std::endl;
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
		// If user specified GPU IDs, use those
		if (!options.gpuIds.empty()) {
			for (int gpuId : options.gpuIds) {
				if (gpuId >= 0 && gpuId < deviceCount) {
					resource_collection.resources.push_back(ARBD::Resource::CUDA(gpuId));
				} else {
					std::cout << "Warning: GPU ID " << gpuId << " is invalid (available: 0-"
							  << (deviceCount - 1) << ")" << std::endl;
				}
			}
		}
		// If user specified number of GPUs, use first N devices
		else if (options.numGpus > 0) {
			int gpusToUse = std::min(options.numGpus, deviceCount);
			for (int i = 0; i < gpusToUse; ++i) {
				resource_collection.resources.push_back(ARBD::Resource::CUDA(i));
			}
		}
		// Default: use all available GPUs
		else {
			for (int i = 0; i < deviceCount; ++i) {
				resource_collection.resources.push_back(ARBD::Resource::CUDA(i));
			}
		}
	}

	// Fallback to CPU if no GPUs available or selected
	if (resource_collection.resources.empty()) {
		std::cout << "No GPUs available. Falling back to CPU." << std::endl;
		resource_collection.resources.push_back(ARBD::Resource::CPU());
	}

#elif defined(USE_SYCL)
	std::cout << "ARBD compiled with SYCL support." << std::endl;

	if (!options.gpuIds.empty()) {
		for (int gpuId : options.gpuIds) {
			resource_collection.resources.push_back(ARBD::Resource::SYCL(gpuId));
		}
	}
	// If user specified number of GPUs, try first N devices
	else if (options.numGpus > 0) {
		for (int i = 0; i < options.numGpus; ++i) {
			resource_collection.resources.push_back(ARBD::Resource::SYCL(i));
		}
	}
	// Default: try device 0
	else {
		resource_collection.resources.push_back(ARBD::Resource::SYCL(0));
	}

	// Fallback to CPU if no SYCL devices available
	if (resource_collection.resources.empty()) {
		std::cout << "No SYCL devices available. Falling back to CPU." << std::endl;
		resource_collection.resources.push_back(ARBD::Resource::CPU());
	}

#elif defined(USE_METAL)
	std::cout << "ARBD compiled with METAL support." << std::endl;
	// For SYCL/Metal/OpenMP, force GPU ID to 0 and ignore user GPU specifications
	options.gpuIds.clear();
	options.gpuIds.push_back(0);
	options.numGpus = 1;

	resource_collection.resources.push_back(ARBD::Resource::METAL(0));

#else
	std::cout << "ARBD compiled with CPU-only support." << std::endl;
	// For CPU-only builds, ignore GPU specifications
	options.gpuIds.clear();
	options.numGpus = 0;
	resource_collection.resources.push_back(ARBD::Resource::CPU());
#endif

	// Validate all selected resources and remove invalid ones
	std::cout << "Validating " << resource_collection.resources.size() << " compute resource(s)..."
			  << std::endl;
	std::vector<ARBD::Resource> validResources;

	for (const auto& res : resource_collection.resources) {
		try {
			// Create a copy to validate (validate() is const)
			ARBD::Resource resCopy = res;
			resCopy.validate();
			validResources.push_back(res);
			std::cout << "✓ " << res.toString() << " validated successfully" << std::endl;
		} catch (const ARBD::Exception& e) {
			std::cout << "✗ " << res.toString() << " validation failed: " << e.what() << std::endl;
		}
	}

	// Update resources with only valid ones
	resource_collection.resources = validResources;

	// If no valid resources, fallback to CPU
	if (resource_collection.resources.empty()) {
		std::cout << "No valid compute resources found. Falling back to CPU." << std::endl;
		resource_collection.resources.push_back(ARBD::Resource::CPU());
	}

	std::cout << "Selected " << resource_collection.resources.size() << " compute resource(s): ";
	for (const auto& res : resource_collection.resources) {
		std::cout << res.toString() << " ";
	}
	std::cout << std::endl;

	// Load and validate configuration → convert to runtime config
	// Pseudocode: cfg = Configuration::Load(options.configFile)
	//            conf = cfg.to_sim_conf()
	// ARBD::Configuration cfg = ARBD::Configuration::Load(options.configFile);
	// ARBD::SimSystem::Conf conf = cfg.to_sim_conf();

	// Build system and manager
	// ARBD::SimSystem sys(conf, resource_collection);
	// ARBD::SimManager manager(sys, resource_collection);

	// Single initialization-time domain decomposition
	// sys.decompose_system();

	// Main simulation loop orchestration lives in SimManager::run()
	// Pseudocode inside run(): build neighbor lists, schedule patch ops,
	// halo exchange (if multi-resource), integrate, write outputs
	// manager.run();

	/*
	int replicas = 1;
	unsigned int imd_port = 0;
	bool imd_on = false;
	// ... (rest of the original complex parsing loop) ...

	char* configFile = options.configFile.data(); // Unsafe if string is empty
	char* outArg = options.outputFile.data();   // Unsafe

	// ... (original CUDA selection logic) ...

	// Configuration config(configFile, replicas, debug);
	// config.copyToCUDA();
	// GrandBrownTown brown(config, outArg,
	//      debug, imd_on, imd_port, replicas);
	// brown.run();
	*/

#ifdef USE_MPI
	MPI_Finalize();
#endif

	return 0;
}
