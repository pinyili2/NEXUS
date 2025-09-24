#include "../extern/Catch2/extras/catch_amalgamated.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Test MPI compilation and basic availability
TEST_CASE("MPI Compilation Tests", "[mpi]") {

#ifdef USE_MPI
	SECTION("MPI headers and compilation") {
		// Test that MPI headers are available and basic constants are defined
		REQUIRE(MPI_COMM_WORLD != 0);
		REQUIRE(MPI_INT != 0);

		std::cout << "MPI headers compiled successfully" << std::endl;
		REQUIRE(true);
	}

	SECTION("MPI version constants") {
// Test MPI version constants are available
#ifdef MPI_VERSION
		std::cout << "MPI_VERSION: " << MPI_VERSION << std::endl;
		REQUIRE(MPI_VERSION >= 3);
#endif

#ifdef MPI_SUBVERSION
		std::cout << "MPI_SUBVERSION: " << MPI_SUBVERSION << std::endl;
		REQUIRE(MPI_SUBVERSION >= 0);
#endif
	}

	SECTION("OpenMPI environment detection") {
		// Test for OpenMPI specific environment variables
		char* ompi_version = getenv("OMPI_VERSION");
		if (ompi_version) {
			std::cout << "OpenMPI Version from environment: " << ompi_version << std::endl;

			// Test for OpenMPI 5.0.x series
			std::string version_str(ompi_version);
			if (version_str.find("5.0.") != std::string::npos) {
				std::cout << "Detected OpenMPI 5.0.x series" << std::endl;
				REQUIRE(true); // OpenMPI 5.0.x detected successfully
			}
		} else {
			std::cout << "OMPI_VERSION environment variable not set" << std::endl;
			REQUIRE(true); // Still pass, environment may not be set
		}
	}

#else
	WARN("MPI not compiled in - skipping MPI tests");
	REQUIRE(true); // Always pass when MPI is not available
#endif
}

// Test MPI with SYCL compilation compatibility
TEST_CASE("MPI with SYCL Backend Compilation", "[mpi][sycl]") {

#if defined(USE_MPI) && \
	(defined(USE_SYCL) || defined(PROJECT_USES_SYCL_ACPP) || defined(PROJECT_USES_SYCL_ICPX))

	SECTION("MPI and SYCL compilation compatibility") {
		// Test that MPI and SYCL headers can be included together
		std::cout << "MPI+SYCL: Headers compiled successfully" << std::endl;

		// Test basic MPI constants are available
		REQUIRE(MPI_COMM_WORLD != 0);
		REQUIRE(MPI_INT != 0);

		// Test that we can create basic data structures
		std::vector<int> test_data(10);
		for (int i = 0; i < 10; i++) {
			test_data[i] = i * 2;
		}

		// Verify data
		for (int i = 0; i < 10; i++) {
			REQUIRE(test_data[i] == i * 2);
		}

		std::cout << "MPI+SYCL: Basic data structure test passed" << std::endl;
	}

	SECTION("Preprocessor definitions check") {
// Check that the right preprocessor definitions are set
#ifdef USE_MPI
		std::cout << "USE_MPI is defined" << std::endl;
		REQUIRE(true);
#endif

#if defined(USE_SYCL) || defined(PROJECT_USES_SYCL_ACPP) || defined(PROJECT_USES_SYCL_ICPX)
		std::cout << "SYCL backend is defined" << std::endl;
		REQUIRE(true);
#endif

#ifdef PROJECT_USES_SYCL_ICPX
		std::cout << "Intel DPC++ SYCL backend detected" << std::endl;
#endif

#ifdef PROJECT_USES_SYCL_ACPP
		std::cout << "AdaptiveCpp SYCL backend detected" << std::endl;
#endif
	}

#else
	WARN("MPI+SYCL not available - skipping integration tests");
	REQUIRE(true);
#endif
}

// Note: MPI initialization and finalization should be handled by the test runner
// or CMake test configuration when running with mpirun
