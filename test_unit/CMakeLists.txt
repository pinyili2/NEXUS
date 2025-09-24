#find_package(Catch2 3 REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2/extras)
include(Catch) # contains some cmake macros that will help with discovery of tests
include(CTest)

# Common source files for core tests
set(CORE_TEST_SOURCES
    Buffer/generic_Buffer.cpp
    Buffer/buffer_test.cpp
    Buffer/Buffer.cpp
    Buffer/Pinned_Unified.cpp
    Buffer/async_buffer_tests.cpp
    backend_tests.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)

# Math folder sources - explicitly list what to include in math tests
set(MATH_SOURCES
    Math/RandomDevice.cpp
    Math/RandomProfile.cpp
    Math/Kernel_for_test.h
    # Add more math sources here as needed
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)

# Common libraries and include directories for all tests
set(COMMON_TEST_LIBS "lib${PROJECT_NAME}" Catch2::Catch2WithMain)
if(USE_CUDA)
    list(APPEND COMMON_TEST_LIBS curand)
endif()
if(USE_NCCL)
    list(APPEND COMMON_TEST_LIBS nccl)
endif()

# Common test executables list
set(COMMON_TEST_EXECUTABLES)

# Configuration parsing test - platform independent
add_executable(arbd_test_config_parse
    parse_config.cpp
)
target_include_directories(arbd_test_config_parse PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(arbd_test_config_parse PRIVATE "lib${PROJECT_NAME}")

# Add test
add_test(NAME Config_Parse_Tests COMMAND arbd_test_config_parse)
set_tests_properties(Config_Parse_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES arbd_test_config_parse)
message(STATUS "Added configuration parsing test executable: arbd_test_config_parse")

# Backend comprehensive tests - platform independent, tests all backend abstractions
add_executable(arbd_test_kernel
    kernels.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(arbd_test_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(arbd_test_kernel PRIVATE "lib${PROJECT_NAME}")

# Add backend test
add_test(NAME Backend_Comprehensive_Tests COMMAND arbd_test_kernel)
set_tests_properties(Backend_Comprehensive_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES arbd_test_kernel)
message(STATUS "Added backend comprehensive test executable: arbd_test_kernel")

# BaseGrid tests - comprehensive grid operations and device transfers
add_executable(arbd_test_basegrid
    BaseGrid_tests.cpp
    catch_boiler.h
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(arbd_test_basegrid PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(arbd_test_basegrid PRIVATE "lib${PROJECT_NAME}")

# Add BaseGrid test
add_test(NAME BaseGrid_Comprehensive_Tests COMMAND arbd_test_basegrid)
set_tests_properties(BaseGrid_Comprehensive_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES arbd_test_basegrid)
message(STATUS "Added BaseGrid comprehensive test executable: arbd_test_basegrid")

# Metal Tests
if(USE_METAL AND APPLE)
    # Common Metal test settings
    set(METAL_TEST_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )
    set(METAL_TEST_LIBS
        "lib${PROJECT_NAME}"
        METAL_CPP
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
    )

    # Metal backend tests - use core sources + Metal-specific sources
    add_executable(arbd_test_metal
        Backends/metal_backend.cpp
        Backends/metal_events.cpp
        Backends/metal_manager.cpp
        Kernels/metal_kernels.cpp
        Kernels/metal_matmul.cpp
        )
    target_include_directories(arbd_test_metal PRIVATE ${METAL_TEST_INCLUDES})
    target_link_libraries(arbd_test_metal PRIVATE ${METAL_TEST_LIBS})
    # Make test depend on main project's Metal shaders
    if(TARGET metal_shaders_main)
        add_dependencies(arbd_test_metal metal_shaders_main)
    endif()

    add_test(NAME Metal_Tests COMMAND arbd_test_metal)
    set_tests_properties(Metal_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Metal math tests - use all math sources
    add_executable(arbd_test_metal_math
        ${MATH_SOURCES}
        Math/metal_bitmask.cpp
        Math/metal_math.cpp
    )
    target_include_directories(arbd_test_metal_math PRIVATE ${METAL_TEST_INCLUDES})
    target_link_libraries(arbd_test_metal_math PRIVATE ${METAL_TEST_LIBS})
    # Make test depend on main project's Metal shaders
    if(TARGET metal_shaders_main)
        add_dependencies(arbd_test_metal_math metal_shaders_main)
    endif()

    add_test(NAME Metal_Math_Tests COMMAND arbd_test_metal_math)
    set_tests_properties(Metal_Math_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES arbd_test_metal arbd_test_metal_math)
    message(STATUS "Added Metal test executables: arbd_test_metal, arbd_test_metal_math")

elseif(USE_METAL)
    message(WARNING "USE_METAL is ON but not on Apple platform - Metal tests will be excluded")
endif()

# SYCL Tests
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # SYCL backend tests
    add_executable(arbd_test_sycl
        Backends/sycl_backend.cpp
        Backends/sycl_manager.cpp
        Backends/sycl_omp.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(arbd_test_sycl PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

    # SYCL math tests - use all math sources
    add_executable(arbd_test_sycl_math
        ${MATH_SOURCES})
    target_include_directories(arbd_test_sycl_math PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl_math PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

    if(USE_SYCL_ACPP)
        target_link_libraries(arbd_test_sycl PRIVATE stdc++)
        target_link_libraries(arbd_test_sycl_backend PRIVATE stdc++)
        target_link_libraries(arbd_test_sycl_math PRIVATE stdc++)
        # Also link Boost libraries for AdaptiveCpp
        target_link_directories(arbd_test_sycl PRIVATE "${BOOST_LIB_DIR}")
        target_link_directories(arbd_test_sycl_backend PRIVATE "${BOOST_LIB_DIR}")
        target_link_directories(arbd_test_sycl_math PRIVATE "${BOOST_LIB_DIR}")
    endif()

    # Intel DPC++ specific linking for test executables
    if(USE_SYCL_ICPX AND CMAKE_CXX_COMPILER MATCHES "icpx")
        # Find Intel OneAPI SYCL installation directory
        get_filename_component(ICPX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ICPX_INSTALL_DIR "${ICPX_BIN_DIR}" DIRECTORY)
        set(ICPX_LIB_DIR "${ICPX_INSTALL_DIR}/lib")

        # Find and link the essential Intel DPC++ SYCL runtime libraries for tests
        find_library(SYCL_LIB NAMES "sycl" PATHS "${ICPX_LIB_DIR}" NO_DEFAULT_PATH)

        if(SYCL_LIB)
            # Add the Intel DPC++ library directory to linker search paths for tests
            target_link_directories(arbd_test_sycl PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(arbd_test_sycl_math PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(arbd_test_kernel PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(arbd_test_basegrid PUBLIC "${ICPX_LIB_DIR}")

            # Link SYCL runtime library to all SYCL test executables
            target_link_libraries(arbd_test_sycl PRIVATE ${SYCL_LIB})
            target_link_libraries(arbd_test_sycl_math PRIVATE ${SYCL_LIB})
            target_link_libraries(arbd_test_kernel PRIVATE ${SYCL_LIB})
            target_link_libraries(arbd_test_basegrid PRIVATE ${SYCL_LIB})

            message(STATUS "Linked Intel DPC++ SYCL runtime to test executables")
        else()
            message(WARNING "Could not find Intel DPC++ SYCL runtime library for tests in ${ICPX_LIB_DIR}")
        endif()

        # Link OpenCL for Intel GPU support in tests
        find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
        if(OPENCL_LIB)
            target_link_libraries(arbd_test_sycl PRIVATE ${OPENCL_LIB})
            target_link_libraries(arbd_test_sycl_math PRIVATE ${OPENCL_LIB})
            target_link_libraries(arbd_test_kernel PRIVATE ${OPENCL_LIB})
            target_link_libraries(arbd_test_basegrid PRIVATE ${OPENCL_LIB})
            message(STATUS "Linked OpenCL to SYCL test executables")
        endif()
    endif()

    # Add tests
    add_test(NAME SYCL_Tests COMMAND arbd_test_sycl)
    set_tests_properties(SYCL_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_test(NAME SYCL_Backend_Tests COMMAND arbd_test_sycl_backend)
    set_tests_properties(SYCL_Backend_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_test(NAME SYCL_Math_Tests COMMAND arbd_test_sycl_math)
    set_tests_properties(SYCL_Math_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES arbd_test_sycl arbd_test_sycl_math)
    message(STATUS "Added SYCL test executables: arbd_test_sycl, arbd_test_sycl_math")
endif()

# CUDA Tests
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA core tests - use core sources + CUDA-specific sources
    add_executable(arbd_test_cuda
        ${CORE_TEST_SOURCES}
        #Kernels/cuda_kernels.cu
        #Kernels/cuda_kernels.cpp
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --allow-unsupported-compiler -Xcompiler=-Wno-deprecated-declarations")
    target_include_directories(arbd_test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_cuda PRIVATE
        "lib${PROJECT_NAME}"
        CUDA::cudart
        CUDA::cuda_driver
        curand)

    # CUDA math tests - use all math sources
    add_executable(arbd_test_cuda_math
        ${MATH_SOURCES}
        #Math/RandomTestKernels.cu
        Math/Kernel_for_test.h
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --allow-unsupported-compiler -Xcompiler=-Wno-deprecated-declarations")
    target_include_directories(arbd_test_cuda_math PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_cuda_math PRIVATE
        "lib${PROJECT_NAME}"
        CUDA::cudart
        CUDA::cuda_driver
        curand)

    # Set CUDA properties for math test files
    set_source_files_properties(
        ${MATH_SOURCES}
        Math/RandomTestKernels.cu
        Math/Kernel_for_test.h
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )

    if(USE_NCCL)
        target_link_libraries(arbd_test_cuda PRIVATE nccl)
    endif()

    # Add tests
    catch_discover_tests(arbd_test_cuda)

    add_test(NAME CUDA_Math_Tests COMMAND arbd_test_cuda_math)
    set_tests_properties(CUDA_Math_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES arbd_test_cuda arbd_test_cuda_math)
    message(STATUS "Added CUDA test executables: arbd_test_cuda, arbd_test_cuda_math")
endif()

# MPI Tests
if(USE_MPI)
    # Create MPI test executable
    add_executable(arbd_test_mpi
        mpitest.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(arbd_test_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_mpi PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        # When using MPI compiler wrapper, linking is automatic
        message(STATUS "MPI test linking handled automatically by compiler wrapper")
    else()
        # Manual MPI linking when not using wrapper
        target_link_libraries(arbd_test_mpi
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories(arbd_test_mpi
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to test executable")
    endif()

    # Ensure MPI test uses only CXX, not CUDA compiler
    set_target_properties(arbd_test_mpi PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    # Explicitly set this as a CXX-only target to avoid CUDA flag inheritance
    set_source_files_properties(mpitest.cpp PROPERTIES LANGUAGE CXX)

    # Fix C++ standard library linking issues (same as main executable)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(arbd_test_mpi PRIVATE -Wl,--whole-archive stdc++ -Wl,--no-whole-archive)
        target_link_libraries(arbd_test_mpi PRIVATE gcc_s gcc m pthread dl)
        target_link_options(arbd_test_mpi PRIVATE -Wl,--no-as-needed)
    endif()

    # Note: Since we removed custom main from mpitest.cpp, we need to use a different approach
    # for MPI tests that require MPI_Init/MPI_Finalize. For now, create basic tests that can
    # run without MPI initialization (testing compilation and basic functionality)

    # Add MPI-specific tests - these will test MPI availability and basic functionality
    add_test(NAME MPI_Tests COMMAND arbd_test_mpi)
    set_tests_properties(MPI_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Set test properties for OpenMPI 5.0.7 compatibility
    set_tests_properties(MPI_Tests PROPERTIES
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES arbd_test_mpi)
    message(STATUS "Added MPI test executable: arbd_test_mpi with OpenMPI 5.0.7 support")
endif()

# Install all test executables at once
install(TARGETS ${COMMON_TEST_EXECUTABLES})

# Print summary of all test executables
message(STATUS "All test executables: ${COMMON_TEST_EXECUTABLES}")
