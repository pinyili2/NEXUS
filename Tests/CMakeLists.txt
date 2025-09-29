
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2/extras)
include(Catch) # contains some cmake macros that will help with discovery of tests
include(CTest)

# Common source files for core tests
set(CORE_TEST_SOURCES
    async_buffer_tests.cpp
    pinned_unified_tests.cpp
    production_buffer_tests.cpp
    mpitest.cpp
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


# Backend comprehensive test suite
add_executable(test_backend
    backend_tests.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(test_backend PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_backend PRIVATE "lib${PROJECT_NAME}")

# Add test
add_test(NAME Backend_Tests COMMAND test_backend)
set_tests_properties(Backend_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES test_backend)
message(STATUS "Added backend comprehensive test executable: test_backend")

# Backend comprehensive tests - platform independent, tests all backend abstractions
add_executable(test_kernel
    kernels.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(test_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_kernel PRIVATE "lib${PROJECT_NAME}")

# Add backend test
add_test(NAME Backend_Comprehensive_Tests COMMAND test_kernel)
set_tests_properties(Backend_Comprehensive_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES test_kernel)
message(STATUS "Added backend comprehensive test executable: test_kernel")

# Buffer tests - comprehensive buffer operations and device transfers
add_executable(test_buffer
    production_buffer_tests.cpp
    async_buffer_tests.cpp
    pinned_unified_tests.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(test_buffer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_buffer PRIVATE "lib${PROJECT_NAME}")

# Add Buffer test
add_test(NAME Buffer_Tests COMMAND test_buffer)
set_tests_properties(Buffer_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES test_buffer)
message(STATUS "Added Buffer comprehensive test executable: test_buffer")

# Texture tests - texture buffer operations
add_executable(test_texture
    texture_tests.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(test_texture PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_texture PRIVATE "lib${PROJECT_NAME}")

# Add Texture test
add_test(NAME Texture_Tests COMMAND test_texture)
set_tests_properties(Texture_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES test_texture)
message(STATUS "Added Texture test executable: test_texture")

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
    add_executable(test_metal

        )
    target_include_directories(test_metal PRIVATE ${METAL_TEST_INCLUDES})
    target_link_libraries(test_metal PRIVATE ${METAL_TEST_LIBS})
    # Make test depend on main project's Metal shaders
    if(TARGET metal_shaders_main)
        add_dependencies(test_metal metal_shaders_main)
    endif()

    add_test(NAME Metal_Tests COMMAND test_metal)
    set_tests_properties(Metal_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_metal)
    message(STATUS "Added Metal test executables: test_metal")

elseif(USE_METAL)
    message(WARNING "USE_METAL is ON but not on Apple platform - Metal tests will be excluded")
endif()

# SYCL Tests
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # SYCL backend tests
    add_executable(test_sycl
        ${CORE_TEST_SOURCES}
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp)
    target_include_directories(test_sycl PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_sycl PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

    if(USE_SYCL_ACPP)
        target_link_libraries(test_sycl PRIVATE stdc++)
        # Also link Boost libraries for AdaptiveCpp
        target_link_directories(test_sycl PRIVATE "${BOOST_LIB_DIR}")
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
            target_link_directories(test_sycl PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(test_kernel PUBLIC "${ICPX_LIB_DIR}")
            target_link_directories(test_buffer PUBLIC "${ICPX_LIB_DIR}")

            # Link SYCL runtime library to all SYCL test executables
            target_link_libraries(test_sycl PRIVATE ${SYCL_LIB})
            target_link_libraries(test_kernel PRIVATE ${SYCL_LIB})
            target_link_libraries(test_buffer PRIVATE ${SYCL_LIB})

            message(STATUS "Linked Intel DPC++ SYCL runtime to test executables")
        else()
            message(WARNING "Could not find Intel DPC++ SYCL runtime library for tests in ${ICPX_LIB_DIR}")
        endif()

        # Link OpenCL for Intel GPU support in tests
        find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
        if(OPENCL_LIB)
            target_link_libraries(test_sycl PRIVATE ${OPENCL_LIB})
            target_link_libraries(test_kernel PRIVATE ${OPENCL_LIB})
            target_link_libraries(test_buffer PRIVATE ${OPENCL_LIB})
            message(STATUS "Linked OpenCL to SYCL test executables")
        endif()
    endif()

    # Add tests
    add_test(NAME SYCL_Tests COMMAND test_sycl)
    set_tests_properties(SYCL_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )


    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_sycl)
    message(STATUS "Added SYCL test executables: test_sycl")
endif()

# CUDA Tests
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CUDA core tests - use core sources + CUDA-specific sources
    add_executable(test_cuda
        ${CORE_TEST_SOURCES}
        #Kernels/cuda_kernels.cu
        #Kernels/cuda_kernels.cpp
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --allow-unsupported-compiler -Xcompiler=-Wno-deprecated-declarations")
    target_include_directories(test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_cuda PRIVATE
        "lib${PROJECT_NAME}"
        CUDA::cudart
        CUDA::cuda_driver
        curand)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --allow-unsupported-compiler -Xcompiler=-Wno-deprecated-declarations")


    if(USE_NCCL)
        target_link_libraries(test_cuda PRIVATE nccl)
    endif()

    # Add tests
    catch_discover_tests(test_cuda)


    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_cuda)
    message(STATUS "Added CUDA test executables: test_cuda")
endif()

# MPI Tests
if(USE_MPI)
    # Create MPI test executable
    add_executable(test_mpi
        mpitest.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_mpi PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        # When using MPI compiler wrapper, linking is automatic
        message(STATUS "MPI test linking handled automatically by compiler wrapper")
    else()
        # Manual MPI linking when not using wrapper
        target_link_libraries(test_mpi
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories(test_mpi
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to test executable")
    endif()

    # Ensure MPI test uses only CXX, not CUDA compiler
    set_target_properties(test_mpi PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    # Explicitly set this as a CXX-only target to avoid CUDA flag inheritance
    set_source_files_properties(mpitest.cpp PROPERTIES LANGUAGE CXX)

    # Fix C++ standard library linking issues (same as main executable)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(test_mpi PRIVATE -Wl,--whole-archive stdc++ -Wl,--no-whole-archive)
        target_link_libraries(test_mpi PRIVATE gcc_s gcc m pthread dl)
        target_link_options(test_mpi PRIVATE -Wl,--no-as-needed)
    endif()

    # Note: Since we removed custom main from mpitest.cpp, we need to use a different approach
    # for MPI tests that require MPI_Init/MPI_Finalize. For now, create basic tests that can
    # run without MPI initialization (testing compilation and basic functionality)

    # Add MPI-specific tests - these will test MPI availability and basic functionality
    add_test(NAME MPI_Tests COMMAND test_mpi)
    set_tests_properties(MPI_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Set test properties for OpenMPI 5.0.7 compatibility
    set_tests_properties(MPI_Tests PROPERTIES
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_mpi)
    message(STATUS "Added MPI test executable: test_mpi with OpenMPI 5.0.7 support")

    # Create MPI Multi-GPU test executable
    add_executable(test_mpi_multi_gpu
        mpi_multi_gpu_test.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_mpi_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_mpi_multi_gpu PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries for multi-GPU test
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        # When using MPI compiler wrapper, linking is automatic
        message(STATUS "MPI multi-GPU test linking handled automatically by compiler wrapper")
    else()
        # Manual MPI linking when not using wrapper
        target_link_libraries(test_mpi_multi_gpu
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories(test_mpi_multi_gpu
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to multi-GPU test executable")
    endif()

    # Ensure MPI multi-GPU test uses only CXX, not CUDA compiler
    set_target_properties(test_mpi_multi_gpu PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    # Explicitly set this as a CXX-only target to avoid CUDA flag inheritance
    set_source_files_properties(mpi_multi_gpu_test.cpp PROPERTIES LANGUAGE CXX)

    # Fix C++ standard library linking issues (same as main executable)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(test_mpi_multi_gpu PRIVATE -Wl,--whole-archive stdc++ -Wl,--no-whole-archive)
        target_link_libraries(test_mpi_multi_gpu PRIVATE gcc_s gcc m pthread dl)
        target_link_options(test_mpi_multi_gpu PRIVATE -Wl,--no-as-needed)
    endif()

    # Add MPI Multi-GPU tests
    add_test(NAME MPI_Multi_GPU_Tests COMMAND test_mpi_multi_gpu)

    # Create SYCL MPI test executable
    add_executable(test_sycl_mpi
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_sycl_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_sycl_mpi PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries for SYCL MPI test
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "SYCL MPI test linking handled automatically by compiler wrapper")
    else()
        target_link_libraries(test_sycl_mpi
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories(test_sycl_mpi
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to SYCL MPI test executable")
    endif()

    # Add SYCL MPI tests
    add_test(NAME SYCL_MPI_Tests COMMAND test_sycl_mpi)

    # Create Standalone MPI GPU test executable
    add_executable(test_mpi_gpu_standalone
        mpi_gpu_standalone_test.cpp
    )
    target_include_directories(test_mpi_gpu_standalone PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_mpi_gpu_standalone PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries for Standalone MPI GPU test
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "Standalone MPI GPU test linking handled automatically by compiler wrapper")
    else()
        target_link_libraries(test_mpi_gpu_standalone
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories(test_mpi_gpu_standalone
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to Standalone MPI GPU test executable")
    endif()

    # Add Standalone MPI GPU tests
    add_test(NAME Standalone_MPI_GPU_Tests COMMAND test_mpi_gpu_standalone)

    # Create Raw MPI test executable
    add_executable(test_raw_mpi
        raw_mpi_test.cpp
        sycl_mpi_comprehensive_test.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_raw_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_raw_mpi PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries for Raw MPI test
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "Raw MPI test linking handled automatically by compiler wrapper")
    else()
        target_link_libraries(test_raw_mpi
            PRIVATE
            MPI::MPI_CXX
        )
        target_include_directories(test_raw_mpi
            PRIVATE
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "Manually linked MPI libraries to Raw MPI test executable")
    endif()

    # Add Raw MPI tests
    add_test(NAME Raw_MPI_Tests COMMAND test_raw_mpi)
    set_tests_properties(MPI_Multi_GPU_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Set test properties for OpenMPI 5.0.7 compatibility
    set_tests_properties(MPI_Multi_GPU_Tests PROPERTIES
        ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_mpi_multi_gpu)
    message(STATUS "Added MPI multi-GPU test executable: test_mpi_multi_gpu")
endif()

# ========================================================================
# OpenMP Multi-GPU Tests
# ========================================================================

# Force enable OpenMP with manual flags for icpx
if(CMAKE_CXX_COMPILER MATCHES "icpx")
    message(STATUS "Using icpx - manually enabling OpenMP")

    # Create OpenMP Multi-GPU test executable
    add_executable(test_omp_multi_gpu
        omp_multi_gpu_test.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_omp_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_omp_multi_gpu PRIVATE "lib${PROJECT_NAME}")

    # Add OpenMP flags manually for icpx (use standard GCC-style flags)
    target_compile_options(test_omp_multi_gpu PRIVATE -fopenmp)
    target_link_options(test_omp_multi_gpu PRIVATE -fopenmp)

    # Add OpenMP Multi-GPU tests
    add_test(NAME OpenMP_Multi_GPU_Tests COMMAND test_omp_multi_gpu)
    set_tests_properties(OpenMP_Multi_GPU_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_omp_multi_gpu)
    message(STATUS "Added OpenMP multi-GPU test executable: test_omp_multi_gpu")

else()
    # Find OpenMP for non-icpx compilers
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - building OpenMP multi-GPU tests")

        # Create OpenMP Multi-GPU test executable
        add_executable(test_omp_multi_gpu
            omp_multi_gpu_test.cpp
            ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
        )
        target_include_directories(test_omp_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_link_libraries(test_omp_multi_gpu PRIVATE "lib${PROJECT_NAME}")

        # Link OpenMP
        target_link_libraries(test_omp_multi_gpu PRIVATE OpenMP::OpenMP_CXX)

        # Add OpenMP Multi-GPU tests
        add_test(NAME OpenMP_Multi_GPU_Tests COMMAND test_omp_multi_gpu)
        set_tests_properties(OpenMP_Multi_GPU_Tests PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )

        # Add to common list
        list(APPEND COMMON_TEST_EXECUTABLES test_omp_multi_gpu)
        message(STATUS "Added OpenMP multi-GPU test executable: test_omp_multi_gpu")

    else()
        message(STATUS "OpenMP not found - skipping OpenMP multi-GPU tests")
    endif()
endif()

# Add Simple Multi-GPU test (no OpenMP, just DeviceBuffer on multiple devices)
add_executable(test_simple_multi_gpu
    simple_multi_gpu_test.cpp
    ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
)
target_include_directories(test_simple_multi_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_simple_multi_gpu PRIVATE "lib${PROJECT_NAME}" Threads::Threads)

# Add Simple Multi-GPU tests
add_test(NAME Simple_Multi_GPU_Tests COMMAND test_simple_multi_gpu)
set_tests_properties(Simple_Multi_GPU_Tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Add to common list
list(APPEND COMMON_TEST_EXECUTABLES test_simple_multi_gpu)
message(STATUS "Added Simple multi-GPU test executable: test_simple_multi_gpu")

# Install all test executables at once
install(TARGETS ${COMMON_TEST_EXECUTABLES})

# Print summary of all test executables
message(STATUS "All test executables: ${COMMON_TEST_EXECUTABLES}")
