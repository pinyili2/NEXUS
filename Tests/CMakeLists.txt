
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2/extras)
include(Catch) # contains some cmake macros that will help with discovery of tests
include(CTest)

# Common source files for core tests
set(CORE_TEST_SOURCES
    async_buffer_tests.cpp
    pinned_unified_tests.cpp
    production_buffer_tests.cpp
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

# Jacobi tests - unified configuration for MPI and non-MPI cases
if(USE_MPI)
    # MPI Jacobi test
    add_executable(test_jacobi_mpi
        Jacobi_mpi.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_jacobi_mpi PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(test_jacobi_mpi PRIVATE "lib${PROJECT_NAME}")

    # Link MPI libraries for MPI Jacobi test
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        message(STATUS "MPI linking for test_jacobi_mpi handled automatically by compiler wrapper")
    else()
        target_link_libraries(test_jacobi_mpi PRIVATE MPI::MPI_CXX)
        target_include_directories(test_jacobi_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})
        message(STATUS "Manually linked MPI libraries to test_jacobi_mpi")
    endif()

    # Add MPI test
    add_test(NAME Jacobi_MPI_Tests COMMAND test_jacobi_mpi)
    set_tests_properties(Jacobi_MPI_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_jacobi_mpi)
    message(STATUS "Added MPI Jacobi test executable: test_jacobi_mpi")
else()
    # Non-MPI Jacobi test
    add_executable(test_jacobi
        Jacobi.cpp
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
    target_include_directories(test_jacobi PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(test_jacobi PRIVATE "lib${PROJECT_NAME}")

    # Add non-MPI test
    add_test(NAME Jacobi_Tests COMMAND test_jacobi)
    set_tests_properties(Jacobi_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Add to common list
    list(APPEND COMMON_TEST_EXECUTABLES test_jacobi)
    message(STATUS "Added Jacobi test executable: test_jacobi")
endif()

# Add backend-specific linking for Jacobi tests
if(USE_MPI)
    # MPI Jacobi test backend linking
    if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
        target_link_libraries(test_jacobi_mpi PRIVATE Threads::Threads)
        if(USE_SYCL_ACPP)
            target_link_libraries(test_jacobi_mpi PRIVATE stdc++)
        endif()

        # Intel DPC++ specific linking for MPI Jacobi test
        if(USE_SYCL_ICPX AND CMAKE_CXX_COMPILER MATCHES "icpx")
            # Find Intel OneAPI SYCL installation directory
            get_filename_component(ICPX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
            get_filename_component(ICPX_INSTALL_DIR "${ICPX_BIN_DIR}" DIRECTORY)
            set(ICPX_LIB_DIR "${ICPX_INSTALL_DIR}/lib")

            # Find and link the essential Intel DPC++ SYCL runtime libraries for MPI Jacobi test
            find_library(SYCL_LIB NAMES "sycl" PATHS "${ICPX_LIB_DIR}" NO_DEFAULT_PATH)

            if(SYCL_LIB)
                target_link_directories(test_jacobi_mpi PRIVATE "${ICPX_LIB_DIR}")
                target_link_libraries(test_jacobi_mpi PRIVATE ${SYCL_LIB})
                message(STATUS "Linked Intel DPC++ SYCL runtime to MPI Jacobi test")
            endif()

            # Link OpenCL for Intel GPU support in MPI Jacobi test
            find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
            if(OPENCL_LIB)
                target_link_libraries(test_jacobi_mpi PRIVATE ${OPENCL_LIB})
                message(STATUS "Linked OpenCL to MPI Jacobi test")
            endif()
        endif()
    endif()

    # CUDA linking for MPI Jacobi test
    if(USE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(test_jacobi_mpi PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            curand)
        if(USE_NCCL)
            target_link_libraries(test_jacobi_mpi PRIVATE nccl)
        endif()
        # Add CUDA kernel implementation for MPI Jacobi test
        target_sources(test_jacobi_mpi PRIVATE JacobiKernels_inst.cu)
        message(STATUS "Linked CUDA libraries to MPI Jacobi test")
    endif()
else()
    # Non-MPI Jacobi test backend linking
    if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
        target_link_libraries(test_jacobi PRIVATE Threads::Threads)
        if(USE_SYCL_ACPP)
            target_link_libraries(test_jacobi PRIVATE stdc++)
        endif()

        # Intel DPC++ specific linking for non-MPI Jacobi test
        if(USE_SYCL_ICPX AND CMAKE_CXX_COMPILER MATCHES "icpx")
            # Find Intel OneAPI SYCL installation directory
            get_filename_component(ICPX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
            get_filename_component(ICPX_INSTALL_DIR "${ICPX_BIN_DIR}" DIRECTORY)
            set(ICPX_LIB_DIR "${ICPX_INSTALL_DIR}/lib")

            # Find and link the essential Intel DPC++ SYCL runtime libraries for non-MPI Jacobi test
            find_library(SYCL_LIB NAMES "sycl" PATHS "${ICPX_LIB_DIR}" NO_DEFAULT_PATH)

            if(SYCL_LIB)
                target_link_directories(test_jacobi PRIVATE "${ICPX_LIB_DIR}")
                target_link_libraries(test_jacobi PRIVATE ${SYCL_LIB})
                message(STATUS "Linked Intel DPC++ SYCL runtime to non-MPI Jacobi test")
            endif()

            # Link OpenCL for Intel GPU support in non-MPI Jacobi test
            find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
            if(OPENCL_LIB)
                target_link_libraries(test_jacobi PRIVATE ${OPENCL_LIB})
                message(STATUS "Linked OpenCL to non-MPI Jacobi test")
            endif()
        endif()
    endif()

    # CUDA linking for non-MPI Jacobi test
    if(USE_CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(test_jacobi PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            curand)
        if(USE_NCCL)
            target_link_libraries(test_jacobi PRIVATE nccl)
        endif()
        # Add CUDA kernel implementation for non-MPI Jacobi test
        target_sources(test_jacobi PRIVATE JacobiKernels_inst.cu)
        message(STATUS "Linked CUDA libraries to non-MPI Jacobi test")
    endif()
endif()

message(STATUS "Added backend comprehensive test executable: test_backend")

# Backend comprehensive tests - platform independent, tests all backend abstractions
if(USE_CUDA)
    add_executable(test_kernel
        ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
    )
else()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cpp)
        add_executable(test_kernel
            kernels.cpp
            ${CMAKE_SOURCE_DIR}/extern/Catch2/extras/catch_amalgamated.cpp
        )
    else()
        message(STATUS "Skipping test_kernel (kernels.cpp not found)")
    endif()
endif()

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
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/texture_tests.cpp)
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
else()
message(STATUS "Skipping texture tests (texture_tests.cpp not found)")
endif()

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
            if(TARGET test_kernel)
                target_link_directories(test_kernel PUBLIC "${ICPX_LIB_DIR}")
            endif()
            target_link_directories(test_buffer PUBLIC "${ICPX_LIB_DIR}")

            # Link SYCL runtime library to all SYCL test executables
            target_link_libraries(test_sycl PRIVATE ${SYCL_LIB})
            if(TARGET test_kernel)
                target_link_libraries(test_kernel PRIVATE ${SYCL_LIB})
            endif()
            target_link_libraries(test_buffer PRIVATE ${SYCL_LIB})

            message(STATUS "Linked Intel DPC++ SYCL runtime to test executables")
        else()
            message(WARNING "Could not find Intel DPC++ SYCL runtime library for tests in ${ICPX_LIB_DIR}")
        endif()

        # Link OpenCL for Intel GPU support in tests
        find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
        if(OPENCL_LIB)
            target_link_libraries(test_sycl PRIVATE ${OPENCL_LIB})
            if(TARGET test_kernel)
                target_link_libraries(test_kernel PRIVATE ${OPENCL_LIB})
            endif()
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

    # Add CUDA tests (keeping basic CUDA test, removing OMP-specific ones)
    add_test(NAME CUDA_Tests COMMAND test_cuda)
    set_tests_properties(CUDA_Tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    # Add to common list (only basic CUDA test, OMP tests are in unified test)
    list(APPEND COMMON_TEST_EXECUTABLES test_cuda)

endif()
