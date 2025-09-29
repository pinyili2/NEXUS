
set(LIB_SOURCES
    Resource.cpp
    #Objects/ParticleProperties.cpp
    #Configuration.cpp
    #Math/BaseGrid.cpp
)

# Conditionally add MPI specific source files to the library
if(USE_MPI)
    list(APPEND LIB_SOURCES
        MPIManager.cpp
    )
    message(STATUS "src: Adding MPI specific sources to lib${PROJECT_NAME}")
endif()

# Conditionally add SYCL specific source files to the library
# We check if either SYCL ACPP or ICPX option is enabled from the top-level CMake.
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "src: Adding SYCL specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        SYCL/SYCLManager.cpp
    )
    # If your SYCL implementation uses a different file extension for device code
    # (though typically it's .cpp for SYCL), list them here too.
endif()

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)
    message(STATUS "src: Adding CUDA specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Legacy/CUDAManager.cpp
    )
    # Ensure the CUDA language is enabled in your top-level project() call
    # when USE_CUDA is ON, so CMake knows how to compile .cu files.
    message(STATUS "src: CUDA is enabled for lib${PROJECT_NAME}. CUDA runtime linking may be handled by compiler or main executable.")
endif()

# Conditionally add Metal specific source files to the library
if(USE_METAL)
    message(STATUS "src: Adding Metal specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        METAL/METALManager.cpp
    )
    message(STATUS "src: Metal is enabled for lib${PROJECT_NAME}. Metal framework linking handled by main executable.")
endif()

# Define the library with all collected sources.
add_library("lib${PROJECT_NAME}" ${LIB_SOURCES})

# Make the library depend on Metal shaders if Metal is enabled
if(USE_METAL AND TARGET metal_shaders_main)
    add_dependencies("lib${PROJECT_NAME}" metal_shaders_main)
endif()

# --- Include Directories ---
# CMAKE_CURRENT_SOURCE_DIR in this file refers to the "Backend" directory.
# We need to include both the Backend directory and the root directory
# so that headers can be found correctly.
target_include_directories("lib${PROJECT_NAME}" PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

# --- SYCL specific linking or include directories for the library ---
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # For Intel DPC++ (icpx), linking is usually handled by -fsycl.
    message(STATUS "src: SYCL is enabled for lib${PROJECT_NAME}.")

    # Explicit linking for AdaptiveCpp runtime libraries
    if(USE_SYCL_ACPP AND CMAKE_CXX_COMPILER MATCHES "acpp")
        # Find ACPP installation directory from the compiler path
        get_filename_component(ACPP_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ACPP_INSTALL_DIR "${ACPP_BIN_DIR}" DIRECTORY)
        set(ACPP_LIB_DIR "${ACPP_INSTALL_DIR}/lib")

        # Determine the appropriate library extension for the platform
        if(APPLE)
            set(ACPP_LIB_EXT "dylib")
        else()
            set(ACPP_LIB_EXT "so")
        endif()

        # Find and link the essential ACPP runtime libraries
        find_library(ACPP_RT_LIB NAMES "acpp-rt" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(ACPP_COMMON_LIB NAMES "acpp-common" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)

        if(ACPP_RT_LIB AND ACPP_COMMON_LIB)
            # Add the ACPP library directory to linker search paths
            target_link_directories("lib${PROJECT_NAME}" PUBLIC "${ACPP_LIB_DIR}")

            target_link_libraries("lib${PROJECT_NAME}" PUBLIC
                ${ACPP_RT_LIB}
                ${ACPP_COMMON_LIB}
            )
            message(STATUS "src: Successfully linked ACPP runtime libraries:")
            message(STATUS "  - ACPP Runtime: ${ACPP_RT_LIB}")
            message(STATUS "  - ACPP Common: ${ACPP_COMMON_LIB}")
        else()
            message(WARNING "Could not find required ACPP runtime libraries in ${ACPP_LIB_DIR}")
            message(WARNING "  - ACPP_RT_LIB: ${ACPP_RT_LIB}")
            message(WARNING "  - ACPP_COMMON_LIB: ${ACPP_COMMON_LIB}")
        endif()

        # Optional LLVM/Clang libraries - try to find them but don't fail if missing
        find_library(ACPP_CLANG_LIB NAMES "acpp-clang" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
        if(ACPP_CLANG_LIB)
            target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${ACPP_CLANG_LIB})
            message(STATUS "  - ACPP Clang: ${ACPP_CLANG_LIB}")

            # Link LLVM libraries required by libacpp-clang.so
            find_library(LLVM_LIB NAMES "LLVM-17" "LLVM" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
            find_library(CLANG_CPP_LIB NAMES "clang-cpp" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)

            if(LLVM_LIB)
                target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${LLVM_LIB})
                message(STATUS "  - LLVM: ${LLVM_LIB}")
            else()
                message(WARNING "Could not find LLVM library in ${ACPP_LIB_DIR}")
            endif()

            if(CLANG_CPP_LIB)
                target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${CLANG_CPP_LIB})
                message(STATUS "  - Clang C++: ${CLANG_CPP_LIB}")
            else()
                message(WARNING "Could not find Clang C++ library in ${ACPP_LIB_DIR}")
            endif()
        endif()

        message(STATUS "Configured lib${PROJECT_NAME} to avoid C++ standard flag conflicts with AdaptiveCpp")
    endif()

    # Explicit linking for Intel DPC++ runtime libraries
    if(USE_SYCL_ICPX AND CMAKE_CXX_COMPILER MATCHES "icpx")
        # Find Intel OneAPI SYCL installation directory
        get_filename_component(ICPX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ICPX_INSTALL_DIR "${ICPX_BIN_DIR}" DIRECTORY)
        set(ICPX_LIB_DIR "${ICPX_INSTALL_DIR}/lib")

        # Find and link the essential Intel DPC++ SYCL runtime libraries
        find_library(SYCL_LIB NAMES "sycl" PATHS "${ICPX_LIB_DIR}" NO_DEFAULT_PATH)

        if(SYCL_LIB)
            # Add the Intel DPC++ library directory to linker search paths
            target_link_directories("lib${PROJECT_NAME}" PUBLIC "${ICPX_LIB_DIR}")

            target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${SYCL_LIB})
            message(STATUS "src: Successfully linked Intel DPC++ SYCL runtime library:")
            message(STATUS "  - SYCL Runtime: ${SYCL_LIB}")
        else()
            message(WARNING "Could not find required Intel DPC++ SYCL runtime library in ${ICPX_LIB_DIR}")
            message(WARNING "  - SYCL_LIB: ${SYCL_LIB}")
        endif()

        # Link OpenCL for Intel GPU support
        find_library(OPENCL_LIB NAMES "OpenCL" PATHS "${ICPX_LIB_DIR}" "/usr/lib64" "/usr/lib" NO_DEFAULT_PATH)
        if(OPENCL_LIB)
            target_link_libraries("lib${PROJECT_NAME}" PUBLIC ${OPENCL_LIB})
            message(STATUS "  - OpenCL: ${OPENCL_LIB}")
        else()
            message(WARNING "Could not find OpenCL library for Intel DPC++")
        endif()

        message(STATUS "Configured lib${PROJECT_NAME} with Intel DPC++ SYCL runtime linking")
    endif()

    # Link OpenMP for AdaptiveCpp
    if(USE_SYCL_ACPP AND OpenMP_CXX_FOUND)
        target_link_libraries("lib${PROJECT_NAME}" PUBLIC OpenMP_Custom)
        message(STATUS "src: Linked custom OpenMP to lib${PROJECT_NAME} for AdaptiveCpp backend")
    endif()
endif()

# --- CUDA specific linking for the library ---
if(USE_CUDA)
    # Set CUDA properties for .cu files
    set_source_files_properties(
        #Random/Random.cu
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Find and link CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries("lib${PROJECT_NAME}"
        PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    )

    # Set CUDA architectures (adjust for your target GPUs)
    set_target_properties("lib${PROJECT_NAME}" PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86"
    )

    # Explicitly link C++ standard library for CUDA targets
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries("lib${PROJECT_NAME}" PRIVATE stdc++ gcc_s gcc m pthread dl)
        target_link_options("lib${PROJECT_NAME}" PRIVATE -Wl,--no-as-needed)
    else()
        target_link_libraries("lib${PROJECT_NAME}" PRIVATE stdc++)
    endif()

    message(STATUS "src: CUDA configured with separable compilation and toolkit linking")
endif()

# --- Metal specific linking and configuration for the library ---
if(USE_METAL)
    # Link Metal, Foundation, and QuartzCore frameworks to the library
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND QUARTZCORE_FRAMEWORK)
        target_link_libraries("lib${PROJECT_NAME}"
            PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${QUARTZCORE_FRAMEWORK}
        )
        message(STATUS "src: Linked Metal frameworks to lib${PROJECT_NAME}")
    else()
        message(FATAL_ERROR "Required frameworks not found for lib${PROJECT_NAME}")
    endif()

    # Add metal-cpp include directories
    target_include_directories("lib${PROJECT_NAME}"
        PUBLIC
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )

    # Add Metal-specific compile definitions
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC USE_METAL)

    # Ensure ARC is enabled for Objective-C++ files
    target_compile_options("lib${PROJECT_NAME}" PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )

    message(STATUS "src: Metal is enabled for lib${PROJECT_NAME}. Framework linking and ARC configured.")
endif()

# --- MPI specific linking and configuration for the library ---
if(USE_MPI)
    # Link MPI to the library
    if(CMAKE_CXX_COMPILER MATCHES "mpi")
        # When using MPI compiler wrapper, linking is automatic
        message(STATUS "src: MPI linking handled automatically by compiler wrapper for lib${PROJECT_NAME}")
    else()
        # Manual MPI linking when not using wrapper
        target_link_libraries("lib${PROJECT_NAME}"
            PUBLIC
            MPI::MPI_CXX
        )
        target_include_directories("lib${PROJECT_NAME}"
            PUBLIC
            ${MPI_CXX_INCLUDE_DIRS}
        )
        message(STATUS "src: Manually linked MPI libraries to lib${PROJECT_NAME}")
    endif()

    # Add MPI compile definitions to the library
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC USE_MPI)

    message(STATUS "src: MPI is enabled for lib${PROJECT_NAME}. MPI linking configured.")
endif()
