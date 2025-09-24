#pragma once

#if defined(USE_CUDA) && defined(USE_NVTX)
#include <nvToolsExt.h>
#include <string>
#include <string_view>
#include <array>
#include <cstdint>

namespace ARBD {
namespace CUDA {

/**
 * @brief Modern C++ wrapper for NVIDIA Tools Extension (NVTX) profiling
 * 
 * This class provides a convenient interface for adding profiling markers to CUDA applications.
 * NVTX markers help visualize application behavior in profilers like Nsight Systems and Nsight Compute.
 * 
 * Features:
 * - RAII-based range management for automatic cleanup
 * - Predefined color palette for consistent visualization
 * - Thread-safe operation
 * - Zero-overhead when NVTX is disabled
 * - Support for both C-style and C++ string types
 * 
 * @example Basic Usage:
 * ```cpp
 * // Manual range management
 * ARBD::CUDA::Profiler::push_range("Kernel Launch", ARBD::CUDA::Profiler::Color::Blue);
 * // ... CUDA operations ...
 * ARBD::CUDA::Profiler::pop_range();
 * 
 * // RAII-based automatic range management (recommended)
 * {
 *     auto range = ARBD::CUDA::Profiler::ScopedRange("Matrix Multiplication", 
 *                                                    ARBD::CUDA::Profiler::Color::Green);
 *     // ... CUDA operations ...
 * } // Range automatically ends here
 * 
 * // Mark specific events
 * ARBD::CUDA::Profiler::mark("Synchronization Point");
 * ```
 * 
 * @example Integration with Functions:
 * ```cpp
 * void my_cuda_function() {
 *     NVTX_RANGE("my_cuda_function", ARBD::CUDA::Profiler::Color::Red);
 *     // Function implementation
 * }
 * 
 * __global__ void my_kernel() {
 *     // Kernel implementation
 * }
 * 
 * void launch_kernel() {
 *     NVTX_RANGE("Kernel Launch");
 *     my_kernel<<<blocks, threads>>>();
 *     CUDA_CHECK(cudaGetLastError());
 * }
 * ```
 * 
 * @note When USE_NVTX is not defined, all operations become no-ops with zero overhead
 */
class Profiler {
public:
    /**
     * @brief Predefined color palette for consistent profiling visualization
     */
    enum class Color : uint32_t {
        Green      = 0xFF00FF00,
        Blue       = 0xFF0000FF,
        Yellow     = 0xFFFFFF00,
        Magenta    = 0xFFFF00FF,
        Cyan       = 0xFF00FFFF,
        Red        = 0xFFFF0000,
        White      = 0xFFFFFFFF,
        Orange     = 0xFFFFA500,
        Purple     = 0xFF800080,
        Pink       = 0xFFFFC0CB,
        Brown      = 0xFFA52A2A,
        Gray       = 0xFF808080
    };

    /**
     * @brief RAII wrapper for automatic NVTX range management
     * 
     * This class ensures that NVTX ranges are properly closed when the object
     * goes out of scope, preventing unmatched push/pop calls.
     * 
     * @example Usage:
     * ```cpp
     * void process_data() {
     *     auto range = ARBD::CUDA::Profiler::ScopedRange("Data Processing");
     *     
     *     // Nested ranges
     *     {
     *         auto sub_range = ARBD::CUDA::Profiler::ScopedRange("Preprocessing", 
     *                                                           ARBD::CUDA::Profiler::Color::Blue);
     *         // ... preprocessing code ...
     *     } // sub_range automatically closed
     *     
     *     // ... main processing code ...
     * } // range automatically closed
     * ```
     */
    class ScopedRange {
    public:
        /**
         * @brief Create a scoped NVTX range with message only
         * @param message Range description
         */
        explicit ScopedRange(std::string_view message) {
            push_range(message);
        }
        
        /**
         * @brief Create a scoped NVTX range with message and color
         * @param message Range description
         * @param color Range color from predefined palette
         */
        ScopedRange(std::string_view message, Color color) {
            push_range(message, color);
        }
        
        /**
         * @brief Create a scoped NVTX range with message and custom color
         * @param message Range description
         * @param color_value Custom ARGB color value
         */
        ScopedRange(std::string_view message, uint32_t color_value) {
            push_range(message, color_value);
        }
        
        /**
         * @brief Destructor automatically pops the range
         */
        ~ScopedRange() {
            pop_range();
        }
        
        // Prevent copying and moving to avoid double-pop
        ScopedRange(const ScopedRange&) = delete;
        ScopedRange& operator=(const ScopedRange&) = delete;
        ScopedRange(ScopedRange&&) = delete;
        ScopedRange& operator=(ScopedRange&&) = delete;
    };

    /**
     * @brief Push a new NVTX range with default color
     * @param message Range description
     */
    static void push_range(std::string_view message);
    
    /**
     * @brief Push a new NVTX range with specified color
     * @param message Range description
     * @param color Range color from predefined palette
     */
    static void push_range(std::string_view message, Color color);
    
    /**
     * @brief Push a new NVTX range with custom color
     * @param message Range description
     * @param color_value Custom ARGB color value
     */
    static void push_range(std::string_view message, uint32_t color_value);
    
    /**
     * @brief Pop the current NVTX range
     */
    static void pop_range();
    
    /**
     * @brief Mark a specific point in the timeline
     * @param message Event description
     */
    static void mark(std::string_view message);
    
    /**
     * @brief Mark a specific point with color
     * @param message Event description
     * @param color Event color from predefined palette
     */
    static void mark(std::string_view message, Color color);
    
    /**
     * @brief Mark a specific point with custom color
     * @param message Event description
     * @param color_value Custom ARGB color value
     */
    static void mark(std::string_view message, uint32_t color_value);

    /**
     * @brief Set name for the current thread (useful for multi-threaded applications)
     * @param name Thread name
     */
    static void name_thread(std::string_view name);
    
    /**
     * @brief Set category for subsequent ranges (helps organize profiling data)
     * @param category Category name
     */
    static void set_category(std::string_view category);

private:
    /**
     * @brief Get next color from rotation (for automatic coloring)
     * @return Color value from predefined palette
     */
    static uint32_t get_next_color();
    
    /**
     * @brief Create NVTX event attributes
     * @param message Event message
     * @param color_value Color value
     * @return Configured nvtxEventAttributes_t structure
     */
    static nvtxEventAttributes_t create_attributes(std::string_view message, uint32_t color_value);

    // Color rotation state
    static thread_local size_t color_index_;
    
    // Predefined color palette
    static constexpr std::array<uint32_t, 12> color_palette_ = {{
        static_cast<uint32_t>(Color::Green),
        static_cast<uint32_t>(Color::Blue),
        static_cast<uint32_t>(Color::Yellow),
        static_cast<uint32_t>(Color::Magenta),
        static_cast<uint32_t>(Color::Cyan),
        static_cast<uint32_t>(Color::Red),
        static_cast<uint32_t>(Color::Orange),
        static_cast<uint32_t>(Color::Purple),
        static_cast<uint32_t>(Color::Pink),
        static_cast<uint32_t>(Color::Brown),
        static_cast<uint32_t>(Color::Gray),
        static_cast<uint32_t>(Color::White)
    }};
};

} // namespace CUDA
} // namespace ARBD

// Convenience macros for common usage patterns

/**
 * @brief Create a scoped NVTX range for the current scope
 * @param name Range name (string literal or std::string)
 * @param ... Optional color parameter
 */
#define NVTX_RANGE(name, ...) \
    auto nvtx_range_##__LINE__ = ARBD::CUDA::Profiler::ScopedRange(name, ##__VA_ARGS__)

/**
 * @brief Mark a point in the timeline
 * @param name Mark description
 * @param ... Optional color parameter
 */
#define NVTX_MARK(name, ...) \
    ARBD::CUDA::Profiler::mark(name, ##__VA_ARGS__)

/**
 * @brief Create a function-scoped NVTX range using the function name
 */
#define NVTX_FUNCTION() \
    NVTX_RANGE(__FUNCTION__)

#elif defined(USE_CUDA) && !defined(USE_NVTX)
// No-op implementations when NVTX is not available
#include <string>
#include <string_view>
#include <array>
#include <cstdint>
namespace ARBD {
namespace CUDA {
class Profiler {
public:
    enum class Color : uint32_t { Green = 0, Blue = 0, Yellow = 0, Magenta = 0, Cyan = 0, Red = 0, White = 0 };
    
    class ScopedRange {
    public:
        explicit ScopedRange(std::string_view) {}
        ScopedRange(std::string_view, Color) {}
        ScopedRange(std::string_view, uint32_t) {}
    };
    
    static void push_range(std::string_view) {}
    static void push_range(std::string_view, Color) {}
    static void push_range(std::string_view, uint32_t) {}
    static void pop_range() {}
    static void mark(std::string_view) {}
    static void mark(std::string_view, Color) {}
    static void mark(std::string_view, uint32_t) {}
    static void name_thread(std::string_view) {}
    static void set_category(std::string_view) {}
};
} // namespace CUDA
} // namespace ARBD

// No-op macros
#define NVTX_RANGE(name, ...)
#define NVTX_MARK(name, ...)
#define NVTX_FUNCTION()

#endif // USE_CUDA && USE_NVTX