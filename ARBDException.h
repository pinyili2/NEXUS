#pragma once

/**
 * @file ARBDException.h
 * @brief Advanced exception handling system for the ARBD library with source
 * location tracking and formatted messages.
 *
 * This header provides a robust exception handling system that includes:
 * - Automatic source location tracking (file, line, function)
 * - Printf-style formatting support (universal compatibility)
 * - Type-safe exception categories
 * - Convenient factory functions for common exceptions
 *
 * @example Basic Usage:
 * ```cpp
 * void your_function() {
 *    // Simplest way - just throw with a message
 *    throw ARBD::Exception(ARBD::ExceptionType::ValueError, "Something went
 * wrong");
 *
 *    // Or use the convenient macro (recommended)
 *    ARBD_Exception(ARBD::ExceptionType::ValueError, "Invalid value: %d", 42);
 * }
 * ```
 * @example Using Convenience Functions:
 * ```cpp
 * void your_function() {
 *    // Using convenience functions
 *    if (value < 0) {
 *        ARBD::throw_value_error("Expected positive value, got %d",
 * negative_value);
 *    }
 *    if (!feature_implemented) {
 *        ARBD::throw_not_implemented("Feature %s is not yet implemented",
 * feature_name.c_str());
 *    }
 * }
 * ```
 * @example Using the Macro:
 * ```cpp
 * void your_function() {
 *    // Using the macro with formatted message
 *    ARBD_Exception(ARBD::ExceptionType::ValueError, "Invalid value: %d", 42);
 * }
 * ```
 *
 * @note The exception system automatically captures:
 * - Exception type and message
 * - Source file name
 * - Line number
 * - Function name (when available)
 *
 * Available Exception Types:
 * - UnspecifiedError (0): Generic error type
 * - NotImplementedError (1): Feature or function not yet implemented
 * - ValueError (2): Invalid value or parameter
 * - DivideByZeroError (3): Mathematical division by zero
 * - CUDARuntimeError (4): CUDA-specific runtime errors
 * - SYCLRuntimeError (5): SYCL-specific runtime errors
 * - MetalRuntimeError (6): Metal-specific runtime errors
 * - FileIoError (7): General file I/O errors
 * - FileOpenError (8): Specific file opening errors
 *
 * Advanced Usage with Formatting:
 * ```cpp
 * void process_data(int value, const std::string& name) {
 *     if (value < 0) {
 *         // Using printf-style formatting with multiple arguments
 *         ARBD_Exception(ARBD::ExceptionType::ValueError,
 *             "Invalid value %d for parameter '%s'. Must be non-negative.",
 *             value, name.c_str());
 *     }
 *
 *     if (!feature_implemented) {
 *         // Using convenience function with formatting
 *         ARBD::throw_not_implemented("Feature '%s' planned for version %d.%d",
 *             feature_name.c_str(), major_version, minor_version);
 *     }
 * }
 * ```
 */
#ifndef __METAL_VERSION__
#include <cstdio>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace ARBD {

// Simple source location for universal compatibility
struct SourceLocation {
	const char* file_name;
	int line;
	const char* function_name;

	constexpr SourceLocation(const char* file = __builtin_FILE(),
							 int line_num = __builtin_LINE(),
							 const char* func = __builtin_FUNCTION())
		: file_name(file), line(line_num), function_name(func) {}
};

enum class ExceptionType {
	UnspecifiedError = 0,
	NotImplementedError = 1,
	ValueError = 2,
	DivideByZeroError = 3,
	CUDARuntimeError = 4,
	SYCLRuntimeError = 5,
	MetalRuntimeError = 6,
	FileIoError = 7,
	FileOpenError = 8,
	RuntimeError = 9
};

class Exception : public std::exception {
  private:
	std::string _error_message;
	ExceptionType _type;
	SourceLocation _location;

	static std::string type_to_str(ExceptionType type) {
		switch (type) {
		case ExceptionType::UnspecifiedError:
			return "Unspecified Error";
		case ExceptionType::NotImplementedError:
			return "Not Implemented Error";
		case ExceptionType::ValueError:
			return "Value Error";
		case ExceptionType::DivideByZeroError:
			return "Divide By Zero Error";
		case ExceptionType::CUDARuntimeError:
			return "CUDA Runtime Error";
		case ExceptionType::SYCLRuntimeError:
			return "SYCL Runtime Error";
		case ExceptionType::MetalRuntimeError:
			return "Metal Runtime Error";
		case ExceptionType::FileIoError:
			return "File IO Error";
		case ExceptionType::FileOpenError:
			return "File Open Error";
		case ExceptionType::RuntimeError:
			return "Unified Runtime Error";
		default:
			return "Unknown Error Code (" + std::to_string(static_cast<int>(type)) + ")";
		}
	};

  public:
	// Printf-style variadic template constructor
	template<typename... Args>
	Exception(ExceptionType type, const SourceLocation& location, const char* fmt, Args&&... args)
		: _type(type), _location(location) {
		std::ostringstream oss;
		oss << type_to_str(type) << ": ";

		// Use printf-style formatting for universal compatibility
		char buffer[1024];
		if constexpr (sizeof...(args) == 0) {
			std::snprintf(buffer, sizeof(buffer), "%s", fmt);
		} else {
			// Convert any std::string arguments to const char*
			std::snprintf(buffer,
						  sizeof(buffer),
						  fmt,
						  convert_to_cstr(std::forward<Args>(args))...);
		}
		oss << buffer;

		oss << " [" << _location.file_name << "(" << _location.line << ") '"
			<< _location.function_name << "']";

		_error_message = oss.str();
	}

	// Helper function to convert std::string to const char* if needed
	template<typename T>
	static const T& convert_to_cstr(const T& arg) {
		return arg;
	}

	static const char* convert_to_cstr(const std::string& str) {
		return str.c_str();
	}

	// Convenience constructor for simple string messages
	Exception(ExceptionType type,
			  const std::string& message,
			  const SourceLocation& location = SourceLocation())
		: _type(type), _location(location) {
		std::ostringstream oss;
		oss << type_to_str(type) << ": " << message << " [" << _location.file_name << "("
			<< _location.line << ") '" << _location.function_name << "']";
		_error_message = oss.str();
	}

	virtual const char* what() const noexcept override {
		return _error_message.c_str();
	}

	ExceptionType type() const noexcept {
		return _type;
	}
	const SourceLocation& where() const noexcept {
		return _location;
	}
};

// Factory functions for common exceptions
template<typename... Args>
[[noreturn]] inline void throw_not_implemented(const char* fmt, Args&&... args) {
	throw Exception(ExceptionType::NotImplementedError,
					SourceLocation(),
					fmt,
					std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_value_error(const char* fmt, Args&&... args) {
	throw Exception(ExceptionType::ValueError, SourceLocation(), fmt, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_cuda_error(const char* fmt, Args&&... args) {
	throw Exception(ExceptionType::CUDARuntimeError,
					SourceLocation(),
					fmt,
					std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_sycl_error(const char* fmt, Args&&... args) {
	throw Exception(ExceptionType::SYCLRuntimeError,
					SourceLocation(),
					fmt,
					std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_metal_error(const char* fmt, Args&&... args) {
	throw Exception(ExceptionType::MetalRuntimeError,
					SourceLocation(),
					fmt,
					std::forward<Args>(args)...);
}

// Simple string versions (no formatting)
[[noreturn]] inline void throw_not_implemented(const std::string& message) {
	throw Exception(ExceptionType::NotImplementedError, message, SourceLocation());
}

[[noreturn]] inline void throw_value_error(const std::string& message) {
	throw Exception(ExceptionType::ValueError, message, SourceLocation());
}

[[noreturn]] inline void throw_cuda_error(const std::string& message) {
	throw Exception(ExceptionType::CUDARuntimeError, message, SourceLocation());
}

[[noreturn]] inline void throw_sycl_error(const std::string& message) {
	throw Exception(ExceptionType::SYCLRuntimeError, message, SourceLocation());
}

[[noreturn]] inline void throw_metal_error(const std::string& message) {
	throw Exception(ExceptionType::MetalRuntimeError, message, SourceLocation());
}

} // namespace ARBD
#define ARBD_Exception(type, ...) throw ARBD::Exception(type, ARBD::SourceLocation(), __VA_ARGS__)
#endif // __METAL_VERSION__
