#pragma once

#include "ARBDException.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#define FMT_HEADER_ONLY
#include "../extern/fmt/include/fmt/format.h"

#ifndef MIN_DEBUG_LEVEL
#define MIN_DEBUG_LEVEL 0
#endif
#ifndef MAX_DEBUG_LEVEL
#define MAX_DEBUG_LEVEL 10
#endif
#ifndef STDERR_LEVEL
#define STDERR_LEVEL 5
#endif

namespace ARBD {

enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, CRITICAL = 5 };

class Logger {
  public:
	static LogLevel current_level;

	// Host-side logging methods
	static void log(LogLevel level, const SourceLocation& loc, std::string_view message) {
		if (level < current_level)
			return;

		auto now = std::chrono::system_clock::now();
		auto time_t = std::chrono::system_clock::to_time_t(now);
		auto& stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;

		stream << "[" << get_level_name(level) << "] " << loc.file_name << ":" << loc.line << " "
			   << message << std::endl;
	}

	template<typename... Args>
	static void
	log(LogLevel level, const SourceLocation& loc, std::string_view fmt_str, Args&&... args) {
		if (level < current_level)
			return;

		try {
			std::string formatted =
				fmt::format(fmt::runtime(std::string(fmt_str)), std::forward<Args>(args)...);
			log(level, loc, formatted);
		} catch (const fmt::format_error& e) {
			log(LogLevel::ERROR, loc, fmt::format("FORMAT_ERROR: {} ({})", fmt_str, e.what()));
		}
	}

	static void set_level(LogLevel level) {
		current_level = level;
	}

  private:
	static const char* get_level_name(LogLevel level) {
		switch (level) {
		case LogLevel::TRACE:
			return "TRACE";
		case LogLevel::DEBUG:
			return "DEBUG";
		case LogLevel::INFO:
			return "INFO";
		case LogLevel::WARN:
			return "WARN";
		case LogLevel::ERROR:
			return "ERROR";
		case LogLevel::CRITICAL:
			return "CRITICAL";
		default:
			return "UNKNOWN";
		}
	}
};

inline LogLevel Logger::current_level = LogLevel::INFO;

} // namespace ARBD

// Debug.h compatibility layer
#ifdef DEBUGMSG
#define Debug(x) (x)
#define DebugMsg(level, ...)                                           \
	ARBD::Logger::log(static_cast<ARBD::LogLevel>(std::min(level, 5)), \
					  ARBD::SourceLocation(),                          \
					  __VA_ARGS__)
#define DebugMessage(level, message)                                   \
	ARBD::Logger::log(static_cast<ARBD::LogLevel>(std::min(level, 5)), \
					  ARBD::SourceLocation(),                          \
					  std::string_view(message))
#else
#define Debug(x) static_cast<void>(0)
#define DebugMsg(level, ...) static_cast<void>(0)
#define DebugMessage(level, message) static_cast<void>(0)
#endif

// ============================================================================
// HOST-ONLY LOGGING MACROS (for .cpp files and host-side code)
// ============================================================================
#define LOGTRACE(...) ARBD::Logger::log(ARBD::LogLevel::TRACE, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGDEBUG(...) ARBD::Logger::log(ARBD::LogLevel::DEBUG, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGINFO(...) ARBD::Logger::log(ARBD::LogLevel::INFO, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGWARN(...) ARBD::Logger::log(ARBD::LogLevel::WARN, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGERROR(...) ARBD::Logger::log(ARBD::LogLevel::ERROR, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGCRITICAL(...) \
	ARBD::Logger::log(ARBD::LogLevel::CRITICAL, ARBD::SourceLocation(), __VA_ARGS__)

// ============================================================================
// DEVICE-ONLY LOGGING MACROS (for .cu files and device kernels)
// ============================================================================
// Simple printf-based logging for CUDA devices
// Note: Use printf-style format strings (%d, %s, %f, etc.) in device code
#define DEVICETRACE(fmt, ...) printf("[CUDA-TRACE]: " fmt "\n", ##__VA_ARGS__)
#define DEVICEDEBUG(fmt, ...) printf("[CUDA-DEBUG]: " fmt "\n", ##__VA_ARGS__)
#define DEVICEINFO(fmt, ...) printf("[CUDA-INFO]: " fmt "\n", ##__VA_ARGS__)
#define DEVICEWARN(fmt, ...) printf("[CUDA-WARN]: " fmt "\n", ##__VA_ARGS__)
#define DEVICEERROR(fmt, ...) printf("[CUDA-ERROR]: " fmt "\n", ##__VA_ARGS__)
#define DEVICECRITICAL(fmt, ...) printf("[CUDA-CRITICAL]: " fmt "\n", ##__VA_ARGS__)
