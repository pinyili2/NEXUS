#pragma once

#ifdef USE_SYCL
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Header.h"
#include <array>
#include <chrono>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace ARBD {
inline void check_sycl_error(const sycl::exception &e, std::string_view file,
                             int line) {
  ARBD_Exception(ExceptionType::SYCLRuntimeError, "SYCL error at {}:{}: {}",
                 file, line, e.what());
}

#define SYCL_CHECK(call)                                                       \
  try {                                                                        \
    call;                                                                      \
  } catch (const sycl::exception &e) {                                         \
    check_sycl_error(e, __FILE__, __LINE__);                                   \
  }

namespace SYCL {

class Manager {
public:
  // Simplified interface - discovery only
  static void init();
  static void load_info();
  static void finalize();

  // Device access (no Device class needed)
  static sycl::device get_device_by_id(size_t device_id);
  static std::vector<sycl::device> get_all_devices();
  static size_t device_count();

private:
  static std::vector<sycl::device> all_devices_;
  static bool initialized_;

  static void discover_devices();
};

} // namespace SYCL
} // namespace ARBD
#endif // PROJECT_USES_SYCL
