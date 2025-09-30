#pragma once

#ifdef USE_SYCL
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Header.h"
#include <mutex>
#include <omp.h>
#include <string_view>
#include <sycl/sycl.hpp>
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

  static sycl::device get_device_by_id(size_t device_id);
  static std::vector<sycl::device> get_all_devices();
  static size_t device_count();
  static void init_for_rank(int local_rank = 0, int ranks_per_node = 1,
                            int threads_per_rank = -1, bool verbose = false);
  static void init_for_omp_thread();
  static int get_thread_device();
  static void set_omp_device_affinity(const std::string &strategy = "block");
  static std::vector<int> get_rank_devices() { return rank_devices_; }
  static bool is_omp_enabled() { return omp_threads_ > 1; }

private:
  static std::vector<sycl::device> all_devices_;
  static bool initialized_;

  static void discover_devices();
  static std::vector<int> rank_devices_;
  static bool multi_rank_mode_;
  static int rank_id_;
  static int omp_threads_;
  static std::vector<int> thread_device_map_;
  static std::string device_affinity_strategy_;

  static void setup_omp_device_mapping();
  static std::mutex mtx_;
};

} // namespace SYCL
} // namespace ARBD
#endif // USE_SYCL
