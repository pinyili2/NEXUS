#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "ReductionKernels.h"
#include "catch_boiler.h"
#include <numeric>
#include <vector>

using namespace ARBD;
using namespace Tests;

TEST_CASE("Reduction Kernel with WorkItem", "[backend][reduction][workitem]") {
  initialize_backend_once();

  if (!Tests::Global::backend_available) {
    SKIP("Backend not available");
  }

  Resource device(0);

  // Test data size
  const unsigned int N = 10000;
  const unsigned int block_size = 256;

  // Create test input data
  std::vector<int> input_data(N);
  std::iota(input_data.begin(), input_data.end(), 1); // 1, 2, 3, ..., N
  int expected_sum = N * (N + 1) / 2;                 // Sum of 1..N

  // Allocate device buffers
  DeviceBuffer<int> input_buf(N, device.id());
  input_buf.copy_from_host(input_data.data(), N);

  // First pass: reduce to block sums
  // simple_reduction_kernel processes one element per thread, so we need
  // ceil(N / block_size) blocks
  unsigned int num_blocks = (N + block_size - 1) / block_size;
  DeviceBuffer<int> partial_sums(num_blocks, device.id());

  KernelConfig cfg_pass1 = KernelConfig::for_1d(N, device);
  cfg_pass1.block_size.x = block_size;
  cfg_pass1.shared_memory = block_size * sizeof(int);

  // Launch first reduction pass
#ifdef USE_CUDA
  cuda_simple_reduction_kernel(device, cfg_pass1, input_buf.data(),
                               partial_sums.data(), N, block_size);
#else
  launch_kernel_with_workitem(device, cfg_pass1, simple_reduction_kernel{},
                              input_buf.data(), partial_sums.data(), N,
                              block_size);
#endif

  device.synchronize_streams();

  // Second pass: reduce block sums to single value
  DeviceBuffer<int> final_result(1, device.id());

  KernelConfig cfg_pass2 = KernelConfig::for_1d(num_blocks, device);
  cfg_pass2.block_size.x = block_size;
  cfg_pass2.shared_memory = block_size * sizeof(int);
#ifdef USE_CUDA
  cuda_simple_reduction_kernel(device, cfg_pass2, partial_sums.data(),
                               final_result.data(), num_blocks, block_size);
#else
  launch_kernel_with_workitem(device, cfg_pass2, simple_reduction_kernel{},
                              partial_sums.data(), final_result.data(),
                              num_blocks, block_size);
#endif

  device.synchronize_streams();

  // Verify result
  std::vector<int> result(1);
  final_result.copy_to_host(result.data(), 1);

  INFO("Input size: " << N);
  INFO("Expected sum: " << expected_sum);
  INFO("Computed sum: " << result[0]);

  REQUIRE(result[0] == expected_sum);
}

TEST_CASE("Reduction6 Pattern with WorkItem",
          "[backend][reduction][workitem][advanced]") {
  initialize_backend_once();

  if (!Tests::Global::backend_available) {
    SKIP("Backend not available");
  }

  Resource device(0);

  const unsigned int N = 10000;
  const unsigned int block_size = 256;

  // Create test input
  std::vector<int> input_data(N);
  std::iota(input_data.begin(), input_data.end(), 1);
  int expected_sum = N * (N + 1) / 2;

  DeviceBuffer<int> input_buf(N, device.id());
  input_buf.copy_from_host(input_data.data(), N);

  // First pass
  unsigned int num_blocks = (N + block_size * 2 - 1) / (block_size * 2);
  DeviceBuffer<int> partial_sums(num_blocks, device.id());

  KernelConfig cfg1 = KernelConfig::for_1d(N, device);
  cfg1.block_size.x = block_size;
  cfg1.grid_size.x = num_blocks;
  cfg1.shared_memory = block_size * sizeof(int);
  // grid_size parameter for kernel: total threads processing data
  unsigned int grid_size1 = num_blocks * block_size * 2;

#ifdef USE_CUDA
  cuda_reduction_kernel(device, cfg1, input_buf.data(), partial_sums.data(), N,
                        block_size, grid_size1);
#else
  launch_kernel_with_workitem(device, cfg1, reduction_kernel{},
                              input_buf.data(), partial_sums.data(), N,
                              block_size, grid_size1);
#endif

  device.synchronize_streams();

  // Second pass (single block)
  DeviceBuffer<int> final_result(1, device.id());

  KernelConfig cfg2 = KernelConfig::for_1d(num_blocks, device);
  cfg2.block_size.x = block_size;
  cfg2.grid_size.x = 1; // Single block for final reduction
  cfg2.shared_memory = block_size * sizeof(int);
  // grid_size parameter: total threads processing data (1 block * 2*block_size)
  unsigned int grid_size2 = block_size * 2;

#ifdef USE_CUDA
  cuda_reduction_kernel(device, cfg2, partial_sums.data(), final_result.data(),
                        num_blocks, block_size, grid_size2);
#else
  launch_kernel_with_workitem(device, cfg2, reduction_kernel{},
                              partial_sums.data(), final_result.data(),
                              num_blocks, block_size, grid_size2);
#endif

  device.synchronize_streams();

  // Debug: Check partial sums
  std::vector<int> partial_host(num_blocks);
  partial_sums.copy_to_host(partial_host.data(), num_blocks);
  int partial_sum_total = 0;
  for (unsigned int i = 0; i < num_blocks; i++) {
    INFO("Partial sum [" << i << "]: " << partial_host[i]);
    partial_sum_total += partial_host[i];
  }
  INFO("Sum of ALL " << num_blocks << " partials: " << partial_sum_total);

  // Verify
  std::vector<int> result(1);
  final_result.copy_to_host(result.data(), 1);

  INFO("Input size: " << N);
  INFO("Expected sum: " << expected_sum);
  INFO("Computed sum: " << result[0]);
  INFO("Num blocks: " << num_blocks);

  REQUIRE(result[0] == expected_sum);
}
