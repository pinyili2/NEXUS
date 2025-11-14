#ifdef USE_SYCL
#include "SimpleRadixSort.h"
#include "Backend/Resource.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

namespace ARBD {

// Simple radix sort for debugging
void simple_radix_sort_pairs(const Resource &device,
                             DeviceBuffer<uint32_t> &keys,
                             DeviceBuffer<uint32_t> &payloads) {
  const uint32_t size = keys.size();
  sycl::queue &q = *static_cast<sycl::queue *>(device.get_stream());

  // Get USM pointers from DeviceBuffer
  uint32_t *keys_ptr = keys.data();
  uint32_t *payloads_ptr = payloads.data();

  // Allocate temporary arrays
  uint32_t *temp_keys = sycl::malloc_device<uint32_t>(size, q);
  uint32_t *temp_payloads = sycl::malloc_device<uint32_t>(size, q);
  uint32_t *histogram = sycl::malloc_device<uint32_t>(256, q);

  // Track which buffer is current input/output
  uint32_t *input_keys = keys_ptr;
  uint32_t *input_payloads = payloads_ptr;
  uint32_t *output_keys = temp_keys;
  uint32_t *output_payloads = temp_payloads;

  // Process 4 radix passes (8 bits each)
  for (int pass = 0; pass < 4; pass++) {
    const int shift = pass * 8;

    // Clear histogram
    q.memset(histogram, 0, 256 * sizeof(uint32_t));
    q.wait();

    // Step 1: Count occurrences of each digit
    // Capture input_keys pointer by value (captured automatically with [=])
    uint32_t *keys_for_hist = input_keys;
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
      uint32_t digit = (keys_for_hist[i] >> shift) & 0xFF;
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          a(histogram[digit]);
      a.fetch_add(1);
    });
    q.wait();

    // Step 2: Compute exclusive prefix sums on host (simple for debugging)
    std::vector<uint32_t> h_histogram(256);
    q.memcpy(h_histogram.data(), histogram, 256 * sizeof(uint32_t));
    q.wait();

    // Debug: Print histogram for all passes
    std::vector<uint32_t> h_keys_sample; // Declare outside if block
    if (pass < 4) {                      // Debug all passes
      std::cout << "Pass " << pass << " - Histogram counts (first 20 digits): ";
      uint32_t total = 0;
      for (int i = 0; i < 20 && i < 256; i++) {
        std::cout << "[" << i << "]=" << h_histogram[i] << " ";
        total += h_histogram[i];
      }
      // Count total across ALL digits, not just first 20
      uint32_t total_all = 0;
      for (int i = 0; i < 256; i++) {
        total_all += h_histogram[i];
      }
      std::cout << "\nPass " << pass << " - Total counted (first 20): " << total
                << ", Total counted (all 256): " << total_all
                << " (expected: " << size << ")" << std::endl;

      // Print original keys (first 20) - populate for all passes
      h_keys_sample.resize(20);
      q.memcpy(h_keys_sample.data(), input_keys,
               std::min(20u, size) * sizeof(uint32_t));
      q.wait();
      std::cout << "Pass " << pass << " - Original keys (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << h_keys_sample[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Original keys HEX (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << h_keys_sample[i] << std::dec << " ";
      }
      std::cout << std::endl;
    }

    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) {
      uint32_t count = h_histogram[i];
      h_histogram[i] = sum; // exclusive scan
      sum += count;
    }

    // Debug: Print exclusive prefix sums for all passes
    if (pass < 4) { // Debug all passes
      std::cout << "Pass " << pass
                << " - Exclusive prefix sums (first 20 digits): ";
      for (int i = 0; i < 20 && i < 256; i++) {
        std::cout << "[" << i << "]=" << h_histogram[i] << " ";
      }
      std::cout << std::endl;
    }

    q.memcpy(histogram, h_histogram.data(), 256 * sizeof(uint32_t));
    q.wait();

    // Step 3: Scatter keys and payloads to output arrays (stable version)
    // To ensure stability, we process keys sequentially within work-groups
    // using a barrier to ensure order.
    uint32_t *current_input_keys = input_keys;
    uint32_t *current_input_payloads = input_payloads;
    uint32_t *current_output_keys = output_keys;
    uint32_t *current_output_payloads = output_payloads;

    // Process keys sequentially to ensure stability
    // This is slower but guarantees that keys with the same digit
    // maintain their relative order from the input
    q.single_task([=]() {
      for (uint32_t i = 0; i < size; i++) {
        uint32_t key = current_input_keys[i];
        uint32_t digit = (key >> shift) & 0xFF;

        // Get position atomically - sequential processing ensures stability
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            a(histogram[digit]);
        uint32_t pos = a.fetch_add(1);

        if (pos < size) {
          current_output_keys[pos] = key;
          current_output_payloads[pos] = current_input_payloads[i];
        }
      }
    });
    q.wait();

    // Debug: Print scattered keys for all passes
    if (pass < 4) { // Debug all passes
      std::vector<uint32_t> h_output_sample(20);
      q.memcpy(h_output_sample.data(), current_output_keys,
               std::min(20u, size) * sizeof(uint32_t));
      q.wait();
      std::cout << "Pass " << pass << " - Scattered keys (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << h_output_sample[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Scattered keys HEX (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << h_output_sample[i] << std::dec << " ";
      }
      std::cout << std::endl;
      // Check if scattered keys are sorted by current byte
      std::cout << "Pass " << pass << " - Byte " << pass
                << " of scattered keys (first 20, decimal): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << ((h_output_sample[i] >> (pass * 8)) & 0xFF) << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Byte " << pass
                << " of scattered keys (first 20, hex): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << std::setfill('0') << std::setw(2)
                  << ((h_output_sample[i] >> (pass * 8)) & 0xFF) << std::dec
                  << " ";
      }
      std::cout << std::endl;
      // Check current byte of original keys for comparison
      std::cout << "Pass " << pass << " - Byte " << pass
                << " of original keys (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << ((h_keys_sample[i] >> (pass * 8)) & 0xFF) << " ";
      }
      std::cout << std::endl;
    }

    // Step 4: Swap input/output for next pass
    std::swap(input_keys, output_keys);
    std::swap(input_payloads, output_payloads);
  }

  // Copy result back if final output is in temp arrays
  if (input_keys != keys_ptr) {
    q.memcpy(keys_ptr, input_keys, size * sizeof(uint32_t));
    q.memcpy(payloads_ptr, input_payloads, size * sizeof(uint32_t));
    q.wait();
  }

  // Debug: Print final sorted keys (first 20)
  std::vector<uint32_t> h_final_sample(20);
  q.memcpy(h_final_sample.data(), keys_ptr,
           std::min(20u, size) * sizeof(uint32_t));
  q.wait();
  std::cout << "Final sorted keys (first 20): ";
  for (int i = 0; i < 20 && i < (int)size; i++) {
    std::cout << h_final_sample[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Final sorted keys HEX (first 20): ";
  for (int i = 0; i < 20 && i < (int)size; i++) {
    std::cout << std::hex << "0x" << h_final_sample[i] << std::dec << " ";
  }
  std::cout << std::endl;

  // Free memory
  sycl::free(temp_keys, q);
  sycl::free(temp_payloads, q);
  sycl::free(histogram, q);
}

} // namespace ARBD
#endif // USE_SYCL
