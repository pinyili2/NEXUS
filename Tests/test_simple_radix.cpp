#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include "Random/Random.h"
#include "SimpleRadixSort.h"
#include "catch_boiler.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace ARBD;
using namespace Tests;

TEST_CASE("Simple Radix Sort", "[simple_radix]") {
  initialize_backend_once();
  Resource device(0);

  SECTION("Small test") {
    const uint32_t size = 1024;
    std::vector<uint32_t> h_keys(size);
    std::vector<uint32_t> h_payloads(size);

    // Generate random keys
    ARBD::Random<ARBD::Resource> rng(device, 12345);
    ARBD::DeviceBuffer<uint32_t> d_keys_gen(size, device.id());
    rng.generate_uniform(d_keys_gen, 0u, 0xFFFFFFFFu);
    d_keys_gen.copy_to_host(h_keys.data(), size);

    // Initialize payloads
    std::iota(h_payloads.begin(), h_payloads.end(), 0);

    // Copy to device
    ARBD::DeviceBuffer<uint32_t> d_keys(size, device.id());
    ARBD::DeviceBuffer<uint32_t> d_payloads(size, device.id());
    d_keys.copy_from_host(h_keys.data(), size);
    d_payloads.copy_from_host(h_payloads.data(), size);

    // Sort
    ARBD::simple_radix_sort_pairs(device, d_keys, d_payloads);

    // Copy back
    std::vector<uint32_t> sorted_keys(size);
    std::vector<uint32_t> sorted_payloads(size);
    d_keys.copy_to_host(sorted_keys.data(), size);
    d_payloads.copy_to_host(sorted_payloads.data(), size);

    // Verify
    bool sorted = true;
    for (uint32_t i = 1; i < size; i++) {
      if (sorted_keys[i] < sorted_keys[i - 1]) {
        sorted = false;
        std::cout << "Not sorted at " << i << ": " << sorted_keys[i - 1]
                  << " > " << sorted_keys[i] << std::endl;
        break;
      }
    }
    REQUIRE(sorted);

    // Verify stability (payloads match original keys)
    for (uint32_t i = 0; i < size; i++) {
      uint32_t original_idx = sorted_payloads[i];
      REQUIRE(sorted_keys[i] == h_keys[original_idx]);
    }
  }
}
