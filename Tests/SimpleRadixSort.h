#pragma once
#include "Backend/Buffer.h"
#include "Backend/Resource.h"
#include <cstdint>

namespace ARBD {

// Simple radix sort for debugging - uses host-side prefix sum computation
// Accepts DeviceBuffer to ensure correct queue and memory management
void simple_radix_sort_pairs(const Resource &device,
                             DeviceBuffer<uint32_t> &keys,
                             DeviceBuffer<uint32_t> &payloads);

} // namespace ARBD
