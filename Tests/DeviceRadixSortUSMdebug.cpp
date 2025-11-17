#ifdef USE_SYCL
#include "Backend/Resource.h"
#include "DeviceRadixSortUSM.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

namespace ARBD {

void device_radix_sort_pairs_usm(const Resource &device, uint32_t *keys,
                                 uint32_t *payloads, uint32_t *alt_keys,
                                 uint32_t *alt_payloads,
                                 uint32_t *globalHistogram,
                                 uint32_t *passHistogram, uint32_t size) {
  // Get SYCL queue from resource
  sycl::queue &q = *static_cast<sycl::queue *>(device.get_stream());

  const uint32_t threadBlocks = (size + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
  if (threadBlocks > 10000000) {
    std::cerr << "WARNING: Launching " << threadBlocks
              << " blocks. Input size may be too large." << std::endl;
  }

  std::cout << "DeviceRadixSortUSM: size=" << size
            << ", threadBlocks=" << threadBlocks << std::endl;

  // Four radix passes
  for (uint32_t pass = 0; pass < 4; ++pass) {
    const uint32_t radixShift = pass * 8;

    // Reset histograms
    q.memset(globalHistogram, 0, DRS_RADIX * 4 * sizeof(uint32_t));
    q.memset(passHistogram, 0, DRS_RADIX * threadBlocks * sizeof(uint32_t));
    q.wait();

    uint32_t *input_keys = (pass % 2 == 0) ? keys : alt_keys;
    uint32_t *input_payloads = (pass % 2 == 0) ? payloads : alt_payloads;
    uint32_t *output_keys = (pass % 2 == 0) ? alt_keys : keys;
    uint32_t *output_payloads = (pass % 2 == 0) ? alt_payloads : payloads;

    // ========================================================================
    // Upsweep: Compute histograms per work-group
    // ========================================================================
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<uint32_t, 1> s_globalHist(DRS_RADIX * 2, h);

      h.parallel_for(
          sycl::nd_range<1>(threadBlocks * DRS_UPSWEEP_THREADS,
                            DRS_UPSWEEP_THREADS),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t group_id = item.get_group(0);
            const uint32_t block_dim = item.get_local_range(0);

            // Clear shared memory
            for (uint32_t i = local_id; i < DRS_RADIX * 2; i += block_dim)
              s_globalHist[i] = 0;
            item.barrier(sycl::access::fence_space::local_space);

            // Histogram - 256 threads per histogram
            uint32_t *s_wavesHist = &s_globalHist[(local_id / 256) * DRS_RADIX];

            if (group_id < threadBlocks - 1) {
              const uint32_t partEnd = (group_id + 1) * DRS_VEC_PART_SIZE;
              for (uint32_t i = local_id + (group_id * DRS_VEC_PART_SIZE);
                   i < partEnd; i += block_dim) {
                const uint32_t base_idx = i * 4;
                if (base_idx + 3 < size) {
                  const uint32_t k0 = input_keys[base_idx + 0];
                  const uint32_t k1 = input_keys[base_idx + 1];
                  const uint32_t k2 = input_keys[base_idx + 2];
                  const uint32_t k3 = input_keys[base_idx + 3];

                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>
                      a0(s_wavesHist[(k0 >> radixShift) & DRS_RADIX_MASK]);
                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>
                      a1(s_wavesHist[(k1 >> radixShift) & DRS_RADIX_MASK]);
                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>
                      a2(s_wavesHist[(k2 >> radixShift) & DRS_RADIX_MASK]);
                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>
                      a3(s_wavesHist[(k3 >> radixShift) & DRS_RADIX_MASK]);

                  a0.fetch_add(1);
                  a1.fetch_add(1);
                  a2.fetch_add(1);
                  a3.fetch_add(1);
                }
              }
            }

            if (group_id == threadBlocks - 1) {
              for (uint32_t i = local_id + (group_id * DRS_PART_SIZE); i < size;
                   i += block_dim) {
                const uint32_t t = input_keys[i];
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>
                    a(s_wavesHist[(t >> radixShift) & DRS_RADIX_MASK]);
                a.fetch_add(1);
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Reduce to first hist, pass out, begin prefix sum
            for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
              s_globalHist[i] += s_globalHist[i + DRS_RADIX];
              passHistogram[i * threadBlocks + group_id] = s_globalHist[i];
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Compute exclusive prefix sum within work-group
            for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
              uint32_t val = s_globalHist[i];
              auto sg = item.get_sub_group();
              s_globalHist[i] = sycl::exclusive_scan_over_group(
                  sg, val, sycl::plus<uint32_t>());
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Add to global histogram (just the count, not inclusive sum)
            for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
              uint32_t original = passHistogram[i * threadBlocks + group_id];
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  a(globalHistogram[i + (radixShift << 5)]);
              a.fetch_add(original);
            }
          });
    });
    q.wait();
    std::cout << "Pass " << pass << " upsweep completed" << std::endl;

    // Debug: Print global histogram after upsweep
    if (pass < 4) {
      std::vector<uint32_t> h_globalHist(DRS_RADIX);
      q.memcpy(h_globalHist.data(), &globalHistogram[radixShift << 5],
               DRS_RADIX * sizeof(uint32_t));
      q.wait();

      uint32_t total = 0;
      std::cout << "Pass " << pass << " - Global histogram (first 20 digits): ";
      for (int i = 0; i < 20 && i < 256; i++) {
        std::cout << "[" << i << "]=" << h_globalHist[i] << " ";
        total += h_globalHist[i];
      }
      uint32_t total_all = 0;
      for (int i = 0; i < 256; i++) {
        total_all += h_globalHist[i];
      }
      std::cout << "\nPass " << pass
                << " - Total in global histogram: " << total_all
                << " (expected: " << size << ")" << std::endl;
    }

    // Debug: Print input keys before scan
    if (pass < 4) {
      std::vector<uint32_t> h_input_sample(20);
      q.memcpy(h_input_sample.data(), input_keys,
               std::min(20u, size) * sizeof(uint32_t));
      q.wait();
      std::cout << "Pass " << pass << " - Input keys (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << h_input_sample[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Input keys HEX (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << h_input_sample[i] << std::dec << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Byte " << pass
                << " of input keys (first 20, hex): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << std::setfill('0') << std::setw(2)
                  << ((h_input_sample[i] >> radixShift) & DRS_RADIX_MASK)
                  << std::dec << " ";
      }
      std::cout << std::endl;
    }

    // ========================================================================
    // Scan: Compute prefix sums across work-groups
    // ========================================================================
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<uint32_t, 1> s_scan(DRS_SCAN_THREADS, h);

      h.parallel_for(
          sycl::nd_range<1>(DRS_RADIX * DRS_SCAN_THREADS, DRS_SCAN_THREADS),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t group_id = item.get_group(0);
            const uint32_t block_dim = item.get_local_range(0);
            auto group = item.get_group(); // ← Use work-group, not sub-group!

            uint32_t reduction = 0;
            const uint32_t partitionsEnd = threadBlocks / block_dim * block_dim;
            const uint32_t digitOffset = group_id * threadBlocks;

            uint32_t i = local_id;
            for (; i < partitionsEnd; i += block_dim) {
              uint32_t original = passHistogram[i + digitOffset];

              // Exclusive scan over the ENTIRE work-group (all 128 threads)
              uint32_t exclusive = sycl::exclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              // Also get the total sum (for reduction)
              uint32_t inclusive = sycl::inclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              passHistogram[i + digitOffset] = exclusive + reduction;

              // Use the last thread's inclusive value as the total
              uint32_t total =
                  sycl::group_broadcast(group, inclusive, block_dim - 1);
              reduction += total;
            }

            // Handle remaining elements
            if (i < threadBlocks) {
              uint32_t original = passHistogram[i + digitOffset];

              uint32_t exclusive = sycl::exclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              uint32_t inclusive = sycl::inclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              passHistogram[i + digitOffset] = exclusive + reduction;
            }
          });
    });
    q.wait();
    std::cout << "Pass " << pass << " scan completed" << std::endl;

    // Compute exclusive prefix sum of global histogram on device
    // This is needed because globalHistogram is used as base offset in
    // downsweep
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<uint32_t, 1> s_counts(DRS_RADIX, h);

      h.parallel_for(
          sycl::nd_range<1>(DRS_RADIX, DRS_RADIX), [=](sycl::nd_item<1> item) {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t global_offset = radixShift << 5;

            // Load counts into local memory
            s_counts[local_id] = globalHistogram[global_offset + local_id];
            item.barrier(sycl::access::fence_space::local_space);

            // Compute exclusive prefix sum sequentially (single thread)
            if (local_id == 0) {
              uint32_t sum = 0;
              for (uint32_t i = 0; i < DRS_RADIX; i++) {
                uint32_t count = s_counts[i];
                s_counts[i] = sum; // exclusive prefix sum
                sum += count;
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Write back to global memory
            globalHistogram[global_offset + local_id] = s_counts[local_id];
          });
    });
    q.wait();

    // Debug: Print pass histogram prefix sums after scan
    if (pass < 4 && threadBlocks <= 4) {
      std::vector<uint32_t> h_passHist(DRS_RADIX * threadBlocks);
      q.memcpy(h_passHist.data(), passHistogram,
               DRS_RADIX * threadBlocks * sizeof(uint32_t));
      q.wait();
      std::cout << "Pass " << pass
                << " - Pass histogram prefix sums (first digit, all blocks): ";
      for (uint32_t block = 0; block < threadBlocks; block++) {
        std::cout << "block[" << block << "]=" << h_passHist[block] << " ";
      }
      std::cout << std::endl;
    }

    // ========================================================================
    // Downsweep: Scatter keys and payloads
    // ========================================================================
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<uint32_t, 1> s_warpHistograms(DRS_BIN_PART_SIZE, h);
      sycl::local_accessor<uint32_t, 1> s_localHistogram(DRS_RADIX, h);

      h.parallel_for(
          sycl::nd_range<1>(threadBlocks * DRS_BIN_THREADS, DRS_BIN_THREADS),
          [=](sycl::nd_item<1> item) {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t group_id = item.get_group(0);
            auto sg = item.get_sub_group();
            const uint32_t warp_index = local_id / 32;
            const uint32_t lane_id = sg.get_local_linear_id();

            uint32_t *s_warpHist = &s_warpHistograms[warp_index * DRS_RADIX];

            // Clear shared memory
            for (uint32_t i = local_id; i < DRS_BIN_HISTS_SIZE;
                 i += DRS_BIN_THREADS)
              s_warpHistograms[i] = 0;
            item.barrier(sycl::access::fence_space::local_space);

            // Load keys
            uint32_t keys[DRS_BIN_KEYS_PER_THREAD];
            const uint32_t bin_sub_part_start =
                warp_index * DRS_BIN_SUB_PART_SIZE;
            const uint32_t bin_part_start = group_id * DRS_BIN_PART_SIZE;

            if (group_id < threadBlocks - 1) {
#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32)
                keys[i] = input_keys[t];
            }

            if (group_id == threadBlocks - 1) {
#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32)
                keys[i] = t < size ? input_keys[t] : 0xffffffff;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Simple histogram-based offset calculation
            uint16_t offsets[DRS_BIN_KEYS_PER_THREAD];
#pragma unroll
            for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
              const uint32_t digit = (keys[i] >> radixShift) & DRS_RADIX_MASK;
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group>
                  a(s_warpHist[digit]);
              offsets[i] = a.fetch_add(1);
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Exclusive prefix sum up the warp histograms
            if (local_id < DRS_RADIX) {
              uint32_t reduction = s_warpHistograms[local_id];
              const uint32_t num_warps = DRS_BIN_HISTS_SIZE / DRS_RADIX;
              for (uint32_t warp = 1; warp < num_warps; ++warp) {
                const uint32_t idx = local_id + warp * DRS_RADIX;
                const uint32_t warp_val = s_warpHistograms[idx];
                reduction += warp_val;
                s_warpHistograms[idx] = reduction - warp_val;
              }
              s_localHistogram[local_id] = reduction;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Exclusive scan: use sub-group scan for each digit (256 threads, 8
            // warps) Each sub-group handles 32 consecutive digits, but we need
            // all 256 So we'll do a simple sequential scan in shared memory
            if (local_id == 0) {
              uint32_t sum = 0;
              for (uint32_t i = 0; i < DRS_RADIX; ++i) {
                uint32_t val = s_localHistogram[i];
                s_warpHistograms[i] = sum; // exclusive
                sum += val;
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Update offsets
            if (warp_index) {
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
                const uint32_t t2 = (keys[i] >> radixShift) & DRS_RADIX_MASK;
                offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
              }
            } else {
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
                offsets[i] +=
                    s_warpHistograms[(keys[i] >> radixShift) & DRS_RADIX_MASK];
            }

            // Load threadblock reductions
            if (local_id < DRS_RADIX) {
              s_localHistogram[local_id] =
                  globalHistogram[local_id + (radixShift << 5)] +
                  passHistogram[local_id * threadBlocks + group_id] -
                  s_warpHistograms[local_id];
            }
            item.barrier(sycl::access::fence_space::local_space);

        // Scatter keys into shared memory
#pragma unroll
            for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
              s_warpHistograms[offsets[i]] = keys[i];
            item.barrier(sycl::access::fence_space::local_space);

            // Scatter keys to device memory
            uint8_t digits[DRS_BIN_KEYS_PER_THREAD];
            if (group_id < threadBlocks - 1) {
#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                digits[i] =
                    (s_warpHistograms[t] >> radixShift) & DRS_RADIX_MASK;
                output_keys[s_localHistogram[digits[i]] + t] =
                    s_warpHistograms[t];
              }
              item.barrier(sycl::access::fence_space::local_space);

          // Load payloads
#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32) {
                keys[i] = input_payloads[t];
              }

          // Scatter payloads into shared memory
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
                s_warpHistograms[offsets[i]] = keys[i];
              item.barrier(sycl::access::fence_space::local_space);

          // Scatter payloads to device
#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                output_payloads[s_localHistogram[digits[i]] + t] =
                    s_warpHistograms[t];
              }
            }

            if (group_id == threadBlocks - 1) {
              const uint32_t finalPartSize = size - bin_part_start;
#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                if (t < finalPartSize) {
                  digits[i] =
                      (s_warpHistograms[t] >> radixShift) & DRS_RADIX_MASK;
                  output_keys[s_localHistogram[digits[i]] + t] =
                      s_warpHistograms[t];
                }
              }
              item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += 32) {
                if (t < size)
                  keys[i] = input_payloads[t];
              }

#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i)
                s_warpHistograms[offsets[i]] = keys[i];
              item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                if (t < finalPartSize)
                  output_payloads[s_localHistogram[digits[i]] + t] =
                      s_warpHistograms[t];
              }
            }
          });
    });
    q.wait();
    std::cout << "Pass " << pass << " downsweep completed" << std::endl;

    // Debug: Print output keys after downsweep
    if (pass < 4) {
      std::vector<uint32_t> h_output_sample(20);
      q.memcpy(h_output_sample.data(), output_keys,
               std::min(20u, size) * sizeof(uint32_t));
      q.wait();
      std::cout << "Pass " << pass << " - Output keys (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << h_output_sample[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Output keys HEX (first 20): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << h_output_sample[i] << std::dec << " ";
      }
      std::cout << std::endl;
      std::cout << "Pass " << pass << " - Byte " << pass
                << " of output keys (first 20, hex): ";
      for (int i = 0; i < 20 && i < (int)size; i++) {
        std::cout << std::hex << "0x" << std::setfill('0') << std::setw(2)
                  << ((h_output_sample[i] >> radixShift) & DRS_RADIX_MASK)
                  << std::dec << " ";
      }
      std::cout << std::endl;

      // Check if output is sorted by current byte
      bool sorted_by_byte = true;
      for (int i = 1; i < 20 && i < (int)size; i++) {
        uint32_t byte_i = (h_output_sample[i] >> radixShift) & DRS_RADIX_MASK;
        uint32_t byte_prev =
            (h_output_sample[i - 1] >> radixShift) & DRS_RADIX_MASK;
        if (byte_prev > byte_i) {
          sorted_by_byte = false;
          std::cout << "Pass " << pass
                    << " - WARNING: Byte order violation at index " << i - 1
                    << "->" << i << ": " << byte_prev << " > " << byte_i
                    << std::endl;
          break;
        }
      }
      if (sorted_by_byte) {
        std::cout << "Pass " << pass << " - Output keys are sorted by byte "
                  << pass << std::endl;
      }
    }
  }

  // Determine final output location (after 4 passes: pass 0->alt, 1->keys,
  // 2->alt, 3->keys) So after pass 3, output is in keys
  uint32_t *final_keys = keys;
  uint32_t *final_payloads = payloads;

  // Debug: Print final sorted keys
  std::vector<uint32_t> h_final_sample(20);
  q.memcpy(h_final_sample.data(), final_keys,
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

  // Check for sorting violations in final output
  std::vector<uint32_t> h_final_all(size);
  q.memcpy(h_final_all.data(), final_keys, size * sizeof(uint32_t));
  q.wait();
  bool sorted = true;
  for (uint32_t i = 1; i < size; i++) {
    if (h_final_all[i - 1] > h_final_all[i]) {
      sorted = false;
      std::cout << "SORTING ERROR at index " << i - 1 << "->" << i << ": "
                << h_final_all[i - 1] << " (0x" << std::hex
                << h_final_all[i - 1] << std::dec << ") > " << h_final_all[i]
                << " (0x" << std::hex << h_final_all[i] << std::dec << ")"
                << std::endl;
      // Print surrounding context
      uint32_t start = (i > 5) ? i - 5 : 0;
      uint32_t end = (i + 5 < size) ? i + 5 : size;
      std::cout << "Context (indices " << start << "-" << end << "): ";
      for (uint32_t j = start; j < end; j++) {
        std::cout << h_final_all[j] << " ";
      }
      std::cout << std::endl;
      break;
    }
  }
  if (sorted) {
    std::cout << "Final keys are correctly sorted!" << std::endl;
  }
}
} // namespace ARBD
#endif // USE_SYCL
