#include <metal_stdlib>
#include "philox_metal.h" // For openrand::PhiloxRNG
#include "Types/METAL/Vector3.h" // For Vector3_t
using namespace ARBD;
using namespace metal;

// Helper function for converting uint32 to float (0.0 to 1.0)
inline float int2float(uint32_t i) {
    constexpr float factor = 1.0f / 4294967296.0f; // 1.0f / 2^32
    constexpr float halffactor = 0.5f * factor;
    return static_cast<float>(i) * factor + halffactor;
}

// Uniform random number generation kernel
kernel void uniform_functor_kernel(constant float& min_val [[buffer(3)]],
                                  constant float& max_val [[buffer(4)]],
                                  constant uint64_t& base_seed [[buffer(5)]],
                                  constant uint32_t& base_ctr [[buffer(6)]],
                                  constant uint32_t& global_seed [[buffer(7)]],
                                  device float* output [[buffer(8)]],
                                  uint index [[thread_position_in_grid]]) {
    
    // Create a fresh PhiloxRNG instance with deterministic parameters
    // Each thread gets a unique counter value based on its index
    openrand::PhiloxRNG rng(base_seed, base_ctr + index, global_seed);
    
    uint32_t random_int = rng.draw();
    
    float random_float_01 = int2float(random_int);
    output[index] = min_val + random_float_01 * (max_val - min_val);
}

// Gaussian random number generation kernel
kernel void gaussian_functor_kernel(constant float& mean [[buffer(3)]],
                                   constant float& stddev [[buffer(4)]],
                                   constant uint64_t& base_seed [[buffer(5)]],
                                   constant uint32_t& base_ctr [[buffer(6)]],
                                   constant uint32_t& global_seed [[buffer(7)]],
                                   device float* output [[buffer(8)]],
                                   uint index [[thread_position_in_grid]]) {
    // Create a fresh PhiloxRNG instance with deterministic parameters
    openrand::PhiloxRNG rng(base_seed, base_ctr + index, global_seed);
    uint32_t i1 = rng.draw();
    uint32_t i2 = rng.draw();
    
    float u1 = (int2float(i1) < 1e-7f) ? 1e-7f : int2float(i1);
    float u2 = (int2float(i2) < 1e-7f) ? 1e-7f : int2float(i2);
    
    // Box-Muller transform - generate one value per thread
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * 3.1415926535f * u2;
    float gaussian_val = r * cos(theta);
    
    output[index] = mean + stddev * gaussian_val;
}

// Vector3 Gaussian random number generation kernel
kernel void gaussian_vector3_functor_kernel(constant float& mean_x [[buffer(3)]],
                                           constant float& mean_y [[buffer(4)]],
                                           constant float& mean_z [[buffer(5)]],
                                           constant float& stddev_x [[buffer(6)]],
                                           constant float& stddev_y [[buffer(7)]],
                                           constant float& stddev_z [[buffer(8)]],
                                           constant uint64_t& base_seed [[buffer(9)]],
                                           constant uint32_t& base_ctr [[buffer(10)]],
                                           constant uint32_t& global_seed [[buffer(11)]],
                                           device ARBD::Vector3_t<float>* output [[buffer(12)]],
                                           uint index [[thread_position_in_grid]]) {
    // Create a fresh PhiloxRNG instance with deterministic parameters
    openrand::PhiloxRNG rng(base_seed, base_ctr + index, global_seed);
    
    uint32_t i1 = rng.draw();
    uint32_t i2 = rng.draw();
    uint32_t i3 = rng.draw();
    uint32_t i4 = rng.draw();
    
    // Generate three Gaussian values using Box-Muller (needs 4 uniform values for 2 pairs)
    float u1_x = (int2float(i1) < 1e-7f) ? 1e-7f : int2float(i1);
    float u2_x = (int2float(i2) < 1e-7f) ? 1e-7f : int2float(i2);
    float u1_y = (int2float(i3) < 1e-7f) ? 1e-7f : int2float(i3);
    float u2_y = (int2float(i4) < 1e-7f) ? 1e-7f : int2float(i4);
    
    // Box-Muller transform for generating pairs of Gaussian values
    float r1 = sqrt(-2.0f * log(u1_x));
    float theta1 = 2.0f * 3.1415926535f * u2_x;
    float r2 = sqrt(-2.0f * log(u1_y));
    float theta2 = 2.0f * 3.1415926535f * u2_y;
    
    // Generate two independent Gaussian pairs
    float gauss1_x = r1 * cos(theta1);
    float gauss1_y = r1 * sin(theta1);
    float gauss2_x = r2 * cos(theta2);
    // float gauss2_y = r2 * sin(theta2); // Not used for z-component
    
    // Use three independent values for x, y, z components
    output[index] = ARBD::Vector3_t<float>(mean_x + stddev_x * gauss1_x,
                                           mean_y + stddev_y * gauss1_y,
                                           mean_z + stddev_z * gauss2_x);
}

// Integer uniform random number generation kernel
kernel void uniform_integer_kernel(constant int& min_val [[buffer(3)]],
                                  constant int& max_val [[buffer(4)]],
                                  constant uint64_t& base_seed [[buffer(5)]],
                                  constant uint32_t& base_ctr [[buffer(6)]],
                                  constant uint32_t& global_seed [[buffer(7)]],
                                  device int* output [[buffer(8)]],
                                  uint index [[thread_position_in_grid]]) {
    // Create a fresh PhiloxRNG instance with deterministic parameters
    openrand::PhiloxRNG rng(base_seed, base_ctr + index, global_seed);
    
    uint32_t random_int = rng.draw();
    
    // Convert to float in [0, 1) range
    float random_float_01 = int2float(random_int);
    
    // Convert to integer range [min_val, max_val]
    // Ensure we don't get values below min_val due to float precision issues
    int range = max_val - min_val + 1;
    float scaled_value = random_float_01 * range;
    
    // Use floor to ensure we don't get values above max_val
    int result = min_val + static_cast<int>(floor(scaled_value));
    
    // Clamp to ensure we stay within bounds
    if (result < min_val) result = min_val;
    if (result > max_val) result = max_val;
    
    // For debugging: ensure we never output 0
    if (result == 0) {
        result = min_val; // Force to minimum value
    }
    
    output[index] = result;
}