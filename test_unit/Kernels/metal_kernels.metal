#include <metal_stdlib>
using namespace metal;

// Simple vector addition kernel for testing
kernel void scalar_add(device const float* a [[buffer(3)]],
                      device const float* b [[buffer(4)]],
                      device float* result [[buffer(5)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

// Simple scalar multiplication kernel
kernel void scalar_multiply(device const float* input [[buffer(3)]],
                           device float* output [[buffer(4)]],
                           constant float& scalar [[buffer(5)]],
                           uint index [[thread_position_in_grid]]) {
    output[index] = input[index] * scalar;
}

// Simple reduction sum kernel (partial reduction)
kernel void reduce_sum(device const float* input [[buffer(3)]],
                      device float* output [[buffer(4)]],
                      uint index [[thread_position_in_grid]],
                      uint threads_per_group [[threads_per_threadgroup]]) {
    threadgroup float shared_data[256]; // Shared memory for the threadgroup
    
    uint tid = index % threads_per_group;
    uint gid = index / threads_per_group;
    
    // Load data into shared memory
    if (index < 1024) { // Assuming max 1024 elements for simplicity
        shared_data[tid] = input[index];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform reduction in shared memory
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this threadgroup
    if (tid == 0) {
        output[gid] = shared_data[0];
    }
}

// Matrix multiplication kernel (simple version)
kernel void matrix_multiply(device const float* A [[buffer(3)]],
                           device const float* B [[buffer(4)]],
                           device float* C [[buffer(5)]],
                           constant uint& grid_width [[buffer(0)]],
                           constant uint& grid_height [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    // Use grid dimensions from kernel config
    uint N = grid_width;  // Matrix dimension = grid width
    
    if (row >= N || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

// Simple fill kernel
kernel void fill_buffer(device float* output [[buffer(3)]],
                       constant float& value [[buffer(4)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = value;
} 


kernel void zero_buffer(device float* buffer [[buffer(3)]],
                       uint index [[thread_position_in_grid]]) {
    buffer[index] = 0.0f;
}

kernel void add_arrays(device const float* a [[buffer(3)]],
                      device const float* b [[buffer(4)]],
                      device float* result [[buffer(5)]],
                      uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

// Debug version that writes buffer values directly to help diagnose the issue
kernel void debug_add_arrays(device const float* a [[buffer(3)]],
                             device const float* b [[buffer(4)]],
                             device float* result [[buffer(5)]],
                             uint index [[thread_position_in_grid]]) {
    // Write debugging information - every thread writes to its index to prove execution
    result[index] = 777.0f + (float)index; // Proof that kernel executed
    
    // First thread writes additional debug info
    if (index == 0) {
        result[0] = a[0];  // First element of buffer a - this will overwrite the 777
    }
}

// Simple debug kernel that just writes a constant value
kernel void debug_write_constant(device float* result [[buffer(0)]],
                                uint index [[thread_position_in_grid]]) {
    // Write to the thread's index to test if kernel executes
    result[index] = 42.0f + index;  // Write a different value to each index
}

// Packed version: input buffer contains [a_data..., b_data...]
// Assumes packed_input has size 2*n, result has size n, and we launch n threads
kernel void add_arrays_packed(device const float* packed_input [[buffer(0)]],
                             device float* result [[buffer(1)]],
                             uint index [[thread_position_in_grid]]) {
    // We need to know n (array size). Let's pass it as the first element of packed_input
    // Layout: [n, a_0, a_1, ..., a_{n-1}, b_0, b_1, ..., b_{n-1}]
    uint n = (uint)packed_input[0];
    
    if (index >= n) return;
    
    // Data starts at index 1, first n elements are 'a', next n are 'b'
    float a_val = packed_input[1 + index];      // a[index]
    float b_val = packed_input[1 + n + index];  // b[index] 
    
    result[index] = a_val + b_val;
}
