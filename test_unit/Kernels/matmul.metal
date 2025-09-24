#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(constant uint& grid_width [[buffer(0)]],
                         constant uint& grid_height [[buffer(1)]],
                         constant uint& grid_depth [[buffer(2)]],
                         device const float* A [[buffer(3)]],
                         device const float* B [[buffer(4)]],
                         device float* C [[buffer(5)]],
                         constant uint& K [[buffer(6)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    uint M = grid_height;  // from config.grid_size.y
    uint N = grid_width;   // from config.grid_size.x
    
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}