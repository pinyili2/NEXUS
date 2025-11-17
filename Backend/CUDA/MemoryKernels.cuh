#pragma once
#include <cuda_runtime.h>

namespace ARBD {
namespace CUDA {

template <typename T>
__global__ void fill_kernel(T *__restrict__ ptr, T value, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    ptr[idx] = value;
  }
}

template <typename T>
void fill_impl(void *dst, T value, size_t num_elements, void *queue,
               bool sync) {
  if (!dst || num_elements == 0)
    return;

  cudaStream_t stream = static_cast<cudaStream_t>(queue);

  if (value == T{}) { // T{} is 0 for numbers, false for bool, etc.
    size_t bytes = num_elements * sizeof(T);
    if (sync) {
      cudaMemset(dst, 0, bytes);
    } else {
      cudaMemsetAsync(dst, 0, bytes, stream);
    }
  } else {
    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fill_kernel<T><<<num_blocks, block_size, 0, stream>>>(static_cast<T *>(dst),
                                                          value, num_elements);

    if (sync) {
      cudaStreamSynchronize(stream);
    }
  }
}

} // namespace CUDA
} // namespace ARBD
