#pragma once

#ifdef USE_CUDA

#include <cstddef>
#include <cuda_runtime.h>

namespace ARBD {

struct ScaleKernel {
	__device__ void operator()(size_t i, const float* input, float* output) {
		output[i] = input[i] * 3.0f;
	}
};

struct MultiplyKernel {
	__device__ void operator()(size_t i, const float* input, float* output, float factor) {
		output[i] = input[i] * factor;
	}
};

struct SquareKernel {
	__device__ void operator()(size_t i, const float* input, float* output) {
		output[i] = input[i] * input[i];
	}
};

} // namespace ARBD

#endif
