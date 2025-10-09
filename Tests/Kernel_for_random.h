#pragma once
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Header.h"
#include <Random/RandomKernels.h>
#include <cmath>
#include <numeric>
#include <vector>

inline double calculate_correlation(const std::vector<float> &x,
                                    const std::vector<float> &y) {
  if (x.size() != y.size() || x.empty())
    return 0.0;

  double sum_x = std::accumulate(x.begin(), x.end(), 0.0);
  double sum_y = std::accumulate(y.begin(), y.end(), 0.0);
  double mean_x = sum_x / x.size();
  double mean_y = sum_y / y.size();

  double numerator = 0.0;
  double sum_sq_x = 0.0;
  double sum_sq_y = 0.0;

  for (size_t i = 0; i < x.size(); ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    numerator += dx * dy;
    sum_sq_x += dx * dx;
    sum_sq_y += dy * dy;
  }

  double denominator = std::sqrt(sum_sq_x * sum_sq_y);
  return (denominator > 1e-10) ? (numerator / denominator) : 0.0;
}

struct TransformKernel {
  HOST DEVICE void operator()(size_t i, const float *input,
                              float *output) const {
    // Transform: y = 2*x + 1
    output[i] = 2.0f * input[i] + 1.0f;
  }
};

struct CombineKernel {
  HOST DEVICE void operator()(size_t i, const float *uniform,
                              const float *gaussian, float *combined) const {
    // Simple combination: 70% uniform + 30% gaussian
    combined[i] = 0.7f * uniform[i] + 0.3f * gaussian[i];
  }
};

struct SimpleKernel {
  HOST DEVICE void operator()(size_t i, const float *input,
                              float *output) const {
    output[i] = static_cast<float>(i);
  }
};
struct SmoothingFilterKernel {
  size_t GRID_SIZE;

  HOST DEVICE void operator()(size_t i, const float *input,
                              float *output) const {
    size_t x = i % GRID_SIZE;
    size_t y = i / GRID_SIZE;

    // Simple 3x3 averaging filter
    float sum = 0.0f;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = static_cast<int>(x) + dx;
        int ny = static_cast<int>(y) + dy;

        if (nx >= 0 && nx < static_cast<int>(GRID_SIZE) && ny >= 0 &&
            ny < static_cast<int>(GRID_SIZE)) {

          size_t idx = ny * GRID_SIZE + nx;
          sum += input[idx];
          count++;
        }
      }
    }

    output[i] = (count > 0) ? sum / count : input[i];
  }
};

struct GradientCalculationKernel {
  size_t GRID_SIZE;

  HOST DEVICE void operator()(size_t i, const float *input,
                              float *output) const {
    size_t x = i % GRID_SIZE;
    size_t y = i / GRID_SIZE;

    float grad_x = 0.0f, grad_y = 0.0f;

    // Calculate finite difference gradients
    if (x > 0 && x < GRID_SIZE - 1) {
      size_t left_idx = y * GRID_SIZE + (x - 1);
      size_t right_idx = y * GRID_SIZE + (x + 1);
      grad_x = (input[right_idx] - input[left_idx]) * 0.5f;
    }

    if (y > 0 && y < GRID_SIZE - 1) {
      size_t top_idx = (y - 1) * GRID_SIZE + x;
      size_t bottom_idx = (y + 1) * GRID_SIZE + x;
      grad_y = (input[bottom_idx] - input[top_idx]) * 0.5f;
    }

    output[i] = std::sqrt(grad_x * grad_x + grad_y * grad_y);
  }
};

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template <typename T>
struct sycl::is_device_copyable<ARBD::GaussianFunctor<T>> : std::true_type {};
template <typename T>
struct sycl::is_device_copyable<ARBD::UniformFunctor<T>> : std::true_type {};
template <>
struct sycl::is_device_copyable<TransformKernel> : std::true_type {};
template <> struct sycl::is_device_copyable<CombineKernel> : std::true_type {};
template <> struct sycl::is_device_copyable<SimpleKernel> : std::true_type {};
template <>
struct sycl::is_device_copyable<SmoothingFilterKernel> : std::true_type {};
template <>
struct sycl::is_device_copyable<GradientCalculationKernel> : std::true_type {};
#endif

// Forward declarations for wrapper functions (implementations in .cu file)
namespace ARBD {

template <typename T>
Event launch_transform_kernel(const Resource &resource,
                              const KernelConfig &config,
                              const TransformKernel &func,
                              DeviceBuffer<T> &input, DeviceBuffer<T> &output);

template <typename T>
Event launch_combine_kernel(const Resource &resource,
                            const KernelConfig &config,
                            const CombineKernel &func, DeviceBuffer<T> &uniform,
                            DeviceBuffer<T> &gaussian,
                            DeviceBuffer<T> &combined);

} // namespace ARBD
