// @HEADER
// *******************************************************************************
//                                OpenRAND                                       *
//   A Performance Portable, Reproducible Random Number Generation Library       *
//                                                                               *
// Copyright (c) 2023, Michigan State University                                 *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy  *
// of this software and associated documentation files (the "Software"), to deal *
// in the Software without restriction, including without limitation the rights  *
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
// copies of the Software, and to permit persons to whom the Software is         *
// furnished to do so, subject to the following conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in    *
// all copies or substantial portions of the Software.                           *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
// SOFTWARE.                                                                     *
//********************************************************************************
// @HEADER
//Modified by Pin-Yi on 9/2/25 for metal compatibility

#ifndef OPENRAND_UTIL_H_
#define OPENRAND_UTIL_H_

// Platform-specific includes
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cmath>
#include <cstdint>
#include <type_traits>
#endif

// Device portability macro
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define OPENRAND_DEVICE __host__ __device__
#elif defined(__METAL_VERSION__)
#define OPENRAND_DEVICE
#else
#define OPENRAND_DEVICE
#endif

namespace openrand {
#ifdef __METAL_VERSION__
static constant uint32_t DEFAULT_GLOBAL_SEED = 0xAAAAAAAA;
#else
constexpr uint32_t DEFAULT_GLOBAL_SEED = 0xAAAAAAAA;
#endif

// --- Cross-Platform Math Functions ---
#ifdef __METAL_VERSION__
// In Metal, math functions are already overloaded for float, half, etc.
using metal::cos;
using metal::log;
using metal::sin;
using metal::sqrt;
#else // For C++ (CUDA/HIP/CPU)
template<typename T>
inline OPENRAND_DEVICE T sin(T x) {
    if constexpr (std::is_same_v<T, float>)
        return ::sinf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::sin(x);
}

template<typename T>
inline OPENRAND_DEVICE T cos(T x) {
    if constexpr (std::is_same_v<T, float>)
        return ::cosf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::cos(x);
}

template<typename T>
inline OPENRAND_DEVICE T log(T x) {
    if constexpr (std::is_same_v<T, float>)
        return ::logf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::log(x);
}

template<typename T>
inline OPENRAND_DEVICE T sqrt(T x) {
    if constexpr (std::is_same_v<T, float>)
        return ::sqrtf(x);
    else if constexpr (std::is_same_v<T, double>)
        return std::sqrt(x);
}
#endif

// --- Cross-Platform Vector Types ---
template<typename T> struct vec2 { T x, y; };
template<typename T> struct vec3 { T x, y, z; };
template<typename T> struct vec4 { T x, y, z, w; };

// Type aliases for convenience
using uint2 = vec2<uint32_t>;
using uint3 = vec3<uint32_t>;
using uint4 = vec4<uint32_t>;

using float2 = vec2<float>;
using float3 = vec3<float>;
using float4 = vec4<float>;

#ifndef __METAL_VERSION__
using double2 = vec2<double>;
using double3 = vec3<double>;
using double4 = vec4<double>;
#else
using double2 = vec2<float>;
using double3 = vec3<float>;
using double4 = vec4<float>;
#endif

// --- Cross-Platform Type Traits ---
#ifdef __METAL_VERSION__
template<typename T, typename U>
static constant bool is_same_v = metal::is_same<T, U>::value;

template<typename T>
static constant bool is_integral_v = metal::is_integral<T>::value;

template<typename T>
static constant bool is_floating_point_v = metal::is_floating_point<T>::value;
#else
template<typename T, typename U>
constexpr bool is_same_v = std::is_same_v<T, U>;

template<typename T>
constexpr bool is_integral_v = std::is_integral_v<T>;

template<typename T>
constexpr bool is_floating_point_v = std::is_floating_point_v<T>;
#endif


#ifndef __METAL_VERSION__
// CRTP helper: has_counter (Not portable to MSL)
template<typename T, typename = std::void_t<>>
struct has_counter : std::false_type {};

template<typename T>
struct has_counter<T, std::void_t<decltype(std::declval<T>()._ctr)>> : std::true_type {};
#endif

} // namespace openrand

#endif // OPENRAND_UTIL_H_
