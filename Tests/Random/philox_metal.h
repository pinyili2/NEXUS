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

#ifndef OPENRAND_PHILOX_METAL_H_
#define OPENRAND_PHILOX_METAL_H_

#include "util.h"

// Philox constants
#define PHILOX_W0 0x9E3779B9
#define PHILOX_W1 0xBB67AE85
#define PHILOX_M0 0xD2511F53
#define PHILOX_M1 0xCD9E8D57

// Metal compatibility constants
#ifdef __METAL_VERSION__
static constant float M_PI = 3.14159265358979323846f;
static constant float M_PI2 = 6.28318530717958647692f;

// Metal address space qualifiers
#define METAL_DEVICE_PTR device
#define METAL_THREAD_PTR thread
#define METAL_CONSTANT_PTR constant
#else
// For non-Metal compilation, these expand to nothing
#define METAL_DEVICE_PTR
#define METAL_THREAD_PTR
#define METAL_CONSTANT_PTR
#endif

namespace openrand {

/**
 * @class PhiloxRNG
 * @brief Philox generator with integrated random number generation methods
 * @note This is a merged version that combines the base functionality and Philox generator
 * without inheritance for Metal compatibility.
 */
class PhiloxRNG {
  public:
	using result_type = uint32_t;

	static constexpr result_type min() {
		return 0u;
	}

	static constexpr result_type max() {
		return ~((result_type)0);
	}

	/**
	 * @brief Construct a new Philox generator
	 *
	 * @note Internally, global_seed is treated in the same way as other counters,
	 * and can be treated as such depending on the application needs.
	 *
	 * @param seed 64-bit seed
	 * @param ctr 32-bit counter
	 * @param global_seed (Optional) 32-bit global seed.
	 * @param ctr1 (Optional) Another 32-bit counter exposed for advanced use.
	 */
	PhiloxRNG(uint64_t seed,
			  uint32_t ctr,
			  uint32_t global_seed = DEFAULT_GLOBAL_SEED,
			  uint32_t ctr1 = 0x12345)
		: seed_hi((uint32_t)(seed >> 32)), seed_lo((uint32_t)(seed & 0xFFFFFFFF)), ctr0(ctr),
		  ctr1(ctr1), ctr2(global_seed), _ctr(0) {}

	/**
	 * @brief Generates a 32 bit unsigned integer from a uniform distribution.
	 *
	 * This function is needed to conform to C++ engine interface
	 *
	 * @return uint32_t random number from a uniform distribution
	 */
	result_type operator()() {
		return draw<uint32_t>();
	}

	/**
	 * @brief Generates a random number from a uniform distribution between 0
	 * and 1.
	 *
	 * @note Some generators may expose a more efficient version of this function
	 * that returns multiple values at once.
	 *
	 * @tparam T Data type to be returned. Can be 32 or 64 bit integer, float.
	 * @return T random number from a uniform distribution between 0 and 1
	 */
	template<typename T = float>
	T rand() {
		if constexpr (sizeof(T) <= 4) {
			const uint32_t x = draw<uint32_t>();
			if constexpr (is_integral_v<T>)
				return static_cast<T>(x);
			else
				return u01<float, uint32_t>(x);
		} else {
			const uint64_t x = draw<uint64_t>();
			if constexpr (is_integral_v<T>)
				return static_cast<T>(x);
			else
				return u01<float, uint64_t>(x); // Use float instead of double for Metal
		}
	}

	/**
	 * @brief Generates a number from a uniform distribution between a and b.
	 *
	 * @Note For integer types, consider using @ref range for greater control.
	 *
	 * @tparam T floating point type to be returned
	 * @param low lower bound of the uniform distribution
	 * @param high upper bound of the uniform distribution
	 * @return T random number from a uniform distribution between a and b
	 */
	template<typename T = float>
	T uniform(const T low, const T high) {
// TODO: Allow 64 bit integers
#ifdef __METAL_VERSION__
		static_assert(!(is_integral_v<T> && sizeof(T) > sizeof(int32_t)),
					  "64 bit int not yet supported");
#else
		static_assert(!(is_integral_v<T> && sizeof(T) > sizeof(int32_t)),
					  "64 bit int not yet supported");
#endif

		T r = high - low;

		if constexpr (is_floating_point_v<T>) {
			return low + r * rand<T>();
		} else if constexpr (is_integral_v<T>) {
			return low + range<true, T>(r);
		}
	}

	/*
	 * @brief Fills an array with random numbers from a uniform distribution [0, 1)
	 *
	 * @Note This array is filled serially. `N` ideally should not be large.
	 */
	template<typename T = float>
	void fill_random(METAL_DEVICE_PTR T* array, const int N) {
		for (int i = 0; i < N; i++)
			array[i] = rand<T>();
	}

	/**
	 * @brief Generates a random number from a normal distribution with mean 0 and
	 * std 1.
	 *
	 * This function implements box-muller method. This method avoids branching,
	 * and therefore more efficient on GPU.
	 *
	 * @tparam T floating point type to be returned
	 * @return T random number from a normal distribution with mean 0 and std 1
	 */
	template<typename T = float>
	T randn() {
#ifdef __METAL_VERSION__
		static_assert(is_floating_point_v<T>);
		constexpr T M_PI2_local = 2.0f * M_PI;
#else
		static_assert(is_floating_point_v<T>);
		constexpr T M_PI2_local = 2 * static_cast<T>(M_PI);
#endif

		T u = rand<T>();
		T v = rand<T>();
		T r = sqrt(T(-2.0) * log(u));
		T theta = v * M_PI2_local;
		return r * cos(theta);
	}

	/**
	 * @brief More efficient version of @ref randn, returns two values at once.
	 *
	 * @tparam T floating point type to be returned
	 * @return T random number from a normal distribution with mean 0 and std 1
	 */
	template<typename T = float>
	openrand::vec2<T> randn2() {
// Implements box-muller method
#ifdef __METAL_VERSION__
		static_assert(is_floating_point_v<T>);
		constexpr T M_PI2_local = 2.0f * M_PI;
#else
		static_assert(is_floating_point_v<T>);
		constexpr T M_PI2_local = 2 * static_cast<T>(M_PI);
#endif

		T u = rand<T>();
		T v = rand<T>();
		T r = sqrt(T(-2.0) * log(u));
		T theta = v * M_PI2_local;
		return {r * cos(theta), r * sin(theta)};
	}

	/**
	 * @brief Generates a random number from a normal distribution with mean and
	 * std.
	 *
	 * @tparam T floating point type to be returned
	 * @param mean mean of the normal distribution
	 * @param std_dev standard deviation of the normal distribution
	 * @return T random number from a normal distribution with mean and std
	 */
	template<typename T = float>
	T randn(const T mean, const T std_dev) {
		return mean + randn<T>() * std_dev;
	}

	/**
	 * @brief Generates a random integer of certain range
	 *
	 * This uses the method described in [1] to generate a random integer
	 * of range [0..N)
	 *
	 * @attention if using non-biased version, please make sure that N is not
	 * too large [2]
	 *
	 * @tparam biased if true, the faster, but slightly biased variant is used.
	 * @tparam T integer type (<=32 bit) to be returned
	 *
	 * @param N A random integral of range [0..N) will be returned
	 * @return T random number from a uniform distribution between 0 and N
	 *
	 * @note [1] https://lemire.me/blog/2016/06/30/fast-random-shuffling/
	 * @note [2] If N=2^b (b<32), pr(taking the branch) = p = 1/2^(32-b). For N=2^24,
	 * this value is 1/2^8 = .4%, quite negligible. But GPU complicates this simple math.
	 * Assuming a warp size of 32, the probability of a thread taking the branch becomes
	 * 1 - (1-p)^32. For N=2^24, that value is 11.8%. For N=2^20, it's 0.8%.
	 */
	template<bool biased = true, typename T = int>
	T range(const T N) {
		uint32_t x = draw<uint32_t>();
		uint64_t res = static_cast<uint64_t>(x) * static_cast<uint64_t>(N);

		if constexpr (biased) {
			return static_cast<T>(res >> 32);
		} else {
			uint32_t leftover = static_cast<uint32_t>(res);
			if (leftover < N) {
				uint32_t threshold = -N % N;
				while (leftover < threshold) {
					x = draw<uint32_t>();
					res = static_cast<uint64_t>(x) * static_cast<uint64_t>(N);
					leftover = static_cast<uint32_t>(res);
				}
			}
			return static_cast<T>(res);
		}
	}

	/**
	 * @brief Generates a random number from a gamma distribution with shape alpha
	 * and scale b.
	 *
	 * Adapted from the following implementation:
	 * https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
	 *
	 * @tparam T floating point type to be returned
	 * @param alpha shape parameter of the gamma distribution
	 * @param b scale parameter of the gamma distribution
	 * @return T random number from a gamma distribution with shape alpha and
	 * scale b
	 */
	template<typename T = float>
	inline T gamma(T alpha, T b) {
		T d = alpha - T((1. / 3.));
		T c = T(1.) / sqrt(T(9.0) * d);
		T v, x;
		while (true) {
			do {
				x = randn<T>();
				v = T(1.0) + c * x;
			} while (v <= T(0.));
			v = v * v * v;
			T u = rand<T>();

			const T x2 = x * x;
			if (u < 1.0f - 0.0331f * x2 * x2)
				return (d * v * b);

			if (log(u) < 0.5f * x2 + d * (1.0f - v + log(v)))
				return (d * v * b);
		}
	}

	/**
	 * @brief Returns a new generator with the internal state forwarded by a given number
	 *
	 * The new generator's first output will be the n+1th output of the current
	 * generator and so on. This is O(1) operation.
	 *
	 * @param n Number of steps to move the state forward
	 * @return PhiloxRNG A new RNG object with the state moved forward by n steps
	 */
	PhiloxRNG forward_state(int n) const {
		PhiloxRNG rng = *this; // copy
		rng._ctr += n;
		return rng;
	}

	/**
	 * @brief Core Philox draw function
	 */
	template<typename T = uint32_t>
	T draw() {
		generate();

		static_assert(is_same_v<T, uint32_t> || is_same_v<T, uint64_t>);
		if constexpr (is_same_v<T, uint32_t>)
			return _out[0];
		else {
			uint64_t res = (static_cast<uint64_t>(_out[0]) << 32) | static_cast<uint64_t>(_out[1]);
			return static_cast<uint64_t>(res);
		}
	}

	openrand::uint4 draw_int4() {
		generate();
		return openrand::uint4{_out[0], _out[1], _out[2], _out[3]};
	}

	openrand::float4 draw_float4() {
		generate();
		return openrand::float4{u01<float, uint32_t>(_out[0]),
								u01<float, uint32_t>(_out[1]),
								u01<float, uint32_t>(_out[2]),
								u01<float, uint32_t>(_out[3])};
	}

  private:
	/*
	 * @brief Converts a random number integer to a floating point number between [0., 1.)
	 */
	template<typename Ftype, typename Utype>
	inline OPENRAND_DEVICE Ftype u01(const Utype in) const {
		constexpr Ftype factor = Ftype(1.) / (Ftype(~static_cast<Utype>(0)) + Ftype(1.));
		constexpr Ftype halffactor = Ftype(0.5) * factor;
		return static_cast<Ftype>(in) * factor + halffactor;
	}

	void generate() {
		uint32_t key[2] = {seed_hi, seed_lo};
		/**
		 * Philox 4x32 can take upto 4 counters. Here, one counter is part of the
		 * general API, mandatory during instantiation. One is (optional) global seed.
		 * Third one can be optionally set by user. 4th one is interanally managed.
		 *
		 * The internal counter helps to avoid forcing user to increment counter
		 * each time a number is generated.
		 */

		_out[0] = ctr0;
		_out[1] = ctr1;
		_out[2] = ctr2;
		_out[3] = _ctr;

		for (int r = 0; r < 10; r++) {
			if (r > 0) {
				key[0] += PHILOX_W0;
				key[1] += PHILOX_W1;
			}
			round(key, _out);
		}
		_ctr++;
	}

	inline OPENRAND_DEVICE uint32_t mulhilo(uint32_t L,
											uint32_t R,
											METAL_THREAD_PTR uint32_t* hip) {
		uint64_t product = static_cast<uint64_t>(L) * static_cast<uint64_t>(R);
		*hip = static_cast<uint32_t>(product >> 32);
		return static_cast<uint32_t>(product);
	}

	inline OPENRAND_DEVICE void round(METAL_THREAD_PTR const uint32_t* key,
									  METAL_THREAD_PTR uint32_t* ctr) {
		uint32_t hi0;
		uint32_t hi1;
		uint32_t lo0 = mulhilo(PHILOX_M0, ctr[0], &hi0);
		uint32_t lo1 = mulhilo(PHILOX_M1, ctr[2], &hi1);
		ctr[0] = hi1 ^ ctr[1] ^ key[0];
		ctr[1] = lo1;
		ctr[2] = hi0 ^ ctr[3] ^ key[1];
		ctr[3] = lo0;
	}

	// User provided seed and counter, constant throughout
	// the lifetime of the rng object
	const uint32_t seed_hi, seed_lo;
	const uint32_t ctr0, ctr1, ctr2;
	uint32_t _out[4];

  public:
	// internal counter to keep track of numbers generated by this instance of rng
	uint32_t _ctr;
}; // class PhiloxRNG

} // namespace openrand

#endif // OPENRAND_PHILOX_METAL_H_
