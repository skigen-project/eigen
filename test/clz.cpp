// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 The Eigen Authors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename T>
int ref_clz(T val) {
  constexpr int kNumBits = sizeof(T) * CHAR_BIT;
  T kMsbMask = T(1) << (kNumBits - 1);
  int z = 0;
  for (; z < kNumBits && ((val & kMsbMask) == 0); ++z) {
    val <<= 1;
  }
  return z;
}

template <typename T>
int ref_ctz(T val) {
  constexpr int kNumBits = sizeof(T) * CHAR_BIT;
  T kLsbMask = T(1);
  int z = 0;
  for (; z < kNumBits && ((val & kLsbMask) == 0); ++z) {
    val >>= 1;
  }
  return z;
}

template <typename T>
int ref_popcount(T val) {
  constexpr int num_bits = sizeof(T) * CHAR_BIT;
  T lsb_mask = T(1);
  int n = 0;
  for (int i = 0; i < num_bits; ++i) {
    n += (val & lsb_mask) != 0;
    val >>= 1;
  }
  return n;
}

template <typename T>
void test_clz_ctz() {
  T step = sizeof(T) <= 2 ? T(1) : (Eigen::NumTraits<T>::highest() / (T(1) << 16));
  T iters = Eigen::NumTraits<T>::highest() / step;
  for (T i = 0; i < iters; ++i) {
    T val = i * step;
    int expected_clz = ref_clz(val);
    int actual_clz = Eigen::internal::clz(val);
    VERIFY(expected_clz == actual_clz);

    int expected_ctz = ref_ctz(val);
    int actual_ctz = Eigen::internal::ctz(val);
    VERIFY(expected_ctz == actual_ctz);

    int expected_popcount = ref_popcount(val);
    int actual_popcount = Eigen::internal::popcount(val);
    VERIFY(expected_popcount == actual_popcount);
  }

  // The loop above stops one step short of `NumTraits<T>::highest()`; check extrema
  // explicitly for `popcount`.
  VERIFY(Eigen::internal::popcount(T(0)) == 0);
  VERIFY(Eigen::internal::popcount(Eigen::NumTraits<T>::highest()) == int(sizeof(T) * CHAR_BIT));
}

template <typename T>
void test_clz_ctz_random() {
  for (int i = 0; i < 1024 * 1024; ++i) {
    T val = Eigen::internal::random<T>();
    int expected_clz = ref_clz(val);
    int actual_clz = Eigen::internal::clz(val);
    VERIFY(expected_clz == actual_clz);

    int expected_ctz = ref_ctz(val);
    int actual_ctz = Eigen::internal::ctz(val);
    VERIFY(expected_ctz == actual_ctz);

    int expected_popcount = ref_popcount(val);
    int actual_popcount = Eigen::internal::popcount(val);
    VERIFY(expected_popcount == actual_popcount);
  }
}

// The intrinsic specializations take a portable path during constant evaluation.
template <typename T>
constexpr bool clz_ctz_constexpr() {
  constexpr int kNumBits = sizeof(T) * CHAR_BIT;
  constexpr T kAllOnes = Eigen::NumTraits<T>::highest();
  static_assert(Eigen::internal::clz(T(0)) == kNumBits && Eigen::internal::ctz(T(0)) == kNumBits, "");
  static_assert(Eigen::internal::clz(T(1)) == kNumBits - 1 && Eigen::internal::ctz(T(1)) == 0, "");
  static_assert(Eigen::internal::clz(T(kAllOnes >> 1)) == 1 && Eigen::internal::ctz(T(T(1) << 3)) == 3, "");
  static_assert(Eigen::internal::popcount(T(0)) == 0 && Eigen::internal::popcount(kAllOnes) == kNumBits, "");
  static_assert(Eigen::internal::popcount(T(0x5a)) == 4, "");
  // GCC and Clang use builtins for every width, so check the portable path directly.
  using Portable = Eigen::internal::count_bits_portable<T>;
  static_assert(Portable::clz(T(0)) == kNumBits && Portable::ctz(T(0)) == kNumBits, "");
  static_assert(Portable::clz(T(1)) == kNumBits - 1 && Portable::ctz(T(1)) == 0, "");
  static_assert(Portable::clz(kAllOnes) == 0 && Portable::ctz(T(T(1) << (kNumBits - 1))) == kNumBits - 1, "");
  static_assert(Portable::clz(T(kAllOnes >> 1)) == 1 && Portable::ctz(T(T(1) << 3)) == 3, "");
  static_assert(Portable::popcount(kAllOnes) == kNumBits && Portable::popcount(T(0x5a)) == 4, "");
  // Signed types count the bits of their two's complement representation.
  using SignedPortable = Eigen::internal::count_bits_portable<std::make_signed_t<T>>;
  static_assert(SignedPortable::clz(-1) == 0 && SignedPortable::popcount(-1) == kNumBits, "");
  return true;
}
static_assert(clz_ctz_constexpr<uint8_t>(), "");
static_assert(clz_ctz_constexpr<uint16_t>(), "");
static_assert(clz_ctz_constexpr<uint32_t>(), "");
static_assert(clz_ctz_constexpr<uint64_t>(), "");

EIGEN_DECLARE_TEST(clz) {
  CALL_SUBTEST_1(test_clz_ctz<uint8_t>());
  CALL_SUBTEST_2(test_clz_ctz<uint16_t>());
  CALL_SUBTEST_3(test_clz_ctz<uint32_t>());
  CALL_SUBTEST_4(test_clz_ctz<uint64_t>());

  for (int i = 0; i < g_repeat; i++) {
    test_clz_ctz_random<uint32_t>();
    test_clz_ctz_random<uint64_t>();
  }
}
