// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// main.h adds instrumentation which breaks constexpr so we do not run any tests in here,
// this is strictly compile-time.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/OrderingMethods>

#include <algorithm>
#include <array>
#include <numeric>

using namespace Eigen;

inline void error_if_not_constexpr() {}  // not constexpr
#if EIGEN_COMP_CXXVER >= 20
consteval
#else
constexpr
#endif
    void
    assert_constexpr(bool b) {
  if (!b) error_if_not_constexpr();
}

// Bit-level inspection, exact sqrt, and safe scaling need a constexpr numext::bit_cast, which GCC 10 lacks.
#if EIGEN_HAS_CONSTEXPR_BIT_CAST
// Constant evaluation rounds and scales exactly, so these match the runtime values bit for bit, signed zeros included.
template <typename T>
constexpr bool same_bits(T a, T b) {
  return numext::bit_cast<typename numext::get_integer_by_size<sizeof(T)>::unsigned_type>(a) ==
         numext::bit_cast<typename numext::get_integer_by_size<sizeof(T)>::unsigned_type>(b);
}

template <typename T>
constexpr bool rounding() {
  const T inf = NumTraits<T>::infinity();
  bool ok = numext::floor(T(-2.5)) == T(-3) && numext::ceil(T(-2.5)) == T(-2) && numext::trunc(T(-2.5)) == T(-2);
  ok = ok && numext::round(T(2.5)) == T(3) && numext::round(T(-2.5)) == T(-3);
  ok = ok && numext::rint(T(2.5)) == T(2) && numext::rint(T(3.5)) == T(4) && numext::rint(T(-2.5)) == T(-2);
  ok = ok && same_bits(numext::ceil(T(-0.5)), -T(0)) && same_bits(numext::trunc(T(-0.3)), -T(0));
  ok = ok && same_bits(numext::floor(-T(0)), -T(0)) && same_bits(numext::round(T(-0.4)), -T(0));
  ok = ok && same_bits(numext::rint(T(-0.5)), -T(0)) && same_bits(numext::floor(T(0.7)), T(0));
  ok = ok && numext::floor(inf) == inf && (numext::isnan)(numext::ceil(NumTraits<T>::quiet_NaN()));
  const T big = T(uint64_t(1) << (NumTraits<T>::digits() - 1)) + T(1);
  ok = ok && numext::floor(big) == big && numext::rint(-big) == -big;
  return ok;
}

static_assert(rounding<float>());
static_assert(rounding<double>());

template <typename T>
constexpr bool scaling() {
  const T denorm = std::numeric_limits<T>::denorm_min();
  const T inf = NumTraits<T>::infinity();
  constexpr int p = NumTraits<T>::digits();
  constexpr int min_exponent = std::numeric_limits<T>::min_exponent;
  bool ok = numext::ldexp(T(1), min_exponent - p) == denorm && numext::ldexp(T(1), min_exponent - p - 1) == T(0);
  // 1.5 and 2.5 times the smallest subnormal round to even.
  ok = ok && numext::ldexp(T(1.5), min_exponent - p) == 2 * denorm &&
       numext::ldexp(T(2.5), min_exponent - p) == 2 * denorm;
  ok = ok && same_bits(numext::ldexp(T(-1), min_exponent - p - 2), -T(0));
  ok = ok && numext::ldexp(denorm, p - min_exponent) == T(1) && numext::ldexp(T(0.75), 3) == T(6);
  ok = ok && numext::ldexp((std::numeric_limits<T>::max)(), 1) == inf && numext::ldexp(T(-1), 1 << 20) == -inf;
  int n = 0;
  ok = ok && numext::frexp(T(-6), &n) == T(-0.75) && n == 3;
  ok = ok && numext::frexp(denorm, &n) == T(0.5) && n == min_exponent - p + 1;
  T e(0);
  ok = ok && internal::pfrexp(T(-6), e) == T(-0.75) && e == T(3);
  ok = ok && internal::pfrexp(denorm, e) == T(0.5) && e == T(min_exponent - p + 1);
  ok = ok && internal::pldexp(T(0.75), T(3)) == T(6);
  return ok;
}

static_assert(scaling<float>());
static_assert(scaling<double>());
#endif  // EIGEN_HAS_CONSTEXPR_BIT_CAST
#if EIGEN_COMP_CXXVER >= 20
#if EIGEN_HAS_CONSTEXPR_BIT_CAST
#if defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
#define EIGEN_TEST_CONSTEXPR_REAL_VIEW 1
#else
#define EIGEN_TEST_CONSTEXPR_REAL_VIEW 0
#endif
#if defined(__cpp_lib_constexpr_complex) && !(EIGEN_COMP_CLANG && defined(__GLIBCXX__))
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 1
#else
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 0
#endif
#endif  // EIGEN_HAS_CONSTEXPR_BIT_CAST
#endif

// Run tests that aren't explicitly constexpr.
constexpr bool test_nc() {
  return true;
}

// Before C++20, assert_constexpr is not consteval, so only constant evaluation of test_nc() makes a failed check an
// error.
static_assert(test_nc(), "");

int main() { return !test_nc(); }
