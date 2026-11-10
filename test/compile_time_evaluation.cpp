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

// Constant evaluation converts through the software routines, which match the hardware conversions bit for bit except
// that F16C quiets a signaling NaN.
template <typename T>
constexpr bool float16_arithmetic(uint16_t one, uint16_t third, uint16_t tiny) {
  const T a(1.0f), b(2.0f), nan = NumTraits<T>::quiet_NaN();
  bool ok = numext::bit_cast<uint16_t>(a) == one && numext::bit_cast<uint16_t>(a / T(3)) == third;
  ok = ok && numext::bit_cast<uint16_t>(T(1e-30f)) == tiny && static_cast<float>(T(0.375)) == 0.375f;
  ok = ok && T(3) == a + b && a * b - b == T(0) && -a < a && a <= T(1) && b > a && b >= a && a != b;
  ok = ok && T(true) == a && T(2u) == b && T(std::complex<float>(2, 5)) == b && T(0.0f) == -T(0.0f);
  ok = ok && !(nan == -nan) && nan != -nan && !(nan < a) && !(nan >= a);
  ok = ok && (numext::isnan)(nan) && (numext::isinf)(NumTraits<T>::infinity()) && (numext::isfinite)(b);
  ok = ok && numext::abs(-b) == b && (numext::mini)(a, b) == a && (numext::maxi)(a, b) == b;
  ok = ok && numext::fma(a, b, a) == T(3) && numext::madd(a, b, a) == T(3) && a / Index(4) == T(0.25f);
  ok = ok && numext::nextafter(a, b) > a && numext::nextafter(a, b) < T(1.1f);
  ok = ok && internal::cast<float, T>(2.0f) == b && internal::cast<T, float>(b) == 2.0f;
  T c = a;
  c += b;
  c *= b;
  c -= a;
  c /= b;
  ++c;
  c--;
  return ok && c == T(2.5f);
}

// 1e-30 underflows half to zero.
static_assert(float16_arithmetic<half>(0x3c00, 0x3555, 0x0000));
static_assert(float16_arithmetic<bfloat16>(0x3f80, 0x3eab, 0x0da2));
// Rounding to nearest even at the subnormal and overflow boundaries of half, and of bfloat16 into the next binade.
static_assert(numext::bit_cast<uint16_t>(half(0x1p-25f)) == 0x0000);
static_assert(numext::bit_cast<uint16_t>(half(0x1.8p-25f)) == 0x0001);
static_assert(numext::bit_cast<uint16_t>(half(0x1.ffcp-15f)) == 0x0400);
static_assert(numext::bit_cast<uint16_t>(half(65519.0f)) == 0x7bff);
static_assert(numext::bit_cast<uint16_t>(half(65520.0f)) == 0x7c00);
static_assert(numext::bit_cast<uint16_t>(half(100000)) == 0x7c00);
static_assert(static_cast<float>(half(0x1p-24f)) == 0x1p-24f);
static_assert(numext::bit_cast<uint16_t>(bfloat16(0x1.ff8p0f)) == 0x4000);
static_assert(numext::bit_cast<uint16_t>(bfloat16(-NumTraits<float>::quiet_NaN())) == 0xffc0);
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
