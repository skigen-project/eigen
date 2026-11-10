// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of the special functions. The scalar paths run the same kernels as at run time, except lgamma,
// which constant evaluation computes from series (the C library is unavailable there); the references are exact
// values rounded to 21 digits.

#include <unsupported/Eigen/SpecialFunctions>

using namespace Eigen;

// The kernels call <cmath> functions such as std::exp and std::log, which are constexpr from C++26 (P1383R2).
#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L

template <typename T>
constexpr bool near(T actual, double expected) {
  const T reference = T(expected);
  const T difference = actual > reference ? actual - reference : reference - actual;
  return difference <= T(64) * NumTraits<T>::epsilon() * (reference < T(0) ? -reference : reference);
}

template <typename T>
constexpr bool special_functions() {
  bool result = true;
  result = result && near(numext::lgamma(T(0.25)), 1.28802252469807745737);
  result = result && near(numext::lgamma(T(2.5)), 0.284682870472919159632);
  result = result && near(numext::lgamma(T(-3.7)), -1.37973990496582464616);
  result = result && near(numext::lgamma(T(40)), 106.631760260643459126);
  result = result && near(numext::digamma(T(0.3)), -3.50252422220013298896);
  result = result && near(numext::digamma(T(2.5)), 0.703156640645243187226);
  result = result && near(numext::digamma(T(-1.5)), 0.703156640645243187226);
  result = result && near(numext::zeta(T(3), T(1)), 1.2020569031595942854);
  result = result && near(numext::zeta(T(2.5), T(0.5)), 6.24711063456881148529);
  result = result && near(numext::polygamma(T(1), T(2.5)), 0.490357756100234864973);
  result = result && near(numext::erf(T(0.3)), 0.328626759459127427639);
  result = result && near(numext::erf(T(-1.7)), -0.983790458590774563626);
  result = result && near(numext::erfc(T(0.3)), 0.671373240540872572361);
  result = result && near(numext::erfc(T(1.7)), 0.0162095414092254363738);
  result = result && near(numext::erfc(T(5)), 1.53745979442803485019e-12);
  result = result && near(numext::ndtri(T(0.025)), -1.95996398454005423552);
  result = result && near(numext::ndtri(T(0.6)), 0.253347103135799798798);
  result = result && near(numext::igamma(T(2), T(1.5)), 0.442174599628925427667);
  result = result && near(numext::igamma(T(0.5), T(3)), 0.985694121564570360474);
  result = result && near(numext::igammac(T(2), T(1.5)), 0.557825400371074572333);
  result = result && near(numext::igamma_der_a(T(2), T(1.5)), -0.31348863782507932431);
  result = result && near(numext::gamma_sample_der_alpha(T(2), T(1.5)), 0.936639067877217373585);
  result = result && near(numext::betainc(T(2), T(3), T(0.4)), 0.5248);
  result = result && near(numext::bessel_i0(T(0.7)), 1.12630301830680919805);
  result = result && near(numext::bessel_i0(T(3.5)), 7.37820343222547966034);
  result = result && near(numext::bessel_i0e(T(0.7)), 0.559305526507068317706);
  result = result && near(numext::bessel_i0e(T(3.5)), 0.222802438010779163313);
  result = result && near(numext::bessel_i1(T(0.7)), 0.371879677777008654743);
  result = result && near(numext::bessel_i1(T(3.5)), 6.20583492225836547362);
  result = result && near(numext::bessel_i1e(T(0.7)), 0.184669982762747324567);
  result = result && near(numext::bessel_i1e(T(3.5)), 0.187399976603049987368);
  result = result && near(numext::bessel_j0(T(0.7)), 0.881200888607405280839);
  result = result && near(numext::bessel_j0(T(3.5)), -0.380127739987263377379);
  result = result && near(numext::bessel_j1(T(0.7)), 0.328995741540058947849);
  result = result && near(numext::bessel_j1(T(3.5)), 0.137377527362327185716);
  result = result && near(numext::bessel_y0(T(0.7)), -0.190664929337395067434);
  result = result && near(numext::bessel_y0(T(3.5)), 0.189021943920826506752);
  result = result && near(numext::bessel_y1(T(0.7)), -1.10324987190763336966);
  result = result && near(numext::bessel_y1(T(3.5)), 0.410188417887511882872);
  result = result && near(numext::bessel_k0(T(0.7)), 0.66051985991510154874);
  result = result && near(numext::bessel_k0(T(3.5)), 0.0195988971703684891076);
  result = result && near(numext::bessel_k0e(T(0.7)), 1.33012365624205562011);
  result = result && near(numext::bessel_k0e(T(3.5)), 0.649026337688688428208);
  result = result && near(numext::bessel_k1(T(0.7)), 1.05028353531291795143);
  result = result && near(numext::bessel_k1(T(3.5)), 0.0222393929259238337386);
  result = result && near(numext::bessel_k1e(T(0.7)), 2.11501131284805236117);
  result = result && near(numext::bessel_k1e(T(3.5)), 0.736467548028912406465);
  // At zero the kernels must skip their large-argument branch, which divides by zero.
  result = result && numext::bessel_i0(T(0)) == T(1) && numext::bessel_i0e(T(0)) == T(1);
  result = result && numext::bessel_i1(T(0)) == T(0) && numext::bessel_i1e(T(0)) == T(0);
  // Exact values and special cases of the compile-time lgamma.
  result = result && numext::lgamma(T(1)) == T(0) && numext::lgamma(T(2)) == T(0);
  result = result && numext::lgamma(T(-2)) == NumTraits<T>::infinity();
  result = result && (numext::isnan)(numext::lgamma(NumTraits<T>::quiet_NaN()));
  return result;
}

static_assert(special_functions<float>(), "");
static_assert(special_functions<double>(), "");

#endif

int main() { return 0; }
