// -*- coding: utf-8
// vim: set fileencoding=utf-8
// SPDX-License-Identifier: MPL-2.0

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NUMERICAL_DIFF_H
#define EIGEN_NUMERICAL_DIFF_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Keeps -ffast-math from folding the rounded evaluation points back into x and h, e.g. (x + h) - x
// into h. EIGEN_OPTIMIZATION_BARRIER is an asm operand constraint restricted to plain types (see
// Macros.h), so class-type scalars go unguarded.
template <typename Scalar, std::enable_if_t<std::is_floating_point<Scalar>::value, int> = 0>
EIGEN_STRONG_INLINE void numerical_diff_barrier(Scalar& x) {
  EIGEN_UNUSED_VARIABLE(x);
  EIGEN_OPTIMIZATION_BARRIER(x)
}
template <typename Scalar, std::enable_if_t<!std::is_floating_point<Scalar>::value, int> = 0>
EIGEN_STRONG_INLINE void numerical_diff_barrier(Scalar&) {}

}  // namespace internal

enum NumericalDiffMode { Forward, Central };

/**
 * This class allows you to add a method df() to your functor, which will
 * use numerical differentiation to compute an approximate of the
 * derivative for the functor. Of course, if you have an analytical form
 * for the derivative, you should rather implement df() by yourself.
 *
 * More information on
 * http://en.wikipedia.org/wiki/Numerical_differentiation
 *
 * Currently only "Forward" and "Central" scheme are implemented.
 */
template <typename Functor_, NumericalDiffMode mode = Forward>
class NumericalDiff : public Functor_ {
 public:
  typedef Functor_ Functor;
  typedef typename Functor::Scalar Scalar;
  typedef typename Functor::InputType InputType;
  typedef typename Functor::ValueType ValueType;
  typedef typename Functor::JacobianType JacobianType;

 private:
  // Fallback/Forward mode helper
  template<NumericalDiffMode M = mode, typename std::enable_if_t<M == Forward, int> = 0>
      static Scalar defaultRelTol() {
          return std::sqrt(NumTraits<Scalar>::epsilon());
      }

  // Central mode helper
  template<NumericalDiffMode M = mode, typename std::enable_if_t<M == Central, int> = 0>
      static Scalar defaultRelTol() {
          return std::cbrt(NumTraits<Scalar>::epsilon());
      }

 public:
  NumericalDiff(Scalar _rel_tol = defaultRelTol(), Scalar _abs_tol = NumTraits<Scalar>::quiet_NaN()) : Functor(), rel_tol(_rel_tol), abs_tol((std::isnan)(_abs_tol) ? _rel_tol : _abs_tol) {}
  NumericalDiff(const Functor& f, Scalar _rel_tol = defaultRelTol(), Scalar _abs_tol = NumTraits<Scalar>::quiet_NaN()) : Functor(f), rel_tol(_rel_tol), abs_tol((std::isnan)(_abs_tol) ? _rel_tol : _abs_tol) {}

  // forward constructors
  template <typename T0>
  NumericalDiff(const T0& a0) : Functor(a0), rel_tol(defaultRelTol()), abs_tol(defaultRelTol()) {}
  template <typename T0, typename T1>
  NumericalDiff(const T0& a0, const T1& a1) : Functor(a0, a1), rel_tol(defaultRelTol()), abs_tol(defaultRelTol()) {}
  template <typename T0, typename T1, typename T2>
  NumericalDiff(const T0& a0, const T1& a1, const T2& a2) : Functor(a0, a1, a2), rel_tol(defaultRelTol()), abs_tol(defaultRelTol()) {}

  enum { InputsAtCompileTime = Functor::InputsAtCompileTime, ValuesAtCompileTime = Functor::ValuesAtCompileTime };

  /**
   * Computes the Jacobian of the functor at \a _x into \a jac and returns the number of functor evaluations.
   *
   * The step along coordinate \c j is <tt>h = max(|x[j]| * rel_tol, abs_tol)</tt> with rel_tol and abs_tol selected by the user 
   * and \c epsilon the machine precision NumTraits<Scalar>::epsilon(); the difference quotient divides by the
   * representable step <tt>fl(x[j] + h) - x[j]</tt> actually applied.
   */
  int df(const InputType& _x, JacobianType& jac) const {
    using std::abs;
    using std::sqrt;
    /* Local variables */
    Scalar h;
    int nfev = 0;
    const typename InputType::Index n = _x.size();
    ValueType val1, val2;
    InputType x = _x;
    // TODO: We should do this only if the size is not already known.
    val1.resize(Functor::values());
    val2.resize(Functor::values());

    // initialization
    switch (mode) {
      case Forward:
        // compute f(x)
        Functor::operator()(x, val1);
        nfev++;
        break;
      case Central:
        // do nothing
        break;
      default:
        eigen_assert(false);
    }

    // Function Body
    for (int j = 0; j < n; ++j) {
      const Scalar x_abs = abs(x[j]);
      h = numext::maxi(x_abs * rel_tol, abs_tol);
      // The functor is evaluated at fl(x[j] + h), so divide by that representable step: the rounding
      // of x[j] + h perturbs h by up to ulp(x[j]) <= epsilon/abs_tol * h <= abs_tol * h, comparable
      // to the error of the difference quotient itself.
      Scalar x_plus = _x[j] + h;
      internal::numerical_diff_barrier(x_plus);
      h = x_plus - _x[j];
      switch (mode) {
        case Forward:
          x[j] = x_plus;
          Functor::operator()(x, val2);
          nfev++;
          x[j] = _x[j];
          jac.col(j) = (val2 - val1) / h;
          break;
        case Central: {
          x[j] = x_plus;
          Functor::operator()(x, val2);
          nfev++;
          // x[j] - h can round (a tie when x[j] < 0 has |x[j]| within h above a power of two), so
          // divide by the separation of the two evaluation points rather than by 2*h.
          Scalar x_minus = _x[j] - h;
          internal::numerical_diff_barrier(x_minus);
          x[j] = x_minus;
          Functor::operator()(x, val1);
          nfev++;
          x[j] = _x[j];
          jac.col(j) = (val2 - val1) / (x_plus - x_minus);
          break;
        }
        default:
          eigen_assert(false);
      }
    }
    return nfev;
  }

 private:
  Scalar rel_tol;
  Scalar abs_tol;

  NumericalDiff& operator=(const NumericalDiff&) = delete;
};

}  // end namespace Eigen

// vim: ai ts=4 sts=4 et sw=4
#endif  // EIGEN_NUMERICAL_DIFF_H
