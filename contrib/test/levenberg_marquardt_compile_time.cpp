// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of the LevenbergMarquardt module, which cannot share a translation unit with
// NonLinearOptimization's solver of the same name.

#include <contrib/Eigen/LevenbergMarquardt>

using namespace Eigen;

// Matrix construction in a constant expression needs C++20.
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_constexpr_dynamic_alloc)

constexpr bool near(double actual, double expected, double tol) { return numext::abs(actual - expected) <= tol; }

// f(x) = (x0^2 - 4, x0 x1 - 2) with the root (2, 1) and the Jacobian {{4, 0}, {1, 2}} there.
struct RootFunctor : DenseFunctor<double> {
  constexpr RootFunctor() : DenseFunctor<double>(2, 2) {}
  constexpr int operator()(const VectorXd& x, VectorXd& f) const {
    f(0) = x(0) * x(0) - 4;
    f(1) = x(0) * x(1) - 2;
    return 0;
  }
  constexpr int df(const VectorXd& x, MatrixXd& j) const {
    j(0, 0) = 2 * x(0);
    j(0, 1) = 0;
    j(1, 0) = x(1);
    j(1, 1) = x(0);
    return 0;
  }
};

// minimize runs lmpar2 for the step; at the root R^T R = P^T J^T J P, so covar turns R into (J^T J)^-1. lmdif1
// approximates the Jacobian by numerical differences.
constexpr bool levenberg_marquardt() {
  RootFunctor f;
  VectorXd x(2);
  x << 3, 3;
  LevenbergMarquardt<RootFunctor> lm(f);
  bool ok = lm.minimize(x) != LevenbergMarquardtSpace::ImproperInputParameters && near(x(0), 2, 1e-10) &&
            near(x(1), 1, 1e-10);
  MatrixXd r = lm.matrixR();
  internal::covar(r, lm.permutation().indices());
  const MatrixXd jac{{4, 0}, {1, 2}};
  ok = ok && (r * (jac.transpose() * jac)).isIdentity(1e-8);
  x << 3, 3;
  Index nfev = 0;
  ok = ok && LevenbergMarquardt<RootFunctor>::lmdif1(f, x, &nfev) != LevenbergMarquardtSpace::ImproperInputParameters;
  return ok && near(x(0), 2, 1e-8) && near(x(1), 1, 1e-8);
}

static_assert(levenberg_marquardt());

#endif

int main() { return 0; }
