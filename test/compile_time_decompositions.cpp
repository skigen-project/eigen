// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of the dense decompositions, split from compile_time_evaluation because instantiating them
// dominates its compile time. Like compile_time_evaluation, this test is strictly compile-time: main.h adds
// instrumentation that constant evaluation rejects.

#include <Eigen/Dense>

using namespace Eigen;

// Dynamic sizes need heap allocation, which constant evaluation allows from C++20.
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_constexpr_dynamic_alloc)
// Dynamic outer products evaluate their operands into a local buffer, and the divide-and-conquer SVD applies plane
// rotations to strided columns.

// A nonsingular test matrix with a_ij = 1/(i+j+1) + 2 delta_ij.
template <typename MatrixType>
constexpr MatrixType shifted_hilbert(Index n) {
  MatrixType a(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) a(i, j) = 1.0 / double(i + j + 1) + (i == j ? 2 : 0);
  return a;
}

// The second-difference matrix tridiag(-1, 2, -1), whose determinant is n + 1. Partial pivoting keeps its diagonal.
template <typename MatrixType>
constexpr MatrixType second_difference(Index n) {
  MatrixType a = MatrixType::Zero(n, n);
  for (Index i = 0; i < n; ++i) {
    a(i, i) = 2;
    if (i > 0) a(i, i - 1) = a(i - 1, i) = -1;
  }
  return a;
}

// Fixed sizes up to 12 take the unrolled LU; dynamic sizes above 16 take the blocked LU.
constexpr bool lu_decompositions() {
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const PartialPivLU<Matrix3d> lu(g);
  bool ok = (g * lu.solve(Vector3d::Ones())).isApprox(Vector3d::Ones());
  const Matrix<double, 5, 5> h = shifted_hilbert<Matrix<double, 5, 5>>(5);
  ok = ok && (h * h.inverse()).isIdentity(1e-12);
  const MatrixXd d = shifted_hilbert<MatrixXd>(20);
  const PartialPivLU<MatrixXd> dlu(d);
  ok = ok && (d * dlu.solve(VectorXd::Ones(20))).isApprox(VectorXd::Ones(20));
  ok = ok && numext::abs(second_difference<Matrix<double, 5, 5>>(5).determinant() - 6) < 1e-12;
  ok = ok && numext::abs(second_difference<MatrixXd>(20).determinant() - 21) < 1e-12;
  return ok;
}

static_assert(lu_decompositions());

// The closed-form inverses of fixed sizes 2, 3 and 4 with an invertibility check.
constexpr bool inverse_with_check() {
  double det = 0;
  bool invertible = false;
  const Matrix2d a2{{4, 1}, {2, 3}};
  Matrix2d i2;
  a2.computeInverseAndDetWithCheck(i2, det, invertible);
  bool ok = invertible && numext::abs(det - 10) < 1e-12 && (a2 * i2).isIdentity(1e-12);
  const Matrix3d a3{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  Matrix3d i3;
  a3.computeInverseAndDetWithCheck(i3, det, invertible);
  ok = ok && invertible && numext::abs(det + 34) < 1e-12 && (a3 * i3).isIdentity(1e-12);
  const Matrix4d a4 = second_difference<Matrix4d>(4);
  Matrix4d i4;
  a4.computeInverseAndDetWithCheck(i4, det, invertible);
  ok = ok && invertible && numext::abs(det - 5) < 1e-12 && (a4 * i4).isIdentity(1e-12);
  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  singular.computeInverseWithCheck(i3, invertible);
  return ok && !invertible;
}

static_assert(inverse_with_check());

constexpr bool householder_qr() {
  // The non-const lvalue selects the in-place constructor, which must keep the copied input.
  Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const HouseholderQR<Matrix3d> qr(g);
  const Matrix3d q = qr.householderQ();
  bool ok = (q * q.transpose()).isIdentity(1e-12) && (g * qr.solve(Vector3d::Ones())).isApprox(Vector3d::Ones());
  MatrixXd d = shifted_hilbert<MatrixXd>(6);
  const HouseholderQR<MatrixXd> dqr(d);
  ok = ok && (d * dqr.solve(VectorXd::Ones(6))).isApprox(VectorXd::Ones(6));
  return ok && numext::abs(qr.absDeterminant() - 34) < 1e-12;
}

static_assert(householder_qr());

// The determinant accessors and the wrapper classes of the complete orthogonal decompositions. Constant evaluation
// takes the unblocked path of RandColPivHouseholderQR regardless of the block size.
constexpr bool decomposition_wrappers() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const double det = 44;
  const LLT<Matrix3d> llt(a);
  bool ok = numext::abs(llt.determinant() - det) < 1e-12 && numext::abs(llt.absDeterminant() - det) < 1e-12 &&
            llt.signDeterminant() == 1;
  const LDLT<Matrix3d> ldlt(a);
  ok = ok && numext::abs(ldlt.determinant() - det) < 1e-12 && numext::abs(ldlt.absDeterminant() - det) < 1e-12 &&
       ldlt.signDeterminant() == 1;
  const Vector3d b(1, 2, 3);
  ok = ok && (a.transpose() * ldlt.transpose().solve(b)).isApprox(b) && (a * ldlt.adjoint().solve(b)).isApprox(b);
  // The size constructor preallocates the workspace of the blocked factorization.
  LDLT<MatrixXd> dynamic_ldlt(3);
  dynamic_ldlt.compute(a);
  ok = ok && numext::abs(dynamic_ldlt.determinant() - det) < 1e-12;
  const Matrix3d na = -a;
  const BunchKaufman<Matrix3d> bk(na);
  ok = ok && numext::abs(bk.determinant() + det) < 1e-12 && numext::abs(bk.absDeterminant() - det) < 1e-12 &&
       bk.signDeterminant() == -1;
  const FullPivLU<Matrix3d> lu(na);
  ok = ok && numext::abs(lu.absDeterminant() - det) < 1e-12 && lu.signDeterminant() == -1 && lu.rank() == 3;
  const FullPivHouseholderQR<Matrix3d> fqr(a);
  ok = ok && numext::abs(fqr.absDeterminant() - det) < 1e-12 && fqr.rank() == 3 && (a * fqr.solve(b)).isApprox(b);
  ok = ok && numext::abs(ColPivHouseholderQR<Matrix3d>(a).absDeterminant() - det) < 1e-12;

  Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  CompleteOrthogonalDecomposition<Matrix3d> cod;
  cod.setThreshold(Default).compute(singular);
  const Matrix3d pinv = cod.pseudoInverse();
  ok = ok && cod.rank() == 2 && (singular * pinv * singular).isApprox(singular);
  // The non-const lvalue selects the in-place constructors.
  const CompleteOrthogonalDecomposition<Matrix3d> cod_inplace(singular);
  ok = ok && cod_inplace.rank() == 2;
  ok = ok && FullPivHouseholderQR<Matrix3d>(singular).rank() == 2 && FullPivLU<Matrix3d>(singular).rank() == 2;

  RandColPivHouseholderQR<Matrix3d> rqr;
  rqr.setSeed(1).setBlockSize(1).compute(singular);
  ok = ok && rqr.rank() == 2 &&
       (rqr.householderQ() * rqr.matrixR().triangularView<Upper>().toDenseMatrix() * rqr.colsPermutation().transpose())
           .isApprox(singular);
  RandColPivHouseholderQR<Matrix3d> rqr_inplace(singular);
  ok = ok && rqr_inplace.rank() == 2;
  const RandCompleteOrthogonalDecomposition<Matrix3d> rcod = singular.randCompleteOrthogonalDecomposition();
  const Matrix3d rpinv = rcod.pseudoInverse();
  ok = ok && rcod.rank() == 2 && (singular * rpinv * singular).isApprox(singular);
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  ok = ok && numext::abs(g.randColPivHouseholderQr().absDeterminant() - 34) < 1e-12;
  return ok;
}

static_assert(decomposition_wrappers());

// Products of a Householder sequence with diagonal, triangular and permutation operands, from either side.
constexpr bool householder_products() {
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const HouseholderQR<Matrix3d> qr(g);
  const auto h = qr.householderQ();
  const Matrix3d hd = h;
  const Vector3d dv(1, 2, 4);
  const auto d = dv.asDiagonal();
  const Matrix3d u = g.triangularView<Upper>();
  PermutationMatrix<3> p;
  p.indices() << 2, 0, 1;
  const Matrix3d pd = p;
  bool ok = Matrix3d(h * d).isApprox(hd * d) && Matrix3d(d * h).isApprox(d * hd);
  ok = ok && Matrix3d(h * g.triangularView<Upper>()).isApprox(hd * u) &&
       Matrix3d(g.triangularView<Upper>() * h).isApprox(u * hd);
  ok = ok && Matrix3d(h * p).isApprox(hd * pd) && Matrix3d(p * h).isApprox(pd * hd);
  ok = ok && Matrix3d(h * p.inverse()).isApprox(hd * pd.transpose()) &&
       Matrix3d(p.inverse() * h).isApprox(pd.transpose() * hd);
  return ok;
}

static_assert(householder_products());
#if defined(__cpp_lib_constexpr_complex) && !(EIGEN_COMP_CLANG && defined(__GLIBCXX__))
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 1
#else
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 0
#endif

// The estimators bound ||A^-1||_1 from below, so they never underestimate rcond = 1 / (||A||_1 ||A^-1||_1); for these
// small matrices they are within a small factor of it.
constexpr bool near_rcond(double estimate, const Matrix3d& a) {
  const Matrix3d inverse = a.inverse();
  const double reference =
      1 / (a.cwiseAbs().colwise().sum().maxCoeff() * inverse.cwiseAbs().colwise().sum().maxCoeff());
  return estimate >= reference * (1 - 1e-12) && estimate <= 3 * reference;
}

constexpr bool condition_estimates() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  bool ok = near_rcond(PartialPivLU<Matrix3d>(g).rcond(), g) && near_rcond(FullPivLU<Matrix3d>(g).rcond(), g);
  ok = ok && near_rcond(LLT<Matrix3d>(a).rcond(), a) && near_rcond(LDLT<Matrix3d>(a).rcond(), a);
  const Matrix3d nearly_singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9.001}};
  return ok && near_rcond(PartialPivLU<Matrix3d>(nearly_singular).rcond(), nearly_singular) &&
         PartialPivLU<Matrix3d>(nearly_singular).rcond() < 1e-3;
}

static_assert(condition_estimates());
#endif

#if EIGEN_COMP_CXXVER >= 20
constexpr bool rank_revealing() {
  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const ColPivHouseholderQR<Matrix3d> qr(singular);
  return qr.rank() == 2 && qr.dimensionOfKernel() == 1 && !qr.isInvertible();
}

static_assert(rank_revealing());
#endif

int main() { return 0; }
