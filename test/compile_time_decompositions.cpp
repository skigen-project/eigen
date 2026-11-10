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
constexpr bool dynamic_decompositions() {
  VectorXd u(3);
  u << 1, 2, 3;
  const MatrixXd outer = u * u.transpose();
  bool ok = outer(1, 2) == 6 && outer(2, 2) == 9;

  Matrix<double, 6, 6> a;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) a(i, j) = 1.0 / (i + j + 1) + (i == j ? 2 : 0);
  BDCSVD<Matrix<double, 6, 6>, ComputeFullU | ComputeFullV> bdc;
  bdc.setSwitchSize(3);
  bdc.compute(a);
  ok = ok && (bdc.matrixU() * bdc.singularValues().asDiagonal() * bdc.matrixV().transpose()).isApprox(a);
  const JacobiSVD<Matrix<double, 6, 6>> jacobi(a);
  ok = ok && jacobi.singularValues().isApprox(bdc.singularValues());
  return ok;
}

static_assert(dynamic_decompositions());

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

constexpr bool eigen_decompositions() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const SelfAdjointEigenSolver<Matrix3d> es(a);
  bool ok = (a * es.eigenvectors()).isApprox(es.eigenvectors() * es.eigenvalues().asDiagonal());
  SelfAdjointEigenSolver<Matrix2d> direct;
  direct.computeDirect(Matrix2d{{2, 1}, {1, 2}});
  ok = ok && direct.eigenvalues().isApprox(Vector2d(1, 3));
  const RealSchur<Matrix3d> schur(g);
  ok = ok && (schur.matrixU() * schur.matrixT() * schur.matrixU().transpose()).isApprox(g);
  const GeneralizedEigenSolver<Matrix3d> ges(a, Matrix3d::Identity());
  ok = ok && ges.info() == Success;
  const BunchKaufman<Matrix3d> bk(a);
  ok = ok && (a * bk.solve(Vector3d::Ones())).isApprox(Vector3d::Ones());
  // Bisection for the eigenvalues, inverse iteration for the eigenvectors.
  const TridiagonalEigenSolver<double> tes(Vector3d(2, 2, 2), Vector2d(1, 1));
  const Matrix3d tri{{2, 1, 0}, {1, 2, 1}, {0, 1, 2}};
  ok = ok && tes.info() == Success &&
       (tri * tes.eigenvectors()).isApprox(tes.eigenvectors() * tes.eigenvalues().asDiagonal());
  return ok;
}

static_assert(eigen_decompositions());

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

// The non-const lvalue constructors decompose in place; the SVDs take triangular and bidiagonal input.
constexpr bool inplace_and_structured_inputs() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  Matrix3d work = g;
  const HessenbergDecomposition<Matrix3d> hess(work);
  bool ok = (hess.matrixQ() * hess.matrixH() * hess.matrixQ().adjoint()).isApprox(g);
  work = a;
  const Tridiagonalization<Matrix3d> tri(work);
  ok = ok && (tri.matrixQ() * tri.matrixT() * tri.matrixQ().adjoint()).isApprox(a);
  work = g;
  const RealSchur<Ref<Matrix3d>> schur(work);
  ok = ok && (schur.matrixU() * schur.matrixT() * schur.matrixU().transpose()).isApprox(g);
  work = a;
  const SelfAdjointEigenSolver<Ref<Matrix3d>> es(work);
  ok = ok && (a * es.eigenvectors()).isApprox(es.eigenvectors() * es.eigenvalues().asDiagonal());
  Matrix3d b = Matrix3d::Identity();
  work = a;
  const GeneralizedEigenSolver<Matrix3d> ges(work, b);
  ok = ok && ges.info() == Success;

  const Matrix3d u = g.triangularView<Upper>();
  const JacobiSVD<Matrix3d> dense_svd(u);
  JacobiSVD<Matrix3d> svd(g.triangularView<Upper>());
  ok = ok && svd.singularValues().isApprox(dense_svd.singularValues());
  svd.compute(g.triangularView<Upper>());
  ok = ok && svd.singularValues().isApprox(dense_svd.singularValues());
  const Matrix3d bidiagonal{{1, 2, 0}, {0, 3, 4}, {0, 0, 5}};
  const BDCSVD<Matrix3d> bdc(Vector3d(1, 3, 5), Vector2d(2, 4));
  ok = ok && bdc.singularValues().isApprox(JacobiSVD<Matrix3d>(bidiagonal).singularValues());

  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const FullPivLU<Matrix3d> lu(singular);
  const MatrixXd kernel = lu.kernel();
  ok = ok && kernel.cols() == 1 && (singular * kernel).isZero(1e-12);
  return ok;
}

static_assert(inplace_and_structured_inputs());

// C++20 makes std::complex arithmetic constexpr, but Clang cannot constant-evaluate libstdc++'s, which is built on
// GCC's __complex__ types.
#if defined(__cpp_lib_constexpr_complex) && !(EIGEN_COMP_CLANG && defined(__GLIBCXX__))
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 1
#else
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 0
#endif

// A V = V D holds for the real block-diagonal D of a real nonsymmetric matrix, including 2x2 blocks for complex pairs.
constexpr bool nonsymmetric_eigen_solver() {
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const EigenSolver<Matrix3d> es(g);
  const Matrix3d v = es.pseudoEigenvectors();
  bool ok = es.info() == Success && (g * v).isApprox(v * es.pseudoEigenvalueMatrix());
  // The eigenvalues of an upper triangular matrix are its diagonal.
  const Matrix3d t{{2, 1, 7}, {0, 5, 1}, {0, 0, 3}};
  const EigenSolver<Matrix3d> triangular(t, false);
  const Matrix3d d = triangular.pseudoEigenvalueMatrix();
  ok = ok && d.isDiagonal() && numext::abs(d.trace() - 10) < 1e-12 && numext::abs(d.diagonal().prod() - 30) < 1e-12;
  // [[1, -2], [2, 1]] has the eigenvalues 1 +- 2i.
  const Matrix2d r{{1, -2}, {2, 1}};
  const EigenSolver<Matrix2d> pair(r, false);
  const auto lambda = pair.eigenvalues();
  ok = ok && numext::abs(lambda(0).real() - 1) < 1e-12 && numext::abs(numext::abs(lambda(0).imag()) - 2) < 1e-12;
  ok = ok && lambda(0).imag() == -lambda(1).imag();
#if EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC
  // The eigenvectors of a complex pair divide complex numbers.
  const EigenSolver<Matrix2d> pair_vectors(r);
  const Matrix2d rv = pair_vectors.pseudoEigenvectors();
  ok = ok && (r * rv).isApprox(rv * pair_vectors.pseudoEigenvalueMatrix());
#endif
  return ok;
}

static_assert(nonsymmetric_eigen_solver());

constexpr bool generalized_selfadjoint_eigen_solver() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d b{{2, 1, 0}, {1, 2, 0}, {0, 0, 1}};
  const GeneralizedSelfAdjointEigenSolver<Matrix3d> ges(a, b);
  const Matrix3d v = ges.eigenvectors();
  // A v = lambda B v, with B-orthonormal eigenvectors.
  bool ok = ges.info() == Success && (a * v).isApprox(b * v * ges.eigenvalues().asDiagonal());
  ok = ok && (v.transpose() * b * v).isIdentity(1e-12);
  // A B v = lambda v.
  const GeneralizedSelfAdjointEigenSolver<Matrix3d> abx(a, b, ComputeEigenvectors | ABx_lx);
  const Matrix3d w = abx.eigenvectors();
  return ok && (a * b * w).isApprox(w * abx.eigenvalues().asDiagonal());
}

static_assert(generalized_selfadjoint_eigen_solver());

constexpr bool eigenvalue_helpers() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const Vector3d lambda = a.selfadjointView<Lower>().eigenvalues();
  bool ok = lambda(0) <= lambda(1) && lambda(1) <= lambda(2) && numext::abs(lambda.sum() - a.trace()) < 1e-12;
  ok = ok && numext::abs(lambda.prod() - 44) < 1e-11;
  // The 2-norm of a symmetric matrix is its largest absolute eigenvalue, and of any matrix its largest singular value.
  ok = ok && numext::abs(a.selfadjointView<Upper>().operatorNorm() - lambda(2)) < 1e-12;
  const double sigma = JacobiSVD<Matrix3d>(g).singularValues()(0);
  ok = ok && numext::abs(g.operatorNorm() - sigma) < 1e-12;
  const auto mu = g.eigenvalues();
  return ok && numext::abs(mu(0).real() + mu(1).real() + mu(2).real() - g.trace()) < 1e-12;
}

static_assert(eigenvalue_helpers());

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

constexpr bool svd_and_lu_solves() {
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const Vector3d b(1, 2, 3);
  const JacobiSVD<Matrix3d, ComputeFullU | ComputeFullV> svd(g);
  bool ok = (g * svd.solve(b)).isApprox(b) && svd.rank() == 3;
  // An overdetermined system: the least-squares solution satisfies the normal equations A^T (A x - b) = 0.
  MatrixXd tall(4, 2);
  tall << 1, 0, 1, 1, 1, 2, 1, 3;
  VectorXd y(4);
  y << 1, 2, 2, 4;
  const VectorXd x = JacobiSVD<MatrixXd, ComputeThinU | ComputeThinV>(tall).solve(y);
  ok = ok && (tall.transpose() * (tall * x - y)).isZero(1e-12);
  const VectorXd z = BDCSVD<MatrixXd, ComputeThinU | ComputeThinV>(tall).solve(y);
  ok = ok && z.isApprox(x) && numext::abs(x(1) - 0.9) < 1e-12;

  const FullPivLU<Matrix3d> lu(g);
  ok = ok && lu.isInvertible() && lu.isInjective() && lu.isSurjective() && (g * lu.solve(b)).isApprox(b);
  ok = ok && (lu.inverse() * g).isIdentity(1e-12) && (lu.reconstructedMatrix()).isApprox(g);
  // The image of a rank-2 matrix: two columns of it that span its column space.
  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const FullPivLU<Matrix3d> slu(singular);
  const MatrixXd image = slu.image(singular);
  Matrix<double, 3, 5> joined;
  joined << singular, image;
  ok = ok && !slu.isInvertible() && image.cols() == 2 && FullPivLU<Matrix<double, 3, 5>>(joined).rank() == 2;
  const Vector3d consistent = singular * Vector3d(1, -1, 2);
  return ok && (singular * slu.solve(consistent)).isApprox(consistent);
}

static_assert(svd_and_lu_solves());

#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L
// std::exp is constexpr from C++26 (P1383R2).
constexpr bool selfadjoint_operator_exp() {
  const Matrix2d a{{0, 0}, {0, 0}};
  const SelfAdjointEigenSolver<Matrix2d> es(a);
  return es.operatorExp().isApprox(Matrix2d::Identity());
}

static_assert(selfadjoint_operator_exp());
#endif
#endif

#if EIGEN_COMP_CXXVER >= 20
constexpr bool rank_revealing() {
  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const ColPivHouseholderQR<Matrix3d> qr(singular);
  return qr.rank() == 2 && qr.dimensionOfKernel() == 1 && !qr.isInvertible();
}

static_assert(rank_revealing());

constexpr bool selfadjoint_eigensolver() {
  const Matrix2d a{{2, 1}, {1, 2}};
  const SelfAdjointEigenSolver<Matrix2d> es(a);
  return es.eigenvalues().isApprox(Vector2d(1, 3)) &&
         (a * es.eigenvectors()).isApprox(es.eigenvectors() * es.eigenvalues().asDiagonal());
}

static_assert(selfadjoint_eigensolver());

// Before C++26, constant evaluation cannot form realView() of std::complex coefficients, which the complex
// decompositions use.
#if defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
constexpr bool complex_eigen_solver() {
  // The non-const lvalue selects the in-place constructor.
  using C = std::complex<double>;
  Matrix2cd c{{C(2, 0), C(1, 0)}, {C(1, 0), C(2, 0)}};
  const ComplexEigenSolver<Matrix2cd> ces(c);
  return (c * ces.eigenvectors()).isApprox(ces.eigenvectors() * ces.eigenvalues().asDiagonal());
}

static_assert(complex_eigen_solver());

constexpr bool complex_jacobi_svd() {
  using C = std::complex<double>;
  const Matrix2cd c{{C(1, 1), C(2, 0)}, {C(0, 1), C(3, -1)}};
  const Vector2d s = JacobiSVD<Matrix2cd>(c).singularValues();
  // s0^2 + s1^2 = ||c||_F^2 = 17 and s0 s1 = |det c| = |(1 + i)(3 - i) - 2i| = 4.
  return numext::abs(s.squaredNorm() - 17) < 1e-12 && numext::abs(s.prod() - 4) < 1e-12;
}

static_assert(complex_jacobi_svd());

// A = U T U^* with T upper triangular, and the generalized form A = Q S Z, B = Q T Z.
constexpr bool complex_schur_and_qz() {
  using C = std::complex<double>;
  const Matrix2cd a{{C(1, 1), C(2, 0)}, {C(0, 1), C(3, -1)}};
  const Matrix2cd b{{C(2, 0), C(0, 1)}, {C(0, -1), C(3, 0)}};
  const ComplexSchur<Matrix2cd> schur(a);
  const Matrix2cd t = schur.matrixT();
  bool ok =
      schur.info() == Success && t(1, 0) == C(0, 0) && (schur.matrixU() * t * schur.matrixU().adjoint()).isApprox(a);
  const ComplexQZ<Matrix2cd> qz(a, b);
  ok = ok && qz.info() == Success && qz.matrixS().isUpperTriangular() && qz.matrixT().isUpperTriangular();
  return ok && (qz.matrixQ() * qz.matrixS() * qz.matrixZ()).isApprox(a) &&
         (qz.matrixQ() * qz.matrixT() * qz.matrixZ()).isApprox(b);
}

static_assert(complex_schur_and_qz());
#endif
#endif

int main() { return 0; }
