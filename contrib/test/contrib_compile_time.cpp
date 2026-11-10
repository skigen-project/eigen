// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of AutoDiff, NonLinearOptimization, NumericalDiff, Polynomials, Riccati and StructuredMatrices.
// main.h adds instrumentation that breaks constant evaluation, so this test only compiles.

#include <contrib/Eigen/AlignedVector3>
#include <contrib/Eigen/AutoDiff>
#include <contrib/Eigen/BVH>
#include <contrib/Eigen/EulerAngles>
#include <contrib/Eigen/KroneckerProduct>
#include <contrib/Eigen/MatrixFunctions>
#include <contrib/Eigen/NNLS>
#include <contrib/Eigen/NonLinearOptimization>
#include <contrib/Eigen/NumericalDiff>
#include <contrib/Eigen/Polynomials>
#include <contrib/Eigen/Riccati>
#include <contrib/Eigen/SparseExtra>
#include <contrib/Eigen/SpecialFunctions>
#include <contrib/Eigen/Splines>
#include <contrib/Eigen/StructuredMatrices>

#include <vector>

using namespace Eigen;

// Matrix construction in a constant expression needs C++20.
#if EIGEN_COMP_CXXVER >= 20

constexpr bool near(double actual, double expected, double tol) { return numext::abs(actual - expected) <= tol; }

// Mixed derivative sizes go through both maybe_coherent_pad_helper specializations: the shorter operand is padded
// and the longer one is passed through.
constexpr bool autodiff() {
  using AD2 = AutoDiffScalar<Vector2d>;
  using AD3 = AutoDiffScalar<Vector3d>;
  const AD2 x(2.0, Vector2d(1, 0));
  const AD3 y(3.0, Vector3d(0, 1, 5));
  const AD3 sum = x + y;
  const AD3 difference = y - x;
  bool ok = sum.value() == 5 && sum.derivatives() == Vector3d(1, 1, 5);
  ok = ok && difference.value() == 1 && difference.derivatives() == Vector3d(-1, 1, 5);
  const AD2 product = x * x + x;
  ok = ok && product.value() == 6 && product.derivatives() == Vector2d(5, 0);
  // d|v|/dv = sign(v).
  const AD2 negative_abs = abs(AD2(-2.0, Vector2d(1, 3)));
  const AD2 positive_abs = abs(AD2(2.0, Vector2d(1, 3)));
  return ok && negative_abs.value() == 2 && negative_abs.derivatives() == Vector2d(-1, -3) &&
         positive_abs.derivatives() == Vector2d(1, 3);
}

static_assert(autodiff());

// The fourth, padding coefficient stays zero through arithmetic and is ignored by the reductions.
constexpr bool aligned_vector3() {
  const AlignedVector3<double> u(1, 2, 2), v(0, 1, -1);
  AlignedVector3<double> w = u * 2.0 - v;
  w += v;
  w /= 2;
  const AlignedVector3<double> c = u.cross(v);
  return w.isApprox(u) && u.dot(v) == 0 && u.squaredNorm() == 9 && u.norm() == 3 && u.sum() == 5 &&
         Vector3d(c) == Vector3d(-4, 1, 1) && Vector3d(u.normalized()).isApprox(Vector3d(1, 2, 2) / 3);
}

static_assert(aligned_vector3());

// f(x) = (x0^2, x0 x1).
struct ProductFunctor {
  using Scalar = double;
  enum { InputsAtCompileTime = 2, ValuesAtCompileTime = 2 };
  using InputType = Vector2d;
  using ValueType = Vector2d;
  using JacobianType = Matrix2d;
  constexpr int inputs() const { return 2; }
  constexpr int values() const { return 2; }
  constexpr int operator()(const InputType& x, ValueType& f) const {
    f(0) = x(0) * x(0);
    f(1) = x(0) * x(1);
    return 0;
  }
};

template <NumericalDiffMode mode>
constexpr bool numerical_diff() {
  NumericalDiff<ProductFunctor, mode> diff;
  Matrix2d jac;
  const int nfev = diff.df(Vector2d(1.5, -2), jac);
  // Truncation error O(h) forward, O(h^2) central, with h = sqrt(eps) or cbrt(eps).
  constexpr double tol = 1e-6;
  return nfev == (mode == Forward ? 3 : 4) && near(jac(0, 0), 3, tol) && near(jac(0, 1), 0, tol) &&
         near(jac(1, 0), -2, tol) && near(jac(1, 1), 1.5, tol);
}

static_assert(numerical_diff<Forward>());
// Central mode's default step uses std::cbrt, which is constexpr from C++26 (P1383R2).
#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L
static_assert(numerical_diff<Central>());
#endif

#if defined(__cpp_lib_constexpr_dynamic_alloc) && defined(__cpp_lib_constexpr_vector) && \
    defined(__cpp_lib_constexpr_algorithms)
// diag(1, 2, 2, 4) + 0.5 z z^T with a repeated pole, which deflates by a rotation.
constexpr bool dpr1() {
  VectorXd d(4), z(4);
  d << 4, 2, 1, 2;
  z << 1, -1, 0.5, 2;
  const double rho = 0.5;
  DPR1EigenSolver<double> es(d, rho, z);
  if (es.info() != Success) return false;
  const MatrixXd a = MatrixXd(d.asDiagonal()) + rho * z * z.transpose();
  const VectorXd& lambda = es.eigenvalues();
  const MatrixXd& v = es.eigenvectors();
  constexpr double tol = 1e-13;
  bool ok = near(lambda.sum(), a.trace(), tol);
  for (Index k = 0; k + 1 < lambda.size(); ++k) ok = ok && lambda[k] <= lambda[k + 1];
  ok = ok && (a * v - v * lambda.asDiagonal()).cwiseAbs().maxCoeff() <= tol;
  return ok && (v.transpose() * v - MatrixXd::Identity(4, 4)).cwiseAbs().maxCoeff() <= tol;
}

static_assert(dpr1());

// With diag = 1, the Gauss-Newton step is R^-1 qtb, the gradient direction g = R^T qtb, and the Cauchy step length
// sgnorm = |g|^3 / |R g|^2. A trust radius delta <= sgnorm takes the gradient step delta g / |g|; one with
// sgnorm < delta < |R^-1 qtb| takes the dogleg point of norm delta.
constexpr bool dogleg() {
  constexpr double tol = 1e-14;
  const VectorXd diag = VectorXd::Ones(2);
  MatrixXd r = MatrixXd::Identity(2, 2);
  VectorXd qtb(2), x(2);
  qtb << 3, 4;
  internal::dogleg<double>(r, diag, qtb, 1.0, x);
  bool ok = near(x[0], 0.6, tol) && near(x[1], 0.8, tol);

  // Gauss-Newton step (1, 0.5) and sgnorm = 5 sqrt(5) / 17 < 1.
  r(1, 1) = 2;
  qtb << 1, 1;
  internal::dogleg<double>(r, diag, qtb, 1.0, x);
  const Vector2d cauchy = 5.0 / 17 * Vector2d(1, 2);
  const Vector2d gauss_newton(1, 0.5);
  const double t = (x[0] - cauchy[0]) / (gauss_newton[0] - cauchy[0]);
  ok = ok && near(x.norm(), 1, tol) && t > 0 && t < 1;
  return ok && near(x[1], cauchy[1] + t * (gauss_newton[1] - cauchy[1]), tol);
}

static_assert(dogleg());

constexpr bool ridders() {
  Ridders<ProductFunctor> diff{ProductFunctor()};
  Matrix2d jac;
  Vector2d error;
  diff.df(Vector2d(1.5, -2), jac, error);
  constexpr double tol = 1e-10;
  return near(jac(0, 0), 3, tol) && near(jac(0, 1), 0, tol) && near(jac(1, 0), -2, tol) && near(jac(1, 1), 1.5, tol);
}

static_assert(ridders());

// f(x) = (x0^2 - 4, x0 x1 - 2) with the root (2, 1) and the Jacobian {{4, 0}, {1, 2}} there.
struct RootFunctor {
  using Scalar = double;
  enum { InputsAtCompileTime = Dynamic, ValuesAtCompileTime = Dynamic };
  using InputType = VectorXd;
  using ValueType = VectorXd;
  using JacobianType = MatrixXd;
  constexpr int inputs() const { return 2; }
  constexpr int values() const { return 2; }
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

// hybrj1 uses the analytic Jacobian; hybrd1 approximates it by forward differences (fdjac1).
constexpr bool hybrid_solver() {
  RootFunctor f;
  VectorXd x(2);
  x << 3, 3;
  HybridNonLinearSolver<RootFunctor> analytic(f);
  bool ok = analytic.hybrj1(x) == HybridNonLinearSolverSpace::RelativeErrorTooSmall && near(x(0), 2, 1e-10) &&
            near(x(1), 1, 1e-10);
  x << 3, 3;
  HybridNonLinearSolver<RootFunctor> numeric(f);
  return ok && numeric.hybrd1(x) == HybridNonLinearSolverSpace::RelativeErrorTooSmall && near(x(0), 2, 1e-8) &&
         near(x(1), 1, 1e-8);
}

static_assert(hybrid_solver());

// At the root R^T R = P^T J^T J P, so covar turns R into (J^T J)^-1.
constexpr bool levenberg_marquardt() {
  RootFunctor f;
  VectorXd x(2);
  x << 3, 3;
  LevenbergMarquardt<RootFunctor> lm(f);
  bool ok =
      lm.lmder1(x) != LevenbergMarquardtSpace::ImproperInputParameters && near(x(0), 2, 1e-10) && near(x(1), 1, 1e-10);
  MatrixXd r = lm.fjac;
  internal::covar(r, lm.permutation.indices());
  const MatrixXd jac{{4, 0}, {1, 2}};
  ok = ok && (r * (jac.transpose() * jac)).isIdentity(1e-8);
  x << 3, 3;
  Index nfev = 0;
  ok = ok && LevenbergMarquardt<RootFunctor>::lmdif1(f, x, &nfev) != LevenbergMarquardtSpace::ImproperInputParameters;
  return ok && near(x(0), 2, 1e-8) && near(x(1), 1, 1e-8);
}

static_assert(levenberg_marquardt());

// chkder's error estimate takes logarithms, and std::log is constexpr from C++26 (P1383R2).
#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L
constexpr bool check_derivatives() {
  RootFunctor f;
  VectorXd x(2), fvec(2), xp(2), fvecp(2), err(2);
  x << 3, 3;
  MatrixXd jac(2, 2);
  f(x, fvec);
  f.df(x, jac);
  internal::chkder(x, fvec, jac, xp, fvecp, 1, err);
  f(xp, fvecp);
  internal::chkder(x, fvec, jac, xp, fvecp, 2, err);
  return err.minCoeff() > 0.9;
}

static_assert(check_derivatives());
#endif

// The angle extraction calls std::atan2 and the rotation std::sin and std::cos, constexpr from C++26 (P1383R2).
#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L
constexpr bool euler_angles() {
  const EulerAnglesXYZd e(0.1, -0.2, 0.3);
  const EulerAnglesXYZd back(e.toRotationMatrix());
  const Matrix3d r = Quaterniond(e).toRotationMatrix();
  return near(back.alpha(), 0.1, 1e-14) && near(back.beta(), -0.2, 1e-14) && near(back.gamma(), 0.3, 1e-14) &&
         r.isApprox(e.toRotationMatrix()) && (r * r.transpose()).isIdentity(1e-14);
}

static_assert(euler_angles());

// The array functions apply the scalar kernels coefficient-wise, so they match numext exactly.
constexpr bool special_function_arrays() {
  const Array3d x(0.5, 1.5, 3.0), a(1.0, 2.0, 0.5);
  const Array3d lg = x.lgamma(), e = x.erf(), i0 = bessel_i0(x), j1 = bessel_j1(x), ig = igamma(a, x);
  const Array3d z = zeta(x + 1.5, a), b = betainc(a, a + 1, x / 4), pg = polygamma(Array3d::Ones(), x);
  bool ok = true;
  for (Index k = 0; k < 3; ++k)
    ok = ok && lg(k) == numext::lgamma(x(k)) && e(k) == numext::erf(x(k)) && i0(k) == numext::bessel_i0(x(k)) &&
         j1(k) == numext::bessel_j1(x(k)) && ig(k) == numext::igamma(a(k), x(k)) &&
         z(k) == numext::zeta(x(k) + 1.5, a(k)) && b(k) == numext::betainc(a(k), a(k) + 1, x(k) / 4) &&
         pg(k) == numext::polygamma(1.0, x(k));
  return ok;
}

static_assert(special_function_arrays());
#endif

// Integer exponents take binary powering and skip the Schur decomposition; it calls std::fmod, which is constexpr
// from C++23 (P0533R9).
#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202202L
constexpr bool matrix_integer_power() {
  const Matrix2d a{{2, 1}, {0, 3}};
  const Matrix2d cube = a.pow(3.0);
  const Matrix2d inverse_square = a.pow(-2.0);
  return cube.isApprox(a * a * a) && (inverse_square * a * a).isIdentity(1e-12);
}

static_assert(matrix_integer_power());
#endif

// Each operator is checked against the dense matrix built from its definition.
constexpr bool vandermonde_and_cauchy() {
  const Vector3d x(1, 2, 4), y(-1, 0.5, 3);
  const Vector3d b(1, 2, 3);
  Matrix3d vd, cd;
  for (Index i = 0; i < 3; ++i)
    for (Index j = 0; j < 3; ++j) {
      vd(i, j) = j == 0 ? 1 : vd(i, j - 1) * x(i);
      cd(i, j) = 1 / (x(i) - y(j));
    }
  const auto v = makeVandermonde(x);
  bool ok = Matrix3d(v).isApprox(vd) && near(v.determinant(), 6, 1e-12) && Vector3d(v * b).isApprox(vd * b);
  const BjorckPereyra<double> bp(v);
  ok = ok && bp.info() == Success && (vd * Vector3d(bp.solve(b))).isApprox(b);
  const auto c = makeCauchy(x, y);
  ok = ok && Matrix3d(c).isApprox(cd) && near(c.determinant(), cd.determinant(), 1e-12) &&
       Vector3d(c * b).isApprox(cd * b);
  const CauchyLU<double> lu(c);
  return ok && lu.info() == Success && (cd * Vector3d(lu.solve(b))).isApprox(b);
}

static_assert(vandermonde_and_cauchy());

constexpr bool low_rank_and_kronecker() {
  const Vector3d d(2, 3, 4), u(1, -1, 2), w(0.5, 1, -1);
  const Vector3d b(1, 2, 3);
  const Matrix3d dense = Matrix3d(d.asDiagonal()) + u * w.transpose();
  const auto dplr = makeDiagonalPlusLowRank(d, u, w);
  bool ok = Matrix3d(dplr).isApprox(dense) && near(dplr.determinant(), dense.determinant(), 1e-12) &&
            Vector3d(dplr * b).isApprox(dense * b) && (dense * dplr.solve(b)).isApprox(b);

  const Matrix2d p{{2, 1}, {0, 3}}, q{{1, -1}, {2, 1}};
  Matrix4d kd;
  for (Index i = 0; i < 4; ++i)
    for (Index j = 0; j < 4; ++j) kd(i, j) = p(i / 2, j / 2) * q(i % 2, j % 2);
  const KroneckerOperator<Matrix2d, Matrix2d> k(p, q);
  const Vector4d e(1, 2, 3, 4);
  return ok && Matrix4d(k).isApprox(kd) && near(k.determinant(), kd.determinant(), 1e-12) &&
         Vector4d(k * e).isApprox(kd * e) && (kd * k.solve(e)).isApprox(e);
}

static_assert(low_rank_and_kronecker());

// The leading 2x2 block of this Toeplitz matrix is singular, which the look-ahead steps over.
constexpr bool look_ahead_levinson() {
  VectorXd col(4), row(4);
  col << 1, 1, 2, 0.5;
  row << 1, 1, -1, 3;
  MatrixXd dense(4, 4);
  for (Index i = 0; i < 4; ++i)
    for (Index j = 0; j < 4; ++j) dense(i, j) = i >= j ? col(i - j) : row(j - i);
  const Toeplitz<double> t(col, row);
  const LookAheadLevinson<double> lev(t);
  const VectorXd b = VectorXd::LinSpaced(4, 1, 4);
  return lev.info() == Success && (dense * VectorXd(lev.solve(b))).isApprox(b);
}

static_assert(look_ahead_levinson());

// H(i, j) = h(i + j); the solve reverses the columns into a Toeplitz system for LookAheadLevinson.
constexpr bool hankel() {
  const Vector3d col(1, 2, 4), row(4, -1, 3);
  Matrix3d dense;
  for (Index i = 0; i < 3; ++i)
    for (Index j = 0; j < 3; ++j) dense(i, j) = i + j < 3 ? col(i + j) : row(i + j - 2);
  const Hankel<double, 3, 3> hk(col, row);
  const Vector3d b(1, 2, 3);
  return Matrix3d(hk).isApprox(dense) && Vector3d(hk * b).isApprox(dense * b) &&
         Matrix3d(hk.transpose()).isApprox(dense.transpose()) && (dense * hk.solve(b)).isApprox(b);
}

static_assert(hankel());

// A (+) B = A (x) I + I (x) B, whose product is vec(B X + X A^T); the solve runs Bartels-Stewart on the Schur forms.
constexpr bool kronecker_sum() {
  const Matrix2d a{{2, 1}, {0, 3}};
  const Matrix3d b{{1, 0, 0}, {1, 4, 0}, {2, 1, 5}};
  MatrixXd dense(6, 6);
  for (Index i = 0; i < 6; ++i)
    for (Index j = 0; j < 6; ++j)
      dense(i, j) = (i % 3 == j % 3 ? a(i / 3, j / 3) : 0) + (i / 3 == j / 3 ? b(i % 3, j % 3) : 0);
  const auto s = makeKroneckerSum(a, b);
  const VectorXd x = VectorXd::LinSpaced(6, 1, 6);
  bool ok = MatrixXd(s).isApprox(dense) && VectorXd(s * x).isApprox(dense * x) && (dense * s.solve(x)).isApprox(x);
  const SparseMatrix<double> sparse = s.sparseView();
  ok = ok && MatrixXd(sparse).isApprox(dense);
  // kroneckerProduct materializes A (x) B, dense or sparse.
  MatrixXd kd(6, 6);
  for (Index i = 0; i < 6; ++i)
    for (Index j = 0; j < 6; ++j) kd(i, j) = a(i / 3, j / 3) * b(i % 3, j % 3);
  const Matrix<double, 6, 6> kp = kroneckerProduct(a, b);
  const SparseMatrix<double> ks = kroneckerProduct(a.sparseView(), b.sparseView());
  return ok && kp.isApprox(kd) && MatrixXd(ks).isApprox(kd) && ks.nonZeros() == 3 * 6;
}

static_assert(kronecker_sum());

// The inverse is computed only on the sparsity pattern of A; for a tridiagonal A those entries match the dense inverse.
constexpr bool sparse_inverse() {
  SparseMatrix<double> a(3, 3);
  a.insert(0, 0) = 4;
  a.insert(1, 0) = -1;
  a.insert(0, 1) = -1;
  a.insert(1, 1) = 4;
  a.insert(2, 1) = -1;
  a.insert(1, 2) = -1;
  a.insert(2, 2) = 4;
  a.makeCompressed();
  const SparseMatrix<double> inv = SparseInverse<double>().compute(a).inverse();
  const MatrixXd reference = MatrixXd(a).inverse();
  bool ok = inv.nonZeros() == 7;
  for (Index j = 0; j < 3; ++j)
    for (SparseMatrix<double>::InnerIterator it(inv, j); it; ++it)
      ok = ok && near(it.value(), reference(it.row(), it.col()), 1e-14);
  return ok;
}

static_assert(sparse_inverse());

// A cubic spline through samples of the parabola (u, u^2) reproduces it, with derivative (1, 2u).
constexpr bool spline_interpolation() {
  Matrix<double, 2, 5> points;
  Spline2d::KnotVectorType parameters(5);
  for (Index i = 0; i < 5; ++i) {
    const double u = 0.25 * double(i);
    points.col(i) = Vector2d(u, u * u);
    parameters(i) = u;
  }
  const Spline2d spline = SplineFitting<Spline2d>::Interpolate(points, 3, parameters);
  const Vector2d p = spline(0.3);
  const auto d = spline.derivatives(0.3, 1);
  return near(p(0), 0.3, 1e-13) && near(p(1), 0.09, 1e-13) && near(d(0, 1), 1, 1e-12) && near(d(1, 1), 0.6, 1e-12);
}

static_assert(spline_interpolation());

// The real Schur form of an SPD matrix is diagonal up to rounding, so sqrt takes the 1x1-block path.
constexpr bool matrix_sqrt() {
  const Matrix2d a{{5, 4}, {4, 5}};
  const Matrix2d r = a.sqrt();
  // sqrt(a) = {{2, 1}, {1, 2}}, from the eigenvalues 9 and 1.
  bool ok = r.isApprox(Matrix2d{{2, 1}, {1, 2}}, 1e-12);
  const Matrix3d t{{4, 1, 2}, {0, 9, 3}, {0, 0, 16}};
  // Only the upper triangle of the result is written.
  Matrix3d s = Matrix3d::Zero();
  matrix_sqrt_triangular(t, s);
  return ok && (s * s).isApprox(t, 1e-12) && s(0, 0) == 2 && s(1, 1) == 3 && s(2, 2) == 4;
}

static_assert(matrix_sqrt());

// The least-squares solution of {{1, 1}, {1, 2}, {1, 3}} x = b is (1/3, 1/2) for b = (1, 1, 2), which is
// nonnegative, and (7/3, -1/2) for b = (2, 1, 1), where NNLS drops the second column and refits the first:
// x0 = (1, 1, 1) . b / 3 = 4/3.
constexpr bool nnls() {
  const Matrix<double, 3, 2> a{{1, 1}, {1, 2}, {1, 3}};
  NNLS<Matrix<double, 3, 2>> solver(a);
  Vector2d x = solver.solve(Vector3d(1, 1, 2));
  bool ok = solver.info() == Success && near(x(0), 1.0 / 3, 1e-12) && near(x(1), 0.5, 1e-12);
  x = solver.solve(Vector3d(2, 1, 1));
  return ok && solver.info() == Success && near(x(0), 4.0 / 3, 1e-12) && x(1) == 0;
}

static_assert(nnls());

// KdBVH finds an object's volume through bounding_box(object), looked up in namespace Eigen.
namespace Eigen {
constexpr AlignedBox2d bounding_box(const Vector2d& v) { return AlignedBox2d(v, v); }
}  // namespace Eigen

// Counts the points inside a query box, pruning the tree's volumes that miss it.
struct BoxCounter {
  AlignedBox2d query;
  int count = 0;
  constexpr bool intersectVolume(const AlignedBox2d& box) const { return query.intersects(box); }
  constexpr bool intersectObject(const Vector2d& p) {
    if (query.contains(p)) ++count;
    return false;
  }
};

// Counts the pairs of points, one from each tree, less than 1 apart in the max norm.
struct PairCounter {
  int count = 0;
  constexpr bool intersectVolumeVolume(const AlignedBox2d& a, const AlignedBox2d& b) const {
    return !(a.min().array() > b.max().array() + 1).any() && !(b.min().array() > a.max().array() + 1).any();
  }
  constexpr bool intersectVolumeObject(const AlignedBox2d& a, const Vector2d& p) const {
    return intersectVolumeVolume(a, AlignedBox2d(p, p));
  }
  constexpr bool intersectObjectVolume(const Vector2d& p, const AlignedBox2d& b) const {
    return intersectVolumeVolume(AlignedBox2d(p, p), b);
  }
  constexpr bool intersectObjectObject(const Vector2d& p, const Vector2d& q) {
    if ((p - q).cwiseAbs().maxCoeff() < 1) ++count;
    return false;
  }
};

// The tree over the 5 x 5 integer grid holds 4 points in [0.5, 2.5]^2. Each point (i + 0.5, j + 0.5) of the 4 x 4
// offset grid is within 1 of its 4 neighbours on the integer grid, so the two trees share 16 * 4 = 64 close pairs.
constexpr bool kd_bvh() {
  std::vector<Vector2d> grid;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) grid.emplace_back(i, j);
  const KdBVH<double, 2, Vector2d> tree(grid.begin(), grid.end());
  BoxCounter counter{AlignedBox2d(Vector2d(0.5, 0.5), Vector2d(2.5, 2.5))};
  BVIntersect(tree, counter);
  std::vector<Vector2d> offset;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) offset.emplace_back(i + 0.5, j + 0.5);
  const KdBVH<double, 2, Vector2d> offset_tree(offset.begin(), offset.end());
  PairCounter pairs;
  BVIntersect(tree, offset_tree, pairs);
  return counter.count == 4 && pairs.count == 64;
}

static_assert(kd_bvh());

// The DARE residual A^* X A - E^* X E - (A^* X B + S) (R + B^* X B)^-1 (B^* X A + S^*) + Q relative to ||Q||.
template <typename MatrixA, typename MatrixB, typename MatrixR>
constexpr double dare_residual(const MatrixA& a, const MatrixB& b, const MatrixA& q, const MatrixR& r,
                               const typename MatrixA::PlainObject& e, const typename MatrixB::PlainObject& s,
                               const MatrixA& x) {
  const MatrixB axbs = a.adjoint() * x * b + s;
  const MatrixA g = axbs * (r + b.adjoint() * x * b).partialPivLu().solve(axbs.adjoint());
  return (a.adjoint() * x * a - e.adjoint() * x * e - g + q).norm() / q.norm();
}

// A sampled double integrator through the balanced pencil.
constexpr bool discrete_algebraic_riccati() {
  const Matrix2d a{{1, 0.1}, {0, 1}};
  const Vector2d b(0.005, 0.1);
  const Matrix2d q = Matrix2d::Identity();
  const Matrix<double, 1, 1> r(0.1);
  const DiscreteAlgebraicRiccatiSolver<Matrix2d> dare(a, b, q, r);
  return dare.info() == Success && dare.solution().isApprox(dare.solution().transpose()) &&
         dare_residual(a, b, q, r, Matrix2d::Identity(), Vector2d::Zero(), dare.solution()) < 1e-12;
}

static_assert(discrete_algebraic_riccati());

// The generalized equation with E and S, through the unbalanced pencil.
constexpr bool generalized_discrete_algebraic_riccati() {
  const Matrix2d a{{1, 0.1}, {0, 1}};
  const Vector2d b(0.005, 0.1);
  const Matrix2d q = Matrix2d::Identity();
  const Matrix<double, 1, 1> r(0.1);
  const Matrix2d e{{1, 0.2}, {0, 2}};
  const Vector2d s(0.01, 0.02);
  const DiscreteAlgebraicRiccatiSolver<Matrix2d> dare(a, b, q, r, e, s, false);
  return dare.info() == Success && dare_residual(a, b, q, r, e, s, dare.solution()) < 1e-12;
}

static_assert(generalized_discrete_algebraic_riccati());

// C++20 makes std::complex arithmetic constexpr, but Clang cannot constant-evaluate libstdc++'s, which is built on
// GCC's __complex__ types. Before C++26, constant evaluation also cannot form realView() of std::complex coefficients,
// which the complex stableNorm() behind the Schur reordering uses.
#if defined(__cpp_lib_constexpr_complex) && !(EIGEN_COMP_CLANG && defined(__GLIBCXX__)) && \
    defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
// A complex A goes through ComplexQZ.
constexpr bool complex_discrete_algebraic_riccati() {
  using C = std::complex<double>;
  const Matrix2cd a{{C(0.9, 0.2), C(0.1, 0)}, {C(0, 0.1), C(1.1, -0.1)}};
  const Vector2cd b(C(0, 0), C(1, 0.5));
  const Matrix2cd q = Matrix2cd::Identity();
  const Matrix<C, 1, 1> r(C(1, 0));
  const DiscreteAlgebraicRiccatiSolver<Matrix2cd> dare(a, b, q, r);
  return dare.info() == Success && dare.solution().isApprox(dare.solution().adjoint()) &&
         dare_residual(a, b, q, r, Matrix2cd::Identity(), Vector2cd::Zero(), dare.solution()) < 1e-12;
}

static_assert(complex_discrete_algebraic_riccati());
#endif

// Before C++26, constant evaluation cannot form realView() of std::complex coefficients, which the complex
// eigensolver behind PolynomialSolver uses.
#if defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
// (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6.
constexpr bool polynomial_roots() {
  const Vector4d p(6, -7, 0, 1);
  const PolynomialSolver<double, 3> solver(p);
  bool real = false;
  const double greatest = solver.greatestRealRoot(real);
  return real && near(greatest, 2, 1e-12) && near(solver.smallestRealRoot(real), -3, 1e-12) &&
         near(solver.absGreatestRealRoot(real), -3, 1e-12) && near(solver.absSmallestRealRoot(real), 1, 1e-12);
}

static_assert(polynomial_roots());
#endif
#endif

// 1 + 2x + 3x^2: 1 + (1 + 2) / 3 and 1 / (1 + (2 + 3) / 1) bound the moduli of its roots, both 1/sqrt(3).
static_assert(cauchy_max_bound(Vector3d(1, 2, 3)) == 2);
static_assert(cauchy_min_bound(Vector3d(1, 2, 3)) == 1.0 / 6);
// Leading zero coefficients are skipped: 1 / (1 + 4 / 2).
static_assert(cauchy_min_bound(Vector3d(0, -2, 4)) == 1.0 / 3);

static_assert(!(isnan)(AutoDiffScalar<Vector2d>(1.0, Vector2d(1, 0))));

#endif

int main() { return 0; }
