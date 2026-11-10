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

constexpr bool zeroSized() {
  constexpr Matrix<float, 0, 0> m0;
  static_assert(m0.size() == 0, "");

  constexpr Matrix<float, 0, 0> m1;
  static_assert(m0 == m1, "");
  static_assert(!(m0 != m1), "");

  constexpr Array<int, 0, 0> a0;
  static_assert(a0.size() == 0, "");

  constexpr Array<int, 0, 0> a1;
  static_assert((a0 == a1).all(), "");
  static_assert((a0 != a1).count() == 0, "");

  constexpr Array<float, 0, 0> af;
  static_assert(m0 == af.matrix(), "");
  static_assert((m0.array() == af).all(), "");
  static_assert(m0.array().matrix() == m0, "");

  return true;
}

static_assert(zeroSized(), "");

static constexpr double static_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

constexpr bool maps() {
  constexpr Map<const Vector4d> m(static_data);
  static_assert(m(0) == 1, "");
  constexpr Map<const Array<double, 4, 1>> a(static_data);
  static_assert(m == a.matrix(), "");
  static_assert(m.size() == 4, "");
  static_assert(a.size() == 4, "");
  static_assert(m.rows() == 4 && m.cols() == 1, "");
  return true;
}

static_assert(maps(), "");

constexpr bool nc_maps() {
  bool result = true;

  double d[] = {1, 2, 3, 4};
  Map<Vector4d> m(d);
  result = result && (m.x() == 1 && m.y() == 2 && m.z() == 3 && m.w() == 4);

  float array[3] = {};
  auto v = Vector3f::Map(array);
  v.fill(10);
  result = result && (v.array() == 10).all();

  return result;
}

constexpr bool blocks() {
  constexpr Map<const Matrix2d> m(static_data);
  constexpr auto block = m.block<2, 1>(0, 1);

  constexpr Map<const Vector2d> v(static_data + 2);
  static_assert(block == v, "");

  return true;
}

static_assert(blocks(), "");

constexpr bool diagonal_row_columns() {
  constexpr Map<const Matrix2d> m(static_data);
  static_assert(m.block<2, 1>(0, 1) == m.col(1), "");
  static_assert(m.block<1, 2>(1, 0) == m.row(1), "");
  static_assert(m.diagonal()(0) == 1 && m.diagonal()(1) == 4, "");
  return true;
}

static_assert(diagonal_row_columns(), "");

static constexpr int static_data_antisym[] = {0, 1, -1, -1, 0, 1, 1, -1, 0};

constexpr bool transpose_unaryminus() {
  constexpr Map<const Matrix<int, 3, 3>> m(static_data_antisym);

  static_assert(m.transpose() == -m, "");
  static_assert(-m.transpose() == m, "");
  static_assert((-m).transpose() == m, "");

  static_assert(m.transpose() != m, "");
  static_assert(-m.transpose() != -m, "");
  static_assert((-m).transpose() != -m, "");

  return true;
}

static_assert(transpose_unaryminus(), "");

constexpr bool reductions() {
  constexpr Map<const Matrix<int, 3, 3>> m(static_data_antisym);
  static_assert(m.size() == 9, "");

  static_assert(m.sum() == 0, "");
  static_assert(m.trace() == 0, "");
  static_assert(m.mean() == 0, "");
  static_assert(m.prod() == 0, "");
  static_assert(m.minCoeff() == -1, "");
  static_assert(m.maxCoeff() == 1, "");

  static_assert(m.rowwise().sum() == Vector3i::Zero(), "");
  static_assert((((m.array() == m.array().transpose()).colwise()).count() == Array<long, 1, 3>::Ones()).all(), "");

  static_assert(m.squaredNorm() == 6, "");

  return true;
}

static_assert(reductions(), "");

constexpr bool scalar_mult_div() {
  constexpr Map<const Matrix2d> m(static_data);

  static_assert((m * 2)(0, 0) == 2, "");
  static_assert((m / 2)(1, 1) == 2 * m(0, 0), "");
#if EIGEN_COMP_CXXVER >= 20
  static_assert((2 * m).sum() == 2 * 4 * 5 / 2, "");
#endif

  constexpr double c = 8;
  static_assert((m * c)(0, 0) == 8, "");
  static_assert((m.array() / c).matrix() == 1 / c * m, "");
  return true;
}

static_assert(scalar_mult_div(), "");

constexpr bool determinant() {
  constexpr Map<const Matrix<double, 1, 1>> m1(static_data);
  static_assert(m1.determinant() == 1, "");

  constexpr Map<const Matrix<double, 2, 2>> m2(static_data);
  static_assert(m2.determinant() == 1 * 4 - 2 * 3, "");

  constexpr Map<const Matrix<double, 3, 3>> m3(static_data);
  static_assert(m3.determinant() == 1 * (5 * 9 - 8 * 6) - 2 * (4 * 9 - 6 * 7) + 3 * (4 * 8 - 5 * 7), "");

  static_assert(Matrix4d::Identity().determinant() == 1, "");

#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix4d M{{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}};
  static_assert(M.determinant() == -1);
#endif

  return true;
}

static_assert(determinant(), "");

constexpr bool constant_identity() {
#if EIGEN_COMP_CXXVER >= 20
  static_assert(Matrix<float, 8, 7>::Constant(5).sum() == 5 * 7 * 8);
#endif
  static_assert(Matrix3f::Zero()(0, 0) == 0, "");
  static_assert(Matrix4d::Ones()(3, 3) == 1, "");
  static_assert(Matrix2i::Identity()(0, 0) == 1 && Matrix2i::Identity()(1, 0) == 0, "");
#if EIGEN_COMP_CXXVER >= 20
  static_assert(Matrix<int, 5, 5>::Identity().trace() == 5);
#endif
  static_assert(Matrix<float, Dynamic, Dynamic>::Ones(2, 3).size() == 6, "");
  static_assert(Matrix<float, Dynamic, 1>::Zero(10).rows() == 10, "");
#if EIGEN_COMP_CXXVER >= 20
  static_assert(Matrix<int, 1, Dynamic>::Constant(20, 20).sum() == 400);
#endif

  return true;
}

static_assert(constant_identity(), "");

constexpr bool dynamic_basics() {
  // This verifies that we only calculate the entry that we need.
  static_assert(Matrix<double, Dynamic, Dynamic>::Identity(50000, 50000).array()(25, 25) == 1, "");

  static_assert(Matrix4d::Identity().block(1, 1, 2, 2)(0, 1) == 0, "");
  static_assert(MatrixXf::Identity(50, 50).transpose() == MatrixXf::Identity(50, 50), "");

  constexpr Map<const MatrixXi> dynMap(static_data_antisym, 3, 3);
  constexpr Map<const Matrix3i> staticMap(static_data_antisym);
  static_assert(dynMap == staticMap, "");
  static_assert(dynMap.transpose() != staticMap, "");
  // e.g. this hits an assertion at compile-time that would otherwise fail at runtime.
  // static_assert(dynMap != staticMap.block(1,2,0,0));

  return true;
}

static_assert(dynamic_basics(), "");

// Heap storage: allocation, copy, move, resize, conservative resize and the temporaries of products and
// decompositions. C++20 allocations must be freed within the evaluation, so the objects are local.
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_constexpr_dynamic_alloc)
constexpr bool dynamic_allocation() {
  MatrixXd m = MatrixXd::Identity(3, 3);
  bool ok = m.rows() == 3 && m(1, 1) == 1 && m(0, 1) == 0;
  VectorXd v(4);
  v << 1, 2, 3, 4;
  ok = ok && v.sum() == 10;
  MatrixXd c = m;
  c.conservativeResize(4, 2);
  ok = ok && c.rows() == 4 && c.cols() == 2 && c(1, 1) == 1 && c(2, 1) == 0;
  c.resize(2, 5);
  c.setZero();
  ok = ok && c.size() == 10 && c.sum() == 0;
  MatrixXd p = m * m + MatrixXd::Ones(3, 3);
  ok = ok && p(0, 0) == 2 && p(0, 1) == 1;
  // GEMM operands with scalar factors: half, whose factors the GEMM combines in float, a cast operand, and a product
  // of two constants.
  using MatrixXh = Matrix<half, Dynamic, Dynamic>;
  const MatrixXh h = MatrixXh::Constant(2, 2, half(1.5f));
  const MatrixXh hh = (half(2) * h) * h;
  ok = ok && hh(0, 1) == half(9);
  const MatrixXf ones = MatrixXf::Ones(3, 3);
  const MatrixXd pc = ones.cast<double>() * m;
  ok = ok && pc(2, 1) == 1;
  const MatrixXd pk = (2.0 * MatrixXd::Constant(3, 3, 1.5)) * m;
  ok = ok && pk(1, 2) == 3;
  const MatrixXd moved = std::move(p);
  ok = ok && moved(2, 2) == 2;
  const MatrixXd a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const VectorXd x = a.llt().solve(VectorXd::Ones(3));
  ok = ok && (a * x).isApprox(VectorXd::Ones(3));
  // Storage only: Clang cannot constant-evaluate libstdc++'s std::complex arithmetic.
  const VectorXcd z = VectorXcd::Constant(2, std::complex<double>(1, 2));
  ok = ok && z.size() == 2 && z(1) == std::complex<double>(1, 2);
  return ok;
}

static_assert(dynamic_allocation());

constexpr bool selfadjoint_updates() {
  const Vector3d u(1, 2, 3);
  Matrix3d r1 = Matrix3d::Zero();
  r1.selfadjointView<Lower>().rankUpdate(u);
  bool ok = r1.triangularView<Lower>().toDenseMatrix() == (u * u.transpose()).triangularView<Lower>().toDenseMatrix();
  const Matrix<double, 3, 2> w{{1, 2}, {0, 1}, {3, -1}};
  Matrix3d rk = Matrix3d::Zero();
  rk.selfadjointView<Upper>().rankUpdate(w, 2.0);
  const Matrix3d wwt = 2.0 * w * w.transpose();
  ok = ok && rk.triangularView<Upper>().toDenseMatrix() == wwt.triangularView<Upper>().toDenseMatrix();
  // Products assigned to a triangular part: the outer product and the general product selectors.
  Matrix3d t = Matrix3d::Ones();
  t.triangularView<StrictlyLower>() = u * Vector3d(1, 1, 2).transpose();
  ok = ok && t(2, 0) == 3 && t(2, 1) == 3 && t(0, 0) == 1 && t(0, 2) == 1;
  t.triangularView<Upper>() = w * w.transpose();
  ok = ok && t(0, 2) == 1 && t(1, 1) == 1 && t(2, 0) == 3;
  return ok;
}

static_assert(selfadjoint_updates());
#endif

constexpr bool sums() {
  constexpr Map<const Matrix<double, 4, 4>> m(static_data);
  constexpr auto b(m.block<2, 2>(0, 0));          // 1 2 5 6
  constexpr Map<const Matrix2d> m2(static_data);  // 1 2 3 4

  static_assert((b + m2).col(0) == 2 * Map<const Vector2d>(static_data), "");
  static_assert(b + m2 == m2 + b, "");

  static_assert((b - m2).col(0) == Vector2d::Zero(), "");
  static_assert((b - m2).col(1) == 2 * Vector2d::Ones(), "");

  static_assert((2 * b - m2).col(0) == b.col(0), "");
  static_assert((b - 2 * m2).col(0) == -b.col(0), "");

  static_assert((b - m2 + b + m2 - 2 * b) == Matrix2d::Zero(), "");

  return true;
}

static_assert(sums(), "");

constexpr bool unit_vectors() {
  static_assert(Vector4d::UnitX()(0) == 1, "");
  static_assert(Vector4d::UnitY()(1) == 1, "");
  static_assert(Vector4d::UnitZ()(2) == 1, "");
  static_assert(Vector4d::UnitW()(3) == 1, "");

  static_assert(Vector4d::UnitX().dot(Vector4d::UnitX()) == 1, "");
  static_assert(Vector4d::UnitX().dot(Vector4d::UnitY()) == 0, "");
  static_assert((Vector4d::UnitX() + Vector4d::UnitZ()).dot(Vector4d::UnitY() + Vector4d::UnitW()) == 0, "");

  return true;
}

static_assert(unit_vectors(), "");

constexpr bool construct_from_other() {
#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix2d m{Map<const Matrix2d>(static_data)};
  static_assert(m == Map<const Matrix2d>(static_data));

  constexpr Matrix2d mt{Map<const Matrix2d>(static_data).transpose()};
  static_assert(m == mt.transpose());
  static_assert(m.transpose() == mt);
  static_assert(m.diagonal() == mt.diagonal());

  constexpr Matrix3i a{Map<const Matrix3i>(static_data_antisym)};
  constexpr Matrix3i a_plus_at{a + a.transpose()};
  static_assert(a_plus_at == Matrix3i::Zero());

  constexpr Matrix3i at{Map<const Matrix<int, 3, 3, RowMajor>>(a.data())};
  static_assert(a + at == Matrix3i::Zero());
  static_assert(a + at.transpose() == 2 * a);
  static_assert(a.diagonal() == at.diagonal());

  constexpr Matrix3i aa{a};
  static_assert(aa == at.transpose());
#endif

  return true;
}

static_assert(construct_from_other(), "");

constexpr bool construct_from_values() {
#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix<float, 1, 1> m11(55);
  static_assert(m11.x() == 55);

  constexpr Vector2i m21(3.1, 42e-2);
  static_assert(m21.sum() == 3);

  constexpr Vector3d m31(2.7, 18e-3, 1e-4);
  static_assert(m31.sum() == 2.7 + 0.018 + 0.0001);

  constexpr Vector4d m41(1, 2, 3, 4);
  static_assert(m41.x() == 1 && m41.y() == 2 && m41.z() == 3 && m41.w() == 4);

  constexpr Matrix<int, 1, 6> a(1, 2, 3, 4, 5, 6);
  static_assert(a.sum() == 6 * 7 / 2);

  constexpr Matrix3d m33{{0, 1, 0}, {0, 0, -1}, {-1, 0, 0}};
  static_assert(m33.determinant() == 1);

  constexpr double data[] = {1, 1, 2, 2};
  constexpr Matrix2d md(data);
  static_assert(md.sum() == 6);
#endif

  return true;
}

static_assert(construct_from_values(), "");

constexpr bool triangular() {
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  constexpr Map<const Matrix4d> m(static_data);
  constexpr Matrix4d m2{m.triangularView<Upper>()};
  constexpr Vector4d m3{m.triangularView<Upper>().solve(m.col(0))};
  result = result && m3(0) != 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) result = result && m2(i, j) == (i < j) ? 0 : m(i, j);
#endif
  return result;
}

static_assert(triangular(), "");

static constexpr int static_data_diag[] = {1, 2, 3};
static constexpr int static_data_perm[] = {2, 0, 1};

constexpr bool diagonal_ops() {
  constexpr Map<const Vector3i> d(static_data_diag);
  constexpr Map<const Matrix3i> m(static_data_antisym);
  static_assert(d.asDiagonal().transpose().diagonal() == d, "");
  static_assert(d.asDiagonal().adjoint().diagonal() == d, "");
  static_assert((m + d.asDiagonal()).diagonal() == d, "");
  static_assert((d.asDiagonal() - m)(0, 1) == -m(0, 1), "");
  static_assert((d.asDiagonal() * m).row(2) == 3 * m.row(2), "");
  static_assert((m * d.asDiagonal()).col(1) == 2 * m.col(1), "");
  return true;
}

static_assert(diagonal_ops(), "");

constexpr bool permutation_ops() {
  constexpr Map<const Vector3i> indices(static_data_perm);
  constexpr auto p = indices.asPermutation();
  // P(i, j) = 1 exactly when i == indices(j).
  static_assert((Matrix3i::Zero() + p)(2, 0) == 1 && (Matrix3i::Zero() + p)(0, 0) == 0, "");
  static_assert((p - Matrix3i::Identity()).sum() == 0, "");
  static_assert((Matrix3i::Zero() + p).colwise().sum() == RowVector3i::Ones(), "");
  return true;
}

static_assert(permutation_ops(), "");

constexpr bool triangular_diagonal_ops() {
  constexpr Map<const Matrix3i> m(static_data_antisym);
  constexpr Map<const Vector3i> d(static_data_diag);
  static_assert((m.triangularView<Upper>() + d.asDiagonal()).coeff(0, 0) == m(0, 0) + 1, "");
  static_assert((m.triangularView<Upper>() + d.asDiagonal()).coeff(0, 2) == m(0, 2), "");
  static_assert((d.asDiagonal() - m.triangularView<Lower>()).coeff(2, 2) == 3 - m(2, 2), "");
  static_assert((d.asDiagonal() - m.triangularView<Lower>()).coeff(1, 0) == -m(1, 0), "");
  static_assert((m.triangularView<Upper>() * d.asDiagonal())(0, 2) == 3 * m(0, 2), "");
  static_assert((m.triangularView<Upper>() * d.asDiagonal())(2, 0) == 0, "");
  return true;
}

static_assert(triangular_diagonal_ops(), "");

constexpr bool abs_values() {
  static_assert(numext::abs(-3) == 3 && numext::abs(3) == 3, "");
  static_assert(numext::abs(-2.5) == 2.5 && numext::abs(2.5f) == 2.5f, "");
  static_assert(numext::absdiff(1.5, 4.0) == 2.5 && numext::absdiff(4.0f, 1.5f) == 2.5f, "");
  static_assert(numext::absdiff(4.0L, 1.5L) == 2.5L && numext::absdiff(2, 7) == 5, "");
  static_assert(numext::norm1(-2.5) == 2.5 && numext::norm1(std::complex<double>(-3, 4)) == 7, "");
  // Both zeros map to +0.0, and absdiff(+0, -0) is +0.0.
#if EIGEN_HAS_CONSTEXPR_BIT_CAST
  static_assert(numext::bit_cast<std::uint64_t>(numext::abs(-0.0)) == 0, "");
  static_assert(numext::bit_cast<std::uint64_t>(numext::abs(0.0)) == 0, "");
  static_assert(numext::bit_cast<std::uint32_t>(numext::absdiff(0.0f, -0.0f)) == 0, "");
#endif
  return true;
}

static_assert(abs_values(), "");

static constexpr double static_data_symm[] = {4, 1, 2, 0, 1, 5, 3, 0, 2, 3, 6, 0, 0, 0, 0, 7};

constexpr bool matrix_norms() {
  // A strided block reduces through the slice-vectorized traversal at run time.
  static_assert(Map<const Matrix4d>(static_data_symm).block(0, 0, 3, 2).sum() == 16, "");
#if EIGEN_COMP_CXXVER >= 20
  constexpr Map<const Matrix3i> a(static_data_antisym);
  constexpr Map<const Matrix4d> s(static_data_symm);
  static_assert(a.lpNorm<1>() == 6);
  // Column sums of the symmetric matrix: 7, 9, 11, 7. Only the selected triangle is read.
  static_assert(s.selfadjointView<Lower>().l1Norm() == 11);
  static_assert(s.selfadjointView<Upper>().l1Norm() == 11);
#endif
  return true;
}

static_assert(matrix_norms(), "");

constexpr bool structured_bindings() {
#if EIGEN_COMP_CXXVER >= 20
  Vector3i v(1, 2, 3);
  auto& [x, y, z] = v;
  z = 10;
  const auto [p, q] = Array2d(0.5, 1.5);
  return x + y == 3 && v(2) == 10 && p + q == 2.0;
#else
  return true;
#endif
}

static_assert(structured_bindings(), "");

static constexpr double static_data_spd[] = {4, 2, 0, 2, 5, 1, 0, 1, 3};

constexpr bool approx_comparisons() {
  constexpr Map<const Matrix3d> a(static_data_spd);
  static_assert(a.isApprox(a), "");
  static_assert(!a.isApprox(2 * a), "");
  static_assert(a.isApprox(a.transpose()), "");
  static_assert(a.col(0).isApprox(a.col(0) * (1 + 1e-14)), "");
  static_assert(Vector2d::UnitX().isOrthogonal(Vector2d::UnitY()), "");
  static_assert(!a.col(0).isOrthogonal(a.col(1)), "");
  return true;
}

static_assert(approx_comparisons(), "");

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

#if EIGEN_COMP_CXXVER >= 20
template <typename T>
constexpr bool float16_matrices() {
  const Matrix<T, 2, 2> m{{T(1), T(2)}, {T(3), T(4)}};
  const Matrix<T, 2, 2> p = m * m;
  return p(0, 0) == T(7) && p(1, 1) == T(22) && m.sum() == T(10) && (m + m).maxCoeff() == T(8);
}

static_assert(float16_matrices<half>());
static_assert(float16_matrices<bfloat16>());
#endif
#endif  // EIGEN_HAS_CONSTEXPR_BIT_CAST

#if EIGEN_COMP_CXXVER >= 20
// Safe scaling and exact sqrt need a constexpr numext::bit_cast too.
#if EIGEN_HAS_CONSTEXPR_BIT_CAST
constexpr bool vector_norms() {
  constexpr Map<const Matrix3d> a(static_data_spd);
  static_assert(Vector2d(3, 4).stableNorm() == 5);
  static_assert(Vector2d(3e200, 4e200).stableNorm() / 5e200 - 1 < 1e-15);
  static_assert(a.colwise().squaredNorm()(1) == 30);
  static_assert(a.colwise().norm()(0) == numext::sqrt(20.0));
  static_assert(Vector2d(3, 4).blueNorm() == 5);
  return true;
}

// Constant evaluation computes sqrt exactly, so these match the correctly rounded runtime values.
static_assert(numext::sqrt(169.0) == 13);
static_assert(numext::sqrt(2.0) == 0x1.6a09e667f3bcdp+0);
static_assert(numext::sqrt(2.0f) == 0x1.6a09e6p+0f);
static_assert(numext::sqrt(0x1p-1074) == 0x1p-537);
static_assert(numext::sqrt(0x1.fffffffffffffp+1023) == 0x1.fffffffffffffp+511);
static_assert(numext::sqrt(0x1p-149f) == 0x1.6a09e6p-75f);

static_assert(vector_norms());

// Before C++26, constant evaluation reaches the components of std::complex coefficients only in squaredNorm() and
// real scaling, which avoid realView(). C++26 forms realView() itself.
#if defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
#define EIGEN_TEST_CONSTEXPR_REAL_VIEW 1
#else
#define EIGEN_TEST_CONSTEXPR_REAL_VIEW 0
#endif

// C++20 makes std::complex arithmetic constexpr, but Clang cannot constant-evaluate libstdc++'s, which is built on
// GCC's __complex__ types.
#if defined(__cpp_lib_constexpr_complex) && !(EIGEN_COMP_CLANG && defined(__GLIBCXX__))
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 1
#else
#define EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC 0
#endif

constexpr bool complex_norms() {
  using C = std::complex<double>;
  const auto near = [](double x, double y) { return numext::abs(x - y) <= 1e-14 * y; };
  bool result = true;
  const Vector2cd v(C(3, 4), C(0, 12));
  const Matrix2cd m{{C(3, 4), C(1, 1)}, {C(0, 12), C(1, 1)}};
  result = result && v.squaredNorm() == 169 && v.norm() == 13;
  result = result && m.col(0).squaredNorm() == 169 && m.row(0).squaredNorm() == 27 && m.squaredNorm() == 173;
  result = result && m.colwise().squaredNorm() == RowVector2d(169, 4);
  result = result && m.real().sum() == 5 && m.imag().sum() == 18 && m.real()(1, 0) == 0;
  result = result && v.blueNorm() == 13;
  const Vector2d u = Vector2d(3, 4).stableNormalized();
  result = result && near(u(0), 0.6) && near(u(1), 0.8);
#if EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC || EIGEN_TEST_CONSTEXPR_REAL_VIEW
  // Before C++26 this uses complex arithmetic.
  Vector2cd w = v;
  w *= 2.0;
  w /= 4.0;
  result = result && w == Vector2cd(C(1.5, 2), C(0, 6));
#endif
#if EIGEN_TEST_CONSTEXPR_REAL_VIEW
  result = result && near(v.stableNorm(), 13) && near(v.hypotNorm(), 13);
  Matrix2cd y = Matrix2cd::Zero();
  y.real() = Matrix2d::Identity();
  y.imag()(0, 1) = 3;
  C z(1, 2);
  numext::real_ref(z) = 5;
  numext::imag_ref(z) = 6;
  result = result && y(0, 0) == C(1, 0) && y(0, 1) == C(0, 3) && y(1, 0) == C(0, 0) && z == C(5, 6);
#if EIGEN_TEST_CONSTEXPR_COMPLEX_ARITHMETIC
  result = result && v.isApprox(Vector2cd(v)) && !v.isApprox(Vector2cd(C(3, 4), C(0, 11)));
  result = result && Vector2cd::Zero().isMuchSmallerThan(v);
  const Vector2cd n = v.normalized();
  Vector2cd x = v;
  x.normalize();
  result = result && x == n && near(n.norm(), 1) && v.stableNormalized().isApprox(n);

  const Vector3cd h(C(1, 1), C(2, 0), C(0, 3));
  Vector2cd essential;
  C tau;
  double beta = 0;
  h.makeHouseholder(essential, tau, beta);
  result = result && near(beta * beta, 15);
#endif
#endif
  return result;
}

static_assert(complex_norms());
#endif  // EIGEN_HAS_CONSTEXPR_BIT_CAST

constexpr bool triangular_solve() {
  const Matrix3d l{{2, 0, 0}, {1, 4, 0}, {3, 1, 8}};
  const Matrix3d b{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
  bool ok = true;
  Matrix3d x = l.triangularView<Lower>().solve(b);
  ok = ok && (l.triangularView<Lower>() * x).isApprox(b);
  x = l.transpose().triangularView<Upper>().solve(b);
  ok = ok && (l.transpose().triangularView<Upper>() * x).isApprox(b);
  // Unit diagonal on the right: X L = B.
  x = l.triangularView<UnitLower>().solve<OnTheRight>(b);
  ok = ok && (x * l.triangularView<UnitLower>()).isApprox(b);
  const Vector3d v = l.triangularView<Lower>().solve(Vector3d(1, 2, 3));
  ok = ok && (l.triangularView<Lower>() * v).isApprox(Vector3d(1, 2, 3));
  const RowVector3d r = l.triangularView<Lower>().solve<OnTheRight>(RowVector3d(1, 2, 3));
  ok = ok && (r * l.triangularView<Lower>()).isApprox(RowVector3d(1, 2, 3));

  Matrix3d inv = l;
  inv.triangularView<Lower>().inverseInPlace();
  ok = ok && (inv.triangularView<Lower>().toDenseMatrix() * l).isIdentity();
  Matrix3d invU = l.transpose();
  invU.triangularView<Upper>().inverseInPlace();
  ok = ok && (invU.triangularView<Upper>().toDenseMatrix() * l.transpose()).isIdentity();
  return ok;
}

static_assert(triangular_solve());

constexpr bool selfadjoint_ops() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  bool ok = (a.llt().inverse() * a).isIdentity(1e-12);
  // P A P^T with P(i, indices(i)) = 1, read from the lower triangle only.
  Matrix3d b = a;
  PermutationMatrix<3> p;
  p.indices() << 2, 0, 1;
  b.selfadjointView<Lower>().applyPermutation(p);
  const Matrix3d expected = p * a * p.transpose();
  ok = ok && b.triangularView<Lower>().toDenseMatrix() == expected.triangularView<Lower>().toDenseMatrix();
  return ok;
}

static_assert(selfadjoint_ops());

struct NullDevice {};

constexpr bool device_wrapper() {
  NullDevice device;
  const Vector3d u(1, 2, 3);
  Vector3d v = Vector3d::Zero();
  v.device(device) = u + u;
  v.device(device) += u;
  v.device(device) -= Vector3d::Ones();
  v.device(device).noalias() = 2 * v;
  return v == Vector3d(4, 10, 16);
}

static_assert(device_wrapper());
#endif

constexpr bool nc_construct_from_values() {
  bool result = true;

#if EIGEN_COMP_CXXVER >= 20
  Matrix<float, 1, 1> m11(55);
  result = result && (m11.x() == 55);

  Matrix<float, 1, 1> maa(22);
  m11.swap(maa);
  result = result && (m11.x() == 22 && maa.x() == 55);
#endif
  return result;
}

constexpr bool nc_crossproduct() {
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  Vector3d X(Vector3d::UnitX());
  Vector3d Y(Vector3d::UnitY());
  Vector3d Z(Vector3d::UnitZ());

  result = result && X.cross(Y) == Z;
#endif
  return result;
}

constexpr bool nc_cast() {
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  constexpr Vector3f v(Vector3d::UnitY().cast<float>());

  result = result && v == Vector3f::UnitY();
#endif

  return result;
}

constexpr bool nc_product() {
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix2i a{{0, -1}, {1, 0}};
  constexpr Matrix2i b{{0, 1}, {-1, 0}};
  Vector2d v{{2, -2}};

  result = result && a * b == Matrix2i::Identity();
  result = result && (a.cast<double>() * v == Vector2d::Constant(2));
  result = result && (2 * a * b) == 2 * Matrix2i::Identity();
  result = result && 2 * a * b == a * 2 * b;

  constexpr Matrix3i c{{-2, 0, 0}, {0, -2, 0}, {0, 0, 2}};
  result = result && (c.block(0, 0, 2, 2) * a == 2 * b);
  result = result && (2 * a == c.block(0, 0, 2, 2) * b);
  result = result && (c.block<2, 2>(1, 1) * a == Matrix2i{{0, 2}, {2, 0}});

  constexpr Map<const Matrix<double, 2, 2, RowMajor>> m(static_data);
  result = result && (m * a.cast<double>() == Matrix2d{{2, -1}, {4, -3}});

  result = result && (a * a.transpose() == Matrix2i::Identity());
  result = result && (a.transpose() * a == Matrix2i::Identity());
  result = result && (a * b).transpose() == Matrix2i::Identity();
  result = result && (a.transpose() * b.transpose()).transpose() == Matrix2i::Identity();
#endif
  return result;
}

static constexpr double static_data_quat[] = {0, 1, 1, 0};

constexpr bool nc_quat_mult() {
  bool result = static_data_quat[3] == 0;  // Silence warning about unused with C++14

#if EIGEN_COMP_CXXVER >= 20
  constexpr Map<const Quaterniond> mqyz(static_data_quat);
  result = result && mqyz.coeffs() == Vector4d::UnitY() + Vector4d::UnitZ();

  Quaterniond q1{1, 0, 0, 0};
  Quaterniond q2{0, 1, 0, 0};

  result = result && q1 * q2 == q2;
  result = result && q1 == Quaternionf::Identity().cast<double>();

  Quaterniond q3{1, -1, 1, -1};
  result = result && q3.squaredNorm() == 4;
  result = result && q3.dot(q1) == 1 && q2.dot(q3) == -1;

  Vector3d vy = Vector3d::UnitY();
  result = result && q2 * vy == -vy;
  result = result && q2.toRotationMatrix() * vy == -vy;

  constexpr double d[] = {0, 0, 1, 0};
  Quaterniond qz(d);
  result = result && qz.z() == 1 && qz * Vector3d::UnitZ() == Vector3d::UnitZ();
  result = result && qz * qz.conjugate() == decltype(qz)::Identity();
  result = result && q3 * qz * qz.conjugate() * q3.inverse() == decltype(qz)::Identity();

  Quaternionf qf{.5, .5, .5, .5};
  Quaterniond qd(qf);
  result = result && qd * Vector3d::UnitX() == Vector3d::UnitY();
#endif
  return result;
}

constexpr bool nc_geometry() {
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  Quaterniond q{1, 2, 3, 4};
  const QuaternionBase<Quaterniond>& qb = q;
  result = result && qb.coeffsScalarFirst() == Vector4d(1, 2, 3, 4);
  result = result && qb.coeffsScalarLast() == Vector4d(2, 3, 4, 1);

  Vector4f a(1, 0, 0, 7);
  Vector4f b(0, 1, 0, -3);
  result = result && a.cross3(b) == Vector4f(0, 0, 1, 0);

  Vector3d v(1, 2, 3);
  result = result && v.homogeneous().sum() == 7;
  result = result && v.homogeneous().maxCoeff() == 3;

  // 90 degrees about z.
  const Matrix3d rz{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
  // 180 degrees about z.
  const Quaterniond qz{0, 0, 0, 1};
  Affine3d t = Affine3d::Identity();
  t.translate(Vector3d(1, 0, 0));
  t.rotate(rz);
  result = result && t * Vector3d::UnitX() == Vector3d(1, 1, 0);
  t.rotate(rz.cast<float>());
  t.prerotate(qz);
  result = result && t.linear() == Matrix3d::Identity() && t.translation() == -Vector3d::UnitX();
  t.prerotate(rz);
  result = result && t.linear() == rz && t.translation() == -Vector3d::UnitY();

  Hyperplane<double, 3> h1(Vector3d::UnitZ(), 2);
  Hyperplane<double, 3> h2(Vector3d(0, 0, -4), -8);
  result = result && h1.isCoincident(h2);
  result = result && !h1.isCoincident(Hyperplane<double, 3>(Vector3d::UnitZ(), 3));

#if defined(__cpp_lib_constexpr_cmath) && __cpp_lib_constexpr_cmath >= 202306L
  // std::sin and std::cos are constexpr from C++26 (P1383R2).
  AngleAxisd aa(0, Vector3d::UnitZ());
  result = result && aa * Vector3d::UnitX() == Vector3d::UnitX();
  Affine2d t2 = Affine2d::Identity();
  t2.rotate(0.0);
  t2.prerotate(0.0);
  result = result && t2.isApprox(Affine2d::Identity());
  // A negative w flips the axis; a pure-real quaternion has no axis and gives the zero rotation about x.
  const AngleAxisd from_q(Quaterniond(-0.6, 0, 0, 0.8));
  result = result && from_q.axis() == -Vector3d::UnitZ() && Quaterniond(from_q).isApprox(Quaterniond(0.6, 0, 0, -0.8));
  const AngleAxisd from_real(Quaterniond(-1, 0, 0, 0));
  result = result && from_real.angle() == 0 && from_real.axis() == Vector3d::UnitX();
#endif
#endif
  return result;
}

#if EIGEN_COMP_CXXVER >= 20
constexpr bool vectorwise_operands() {
  const Matrix3d m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const Vector3d v(1, 2, 4);
  bool ok = Matrix3d(v + m.colwise()) == Matrix3d(m.colwise() + v);
  ok = ok && Matrix3d(v - m.colwise()) == Matrix3d(-(m.colwise() - v));
  ok = ok && Matrix3d(m.colwise().binaryExpr(v, internal::scalar_difference_op<double, double>())) ==
                 Matrix3d(m.colwise() - v);
  const Array33d a = m.array();
  const Array3d w = v.array();
  ok = ok && (Array33d(w * a.colwise()) == Array33d(a.colwise() * w)).all();
  const Array33d q = w / a.colwise();
  ok = ok && q(1, 0) == 0.5 && q(2, 2) == 4.0 / 9;
  const Matrix<double, 6, 3> r = m.colwise().replicate(fix<2>);
  ok = ok && r.topRows<3>() == m && r.bottomRows<3>() == m;
  return ok;
}

static_assert(vectorwise_operands());

constexpr bool skew_symmetric_dense() {
  const Matrix3d m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const SkewSymmetricMatrix3<double> s(Vector3d(1, 2, 3));
  const Matrix3d sd = s.toDenseMatrix();
  bool ok = Matrix3d(m + s) == m + sd && Matrix3d(s + m) == m + sd;
  ok = ok && Matrix3d(m - s) == m - sd && Matrix3d(s - m) == sd - m;
  const Vector3d dv(1, 2, 4);
  const auto d = dv.asDiagonal();
  ok = ok && Matrix3d(s * d) == sd * d && Matrix3d(d * s) == d * sd;
  Matrix3d f = m;
  f += s;
  ok = ok && f == m + sd;
  f -= s;
  return ok && f == m;
}

static_assert(skew_symmetric_dense());

constexpr bool transpositions_assignment() {
  Vector3i indices(2, 1, 2);
  Transpositions<3> t;
  t = TranspositionsWrapper<Vector3i>(indices);
  return t.indices() == indices;
}

static_assert(transpositions_assignment());

// m(i, j) = 4 i + j.
constexpr Matrix4d index_matrix() {
  Matrix4d m;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) m(i, j) = 4 * i + j;
  return m;
}

constexpr bool indexed_views_and_reshaping() {
  using placeholders::all;
  using placeholders::last;
  using placeholders::lastN;
  const Matrix4d m = index_matrix();
  const Matrix<double, 2, 4> even_rows = m(seq(0, 2, 2), all);
  bool ok = even_rows.row(1) == m.row(2) && even_rows(0, 3) == 3;
  const Matrix2d corner = m(lastN(2), lastN(2));
  ok = ok && corner == m.bottomRightCorner<2, 2>();
  const std::array<int, 2> cols{3, 0};
  const Matrix<double, 4, 2> picked = m(all, cols);
  ok = ok && picked.col(0) == m.col(3) && picked.col(1) == m.col(0);
  ok = ok && m(last, last) == 15 && m(seqN(1, 2), 1).sum() == 5 + 9;
  Matrix4d w = m;
  w(seq(1, 2), seq(1, 2)).setZero();
  ok = ok && w.sum() == m.sum() - (5 + 6 + 9 + 10);
  // Column-major reshaping walks down the columns; RowMajor walks along the rows.
  const Matrix<double, 2, 8> r = m.reshaped(2, 8);
  ok = ok && r(1, 0) == 4 && r(0, 1) == 8 && r(1, 7) == 15;
  const Matrix<double, 16, 1> flat = m.reshaped<RowMajor>();
  ok = ok && flat(1) == 1 && flat(4) == 4;
  return ok && m.reshaped().sum() == 120;
}

static_assert(indexed_views_and_reshaping());

constexpr bool reverse_and_select() {
  const Matrix4d m = index_matrix();
  const Matrix4d r = m.reverse();
  bool ok = r(0, 0) == 15 && r(3, 0) == 3 && r(0, 3) == 12;
  ok = ok && Matrix4d(m.colwise().reverse()).row(0) == m.row(3) && Matrix4d(m.rowwise().reverse()).col(0) == m.col(3);
  Vector4d v(1, 2, 3, 4);
  v.reverseInPlace();
  ok = ok && v == Vector4d(4, 3, 2, 1);
  Matrix4d s = m;
  s.reverseInPlace();
  ok = ok && s == r;
  // Matrices with an odd number of rows keep their middle row in place.
  Matrix3d odd{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  odd.colwise().reverseInPlace();
  ok = ok && odd.row(1) == RowVector3d(4, 5, 6) && odd(0, 0) == 7;
  const Matrix4d sel = (m.array() > 7).select(m, -m);
  ok = ok && sel(1, 3) == -7 && sel(2, 0) == 8;
  const Matrix4d sel_scalar = (m.array() < 4).select(m, 0.0);
  return ok && sel_scalar.sum() == 0 + 1 + 2 + 3;
}

static_assert(reverse_and_select());

constexpr bool stl_iterators() {
  Matrix3d m{{3, 1, 2}, {6, 4, 5}, {9, 7, 8}};
  double total = 0;
  for (double x : m.reshaped()) total += x;
  bool ok = total == 45;
  Vector4d v(4, 1, 3, 2);
  std::sort(v.begin(), v.end());
  ok = ok && v == Vector4d(1, 2, 3, 4) && std::accumulate(v.cbegin(), v.cend(), 0.0) == 10;
  ok = ok && *std::max_element(v.begin(), v.end()) == 4 && v.end() - v.begin() == 4;
  // The rows of a row-major matrix and the columns of a column-major one are contiguous.
  Matrix<double, 3, 3, RowMajor> rm = m;
  for (auto row : rm.rowwise()) std::sort(row.begin(), row.end());
  ok = ok && rm == Matrix3d{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  for (auto col : m.colwise()) std::sort(col.begin(), col.end(), [](double a, double b) { return a > b; });
  ok = ok && m == Matrix3d{{9, 7, 8}, {6, 4, 5}, {3, 1, 2}};
  double first_row = 0;
  for (const auto& col : m.colwise()) first_row += col(0);
  ok = ok && first_row == 9 + 7 + 8;
  // Reshaping a column-major matrix in row-major order iterates through indices rather than a pointer.
  const Matrix3d cm = rm;
  int position = 0;
  for (double x : cm.reshaped<RowMajor>()) ok = ok && x == ++position;
  return ok && position == 9;
}

static_assert(stl_iterators());

constexpr bool band_matrix() {
  internal::BandMatrix<double, 4, 4, 1, 2> b(4, 4, 1, 2);
  b.diagonal().setConstant(4);
  b.diagonal(1).setConstant(-1);
  b.diagonal(-1).setConstant(2);
  b.diagonal<-2>().setConstant(3);
  const Matrix4d d = b.toDenseMatrix();
  Matrix4d expected = Matrix4d::Zero();
  for (int i = 0; i < 4; ++i) {
    expected(i, i) = 4;
    if (i + 1 < 4) expected(i, i + 1) = -1;
    if (i >= 1) expected(i, i - 1) = 2;
    if (i >= 2) expected(i, i - 2) = 3;
  }
  internal::TridiagonalMatrix<double, 3, ColMajor> t;
  t.diagonal() = Vector3d(1, 2, 3);
  t.super() = Vector2d(5, 6);
  t.sub() = Vector2d(7, 8);
  const Matrix3d td = t.toDenseMatrix();
  return d == expected && b.rows() == 4 && b.supers() == 1 && b.subs() == 2 && td(0, 1) == 5 && td(2, 1) == 8 &&
         td(1, 1) == 2 && td(2, 0) == 0;
}

static_assert(band_matrix());

// P(indices(j), j) = 1, so (P D)(indices(j), j) = d(j).
constexpr bool scaled_permutation() {
  const Vector3i indices(2, 0, 1);
  const PermutationMatrix<3> p(indices);
  const DiagonalMatrix<double, 3> d(2, 3, 5);
  const ScaledPermutationMatrix<double, 3> s = p * d;
  const Matrix3d pd = p.toDenseMatrix().cast<double>();
  const Matrix3d dense = pd * d.toDenseMatrix();
  bool ok = s.toDenseMatrix() == dense && s.coeff(2, 0) == 2 && s.coeff(0, 1) == 3 && s.coeff(0, 0) == 0;
  ok = ok && Matrix3d(d * p) == d.toDenseMatrix() * pd &&
       Matrix3d(p.inverse() * d) == pd.transpose() * d.toDenseMatrix();
  const Matrix3d m = index_matrix().topLeftCorner<3, 3>();
  ok = ok && Matrix3d(s * m) == dense * m && Matrix3d(m * s) == m * dense;
  Matrix3d acc = m;
  acc += s;
  ok = ok && acc == m + dense;
  ok = ok && Matrix3d(2.0 * s) == 2 * dense && Matrix3d(-s) == -dense;
  const ScaledPermutationMatrix<double, 3> t = d * p;
  ok = ok && Matrix3d(s * t).isApprox(dense * t.toDenseMatrix());
  return ok && (Matrix3d(s.inverse()) * dense).isIdentity();
}

static_assert(scaled_permutation());

struct SumAndArgmaxVisitor {
  double sum = 0;
  double best = 0;
  Index row = -1, col = -1;
  constexpr void init(const double& value, Index i, Index j) {
    sum = best = value;
    row = i;
    col = j;
  }
  constexpr void operator()(const double& value, Index i, Index j) {
    sum += value;
    if (value > best) {
      best = value;
      row = i;
      col = j;
    }
  }
};

constexpr bool visitors_and_reductions() {
  const Matrix3d m{{2, -7, 1}, {5, 0, 9}, {-3, 4, 6}};
  Index i = -1, j = -1;
  bool ok = m.minCoeff(&i, &j) == -7 && i == 0 && j == 1;
  ok = ok && m.maxCoeff(&i, &j) == 9 && i == 1 && j == 2;
  const Vector4d v(3, -1, 8, 8);
  ok = ok && v.maxCoeff(&i) == 8 && i == 2 && v.minCoeff(&i) == -1 && i == 1;
  // NaN propagation: PropagateNaN returns NaN, PropagateNumbers skips it.
  const Vector3d with_nan(1, NumTraits<double>::quiet_NaN(), -2);
  ok = ok && (numext::isnan)(with_nan.maxCoeff<PropagateNaN>()) && with_nan.minCoeff<PropagateNumbers>(&i) == -2 &&
       i == 2;
  SumAndArgmaxVisitor visitor;
  m.visit(visitor);
  ok = ok && visitor.sum == m.sum() && visitor.best == 9 && visitor.row == 1 && visitor.col == 2;
  ok = ok && m.redux([](double a, double b) { return a > b ? a : b; }) == 9;
  ok = ok && m.unaryExpr([](double x) { return x * x; }).sum() == m.squaredNorm();
  const Matrix3d n = Matrix3d::Constant(2);
  ok = ok && m.cwiseProduct(n) == 2 * m && m.cwiseQuotient(n) == m / 2;
  ok = ok && m.cwiseMin(1.0)(1, 2) == 1 && m.cwiseMax(n)(0, 1) == 2 && m.cwiseAbs().sum() == 37;
  ok = ok && m.lazyProduct(m) == Matrix3d(m * m) && m.lpNorm<Infinity>() == 9 && m.lpNorm<1>() == 37;
  ok = ok && (m.array() > 8).any() && !(m.array() > 9).any() && (m.array() > 8).count() == 1;
  ok = ok && m.allFinite() && !m.hasNaN() && with_nan.hasNaN() && !with_nan.allFinite();
  return ok && !Vector2d(1, NumTraits<double>::infinity()).allFinite();
}

static_assert(visitors_and_reductions());

constexpr bool structural_predicates() {
  const Matrix3d m{{2, 1, 0}, {0, 3, 4}, {0, 0, 5}};
  bool ok = m.isUpperTriangular() && !m.isLowerTriangular() && !m.isDiagonal();
  ok = ok && m.transpose().isLowerTriangular() && Matrix3d(m.diagonal().asDiagonal()).isDiagonal();
  ok = ok && Matrix3d::Constant(4).isConstant(4) && !m.isConstant(2) && Matrix3d::Ones().isOnes();
  ok = ok && (1e-14 * m).isZero() && !m.isZero() && Matrix3d::Zero().isZero(0);
  ok = ok && Matrix3d::Identity().isUnitary() && !m.isUnitary();
  const Matrix2d rotation{{0, -1}, {1, 0}};
  return ok && rotation.isUnitary() && (rotation * rotation.transpose()).isIdentity();
}

static_assert(structural_predicates());

constexpr bool lines_and_boxes() {
  // The line through (1, 0, 0) along y.
  const ParametrizedLine<double, 3> line(Vector3d(1, 0, 0), Vector3d::UnitY());
  bool ok = line.distance(Vector3d(4, 7, 4)) == 5 && line.squaredDistance(Vector3d(1, 2, 3)) == 9;
  ok = ok && line.projection(Vector3d(4, 7, 4)) == Vector3d(1, 7, 0) && line.pointAt(2) == Vector3d(1, 2, 0);
  const auto through = ParametrizedLine<double, 3>::Through(Vector3d(0, 0, 0), Vector3d(0, 0, 4));
  ok = ok && through.direction() == Vector3d::UnitZ();
  // The plane y = 3, which the line meets at parameter 3.
  const Hyperplane<double, 3> plane(Vector3d::UnitY(), -3);
  ok = ok && line.intersectionParameter(plane) == 3 && line.intersectionPoint(plane) == Vector3d(1, 3, 0);
  const Hyperplane<double, 2> h = Hyperplane<double, 2>::Through(Vector2d(0, 1), Vector2d(2, 1));
  ok = ok && numext::abs(h.absDistance(Vector2d(5, 4)) - 3) < 1e-14 &&
       h.projection(Vector2d(5, 4)).isApprox(Vector2d(5, 1));

  AlignedBox3d box(Vector3d(0, 0, 0), Vector3d(1, 2, 3));
  ok = ok && box.volume() == 6 && box.center() == Vector3d(0.5, 1, 1.5) && box.sizes() == Vector3d(1, 2, 3);
  ok = ok && box.contains(Vector3d(1, 1, 1)) && !box.contains(Vector3d(1, 3, 1)) && !box.isEmpty();
  // (2, 3, -1) is 1, 1 and 1 outside the faces x = 1, y = 2 and z = 0.
  ok = ok && box.squaredExteriorDistance(Vector3d(2, 3, -1)) == 3 && box.exteriorDistance(Vector3d(0.5, 1, 7)) == 4;
  const AlignedBox3d other(Vector3d(0.5, -1, 1), Vector3d(4, 1, 2));
  const AlignedBox3d meet = box.intersection(other);
  ok = ok && meet.min() == Vector3d(0.5, 0, 1) && meet.max() == Vector3d(1, 1, 2) && box.intersects(other);
  ok = ok && box.merged(other).volume() == 4 * 3 * 3;
  box.extend(Vector3d(-1, 0, 0));
  ok = ok && box.volume() == 12 && box.corner(AlignedBox3d::TopRightCeil) == Vector3d(1, 2, 3);
  AlignedBox2i empty;
  ok = ok && empty.isEmpty();
  empty.extend(Vector2i(3, 4));
  return ok && !empty.isEmpty() && empty.volume() == 0;
}

static_assert(lines_and_boxes());

constexpr bool transform_factors() {
  const Vector3d p(1, 1, 1);
  const Affine3d t = Translation3d(1, 2, 3) * Scaling(2.0, 3.0, 4.0);
  bool ok = t * p == Vector3d(3, 5, 7);
  const Affine3d u = Scaling(2.0) * Translation3d(1, 2, 3);
  ok = ok && u * p == Vector3d(4, 6, 8) && (t.inverse() * Vector3d(3, 5, 7)).isApprox(p);
  const Translation3d move(Vector3d(1, -1, 0));
  ok = ok && (move * move.inverse()).translation() == Vector3d::Zero() && move * p == Vector3d(2, 0, 1);
  const DiagonalMatrix<double, 3> stretch = Scaling(Vector3d(1, 2, 3));
  ok = ok && Matrix3d(stretch).diagonal() == Vector3d(1, 2, 3);
  ok = ok && Vector4d(2, 4, 6, 2).hnormalized() == Vector3d(1, 2, 3) && (Vector3d(1, 2, 3).homogeneous()).w() == 1;
  const Matrix<double, 3, 2> columns{{2, 4}, {4, 8}, {2, 2}};
  ok = ok && Matrix<double, 2, 2>(columns.colwise().hnormalized()) == Matrix2d{{1, 2}, {2, 4}};
  const Vector3d a(1, 2, 3);
  const Vector3d n = a.unitOrthogonal();
  ok = ok && numext::abs(n.dot(a)) < 1e-14 && numext::abs(n.norm() - 1) < 1e-14;
  const Vector2d n2 = Vector2d(3, 4).unitOrthogonal();
  return ok && n2.isApprox(Vector2d(-0.8, 0.6));
}

static_assert(transform_factors());

constexpr bool quaternion_from_two_vectors() {
  Quaterniond q;
  q.setFromTwoVectors(Vector3d(2, 0, 0), Vector3d(0, 3, 0));
  bool ok = (q * Vector3d::UnitX()).isApprox(Vector3d::UnitY()) && numext::abs(q.norm() - 1) < 1e-14;
  const Quaterniond r = Quaterniond::FromTwoVectors(Vector3d(1, 1, 0), Vector3d(0, 0, 1));
  ok = ok && (r * Vector3d(1, 1, 0).normalized()).isApprox(Vector3d::UnitZ());
  // Opposite vectors have no unique rotation; the result is a half turn about an axis orthogonal to both.
  q.setFromTwoVectors(Vector3d::UnitX(), -Vector3d::UnitX());
  ok = ok && (q * Vector3d::UnitX()).isApprox(-Vector3d::UnitX()) && numext::abs(q.w()) < 1e-12;
  return ok && q.vec().dot(Vector3d::UnitX()) == 0;
}

static_assert(quaternion_from_two_vectors());

// Points mapped by x -> 2 R x + t with R the quarter turn about z.
constexpr bool umeyama_similarity() {
  const Matrix<double, 3, 4> src{{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  const Matrix3d rotation{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
  const Vector3d translation(1, 2, 3);
  const Matrix<double, 3, 4> dst = (2 * rotation * src).colwise() + translation;
  const Matrix4d transform = umeyama(src, dst);
  bool ok =
      transform.topLeftCorner<3, 3>().isApprox(2 * rotation) && transform.topRightCorner<3, 1>().isApprox(translation);
  const Matrix4d rigid = umeyama(src, dst, false);
  return ok && rigid.topLeftCorner<3, 3>().isApprox(rotation) && transform.row(3) == RowVector4d(0, 0, 0, 1);
}

static_assert(umeyama_similarity());
#endif

static_assert(numext::copysign(true, false));
static_assert(numext::sign(std::complex<double>(3, 4)) == std::complex<double>(0.6, 0.8));

// Run tests that aren't explicitly constexpr.
constexpr bool test_nc() {
  assert_constexpr(nc_cast());
  assert_constexpr(nc_construct_from_values());
  assert_constexpr(nc_crossproduct());
  assert_constexpr(nc_geometry());
  assert_constexpr(nc_maps());
  assert_constexpr(nc_product());
  assert_constexpr(nc_quat_mult());
  return true;
}

// Before C++20, assert_constexpr is not consteval, so only constant evaluation of test_nc() makes a failed check an
// error.
static_assert(test_nc(), "");

int main() { return !test_nc(); }
