// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of sparse matrices and the sparse solvers. Like compile_time_evaluation, this test is strictly
// compile-time: main.h adds instrumentation that constant evaluation rejects.

#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <vector>

using namespace Eigen;

// Sparse storage is allocated on the heap, which constant evaluation allows from C++20.
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_constexpr_dynamic_alloc)

// The tridiagonal matrix with 4 on the diagonal and -1 off it, which is symmetric positive definite.
template <typename SparseMatrixType>
constexpr SparseMatrixType laplacian(Index n) {
  std::vector<Triplet<double>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, 4.0);
    if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
    if (i + 1 < n) triplets.emplace_back(i, i + 1, -1.0);
  }
  SparseMatrixType a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());
  return a;
}

constexpr bool sparse_basics() {
  SparseMatrix<double> a(3, 3);
  a.insert(0, 0) = 2;
  a.insert(1, 2) = 3;
  a.coeffRef(2, 1) += 4;
  a.makeCompressed();
  bool ok = a.nonZeros() == 3 && a.coeff(1, 2) == 3 && a.coeff(2, 2) == 0 && a.sum() == 9;
  const Vector3d v(1, 2, 3);
  ok = ok && (a * v).isApprox(Vector3d(2, 9, 8));
  const SparseMatrix<double> at = a.transpose();
  ok = ok && at.coeff(2, 1) == 3 && at.coeff(1, 2) == 4;
  const SparseMatrix<double> sum = a + at;
  ok = ok && sum.coeff(1, 2) == 7;
  const SparseMatrix<double> product = a * at;
  ok = ok && MatrixXd(product).isApprox(MatrixXd(a) * MatrixXd(at));
  SparseMatrix<double, RowMajor> row_major = a;
  ok = ok && row_major.coeff(2, 1) == 4;
  SparseVector<double> sv(5);
  sv.insert(3) = 1.5;
  sv.coeffRef(1) = -2;
  ok = ok && sv.nonZeros() == 2 && sv.sum() == -0.5 && sv.dot(sv) == 6.25;
  return ok;
}

static_assert(sparse_basics());

constexpr bool direct_solvers() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(6);
  const VectorXd b = VectorXd::LinSpaced(6, 1, 6);
  const SimplicialLLT<SparseMatrix<double>> llt(a);
  bool ok = llt.info() == Success && (a * llt.solve(b)).isApprox(b);
  const SimplicialLDLT<SparseMatrix<double>> ldlt(a);
  ok = ok && ldlt.info() == Success && (a * ldlt.solve(b)).isApprox(b);
  SparseLU<SparseMatrix<double>> lu(a);
  ok = ok && lu.info() == Success && (a * lu.solve(b)).isApprox(b);
  SparseMatrix<double> c = a;
  c.makeCompressed();
  const SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr(c);
  ok = ok && qr.info() == Success && (a * qr.solve(b)).isApprox(b);
  return ok;
}

static_assert(direct_solvers());

// A dense n x n matrix with a_ij = 1/(1+|i-j|) + n delta_ij, which is symmetric positive definite.
constexpr SparseMatrix<double> dense_spd(Index n) {
  SparseMatrix<double> a(n, n);
  a.reserve(VectorXi::Constant(n, int(n)));
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) a.insert(i, j) = 1.0 / double(1 + (i > j ? i - j : j - i)) + (i == j ? n : 0);
  a.makeCompressed();
  return a;
}

// The first column of the factor has 32 entries below the diagonal, enough for the packet scatter, which constant
// evaluation must skip.
constexpr bool long_columns() {
  const SparseMatrix<double> a = dense_spd(33);
  const VectorXd b = VectorXd::LinSpaced(33, 1, 2);
  const SimplicialLLT<SparseMatrix<double>, Lower, NaturalOrdering<int>> llt(a);
  return llt.info() == Success && (a * llt.solve(b)).isApprox(b);
}

static_assert(long_columns());

// Dense supernodes take SparseLU through its panel and kernel block updates.
constexpr bool supernodes() {
  const SparseMatrix<double> a = dense_spd(12);
  const VectorXd b = VectorXd::LinSpaced(12, 1, 2);
  SparseLU<SparseMatrix<double>, NaturalOrdering<int>> lu(a);
  return lu.info() == Success && (a * lu.solve(b)).isApprox(b);
}

static_assert(supernodes());

// An uncompressed matrix makes Ref<const SparseMatrix, StandardCompressedFormat> hold a compressed copy, and a block
// of rows of a column-major matrix iterates across its outer vectors.
constexpr bool refs_and_blocks() {
  SparseMatrix<double> a = laplacian<SparseMatrix<double>>(5);
  a.uncompress();
  const Ref<const SparseMatrix<double>, StandardCompressedFormat> ref(a);
  bool ok = ref.isCompressed() && ref.nonZeros() == 13 && ref.coeff(3, 2) == -1;
  const SparseMatrix<double, RowMajor> rows = a.middleRows(1, 2);
  ok = ok && rows.nonZeros() == 6 && rows.coeff(0, 1) == 4;
  double sum = 0;
  for (SparseMatrix<double>::InnerIterator it(a, 2); it; ++it) sum += it.value();
  return ok && sum == 2;
}

static_assert(refs_and_blocks());

// Iterators compare by position, and assigning a map rebinds it.
constexpr bool iterator_equality_and_map_assignment() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(4);
  SparseMatrix<double>::InnerIterator it(a, 1), other(a, 1);
  bool ok = it == other && !(it != other);
  ++other;
  ok = ok && it != other && !(it == other);
  SparseMatrix<double>::ReverseInnerIterator rit(a, 1), rother(a, 1);
  ok = ok && rit == rother;
  --rother;
  ok = ok && rit != rother;

  const SparseMatrix<double> b = 2 * a;
  Map<const SparseMatrix<double>> map(a.rows(), a.cols(), a.nonZeros(), a.outerIndexPtr(), a.innerIndexPtr(),
                                      a.valuePtr());
  const Map<const SparseMatrix<double>> map_b(b.rows(), b.cols(), b.nonZeros(), b.outerIndexPtr(), b.innerIndexPtr(),
                                              b.valuePtr());
  map = map_b;
  return ok && map.valuePtr() == b.valuePtr() && map.coeff(1, 1) == 8;
}

static_assert(iterator_equality_and_map_assignment());

// The sizes keep each evaluation within the compilers' default constexpr step limits.
template <typename Solver, Index n = 8>
constexpr bool iterative_solver() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(n);
  const VectorXd b = VectorXd::LinSpaced(n, 1, double(n));
  Solver solver;
  solver.setTolerance(1e-12);
  solver.compute(a);
  const VectorXd x = solver.solve(b);
  return solver.info() == Success && (a * x).isApprox(b, 1e-10);
}

static_assert(iterative_solver<ConjugateGradient<SparseMatrix<double>, Lower | Upper>>());
static_assert(iterative_solver<BiCGSTAB<SparseMatrix<double>>>());
static_assert(iterative_solver<LeastSquaresConjugateGradient<SparseMatrix<double>>>());
static_assert(iterative_solver<GMRES<SparseMatrix<double>>>());
static_assert(iterative_solver<DGMRES<SparseMatrix<double>>>());
static_assert(iterative_solver<IDRS<SparseMatrix<double>>, 4>());
static_assert(iterative_solver<IDRSTABL<SparseMatrix<double>>>());
static_assert(iterative_solver<BiCGSTABL<SparseMatrix<double>>>());
static_assert(iterative_solver<MINRES<SparseMatrix<double>, Lower | Upper>>());
static_assert(iterative_solver<BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>>>());
static_assert(iterative_solver<ConjugateGradient<SparseMatrix<double>, Lower, IncompleteCholesky<double>>>());
static_assert(iterative_solver<LSMR<SparseMatrix<double>>>());
static_assert(iterative_solver<BiCGSTAB<SparseMatrix<double>, IdentityPreconditioner>>());
static_assert(iterative_solver<BiCGSTAB<SparseMatrix<double>, IncompleteLU<double>>>());

// Starting from the exact solution, CG converges without iterating; IterScaling equilibrates rows and columns to
// unit infinity norm.
constexpr bool solve_with_guess_and_scaling() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(5);
  const VectorXd x0 = VectorXd::LinSpaced(5, 1, 5);
  const VectorXd b = a * x0;
  ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg(a);
  const VectorXd x = cg.solveWithGuess(b, x0);
  bool ok = cg.info() == Success && cg.iterations() == 0 && x.isApprox(x0);
  SparseMatrix<double> scaled = a;
  IterScaling<SparseMatrix<double>> scaling;
  scaling.computeRef(scaled);
  const MatrixXd reference = scaling.LeftScaling().asDiagonal() * MatrixXd(a) * scaling.RightScaling().asDiagonal();
  ok = ok && MatrixXd(scaled).isApprox(reference);
  return ok && (MatrixXd(scaled).cwiseAbs().rowwise().maxCoeff().array() - 1).abs().maxCoeff() < 1e-6;
}

static_assert(solve_with_guess_and_scaling());

// Each sparse expression is checked against the same expression on the dense copy.
constexpr bool sparse_expressions() {
  const MatrixXd d{{1, 0, 2, 0}, {0, 1e-20, 0, 3}, {4, 0, 5, 0}, {0, 6, 0, 7}};
  SparseMatrix<double> a = d.sparseView();
  bool ok = a.nonZeros() == 8 && MatrixXd(a) == d;
  ok = ok && a.isApprox(SparseMatrix<double>(d.sparseView())) && !a.isApprox(SparseMatrix<double>(2 * a));
  ok = ok && SparseMatrix<double>(d.sparseView(1, 1e-10)).nonZeros() == 7;
  SparseMatrix<double> pruned = a;
  pruned.prune(1, 1e-10);
  ok = ok && pruned.nonZeros() == 7 && pruned.coeff(1, 1) == 0;
  const Vector4d dv(1, 2, 3, 4);
  ok = ok && MatrixXd(SparseMatrix<double>(a * dv.asDiagonal())) == d * dv.asDiagonal();
  ok = ok && MatrixXd(SparseMatrix<double>(dv.asDiagonal() * a)) == dv.asDiagonal() * d;
  PermutationMatrix<Dynamic, Dynamic, int> p(4);
  p.indices() << 2, 0, 3, 1;
  ok = ok && MatrixXd(SparseMatrix<double>(p * a)) == p * d && MatrixXd(SparseMatrix<double>(a * p)) == d * p;
  // twistedBy(P) of the selfadjoint view is P A P^T with A read from one triangle.
  const MatrixXd s = d + d.transpose();
  const SparseMatrix<double> sl = SparseMatrix<double>(s.sparseView()).triangularView<Lower>();
  SparseMatrix<double> twisted;
  twisted = sl.selfadjointView<Lower>().twistedBy(p);
  ok = ok && MatrixXd(twisted).isApprox(p * s * p.transpose());
  const Vector4d v(1, -1, 2, 0.5);
  ok = ok && Vector4d(sl.selfadjointView<Lower>() * v).isApprox(s * v);
  const SparseMatrix<double> full = sl.selfadjointView<Lower>();
  ok = ok && MatrixXd(SparseMatrix<double>(sl.selfadjointView<Lower>() * a)).isApprox(s * d);
  return ok && MatrixXd(full) == s;
}

static_assert(sparse_expressions());

// A sparse right-hand side takes the sparse-sparse triangular solve.
constexpr bool sparse_triangular_solve() {
  const MatrixXd l{{2, 0, 0, 0}, {1, 4, 0, 0}, {0, 0, 1, 0}, {3, 0, 1, 8}};
  const SparseMatrix<double> ls = l.sparseView();
  SparseMatrix<double> b(4, 2);
  b.insert(0, 0) = 2;
  b.insert(2, 1) = 3;
  SparseMatrix<double> x = b;
  ls.triangularView<Lower>().solveInPlace(x);
  bool ok = MatrixXd(ls * x).isApprox(MatrixXd(b)) && x.coeff(2, 0) == 0 && x.coeff(0, 1) == 0;
  const SparseMatrix<double> u = ls.transpose();
  SparseMatrix<double> y = b;
  u.triangularView<Upper>().solveInPlace(y);
  return ok && MatrixXd(u * y).isApprox(MatrixXd(b));
}

static_assert(sparse_triangular_solve());

// The non-Hermitian factorizations of a symmetric matrix agree with the Hermitian ones for real scalars.
constexpr bool non_hermitian_simplicial() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(6);
  const VectorXd b = VectorXd::LinSpaced(6, 1, 6);
  const SimplicialNonHermitianLLT<SparseMatrix<double>> llt(a);
  const SimplicialNonHermitianLDLT<SparseMatrix<double>> ldlt(a);
  return llt.info() == Success && (a * llt.solve(b)).isApprox(b) && ldlt.info() == Success &&
         (a * ldlt.solve(b)).isApprox(b);
}

static_assert(non_hermitian_simplicial());

// Blocks of the tiled 4x4 dense matrix, with the zero (0, 1) block left out.
constexpr bool block_sparse() {
  using BSM = BlockSparseMatrix<double, ColMajor, 2, 2>;
  const MatrixXd d{{1, 2, 0, 0}, {3, 4, 0, 0}, {5, 6, 7, 8}, {-1, 2, 0, 9}};
  std::vector<BSM::TripletType> triplets;
  triplets.emplace_back(0, 0, Matrix2d(d.block<2, 2>(0, 0)));
  triplets.emplace_back(1, 0, Matrix2d(d.block<2, 2>(2, 0)));
  triplets.emplace_back(1, 1, Matrix2d(d.block<2, 2>(2, 2)));
  BSM a(2, 2);
  a.setFromTriplets(triplets.begin(), triplets.end());
  bool ok = a.nonZeroBlocks() == 3 && MatrixXd(a.toSparse()) == d && a.coeff(3, 0) == -1;
  const BSM round_trip = BSM::fromSparse(a.toSparse());
  ok = ok && MatrixXd(round_trip.toSparse()) == d;
  const BSM sum = a + a;
  const BSM product = a * a;
  ok = ok && MatrixXd(sum.toSparse()) == 2 * d && MatrixXd(product.toSparse()).isApprox(d * d);
  const BSM at = a.transpose();
  ok = ok && MatrixXd(at.toSparse()) == d.transpose();
  const Vector4d v(1, 2, 3, 4);
  ok = ok && Vector4d(a * v).isApprox(d * v);
  BSM id(2, 2);
  id.setIdentity();
  return ok && MatrixXd(id.toSparse()) == MatrixXd::Identity(4, 4);
}

static_assert(block_sparse());

#endif

int main() { return 0; }
