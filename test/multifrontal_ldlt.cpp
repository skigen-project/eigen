// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW

#include "sparse_solver.h"
#include "AnnoyingScalar.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

namespace {

template <typename RealScalar>
RealScalar pivotTolerance(Index size, RealScalar threshold = RealScalar(0.01)) {
  // The pivot tests bound |L| by 1/u, so the normwise backward error is about n * eps / u (Duff and Reid, 1983).
  return RealScalar(numext::maxi(size, Index(1))) * NumTraits<RealScalar>::epsilon() / threshold;
}

struct Inertia {
  Index positive = 0;
  Index negative = 0;
  Index zero = 0;

  void add(int sign) {
    Index& count = sign > 0 ? positive : (sign < 0 ? negative : zero);
    ++count;
  }
  bool operator==(const Inertia& other) const {
    return positive == other.positive && negative == other.negative && zero == other.zero;
  }
};

// Counts the eigenvalues of each sign from a factorization's D.
template <typename Solver>
Inertia inertiaOfD(const Solver& solver) {
  using RealScalar = typename Solver::RealScalar;
  const auto sign = [](const RealScalar& value) {
    return value > RealScalar(0) ? 1 : (value < RealScalar(0) ? -1 : 0);
  };

  Inertia inertia;
  const auto& D = solver.vectorD();
  const auto& sub = solver.subDiagonal();
  for (Index k = 0; k < D.size(); ++k) {
    if (sub[k] != typename Solver::Scalar(0)) {
      // [a b*; b c] has eigenvalues mid +- r. Take the larger magnitude one directly and the other from det / large, so
      // the rounding of a difference can't flip its sign.
      const RealScalar a = D[k];
      const RealScalar c = D[k + 1];
      const RealScalar b = numext::abs(sub[k]);
      const RealScalar mid = (a + c) / RealScalar(2);
      const RealScalar r = numext::hypot(RealScalar((a - c) / RealScalar(2)), b);
      const RealScalar large = mid >= RealScalar(0) ? RealScalar(mid + r) : RealScalar(mid - r);
      inertia.add(sign(large));
      inertia.add(sign((a / large) * c - (b / large) * b));
      ++k;
    } else {
      inertia.add(sign(D[k]));
    }
  }
  return inertia;
}

// Normwise backward error: ||A x - b||_inf / (||A||_inf ||x||_inf + ||b||_inf)
template <typename Mat, typename Rhs, typename Sol>
typename NumTraits<typename Mat::Scalar>::Real relativeResidual(const Mat& A, const Sol& x, const Rhs& b) {
  using RealScalar = typename NumTraits<typename Mat::Scalar>::Real;
  const Matrix<typename Mat::Scalar, Dynamic, Dynamic> dA = A;
  const RealScalar normA = dA.cwiseAbs().rowwise().sum().maxCoeff();
  const RealScalar scale = normA * x.template lpNorm<Infinity>() + b.template lpNorm<Infinity>();
  return (dA * x - b).template lpNorm<Infinity>() / scale;
}

template <typename Mat>
Mat fromTriplets(Index size, const std::vector<Triplet<typename Mat::Scalar, typename Mat::StorageIndex>>& triplets) {
  Mat matrix(size, size);
  matrix.setFromTriplets(triplets.begin(), triplets.end());
  matrix.makeCompressed();
  return matrix;
}

template <typename Mat>
using Triplets = std::vector<Triplet<typename Mat::Scalar, typename Mat::StorageIndex>>;

template <typename Scalar>
Scalar randomUnit() {
  Scalar value;
  do {
    value = internal::random<Scalar>();
  } while (numext::abs(value) < typename NumTraits<Scalar>::Real(0.1));
  return value / numext::abs(value);
}

// Random symmetric or Hermitian matrix with about offDiagonals entries in each triangle and a zero diagonal in every
// second row if zeroDiagonals, which forces 2x2 pivots. Its diagonal is otherwise nonzero with random signs. The
// magnitudes vary so that entries can't cancel into an exactly singular matrix.
template <typename Mat>
Mat randomHermitian(Index size, Index offDiagonals, bool zeroDiagonals) {
  using Scalar = typename Mat::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  const auto randomNonZero = []() {
    return Scalar(randomUnit<Scalar>() * internal::random<RealScalar>(RealScalar(0.5), RealScalar(1)));
  };

  Triplets<Mat> triplets;
  for (Index i = 0; i < size; ++i) {
    if (!zeroDiagonals || i % 2 == 0) triplets.emplace_back(i, i, Scalar(numext::real(randomNonZero())));
  }
  if (zeroDiagonals) {
    for (Index i = 0; i + 1 < size; ++i) {
      const Scalar value = randomNonZero();
      triplets.emplace_back(i + 1, i, value);
      triplets.emplace_back(i, i + 1, numext::conj(value));
    }
  }
  for (Index k = 0; k < offDiagonals; ++k) {
    const Index i = internal::random<Index>(0, size - 1);
    const Index j = internal::random<Index>(0, size - 1);
    if (i != j) {
      const Scalar value = internal::random<Scalar>();
      triplets.emplace_back(i, j, value);
      triplets.emplace_back(j, i, numext::conj(value));
    }
  }
  return fromTriplets<Mat>(size, triplets);
}

// The KKT matrix [H A^*; A 0] with H = diagonal * I and A = phase * B, where B is the (n-1) x n bidiagonal with rows
// e_r^T - e_{r+1}^T, the structure of a chain of dynamics constraints.
template <typename Mat>
Mat kktChain(Index numVariables, const typename Mat::Scalar& diagonal,
             const typename Mat::Scalar& phase = typename Mat::Scalar(1)) {
  using Scalar = typename Mat::Scalar;
  Triplets<Mat> triplets;
  for (Index i = 0; i < numVariables; ++i) {
    if (diagonal != Scalar(0)) triplets.emplace_back(i, i, diagonal);
  }
  for (Index r = 0; r + 1 < numVariables; ++r) {
    const Index row = numVariables + r;
    triplets.emplace_back(row, r, phase);
    triplets.emplace_back(r, row, numext::conj(phase));
    triplets.emplace_back(row, r + 1, Scalar(-phase));
    triplets.emplace_back(r + 1, row, Scalar(-numext::conj(phase)));
  }
  return fromTriplets<Mat>(2 * numVariables - 1, triplets);
}

// L is unit lower triangular, so each leading principal minor of P A P^T that ends on a block boundary of D is the
// product of the determinants of D's blocks up to there. A wrong row in P changes the minors. Their condition numbers
// can be much larger than the matrix's, so this only uses a loose tolerance.
template <typename Solver>
void checkPermutation(const Solver& solver, const typename Solver::MatrixType& full) {
  using Scalar = typename Solver::Scalar;
  using RealScalar = typename Solver::RealScalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  const Index size = full.rows();
  const auto& P = solver.permutationP();
  const auto& Pinv = solver.permutationPinv();
  VERIFY_IS_EQUAL(P.size(), size);
  VERIFY_IS_EQUAL(Pinv.size(), size);
  for (Index k = 0; k < size; ++k) VERIFY_IS_EQUAL(Index(P.indices()[Pinv.indices()[k]]), k);

  const DenseMatrix permuted = P * DenseMatrix(full) * Pinv;
  const auto& D = solver.vectorD();
  const auto& sub = solver.subDiagonal();
  const RealScalar tolerance = numext::sqrt(NumTraits<RealScalar>::epsilon());
  RealScalar product(1);
  for (Index k = 0; k < size; ++k) {
    if (sub[k] != Scalar(0)) {
      product *= D[k] * D[k + 1] - numext::abs2(sub[k]);
      ++k;
    } else {
      product *= D[k];
    }
    const Scalar leadingMinor = permuted.topLeftCorner(k + 1, k + 1).determinant();
    VERIFY(numext::abs(leadingMinor - Scalar(product)) <= tolerance * numext::abs(product));
  }
}

template <typename Solver>
void checkIndefiniteFactorization(Solver& solver, const typename Solver::MatrixType& input,
                                  const typename Solver::MatrixType& full) {
  using Scalar = typename Solver::Scalar;
  using RealScalar = typename Solver::RealScalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  solver.compute(input);
  VERIFY_IS_EQUAL(solver.info(), Success);

  const Index size = full.rows();
  const RealScalar tolerance = pivotTolerance<RealScalar>(size);

  const DenseVector b = DenseVector::Random(size);
  const DenseVector x = solver.solve(b);
  VERIFY(relativeResidual(full, x, b) <= tolerance);

  const DenseMatrix B = DenseMatrix::Random(size, 3);
  const DenseMatrix X = solver.solve(B);
  for (Index col = 0; col < B.cols(); ++col) {
    const DenseVector bCol = B.col(col);
    const DenseVector xCol = X.col(col);
    VERIFY(relativeResidual(full, xCol, bCol) <= tolerance);
    const DenseVector xSingle = solver.solve(bCol);
    VERIFY(relativeResidual(full, xSingle, bCol) <= tolerance);
  }

  // ||P A P^T - L D L^*||_F <= 64 n epsilon (||A||_F + || |L| |D| |L^*| ||_F), the normwise form of the backward error
  // bound of block LDL^T (N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM, 2002,
  // Theorem 11.3).
  const typename Solver::FactorType sparseL = solver.matrixL();
  VERIFY_IS_EQUAL(sparseL.nonZeros(), solver.nonZeros() + size);
  const DenseMatrix L = sparseL;
  VERIFY((L.array() == DenseMatrix(L.template triangularView<UnitLower>()).array()).all());
  VERIFY(DenseMatrix(solver.matrixU()) == L.adjoint());
  DenseMatrix D = DenseMatrix::Zero(size, size);
  D.diagonal() = solver.vectorD().template cast<Scalar>();
  for (Index k = 0; k + 1 < size; ++k) {
    if (solver.subDiagonal()[k] != Scalar(0)) {
      D(k + 1, k) = solver.subDiagonal()[k];
      D(k, k + 1) = numext::conj(solver.subDiagonal()[k]);
      VERIFY_IS_EQUAL(L(k + 1, k), Scalar(0));
    }
  }
  const DenseMatrix permuted = solver.permutationP() * DenseMatrix(full) * solver.permutationPinv();
  const RealScalar scale = (L.cwiseAbs() * D.cwiseAbs() * L.adjoint().cwiseAbs()).norm();
  const RealScalar factorTolerance = RealScalar(64 * size) * NumTraits<RealScalar>::epsilon();
  VERIFY((permuted - L * D * L.adjoint()).norm() <= factorTolerance * (permuted.norm() + scale));

  // An independent reference, skipped near a sign change that the working precision can't resolve.
  const DenseMatrix dense = full;
  const Matrix<RealScalar, Dynamic, 1> eigenvalues =
      SelfAdjointEigenSolver<DenseMatrix>(dense, EigenvaluesOnly).eigenvalues();
  const RealScalar gap = eigenvalues.cwiseAbs().minCoeff();
  if (gap > tolerance * eigenvalues.cwiseAbs().maxCoeff()) {
    Inertia reference;
    for (Index i = 0; i < size; ++i) reference.add(eigenvalues[i] > RealScalar(0) ? 1 : -1);
    VERIFY(inertiaOfD(solver) == reference);
    VERIFY_IS_EQUAL(solver.numPositive(), reference.positive);
    VERIFY_IS_EQUAL(solver.numNegative(), reference.negative);
    VERIFY_IS_EQUAL(solver.numZero(), Index(0));
    VERIFY_IS_EQUAL(solver.isPositive(), reference.negative == 0);
    VERIFY_IS_EQUAL(solver.isNegative(), reference.positive == 0);
  }
}

}  // namespace

// The harness shared with the other sparse solvers, on the Hermitian positive definite matrices it generates.
template <typename Scalar, typename StorageIndex, int Options>
void test_positive_definite() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  MultifrontalLDLT<Mat, Lower> lowerAmd;
  MultifrontalLDLT<Mat, Upper> upperAmd;
  MultifrontalLDLT<Mat, Lower, NaturalOrdering<StorageIndex>> lowerNatural;
  MultifrontalLDLT<Mat, Upper, NaturalOrdering<StorageIndex>> upperNatural;

  check_sparse_spd_solving(lowerAmd);
  check_sparse_spd_solving(upperAmd);
  check_sparse_spd_solving(lowerNatural, (std::min)(100, EIGEN_TEST_MAX_SIZE));
  check_sparse_spd_solving(upperNatural, (std::min)(100, EIGEN_TEST_MAX_SIZE));
  check_sparse_zero_matrix(lowerAmd);
}

// Random indefinite matrices over both triangles and the kinds of ordering, from both a full and a half input.
template <typename Scalar, typename StorageIndex, int Options>
void test_random_indefinite() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  MultifrontalLDLT<Mat, Lower> lowerAmd;
  MultifrontalLDLT<Mat, Upper> upperAmd;
  MultifrontalLDLT<Mat, Lower, NaturalOrdering<StorageIndex>> lowerNatural;
  MultifrontalLDLT<Mat, Upper, COLAMDOrdering<StorageIndex>> upperColamd;

  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    const Index size = internal::random<Index>(1, (std::min)(100, EIGEN_TEST_MAX_SIZE));
    const Index density = internal::random<Index>(0, 3);
    const bool zeroDiagonals = internal::random<bool>() && size > 1;
    const Mat full = randomHermitian<Mat>(size, size * density, zeroDiagonals);
    const Mat lower = full.template triangularView<Lower>();
    const Mat upper = full.template triangularView<Upper>();

    CALL_SUBTEST(checkIndefiniteFactorization(lowerAmd, full, full));
    CALL_SUBTEST(checkIndefiniteFactorization(lowerAmd, lower, full));
    CALL_SUBTEST(checkIndefiniteFactorization(upperAmd, full, full));
    CALL_SUBTEST(checkIndefiniteFactorization(upperAmd, upper, full));
    CALL_SUBTEST(checkIndefiniteFactorization(lowerNatural, full, full));
    CALL_SUBTEST(checkIndefiniteFactorization(lowerNatural, lower, full));
    CALL_SUBTEST(checkIndefiniteFactorization(upperColamd, upper, full));
  }
}

// One dense front with a zero diagonal: every pivot is 2x2, and the pivots swap columns that aren't neighbors.
template <typename Scalar>
void test_dense_front_with_2x2_pivots() {
  using Mat = SparseMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    // Sizes on both sides of a panel's width, and of a dense solve.
    const Index size = 2 * internal::random<Index>(2, 8) * internal::random<Index>(1, 8);
    Triplets<Mat> triplets;
    for (Index col = 0; col < size; ++col) {
      for (Index row = col + 1; row < size; ++row) {
        const Scalar value = internal::random<Scalar>();
        triplets.emplace_back(row, col, value);
        triplets.emplace_back(col, row, numext::conj(value));
      }
    }
    const Mat matrix = fromTriplets<Mat>(size, triplets);
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);

    const DenseVector b = DenseVector::Random(size);
    const DenseVector x = solver.solve(b);
    VERIFY(relativeResidual(matrix, x, b) <= pivotTolerance<RealScalar>(size));
  }
}

// Three dense blocks that share a dense separator, ordered blocks first. The separator's front absorbs one block, which
// adds no explicit zeros, and that leaves two fronts wider than a panel with rows below their pivots and a root that
// assembles both of their contribution blocks. With zero diagonals, the blocks' pivots are 2x2 and can pair columns on
// either side of a panel boundary, or be delayed to the root.
template <typename Scalar, typename StorageIndex, int Options>
void test_large_fronts() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  using Natural = NaturalOrdering<StorageIndex>;

  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    // The first repeat is wide enough to apply each front's factor through dense products in solve().
    const Index blockSize = repeat == 0 ? 80 : internal::random<Index>(1, 80);
    const Index separatorSize = repeat == 0 ? 80 : internal::random<Index>(1, 80);
    const Index numBlocks = 3;
    const Index separator = numBlocks * blockSize;
    const Index size = separator + separatorSize;
    const bool zeroDiagonals = internal::random<bool>();

    Triplets<Mat> triplets;
    for (Index col = 0; col < size; ++col) {
      const bool inSeparator = col >= separator;
      if (!zeroDiagonals || inSeparator) triplets.emplace_back(col, col, Scalar(numext::real(randomUnit<Scalar>())));
      for (Index row = col + 1; row < size; ++row) {
        if (row / blockSize == col / blockSize || row >= separator || inSeparator) {
          const Scalar value = internal::random<Scalar>();
          triplets.emplace_back(row, col, value);
          triplets.emplace_back(col, row, numext::conj(value));
        }
      }
    }
    const Mat full = fromTriplets<Mat>(size, triplets);
    const Mat lower = full.template triangularView<Lower>();

    MultifrontalLDLT<Mat, Lower, Natural> natural;
    CALL_SUBTEST(checkIndefiniteFactorization(natural, lower, full));

    MultifrontalLDLT<Mat> amd;
    CALL_SUBTEST(checkIndefiniteFactorization(amd, full, full));
  }
}

// The KKT matrix [delta I A^*; A h I] with h / |A| above 1 / threshold. Constraint columns fail both pivot tests until
// enough variables are eliminated, while each variable passes its 1x1 test. The dense blocks keep every column in one
// front, up to several panels wide, with the columns that fail first. An optional dense block at the end adds rows
// below that front: half its columns are coupled to every fully summed column, and the other half are too many explicit
// zeros for the front to merge into the block's front. So it's a non-root front that pivots on variables and delays the
// constraint columns that still fail.
template <typename Scalar, typename StorageIndex, int Options>
void test_failing_columns_before_pivots() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using Natural = NaturalOrdering<StorageIndex>;

  const RealScalar eps = NumTraits<RealScalar>::epsilon();
  for (const Index numConstraints : {Index(5), Index(70), Index(150)}) {
    for (const Index numVariables : {Index(5), Index(70), Index(150)}) {
      for (const Index numTail : {Index(0), numConstraints}) {
        for (const RealScalar h : {RealScalar(1e3), RealScalar(1) / numext::sqrt(eps)}) {
          const Index numFully = numConstraints + numVariables;
          const Index numRoot = 2 * numTail;
          const Index size = numFully + numRoot;
          Triplets<Mat> triplets;
          const auto addOffDiagonal = [&](Index row, Index col, const Scalar& value) {
            triplets.emplace_back(row, col, value);
            triplets.emplace_back(col, row, numext::conj(value));
          };
          // The magnitudes vary so that entries can't cancel into an exactly singular matrix.
          const auto randomNonZero = []() {
            return Scalar(randomUnit<Scalar>() * internal::random<RealScalar>(0.5, 1));
          };
          for (Index i = 0; i < numConstraints; ++i) {
            triplets.emplace_back(i, i, Scalar(-eps));
            for (Index j = 0; j < i; ++j) addOffDiagonal(i, j, Scalar(0));
            for (Index j = numConstraints; j < numFully; ++j) {
              addOffDiagonal(j, i, randomNonZero());
            }
          }
          for (Index i = numConstraints; i < numFully; ++i) {
            triplets.emplace_back(i, i, Scalar(h));
            for (Index j = numConstraints; j < i; ++j) addOffDiagonal(i, j, Scalar(0));
          }
          for (Index i = 0; i < numRoot; ++i) {
            triplets.emplace_back(numFully + i, numFully + i, Scalar(RealScalar(i % 2 == 0 ? 1 : -1)));
            for (Index j = 0; j < i; ++j) addOffDiagonal(numFully + i, numFully + j, randomNonZero());
          }
          for (Index i = 0; i < numTail; ++i) {
            for (Index j = 0; j < numFully; ++j) addOffDiagonal(numFully + i, j, randomNonZero());
          }
          const Mat full = fromTriplets<Mat>(size, triplets);

          MultifrontalLDLT<Mat, Lower, Natural> natural;
          CALL_SUBTEST(checkIndefiniteFactorization(natural, full, full));

          MultifrontalLDLT<Mat> amd;
          CALL_SUBTEST(checkIndefiniteFactorization(amd, full, full));
        }
      }
    }
  }
}

// Merged fronts on diagonally dominant matrices, where no pivot is delayed, so nonZeros() counts the analysis's stored
// entries. SimplicialLDLT gives the fill without explicit zeros.
template <typename Scalar, typename StorageIndex, int Options>
void test_relaxed_fronts() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using Natural = NaturalOrdering<StorageIndex>;

  const auto addOffDiagonal = [](Triplets<Mat>& triplets, Index row, Index col) {
    const Scalar value = randomUnit<Scalar>();
    triplets.emplace_back(row, col, value);
    triplets.emplace_back(col, row, numext::conj(value));
  };
  const auto addDiagonal = [](Triplets<Mat>& triplets, Index size, Index value) {
    for (Index i = 0; i < size; ++i) triplets.emplace_back(i, i, Scalar(RealScalar(value)));
  };

  // Returns the number of explicit zeros in L, which no front's limit lets exceed 30% of its entries.
  const auto verifyAgainstExactFill = [](const auto& solver, const auto& exact, const Mat& matrix) {
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(exact.info(), Success);
    VERIFY_IS_EQUAL(solver.numPositive(), matrix.rows());
    const Index size = matrix.rows();
    const Index exactNonZeros = exact.matrixL().nestedExpression().nonZeros();
    const Index stored = solver.nonZeros() + size;
    VERIFY(solver.nonZeros() >= exactNonZeros);
    VERIFY(RealScalar(0.7) * RealScalar(stored) <= RealScalar(exactNonZeros + size));
    const DenseVector b = DenseVector::Random(size);
    const DenseVector x = solver.solve(b);
    VERIFY(relativeResidual(matrix, x, b) <= pivotTolerance<RealScalar>(size));
    return solver.nonZeros() - exactNonZeros;
  };

  {
    // A tridiagonal matrix's fundamental fronts have one pivot and one row below. Merging two adds one explicit zero.
    const Index size = 30;
    Triplets<Mat> triplets;
    addDiagonal(triplets, size, 4);
    for (Index i = 0; i + 1 < size; ++i) addOffDiagonal(triplets, i + 1, i);
    const Mat matrix = fromTriplets<Mat>(size, triplets);
    MultifrontalLDLT<Mat, Lower, Natural> solver(matrix);
    SimplicialLDLT<Mat, Lower, Natural> exact(matrix);
    VERIFY(verifyAgainstExactFill(solver, exact, matrix) > 0);
    CALL_SUBTEST(checkPermutation(solver, matrix));
  }
  {
    // A separator with two children: a dense block coupled to one of its rows, then one column coupled to all of them.
    // The tree's postorder puts the block between the column and the separator. The column's merge adds no explicit
    // zeros, so it moves past the block to be contiguous with the separator. The block's merge would add too many.
    const Index blockSize = 8;
    const Index separatorSize = 8;
    const Index column = blockSize;
    const Index separator = blockSize + 1;
    const Index size = separator + separatorSize;
    Triplets<Mat> triplets;
    addDiagonal(triplets, size, 2 * size);
    for (Index j = 0; j < blockSize; ++j) {
      for (Index i = j + 1; i < blockSize; ++i) addOffDiagonal(triplets, i, j);
      addOffDiagonal(triplets, separator, j);
    }
    for (Index j = separator; j < size; ++j) {
      addOffDiagonal(triplets, j, column);
      for (Index i = j + 1; i < size; ++i) addOffDiagonal(triplets, i, j);
    }
    const Mat matrix = fromTriplets<Mat>(size, triplets);
    MultifrontalLDLT<Mat, Lower, Natural> solver(matrix);
    SimplicialLDLT<Mat, Lower, Natural> exact(matrix);
    VERIFY_IS_EQUAL(verifyAgainstExactFill(solver, exact, matrix), Index(0));
    VERIFY_IS_EQUAL(Index(solver.permutationP().indices()[column]), blockSize);
    CALL_SUBTEST(checkPermutation(solver, matrix));
  }
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    // Random patterns over the front sizes that the limits distinguish.
    const Index size = internal::random<Index>(2, 150);
    Triplets<Mat> triplets;
    addDiagonal(triplets, size, 2 * size);
    const Index offDiagonals = size * internal::random<Index>(0, 4);
    for (Index k = 0; k < offDiagonals; ++k) {
      const Index i = internal::random<Index>(0, size - 1);
      const Index j = internal::random<Index>(0, size - 1);
      if (i != j) addOffDiagonal(triplets, (std::max)(i, j), (std::min)(i, j));
    }
    const Mat matrix = fromTriplets<Mat>(size, triplets);
    MultifrontalLDLT<Mat, Lower, Natural> natural(matrix);
    SimplicialLDLT<Mat, Lower, Natural> exactNatural(matrix);
    verifyAgainstExactFill(natural, exactNatural, matrix);
    MultifrontalLDLT<Mat> amd(matrix);
    SimplicialLDLT<Mat> exactAmd(matrix);
    verifyAgainstExactFill(amd, exactAmd, matrix);
  }
}

// Reference determinants come from dense LU, whose error is bounded by the conditioning.
template <typename Scalar, typename StorageIndex, int Options>
void test_determinant() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  MultifrontalLDLT<Mat, Lower> lowerSolver;
  MultifrontalLDLT<Mat, Upper> upperSolver;
  check_sparse_spd_determinant(lowerSolver);
  check_sparse_spd_determinant(upperSolver);

  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    const Index size = internal::random<Index>(1, 30);
    const Mat full = randomHermitian<Mat>(size, 2 * size, internal::random<bool>() && size > 1);
    const DenseMatrix dense = full;
    lowerSolver.compute(full);
    VERIFY_IS_EQUAL(lowerSolver.info(), Success);
    check_sparse_determinant(lowerSolver, full, dense);
    // A Hermitian matrix has a real determinant.
    VERIFY_IS_EQUAL(numext::imag(lowerSolver.determinant()), RealScalar(0));
  }

  // [0 s; s 0] has determinant -s^2, which must not lose its sign to a 2x2 block's scaling.
  const RealScalar s(4);
  const Mat block = fromTriplets<Mat>(2, {{1, 0, Scalar(s)}, {0, 1, Scalar(s)}});
  lowerSolver.compute(block);
  VERIFY_IS_APPROX(numext::real(lowerSolver.determinant()), RealScalar(-16));

  const Mat singular = kktChain<Mat>(3, Scalar(0));
  lowerSolver.compute(singular);
  VERIFY_IS_EQUAL(lowerSolver.info(), NumericalIssue);
  VERIFY_IS_EQUAL(lowerSolver.determinant(), Scalar(0));

  MultifrontalLDLT<Mat> unfactorized;
  VERIFY_RAISES_ASSERT(unfactorized.determinant());
  unfactorized.analyzePattern(block);
  VERIFY_RAISES_ASSERT(unfactorized.determinant());
}

// Which pivots get delayed depends on the values, so the same analysis must handle fronts of different sizes, and the
// workspace must stop growing at the largest of them.
template <typename Scalar, typename StorageIndex, int Options>
void test_delayed_pivots() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using Natural = NaturalOrdering<StorageIndex>;

  const Index numVariables = 20;
  const Index size = 2 * numVariables - 1;
  const Scalar phase = randomUnit<Scalar>();
  const Mat small = kktChain<Mat>(numVariables, Scalar(RealScalar(1e-6)), phase);
  const Mat large = kktChain<Mat>(numVariables, Scalar(10), phase);
  const DenseVector b = DenseVector::Random(size);

  Inertia reference;
  reference.positive = numVariables;
  reference.negative = numVariables - 1;

  // Each variable's small diagonal fails the pivot test until the column reaches a constraint row it can pair with.
  MultifrontalLDLT<Mat, Lower, Natural> delaying(small);
  VERIFY_IS_EQUAL(delaying.info(), Success);
  VERIFY_IS_EQUAL(delaying.numPositive(), numVariables);
  VERIFY_IS_EQUAL(delaying.numNegative(), numVariables - 1);
  VERIFY_IS_EQUAL(delaying.numZero(), Index(0));
  VERIFY(inertiaOfD(delaying) == reference);
  // The delays take the pivots out of the natural order.
  using IndexVector = Matrix<StorageIndex, Dynamic, 1>;
  VERIFY(delaying.permutationP().indices() != IndexVector::LinSpaced(size, 0, size - 1));
  CALL_SUBTEST(checkPermutation(delaying, small));
  const DenseVector delayedX = delaying.solve(b);
  VERIFY(relativeResidual(small, delayedX, b) <= pivotTolerance<RealScalar>(size));

  MultifrontalLDLT<Mat> solver;
  solver.analyzePattern(small);
  VERIFY_IS_EQUAL(solver.info(), InvalidInput);
  DenseVector unfactorizedX(size);
  VERIFY_RAISES_ASSERT(unfactorizedX = solver.solve(b));
  solver.factorize(small);
  solver.factorize(large);
  for (const Mat* matrix : {&small, &large, &small}) {
    solver.factorize(*matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(inertiaOfD(solver) == reference);
    const DenseVector x = solver.solve(b);
    VERIFY(relativeResidual(*matrix, x, b) <= pivotTolerance<RealScalar>(size));
  }
}

// Pivot rows and D must describe the same matrix, including when 2x2 pivots and delays reorder the pivots.
template <typename Scalar, typename StorageIndex, int Options>
void test_permutation() {
  using Mat = SparseMatrix<Scalar, Options, StorageIndex>;
  using Natural = NaturalOrdering<StorageIndex>;

  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    const Index size = internal::random<Index>(2, 20);
    for (const bool zeroDiagonals : {false, true}) {
      const Mat full = randomHermitian<Mat>(size, 2 * size, zeroDiagonals);
      MultifrontalLDLT<Mat> amd(full);
      VERIFY_IS_EQUAL(amd.info(), Success);
      CALL_SUBTEST(checkPermutation(amd, full));
      MultifrontalLDLT<Mat, Lower, Natural> natural(full);
      VERIFY_IS_EQUAL(natural.info(), Success);
      CALL_SUBTEST(checkPermutation(natural, full));
    }
  }
}

// The threshold trades fill against the bound on |L|, but the factorization must be accurate over its whole range.
template <typename Scalar>
void test_pivot_threshold() {
  using Mat = SparseMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  MultifrontalLDLT<Mat> solver;
  VERIFY_IS_EQUAL(solver.pivotThreshold(), RealScalar(0.01));
  VERIFY_RAISES_ASSERT(solver.setPivotThreshold(RealScalar(0)));
  VERIFY_RAISES_ASSERT(solver.setPivotThreshold(RealScalar(0.75)));

  const Mat matrix = kktChain<Mat>(10, Scalar(RealScalar(1e-2)), randomUnit<Scalar>());
  const DenseVector b = DenseVector::Random(matrix.rows());
  for (const RealScalar threshold : {RealScalar(1e-4), RealScalar(1e-1), RealScalar(0.5)}) {
    VERIFY_IS_EQUAL(solver.setPivotThreshold(threshold).pivotThreshold(), threshold);
    solver.compute(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    const DenseVector x = solver.solve(b);
    VERIFY(relativeResidual(matrix, x, b) <= pivotTolerance<RealScalar>(matrix.rows(), threshold));
  }
}

// With the same ordering, every input form is the same matrix, so its factorization must match exactly.
template <typename Scalar>
void test_input_forms() {
  using ColMat = SparseMatrix<Scalar, ColMajor>;
  using RowMat = SparseMatrix<Scalar, RowMajor>;
  using Natural = NaturalOrdering<typename ColMat::StorageIndex>;

  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    const ColMat full = randomHermitian<ColMat>(30, 60, internal::random<bool>());
    const ColMat lower = full.template triangularView<Lower>();
    const ColMat upper = full.template triangularView<Upper>();
    const RowMat rowFull = full;
    const RowMat rowUpper = upper;
    ColMat uncompressed = lower;
    uncompressed.uncompress();
    VERIFY(!uncompressed.isCompressed());
    // Spare room between the inner vectors moves each entry away from its position in the compressed matrix.
    ColMat uncompressedFull = full;
    uncompressedFull.reserve(VectorXi::Constant(full.outerSize(), 2));
    VERIFY(!uncompressedFull.isCompressed());

    MultifrontalLDLT<ColMat, Lower, Natural> fromFull(full);
    MultifrontalLDLT<ColMat, Lower, Natural> fromLower(lower);
    MultifrontalLDLT<ColMat, Upper, Natural> fromUpper(upper);
    MultifrontalLDLT<ColMat, Lower, Natural> fromUncompressed(uncompressed);
    MultifrontalLDLT<ColMat, Lower, Natural> fromUncompressedFull(uncompressedFull);
    MultifrontalLDLT<RowMat, Lower, Natural> fromRowFull(rowFull);
    MultifrontalLDLT<RowMat, Upper, Natural> fromRowUpper(rowUpper);

    const auto verifySameFactorization = [&](const auto& other) {
      VERIFY_IS_EQUAL(other.info(), Success);
      VERIFY_IS_EQUAL(other.vectorD(), fromFull.vectorD());
      VERIFY_IS_EQUAL(other.subDiagonal(), fromFull.subDiagonal());
      VERIFY_IS_EQUAL(other.permutationP().indices(), fromFull.permutationP().indices());
    };
    VERIFY_IS_EQUAL(fromFull.info(), Success);
    verifySameFactorization(fromLower);
    verifySameFactorization(fromUpper);
    verifySameFactorization(fromUncompressed);
    verifySameFactorization(fromUncompressedFull);
    // The analysis holds for the pattern in either storage mode.
    MultifrontalLDLT<ColMat, Lower, Natural> acrossModes;
    acrossModes.analyzePattern(full);
    acrossModes.factorize(uncompressedFull);
    verifySameFactorization(acrossModes);
    acrossModes.analyzePattern(uncompressedFull);
    acrossModes.factorize(full);
    verifySameFactorization(acrossModes);
    verifySameFactorization(fromRowFull);
    verifySameFactorization(fromRowUpper);
  }
}

template <typename Scalar>
void test_small_and_singular() {
  using Mat = SparseMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using Natural = NaturalOrdering<typename Mat::StorageIndex>;

  const auto inertia = [](Index positive, Index negative, Index zero) {
    Inertia result;
    result.positive = positive;
    result.negative = negative;
    result.zero = zero;
    return result;
  };

  {
    MultifrontalLDLT<Mat> solver;
    VERIFY_RAISES_ASSERT(solver.info());
    VERIFY_RAISES_ASSERT(solver.factorize(Mat(1, 1)));
    VERIFY_RAISES_ASSERT(solver.numPositive());
    VERIFY_RAISES_ASSERT(solver.numNegative());
    VERIFY_RAISES_ASSERT(solver.numZero());
    VERIFY_RAISES_ASSERT(solver.isPositive());
    VERIFY_RAISES_ASSERT(solver.isNegative());
    VERIFY_RAISES_ASSERT(solver.vectorD());
    VERIFY_RAISES_ASSERT(solver.subDiagonal());
    VERIFY_RAISES_ASSERT(solver.permutationP());
    VERIFY_RAISES_ASSERT(solver.permutationPinv());
    VERIFY_RAISES_ASSERT(solver.nonZeros());
  }
  {
    // analyzePattern() leaves no factorization to read, and factorize() checks the analyzed pattern's entry count.
    const Mat diagonal = fromTriplets<Mat>(2, {{0, 0, Scalar(1)}, {1, 1, Scalar(1)}});
    const Mat full = fromTriplets<Mat>(2, {{0, 0, Scalar(1)}, {1, 0, Scalar(1)}, {0, 1, Scalar(1)}, {1, 1, Scalar(2)}});
    MultifrontalLDLT<Mat> solver(diagonal);
    VERIFY_IS_EQUAL(solver.nonZeros(), 0);
    solver.analyzePattern(full);
    VERIFY_RAISES_ASSERT(solver.vectorD());
    VERIFY_RAISES_ASSERT(solver.nonZeros());
    VERIFY_RAISES_ASSERT(solver.factorize(diagonal));
    solver.analyzePattern(diagonal);
    VERIFY_RAISES_ASSERT(solver.factorize(full));
  }
  {
    const Mat matrix(0, 0);
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(solver.rows(), 0);
    VERIFY_IS_EQUAL(solver.vectorD().size(), 0);
    const DenseVector x = solver.solve(DenseVector(0));
    VERIFY_IS_EQUAL(x.size(), 0);
  }
  {
    const Mat matrix = fromTriplets<Mat>(1, {{0, 0, Scalar(2)}});
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(solver.isPositive());
    VERIFY(!solver.isNegative());
    const DenseVector b = DenseVector::Constant(1, Scalar(3));
    const DenseVector x = solver.solve(b);
    VERIFY_IS_EQUAL(x[0], Scalar(RealScalar(1.5)));
  }
  {
    const Mat matrix = fromTriplets<Mat>(1, {{0, 0, Scalar(-2)}});
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(!solver.isPositive());
    VERIFY(solver.isNegative());
  }
  {
    const Mat matrix = fromTriplets<Mat>(1, {{0, 0, Scalar(0)}});
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    VERIFY(inertiaOfD(solver) == inertia(0, 0, 1));
    const DenseVector b = DenseVector::Constant(1, Scalar(3));
    const DenseVector x = solver.solve(b);
    VERIFY_IS_EQUAL(x[0], Scalar(0));
  }
  {
    // Structurally empty diagonal
    const Mat matrix(2, 2);
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    VERIFY(inertiaOfD(solver) == inertia(0, 0, 2));
    VERIFY_IS_EQUAL(solver.numZero(), Index(2));
    // The zero matrix is semidefinite both ways.
    VERIFY(solver.isPositive());
    VERIFY(solver.isNegative());
  }
  for (Index numVariables : {2, 3, 10, 50}) {
    // [0 A^*; A 0] is singular because A has fewer rows than columns.
    const Mat matrix = kktChain<Mat>(numVariables, Scalar(0));
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    VERIFY(inertiaOfD(solver) == inertia(numVariables - 1, numVariables - 1, 1));
    VERIFY_IS_EQUAL(solver.numPositive(), numVariables - 1);
    VERIFY_IS_EQUAL(solver.numNegative(), numVariables - 1);
    VERIFY_IS_EQUAL(solver.numZero(), Index(1));
    VERIFY(!solver.isPositive());
    VERIFY(!solver.isNegative());
  }
  {
    // [I A^*; A 0] where A's two rows are equal. Rounding can leave the zero eigenvalue's pivot tiny instead of exactly
    // zero, so look at the smallest pivot.
    Triplets<Mat> triplets;
    for (Index i = 0; i < 3; ++i) triplets.emplace_back(i, i, Scalar(1));
    for (Index row : {3, 4}) {
      for (Index col : {0, 1, 2}) {
        triplets.emplace_back(row, col, Scalar(1));
        triplets.emplace_back(col, row, Scalar(1));
      }
    }
    const Mat matrix = fromTriplets<Mat>(5, triplets);
    MultifrontalLDLT<Mat> solver(matrix);
    const Inertia counts = inertiaOfD(solver);
    VERIFY_IS_EQUAL(counts.positive, 3);
    VERIFY_IS_EQUAL(counts.negative, 1);
    VERIFY(solver.vectorD().cwiseAbs().minCoeff() <= RealScalar(10) * NumTraits<RealScalar>::epsilon());
  }
  {
    // A 2x2 pivot with det > 0: the tiny diagonal fails the 1x1 test, and both eigenvalues have the sign of the trace.
    const RealScalar tiny(1e-4);
    const RealScalar large(2e5);
    Mat matrix =
        fromTriplets<Mat>(2, {{0, 0, Scalar(tiny)}, {1, 0, Scalar(1)}, {0, 1, Scalar(1)}, {1, 1, Scalar(large)}});
    MultifrontalLDLT<Mat, Lower, Natural> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(numext::abs(solver.subDiagonal()[0]) != RealScalar(0));
    VERIFY(solver.isPositive());
    VERIFY(!solver.isNegative());
    matrix = -matrix;
    solver.compute(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(!solver.isPositive());
    VERIFY(solver.isNegative());
  }
  {
    // Disconnected blocks with a zero diagonal, so each root must take a 2x2 pivot.
    Triplets<Mat> triplets;
    for (Index block = 0; block < 4; ++block) {
      triplets.emplace_back(2 * block + 1, 2 * block, Scalar(block + 1));
      triplets.emplace_back(2 * block, 2 * block + 1, Scalar(block + 1));
    }
    const Mat matrix = fromTriplets<Mat>(8, triplets);
    MultifrontalLDLT<Mat, Lower, Natural> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(inertiaOfD(solver) == inertia(4, 4, 0));
    const DenseVector b = DenseVector::Random(8);
    const DenseVector x = solver.solve(b);
    VERIFY(relativeResidual(matrix, x, b) <= RealScalar(8) * NumTraits<RealScalar>::epsilon());
  }
}

// A non-finite pivot has no inertia, so the factorization fails and the matrix is neither semidefinite nor counted in
// full, whichever of the 1x1 and 2x2 paths it took.
template <typename Mat>
void verifyNonFiniteFactorization(const Mat& matrix) {
  using Natural = NaturalOrdering<typename Mat::StorageIndex>;
  const auto verify = [&matrix](const auto& solver) {
    VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    VERIFY(!solver.isPositive());
    VERIFY(!solver.isNegative());
    VERIFY(solver.numPositive() + solver.numNegative() + solver.numZero() < matrix.rows());
    VERIFY(!(numext::isfinite)(numext::real(solver.determinant())));
  };
  verify(MultifrontalLDLT<Mat, Lower, Natural>(matrix));
  verify(MultifrontalLDLT<Mat, Upper, Natural>(matrix));
}

template <typename Scalar>
void test_extreme_values() {
  using Mat = SparseMatrix<Scalar>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  {
    // s^2 overflows, so a 2x2 pivot must divide by its off-diagonal instead of forming its determinant.
    const RealScalar s = numext::pow(NumTraits<RealScalar>::highest(), RealScalar(0.75));
    const Mat matrix = fromTriplets<Mat>(2, {{1, 0, Scalar(s)}, {0, 1, Scalar(s)}});
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(!solver.isPositive());
    VERIFY(!solver.isNegative());
    DenseVector b(2);
    b << Scalar(s), Scalar(RealScalar(2) * s);
    const DenseVector x = solver.solve(b);
    VERIFY_IS_EQUAL(x[0], Scalar(2));
    VERIFY_IS_EQUAL(x[1], Scalar(1));
  }

  {
    // A 2x2 block of zero diagonals where u * s underflows to zero. u and s are normal, so flush-to-zero gives the same
    // result.
    const RealScalar u = (std::numeric_limits<RealScalar>::min)();
    const RealScalar s = u / NumTraits<RealScalar>::epsilon();
    const Mat matrix = fromTriplets<Mat>(2, {{1, 0, Scalar(s)}, {0, 1, Scalar(s)}});
    MultifrontalLDLT<Mat> solver;
    solver.setPivotThreshold(u).compute(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_EQUAL(solver.numPositive(), 1);
    VERIFY_IS_EQUAL(solver.numNegative(), 1);
    VERIFY_IS_EQUAL(solver.numZero(), 0);
    DenseVector b(2);
    b << Scalar(s), Scalar(RealScalar(2) * s);
    const DenseVector x = solver.solve(b);
    VERIFY_IS_EQUAL(x[0], Scalar(2));
    VERIFY_IS_EQUAL(x[1], Scalar(1));
  }

  {
    // A 2x2 candidate whose off-diagonal is far below sqrt(|d11 d22|): (d11/|d21|)(d22/|d21|) overflows though det
    // doesn't. Its bound on column 0 is about |d11|/u, so the row below rejects it and the column waits for that row.
    const Scalar tinyOffDiagonal =
        randomUnit<Scalar>() * RealScalar(0.1) *
        numext::sqrt(RealScalar(NumTraits<RealScalar>::epsilon() / NumTraits<RealScalar>::highest()));
    const Scalar d11(NumTraits<RealScalar>::epsilon());
    const Mat matrix = fromTriplets<Mat>(3, {{0, 0, d11},
                                             {1, 0, tinyOffDiagonal},
                                             {0, 1, Scalar(numext::conj(tinyOffDiagonal))},
                                             {1, 1, Scalar(1)},
                                             {2, 0, Scalar(1)},
                                             {0, 2, Scalar(1)},
                                             {2, 1, Scalar(1)},
                                             {1, 2, Scalar(1)},
                                             {2, 2, Scalar(1)}});
    MultifrontalLDLT<Mat, Lower, NaturalOrdering<typename Mat::StorageIndex>> solver;
    CALL_SUBTEST(checkIndefiniteFactorization(solver, matrix, matrix));
    VERIFY_IS_EQUAL(solver.numPositive(), 2);
    VERIFY_IS_EQUAL(solver.numNegative(), 1);
  }

  {
    // d22/|d21| is out of range. The tiny diagonal fails the 1x1 test against d21, and the block's scaled determinant
    // underflows, so the 2x2 is rejected in favor of d22 as a 1x1 pivot. det > 0, so both eigenvalues are positive.
    const RealScalar sqrtHighest = numext::sqrt(NumTraits<RealScalar>::highest());
    const RealScalar d21 = RealScalar(1) / sqrtHighest;
    const RealScalar d11 = d21 * RealScalar(1e-4);
    const RealScalar d22 = numext::pow(NumTraits<RealScalar>::highest(), RealScalar(0.75));
    const Scalar offDiagonal = randomUnit<Scalar>() * d21;
    const Mat matrix = fromTriplets<Mat>(
        2, {{0, 0, Scalar(d11)}, {1, 0, offDiagonal}, {0, 1, Scalar(numext::conj(offDiagonal))}, {1, 1, Scalar(d22)}});
    MultifrontalLDLT<Mat, Lower, NaturalOrdering<typename Mat::StorageIndex>> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(solver.isPositive());
    VERIFY(!solver.isNegative());
    VERIFY(numext::real(solver.determinant()) > RealScalar(0));
    DenseVector b(2);
    b << Scalar(d11 + d21), Scalar(d22 + d21);
    const DenseVector x = solver.solve(b);
    VERIFY(x.allFinite());
    VERIFY(relativeResidual(matrix, x, b) <= pivotTolerance<RealScalar>(2));
  }

  // Definite 2x2 pivots: d11 fails the 1x1 test against d21, and nothing else bounds the block.
  for (const RealScalar sign : {RealScalar(1), RealScalar(-1)}) {
    const Scalar offDiagonal = randomUnit<Scalar>() * RealScalar(1e-3);
    const Mat matrix = fromTriplets<Mat>(2, {{0, 0, Scalar(sign * RealScalar(1e-7))},
                                             {1, 0, offDiagonal},
                                             {0, 1, Scalar(numext::conj(offDiagonal))},
                                             {1, 1, Scalar(sign * RealScalar(100))}});
    MultifrontalLDLT<Mat, Lower, NaturalOrdering<typename Mat::StorageIndex>> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(numext::abs(solver.subDiagonal()[0]) > RealScalar(0));
    VERIFY_IS_EQUAL(solver.numPositive(), sign > RealScalar(0) ? Index(2) : Index(0));
    VERIFY_IS_EQUAL(solver.numNegative(), sign > RealScalar(0) ? Index(0) : Index(2));
    VERIFY_IS_EQUAL(solver.isPositive(), sign > RealScalar(0));
    VERIFY_IS_EQUAL(solver.isNegative(), sign < RealScalar(0));
    VERIFY(numext::real(solver.determinant()) > RealScalar(0));
  }

  for (const RealScalar value : {NumTraits<RealScalar>::infinity(), NumTraits<RealScalar>::quiet_NaN()}) {
    verifyNonFiniteFactorization(
        fromTriplets<Mat>(2, {{0, 0, Scalar(value)}, {1, 0, Scalar(1)}, {0, 1, Scalar(1)}, {1, 1, Scalar(1)}}));
    verifyNonFiniteFactorization(
        fromTriplets<Mat>(2, {{0, 0, Scalar(1)}, {1, 0, Scalar(value)}, {0, 1, Scalar(value)}, {1, 1, Scalar(1)}}));

    // No pivot test can pass on a non-finite column, so the root falls back to taking its largest diagonal.
    Triplets<Mat> triplets;
    for (Index col = 0; col < 4; ++col) {
      for (Index row = 0; row < 4; ++row) triplets.emplace_back(row, col, Scalar(value));
    }
    verifyNonFiniteFactorization(fromTriplets<Mat>(4, triplets));

    // A zero diagonal's non-finite column, in the root's own front and in a column delayed to it.
    verifyNonFiniteFactorization(fromTriplets<Mat>(2, {{1, 0, Scalar(value)}, {0, 1, Scalar(value)}}));
    verifyNonFiniteFactorization(
        fromTriplets<Mat>(3, {{1, 0, Scalar(value)}, {0, 1, Scalar(value)}, {2, 1, Scalar(1)}, {1, 2, Scalar(1)}}));

    const Mat single = fromTriplets<Mat>(1, {{0, 0, Scalar(value)}});
    verifyNonFiniteFactorization(single);
    verifyNonFiniteFactorization(Mat(-single));
  }

  // Updates can overflow on finite input. M = 0.6 * highest is in range, and so is every entry below.
  const RealScalar M = RealScalar(0.6) * NumTraits<RealScalar>::highest();
  // The pivot M leaves the Schur complement -2M on the diagonal.
  verifyNonFiniteFactorization(
      fromTriplets<Mat>(2, {{0, 0, Scalar(M)}, {1, 0, Scalar(M)}, {0, 1, Scalar(M)}, {1, 1, Scalar(-M)}}));
  // The pivot M leaves -2M off the diagonal, between the finite pivots 0.5 M. Eliminating either one carries the
  // overflow on to the other.
  const RealScalar d = RealScalar(1.5) * M;
  verifyNonFiniteFactorization(fromTriplets<Mat>(3, {{0, 0, Scalar(M)},
                                                     {1, 0, Scalar(M)},
                                                     {2, 0, Scalar(M)},
                                                     {0, 1, Scalar(M)},
                                                     {1, 1, Scalar(d)},
                                                     {2, 1, Scalar(-M)},
                                                     {0, 2, Scalar(M)},
                                                     {1, 2, Scalar(-M)},
                                                     {2, 2, Scalar(d)}}));
}

// A zero diagonal with finite off-diagonal entries whose magnitude overflows.
template <typename RealScalar>
void test_overflowing_complex_magnitude() {
  using Scalar = std::complex<RealScalar>;
  using Mat = SparseMatrix<Scalar>;
  const RealScalar m = RealScalar(0.8) * NumTraits<RealScalar>::highest();
  const Scalar z(m, m);
  VERIFY(!(numext::isfinite)(numext::abs(z)));
  verifyNonFiniteFactorization(fromTriplets<Mat>(2, {{1, 0, z}, {0, 1, numext::conj(z)}}));
}

// A scalar with no vectorization, non-trivial construction, and only the operations NumTraits documents.
template <typename Scalar>
void test_custom_scalar() {
  using Mat = SparseMatrix<Scalar>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  {
    const Mat matrix = kktChain<Mat>(5, Scalar(0.25f));
    MultifrontalLDLT<Mat> solver(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    const DenseVector b = DenseVector::Random(matrix.rows());
    const DenseVector x = solver.solve(b);
    VERIFY(relativeResidual(matrix, x, b) <= pivotTolerance<Scalar>(matrix.rows()));
  }
  VERIFY_IS_EQUAL(AnnoyingScalar::instances, 0);
}

EIGEN_DECLARE_TEST(multifrontal_ldlt) {
  CALL_SUBTEST_1((test_positive_definite<double, int, ColMajor>()));
  CALL_SUBTEST_2((test_positive_definite<std::complex<double>, int, ColMajor>()));
  CALL_SUBTEST_3((test_positive_definite<double, long int, RowMajor>()));

  CALL_SUBTEST_4((test_random_indefinite<float, int, ColMajor>()));
  CALL_SUBTEST_5((test_random_indefinite<double, int, ColMajor>()));
  CALL_SUBTEST_6((test_random_indefinite<std::complex<float>, int, ColMajor>()));
  CALL_SUBTEST_7((test_random_indefinite<std::complex<double>, long int, ColMajor>()));
  CALL_SUBTEST_8((test_random_indefinite<double, long int, RowMajor>()));
  CALL_SUBTEST_9((test_random_indefinite<std::complex<double>, int, RowMajor>()));

  CALL_SUBTEST_10((test_delayed_pivots<float, int, ColMajor>()));
  CALL_SUBTEST_10((test_delayed_pivots<double, long int, RowMajor>()));
  CALL_SUBTEST_10((test_delayed_pivots<std::complex<double>, int, ColMajor>()));

  CALL_SUBTEST_11(test_dense_front_with_2x2_pivots<float>());
  CALL_SUBTEST_11(test_dense_front_with_2x2_pivots<double>());
  CALL_SUBTEST_11(test_dense_front_with_2x2_pivots<std::complex<float>>());
  CALL_SUBTEST_11(test_dense_front_with_2x2_pivots<std::complex<double>>());

  CALL_SUBTEST_12(test_pivot_threshold<double>());
  CALL_SUBTEST_12(test_pivot_threshold<std::complex<double>>());

  CALL_SUBTEST_13(test_input_forms<double>());
  CALL_SUBTEST_13(test_input_forms<std::complex<double>>());

  CALL_SUBTEST_14(test_small_and_singular<float>());
  CALL_SUBTEST_14(test_small_and_singular<double>());
  CALL_SUBTEST_14(test_small_and_singular<std::complex<double>>());

  CALL_SUBTEST_15(test_extreme_values<float>());
  CALL_SUBTEST_15(test_extreme_values<double>());
  CALL_SUBTEST_15(test_extreme_values<std::complex<double>>());
  CALL_SUBTEST_15(test_overflowing_complex_magnitude<float>());
  CALL_SUBTEST_15(test_overflowing_complex_magnitude<double>());

  CALL_SUBTEST_16(test_custom_scalar<AnnoyingScalar>());

  CALL_SUBTEST_17((test_large_fronts<float, int, ColMajor>()));
  CALL_SUBTEST_18((test_large_fronts<double, int, ColMajor>()));
  CALL_SUBTEST_19((test_large_fronts<std::complex<double>, int, ColMajor>()));
  CALL_SUBTEST_20((test_large_fronts<std::complex<float>, long int, RowMajor>()));

  CALL_SUBTEST_21((test_determinant<double, int, ColMajor>()));
  CALL_SUBTEST_21((test_determinant<std::complex<double>, int, RowMajor>()));

  CALL_SUBTEST_22((test_permutation<double, int, ColMajor>()));
  CALL_SUBTEST_22((test_permutation<std::complex<double>, long int, RowMajor>()));

  CALL_SUBTEST_23((test_failing_columns_before_pivots<float, int, ColMajor>()));
  CALL_SUBTEST_23((test_failing_columns_before_pivots<double, long int, RowMajor>()));
  CALL_SUBTEST_23((test_failing_columns_before_pivots<std::complex<double>, int, ColMajor>()));

  CALL_SUBTEST_24((test_relaxed_fronts<float, int, ColMajor>()));
  CALL_SUBTEST_24((test_relaxed_fronts<double, long int, RowMajor>()));
  CALL_SUBTEST_24((test_relaxed_fronts<std::complex<double>, int, ColMajor>()));
}
