// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_VISITOR_H
#define EIGEN_VISITOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Opt in only when coefficient updates are valid without init/initpacket.
template <typename Visitor, typename = void>
struct visitor_already_initialized : std::false_type {};
template <typename Visitor>
struct visitor_already_initialized<Visitor, void_t<decltype(functor_traits<Visitor>::AlreadyInitialized)>>
    : bool_constant<functor_traits<Visitor>::AlreadyInitialized> {};

template <typename Visitor, typename Derived, int UnrollCount,
          bool Vectorize = (Derived::PacketAccess && functor_traits<Visitor>::PacketAccess), bool LinearAccess = false,
          bool ShortCircuitEvaluation = false>
struct visitor_impl;

template <typename Visitor, bool ShortCircuitEvaluation = false>
struct short_circuit_eval_impl {
  // if short circuit evaluation is not used, do nothing
  static constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Visitor&) { return false; }
};
template <typename Visitor>
struct short_circuit_eval_impl<Visitor, true> {
  // if short circuit evaluation is used, check the visitor
  static constexpr EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Visitor& visitor) { return visitor.done(); }
};

// unrolled inner-outer traversal
template <typename Visitor, typename Derived, int UnrollCount, bool Vectorize, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, UnrollCount, Vectorize, false, ShortCircuitEvaluation> {
  // don't use short circuit evaluation for unrolled version
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr bool RowMajor = Derived::IsRowMajor;
  static constexpr int RowsAtCompileTime = Derived::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = Derived::ColsAtCompileTime;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  static constexpr int InnerSizeAtCompileTime = RowMajor ? ColsAtCompileTime : RowsAtCompileTime;
  static constexpr int OuterSizeAtCompileTime = RowMajor ? RowsAtCompileTime : ColsAtCompileTime;
  static constexpr int PacketOpsPerOuter =
      Vectorize && InnerSizeAtCompileTime >= PacketSize ? InnerSizeAtCompileTime / PacketSize : 0;
  static constexpr int FirstScalarInner = PacketOpsPerOuter * PacketSize;
  static constexpr int ScalarOpsPerOuter = InnerSizeAtCompileTime - FirstScalarInner;
  static constexpr int OpsPerOuter = PacketOpsPerOuter + ScalarOpsPerOuter;
  static constexpr int OpCount = UnrollCount == 0 ? 0 : OuterSizeAtCompileTime * OpsPerOuter;

  template <int Op>
  static constexpr bool IsPacketOp() {
    return OpsPerOuter != 0 && (Op % OpsPerOuter) < PacketOpsPerOuter;
  }

  template <int Op>
  static constexpr int CoeffIndex() {
    return OpsPerOuter == 0 ? 0
                            : (Op / OpsPerOuter) * InnerSizeAtCompileTime +
                                  (IsPacketOp<Op>() ? ((Op % OpsPerOuter) * PacketSize)
                                                    : (FirstScalarInner + (Op % OpsPerOuter) - PacketOpsPerOuter));
  }

  template <int Op, std::enable_if_t<!IsPacketOp<Op>(), bool> = true>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void visit(const Derived& mat, Visitor& visitor) {
    constexpr int K = CoeffIndex<Op>();
    constexpr int R = RowMajor ? (K / ColsAtCompileTime) : (K % RowsAtCompileTime);
    constexpr int C = RowMajor ? (K % ColsAtCompileTime) : (K / RowsAtCompileTime);
    EIGEN_IF_CONSTEXPR (Op == 0) {
      visitor.init(mat.coeff(R, C), R, C);
    } else {
      visitor(mat.coeff(R, C), R, C);
    }
  }

  template <int Op, std::enable_if_t<IsPacketOp<Op>(), bool> = true>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void visit(const Derived& mat, Visitor& visitor) {
    constexpr int K = CoeffIndex<Op>();
    constexpr int R = RowMajor ? (K / ColsAtCompileTime) : (K % RowsAtCompileTime);
    constexpr int C = RowMajor ? (K % ColsAtCompileTime) : (K / RowsAtCompileTime);
    Packet P = mat.template packet<Packet>(R, C);
    EIGEN_IF_CONSTEXPR (Op == 0) {
      visitor.initpacket(P, R, C);
    } else {
      visitor.packet(P, R, C);
    }
  }

  template <int... Ops>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run_impl(const Derived& mat, Visitor& visitor,
                                                             std::integer_sequence<int, Ops...>) {
    int unused[] = {0, (visit<Ops>(mat, visitor), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const Derived& mat, Visitor& visitor) {
    run_impl(mat, visitor, std::make_integer_sequence<int, OpCount>{});
  }
};

// unrolled linear traversal
template <typename Visitor, typename Derived, int UnrollCount, bool Vectorize, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, UnrollCount, Vectorize, true, ShortCircuitEvaluation> {
  // don't use short circuit evaluation for unrolled version
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  static constexpr int PacketOps = Vectorize ? UnrollCount / PacketSize : 0;
  static constexpr int FirstScalar = PacketOps * PacketSize;
  static constexpr int ScalarOps = UnrollCount - FirstScalar;
  static constexpr int OpCount = PacketOps + ScalarOps;

  template <int Op>
  static constexpr bool IsPacketOp() {
    return Op < PacketOps;
  }

  template <int Op>
  static constexpr int CoeffIndex() {
    return IsPacketOp<Op>() ? (Op * PacketSize) : (FirstScalar + Op - PacketOps);
  }

  template <int Op, std::enable_if_t<!IsPacketOp<Op>(), bool> = true>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void visit(const Derived& mat, Visitor& visitor) {
    constexpr int K = CoeffIndex<Op>();
    EIGEN_IF_CONSTEXPR (Op == 0) {
      visitor.init(mat.coeff(K), K);
    } else {
      visitor(mat.coeff(K), K);
    }
  }

  template <int Op, std::enable_if_t<IsPacketOp<Op>(), bool> = true>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void visit(const Derived& mat, Visitor& visitor) {
    constexpr int K = CoeffIndex<Op>();
    Packet P = mat.template packet<Packet>(K);
    EIGEN_IF_CONSTEXPR (Op == 0) {
      visitor.initpacket(P, K);
    } else {
      visitor.packet(P, K);
    }
  }

  template <int... Ops>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run_impl(const Derived& mat, Visitor& visitor,
                                                             std::integer_sequence<int, Ops...>) {
    int unused[] = {0, (visit<Ops>(mat, visitor), 0)...};
    EIGEN_UNUSED_VARIABLE(unused);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const Derived& mat, Visitor& visitor) {
    run_impl(mat, visitor, std::make_integer_sequence<int, OpCount>{});
  }
};

// dynamic scalar outer-inner traversal
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, Dynamic, /*Vectorize=*/false, /*LinearAccess=*/false, ShortCircuitEvaluation> {
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  static constexpr bool RowMajor = Derived::IsRowMajor;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const Derived& mat, Visitor& visitor) {
    const Index innerSize = RowMajor ? mat.cols() : mat.rows();
    const Index outerSize = RowMajor ? mat.rows() : mat.cols();
    if (innerSize == 0 || outerSize == 0) return;
    {
      visitor.init(mat.coeff(0, 0), 0, 0);
      if (short_circuit::run(visitor)) return;
      for (Index i = 1; i < innerSize; ++i) {
        Index r = RowMajor ? 0 : i;
        Index c = RowMajor ? i : 0;
        visitor(mat.coeff(r, c), r, c);
        if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      }
    }
    for (Index j = 1; j < outerSize; j++) {
      for (Index i = 0; i < innerSize; ++i) {
        Index r = RowMajor ? j : i;
        Index c = RowMajor ? i : j;
        visitor(mat.coeff(r, c), r, c);
        if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      }
    }
  }
};

// dynamic vectorized outer-inner traversal
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, Dynamic, /*Vectorize=*/true, /*LinearAccess=*/false, ShortCircuitEvaluation> {
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  static constexpr bool RowMajor = Derived::IsRowMajor;
  static constexpr bool AlreadyInitialized = visitor_already_initialized<Visitor>::value;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const Derived& mat, Visitor& visitor) {
    const Index innerSize = RowMajor ? mat.cols() : mat.rows();
    const Index outerSize = RowMajor ? mat.rows() : mat.cols();
    if (innerSize == 0 || outerSize == 0) return;
    EIGEN_IF_CONSTEXPR (!AlreadyInitialized) {
      Index i = 0;
      if (innerSize < PacketSize) {
        visitor.init(mat.coeff(0, 0), 0, 0);
        i = 1;
      } else {
        Packet p = mat.template packet<Packet>(0, 0);
        visitor.initpacket(p, 0, 0);
        i = PacketSize;
      }
      if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      for (; i + PacketSize - 1 < innerSize; i += PacketSize) {
        Index r = RowMajor ? 0 : i;
        Index c = RowMajor ? i : 0;
        Packet p = mat.template packet<Packet>(r, c);
        visitor.packet(p, r, c);
        if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      }
      for (; i < innerSize; ++i) {
        Index r = RowMajor ? 0 : i;
        Index c = RowMajor ? i : 0;
        visitor(mat.coeff(r, c), r, c);
        if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      }
    }
    const Index packetEnd = innerSize - innerSize % PacketSize;
    for (Index j = AlreadyInitialized ? 0 : 1; j < outerSize; j++) {
      Index i = 0;
      for (; AlreadyInitialized ? i < packetEnd : i + PacketSize - 1 < innerSize; i += PacketSize) {
        Index r = RowMajor ? j : i;
        Index c = RowMajor ? i : j;
        Packet p = mat.template packet<Packet>(r, c);
        visitor.packet(p, r, c);
        if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      }
      for (; i < innerSize; ++i) {
        Index r = RowMajor ? j : i;
        Index c = RowMajor ? i : j;
        visitor(mat.coeff(r, c), r, c);
        if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
      }
    }
  }
};

// dynamic scalar linear traversal
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, Dynamic, /*Vectorize=*/false, /*LinearAccess=*/true, ShortCircuitEvaluation> {
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const Derived& mat, Visitor& visitor) {
    const Index size = mat.size();
    if (size == 0) return;
    visitor.init(mat.coeff(0), 0);
    if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
    for (Index k = 1; k < size; k++) {
      visitor(mat.coeff(k), k);
      if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
    }
  }
};

// dynamic vectorized linear traversal
template <typename Visitor, typename Derived, bool ShortCircuitEvaluation>
struct visitor_impl<Visitor, Derived, Dynamic, /*Vectorize=*/true, /*LinearAccess=*/true, ShortCircuitEvaluation> {
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr int PacketSize = packet_traits<Scalar>::size;
  using short_circuit = short_circuit_eval_impl<Visitor, ShortCircuitEvaluation>;
  static constexpr bool AlreadyInitialized = visitor_already_initialized<Visitor>::value;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const Derived& mat, Visitor& visitor) {
    const Index size = mat.size();
    if (size == 0) return;
    const Index packetEnd = size - size % PacketSize;
    Index k = 0;
    EIGEN_IF_CONSTEXPR (!AlreadyInitialized) {
      if (size < PacketSize) {
        visitor.init(mat.coeff(0), 0);
        k = 1;
      } else {
        Packet p = mat.template packet<Packet>(k);
        visitor.initpacket(p, k);
        k = PacketSize;
      }
      if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
    }
    for (; k < packetEnd; k += PacketSize) {
      Packet p = mat.template packet<Packet>(k);
      visitor.packet(p, k);
      if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
    }
    for (; k < size; k++) {
      visitor(mat.coeff(k), k);
      if EIGEN_PREDICT_FALSE (short_circuit::run(visitor)) return;
    }
  }
};

// evaluator adaptor
template <typename XprType>
class visitor_evaluator {
 public:
  using Evaluator = evaluator<XprType>;
  using Scalar = typename XprType::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  using CoeffReturnType = std::remove_const_t<typename XprType::CoeffReturnType>;

  static constexpr bool PacketAccess = static_cast<bool>(Evaluator::Flags & PacketAccessBit);
  static constexpr bool LinearAccess = static_cast<bool>(Evaluator::Flags & LinearAccessBit);
  static constexpr bool IsRowMajor = static_cast<bool>(XprType::IsRowMajor);
  static constexpr int RowsAtCompileTime = XprType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = XprType::ColsAtCompileTime;
  static constexpr int XprAlignment = Evaluator::Alignment;
  static constexpr int CoeffReadCost = Evaluator::CoeffReadCost;

  EIGEN_DEVICE_FUNC explicit visitor_evaluator(const XprType& xpr) : m_evaluator(xpr), m_xpr(xpr) {}

  EIGEN_DEVICE_FUNC constexpr Index rows() const noexcept { return m_xpr.rows(); }
  EIGEN_DEVICE_FUNC constexpr Index cols() const noexcept { return m_xpr.cols(); }
  EIGEN_DEVICE_FUNC constexpr Index size() const noexcept { return m_xpr.size(); }
  // outer-inner access
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index row, Index col) const {
    return m_evaluator.coeff(row, col);
  }
  template <typename Packet, int Alignment = Unaligned>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index row, Index col) const {
    return m_evaluator.template packet<Alignment, Packet>(row, col);
  }
  // linear access
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const { return m_evaluator.coeff(index); }
  template <typename Packet, int Alignment = XprAlignment>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index index) const {
    return m_evaluator.template packet<Alignment, Packet>(index);
  }

 protected:
  Evaluator m_evaluator;
  const XprType& m_xpr;
};

template <typename T, typename = void>
struct visitor_has_linear_access : std::false_type {};

template <typename T>
struct visitor_has_linear_access<T, void_t<decltype(functor_traits<T>::LinearAccess)>>
    : bool_constant<static_cast<bool>(functor_traits<T>::LinearAccess)> {};

template <typename Derived, typename Visitor, bool ShortCircuitEvaluation>
struct visit_impl {
  using Evaluator = visitor_evaluator<Derived>;
  using Scalar = typename DenseBase<Derived>::Scalar;

  static constexpr bool IsRowMajor = DenseBase<Derived>::IsRowMajor;
  static constexpr int SizeAtCompileTime = DenseBase<Derived>::SizeAtCompileTime;
  static constexpr int RowsAtCompileTime = DenseBase<Derived>::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = DenseBase<Derived>::ColsAtCompileTime;
  static constexpr int InnerSizeAtCompileTime = IsRowMajor ? ColsAtCompileTime : RowsAtCompileTime;
  static constexpr int OuterSizeAtCompileTime = IsRowMajor ? RowsAtCompileTime : ColsAtCompileTime;

  // Linear packets can make an early scalar short-circuit exit more expensive.
  // Preinitialized visitors opt into starting directly with packet traversal.
  static constexpr bool LinearAccess = (!ShortCircuitEvaluation || visitor_already_initialized<Visitor>::value) &&
                                       Evaluator::LinearAccess && visitor_has_linear_access<Visitor>::value;
  static constexpr bool Vectorize = Evaluator::PacketAccess && static_cast<bool>(functor_traits<Visitor>::PacketAccess);

  static constexpr int PacketSize = packet_traits<Scalar>::size;
  static constexpr int VectorOps =
      Vectorize ? (LinearAccess ? (SizeAtCompileTime / PacketSize)
                                : (OuterSizeAtCompileTime * (InnerSizeAtCompileTime / PacketSize)))
                : 0;
  static constexpr int ScalarOps = SizeAtCompileTime - (VectorOps * PacketSize);
  // treat vector op and scalar op as same cost for unroll logic
  static constexpr int TotalOps = VectorOps + ScalarOps;

  static constexpr int UnrollCost = int(Evaluator::CoeffReadCost) + int(functor_traits<Visitor>::Cost);
  static constexpr bool Unroll = (SizeAtCompileTime != Dynamic) && ((TotalOps * UnrollCost) <= EIGEN_UNROLLING_LIMIT);
  static constexpr int UnrollCount = Unroll ? int(SizeAtCompileTime) : Dynamic;

  using impl = visitor_impl<Visitor, Evaluator, UnrollCount, Vectorize, LinearAccess, ShortCircuitEvaluation>;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(const DenseBase<Derived>& mat, Visitor& visitor) {
    Evaluator evaluator(mat.derived());
    impl::run(evaluator, visitor);
  }
};

}  // end namespace internal

/** Applies the visitor \a visitor to the whole coefficients of the matrix or vector.
 *
 * The template parameter \a Visitor is the type of the visitor and provides the following interface:
 * \code
 * struct MyVisitor {
 *   // called for the first coefficient
 *   void init(const Scalar& value, Index i, Index j);
 *   // called for all other coefficients
 *   void operator() (const Scalar& value, Index i, Index j);
 * };
 * \endcode
 *
 * \note compared to one or two \em for \em loops, visitors offer automatic
 * unrolling for small fixed size matrix.
 *
 * \note if the matrix is empty, then the visitor is left unchanged.
 *
 * \sa minCoeff(Index*,Index*), maxCoeff(Index*,Index*), DenseBase::redux()
 */
template <typename Derived>
template <typename Visitor>
EIGEN_DEVICE_FUNC void DenseBase<Derived>::visit(Visitor& visitor) const {
  using impl = internal::visit_impl<Derived, Visitor, /*ShortCircuitEvaluation*/ false>;
  impl::run(derived(), visitor);
}

namespace internal {

// Predicate reductions all_of, any_of and count_if. A predicate tests a scalar with operator() and a packet with
// packetOp, which returns a truth packet: each lane is zero where the predicate is false and, where it is true, the
// lane mask pcmp_* produces (all ones, or one for bool packets). Truth packets combine with pand and por, so the
// traversals below reduce and branch once per block of packets rather than once per packet. Results do not depend on
// the grouping, but a short-circuit may read past the deciding coefficient, up to the end of its block.
template <typename Scalar>
struct nonzero_predicate {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const { return x != Scalar(0); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pandnot(ptrue(x), pcmp_eq(x, pzero(x)));
  }
};
template <typename Scalar>
struct functor_traits<nonzero_predicate<Scalar>> {
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasCmp;
};

template <typename Scalar>
struct isfinite_predicate {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const { return (numext::isfinite)(x); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pisfinite(x);
  }
};
template <typename Scalar>
struct functor_traits<isfinite_predicate<Scalar>> {
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasCmp;
};

// Separate specializations keep custom scalars that support only one of isApprox and isMuchSmallerThan compiling.
template <typename Scalar, bool Approximate>
struct fuzzy_constant_predicate {
  Scalar value;
  typename NumTraits<Scalar>::Real precision;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const {
    return internal::isApprox(x, value, precision);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    const Packet v = pset1<Packet>(value);
    return pcmp_le(pabs(psub(x, v)), pmul(pmin(pabs(x), pabs(v)), pset1<Packet>(precision)));
  }
};
template <typename Scalar>
struct fuzzy_constant_predicate<Scalar, false> {
  Scalar value;
  typename NumTraits<Scalar>::Real precision;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& x) const {
    return internal::isMuchSmallerThan(x, Scalar(1), precision);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return pcmp_le(pabs(x), pset1<Packet>(precision));
  }
};
template <typename Scalar, bool Approximate>
struct functor_traits<fuzzy_constant_predicate<Scalar, Approximate>> {
  // ARMv7 NEON flushes subnormal float operands and results; scalar VFP does not.
  // Explicitly flushing would change exact-zero checks; normal operands can also have a subnormal difference.
  static constexpr bool PacketAccess =
      (std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value) &&
      !(EIGEN_ARCH_ARM && std::is_same<Scalar, float>::value) && packet_traits<Scalar>::HasAbs &&
      packet_traits<Scalar>::HasCmp &&
      (!Approximate ||
       (packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMin && packet_traits<Scalar>::HasMul));
};

// Searches for a coefficient on which the predicate is Target: any_of searches for true, all_of for false.
template <bool Target>
struct predicate_search;
template <>
struct predicate_search<true> {
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet combine(const Packet& a, const Packet& b) {
    return por(a, b);
  }
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool found(const Packet& m) {
    return predux_any(m);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool combine(bool a, bool b) { return a | b; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool found(bool b) { return b; }
};
template <>
struct predicate_search<false> {
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet combine(const Packet& a, const Packet& b) {
    return pand(a, b);
  }
  // pandnot rather than pnot, which some targets implement only bytewise.
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool found(const Packet& m) {
    return predux_any(pandnot(ptrue(m), m));
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool combine(bool a, bool b) { return a & b; }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool found(bool b) { return !b; }
};

// Coefficient access along one inner vector, or along the whole expression with linear access.
template <typename Evaluator, bool Linear = Evaluator::LinearAccess>
struct predicate_segment {
  const Evaluator& eval;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE predicate_segment(const Evaluator& evaluator, Index) : eval(evaluator) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return eval.size(); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Evaluator::CoeffReturnType coeff(Index i) const {
    return eval.coeff(i);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index i) const {
    return eval.template packet<Packet>(i);
  }
};
template <typename Evaluator>
struct predicate_segment<Evaluator, false> {
  const Evaluator& eval;
  Index outer;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE predicate_segment(const Evaluator& evaluator, Index j)
      : eval(evaluator), outer(j) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return Evaluator::IsRowMajor ? eval.cols() : eval.rows(); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Evaluator::CoeffReturnType coeff(Index i) const {
    return Evaluator::IsRowMajor ? eval.coeff(outer, i) : eval.coeff(i, outer);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(Index i) const {
    return Evaluator::IsRowMajor ? eval.template packet<Packet>(outer, i) : eval.template packet<Packet>(i, outer);
  }
};

template <typename Search, typename Segment, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predicate_search_scalar(const Segment& seg, Index i, const Predicate& pred) {
  const Index size = seg.size();
  for (; i + 3 < size; i += 4) {
    bool b = Search::combine(Search::combine(pred(seg.coeff(i)), pred(seg.coeff(i + 1))),
                             Search::combine(pred(seg.coeff(i + 2)), pred(seg.coeff(i + 3))));
    if EIGEN_PREDICT_FALSE (Search::found(b)) return true;
  }
  if (i == size) return false;
  bool b = pred(seg.coeff(i));
  for (++i; i < size; ++i) b = Search::combine(b, pred(seg.coeff(i)));
  return Search::found(b);
}

template <typename Search, typename Packet, bool Vectorize>
struct predicate_search_segment {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Segment& seg, const Predicate& pred) {
    return predicate_search_scalar<Search>(seg, 0, pred);
  }
};
template <typename Search, typename Packet>
struct predicate_search_segment<Search, Packet, true> {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Segment& seg, const Predicate& pred) {
    constexpr Index PacketSize = unpacket_traits<Packet>::size;
    const Index size = seg.size();
    Index i = 0;
    for (; i + 4 * PacketSize <= size; i += 4 * PacketSize) {
      Packet m0 = pred.packetOp(seg.template packet<Packet>(i));
      Packet m1 = pred.packetOp(seg.template packet<Packet>(i + PacketSize));
      Packet m2 = pred.packetOp(seg.template packet<Packet>(i + 2 * PacketSize));
      Packet m3 = pred.packetOp(seg.template packet<Packet>(i + 3 * PacketSize));
      if EIGEN_PREDICT_FALSE (Search::found(Search::combine(Search::combine(m0, m1), Search::combine(m2, m3))))
        return true;
    }
    if (i + PacketSize <= size) {
      Packet m = pred.packetOp(seg.template packet<Packet>(i));
      for (i += PacketSize; i + PacketSize <= size; i += PacketSize)
        m = Search::combine(m, pred.packetOp(seg.template packet<Packet>(i)));
      if (Search::found(m)) return true;
    }
    return predicate_search_scalar<Search>(seg, i, pred);
  }
};

template <typename Packet, bool Vectorize>
struct predicate_count_segment {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index run(const Segment& seg, const Predicate& pred) {
    Index count = 0;
    for (Index i = 0; i < seg.size(); ++i) count += pred(seg.coeff(i)) ? 1 : 0;
    return count;
  }
};
template <typename Packet>
struct predicate_count_segment<Packet, true> {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index run(const Segment& seg, const Predicate& pred) {
    constexpr Index PacketSize = unpacket_traits<Packet>::size;
    const Index size = seg.size();
    Index count = 0;
    Index i = 0;
    for (; i + PacketSize <= size; i += PacketSize)
      count += predux_count(pred.packetOp(seg.template packet<Packet>(i)));
    for (; i < size; ++i) count += pred(seg.coeff(i)) ? 1 : 0;
    return count;
  }
};

template <typename Derived, typename Predicate>
struct predicate_reduction {
  using Evaluator = visitor_evaluator<Derived>;
  using Scalar = typename Derived::Scalar;
  using Packet = typename packet_traits<Scalar>::type;
  static constexpr bool Vectorize = Evaluator::PacketAccess && packet_traits<Scalar>::Vectorizable &&
                                    (unpacket_traits<Packet>::size > 1) &&
                                    static_cast<bool>(functor_traits<Predicate>::PacketAccess);
  static constexpr bool LinearAccess = Evaluator::LinearAccess;
  using Segment = predicate_segment<Evaluator>;

  template <bool Target>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool search(const Derived& xpr, const Predicate& pred) {
    using impl = predicate_search_segment<predicate_search<Target>, Packet, Vectorize>;
    Evaluator eval(xpr);
    EIGEN_IF_CONSTEXPR (LinearAccess) {
      return impl::run(Segment(eval, 0), pred);
    } else {
      const Index outerSize = Evaluator::IsRowMajor ? eval.rows() : eval.cols();
      for (Index j = 0; j < outerSize; ++j)
        if (impl::run(Segment(eval, j), pred)) return true;
      return false;
    }
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index count(const Derived& xpr, const Predicate& pred) {
    using impl = predicate_count_segment<Packet, Vectorize>;
    Evaluator eval(xpr);
    EIGEN_IF_CONSTEXPR (LinearAccess) {
      return impl::run(Segment(eval, 0), pred);
    } else {
      const Index outerSize = Evaluator::IsRowMajor ? eval.rows() : eval.cols();
      Index count = 0;
      for (Index j = 0; j < outerSize; ++j) count += impl::run(Segment(eval, j), pred);
      return count;
    }
  }
};

/** \internal \returns true if \a pred holds for every coefficient of \a xpr. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool all_of(const Derived& xpr, const Predicate& pred) {
  return !predicate_reduction<Derived, Predicate>::template search<false>(xpr, pred);
}

/** \internal \returns true if \a pred holds for some coefficient of \a xpr. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool any_of(const Derived& xpr, const Predicate& pred) {
  return predicate_reduction<Derived, Predicate>::template search<true>(xpr, pred);
}

/** \internal \returns the number of coefficients of \a xpr for which \a pred holds. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index count_if(const Derived& xpr, const Predicate& pred) {
  return predicate_reduction<Derived, Predicate>::count(xpr, pred);
}

template <typename Derived, bool AlwaysTrue = NumTraits<typename traits<Derived>::Scalar>::IsInteger>
struct all_finite_impl {
  static EIGEN_DEVICE_FUNC inline bool run(const Derived& /*derived*/) { return true; }
};
#if !defined(__FINITE_MATH_ONLY__) || !(__FINITE_MATH_ONLY__)
template <typename Derived>
struct all_finite_impl<Derived, false> {
  static EIGEN_DEVICE_FUNC inline bool run(const Derived& derived) {
    return all_of(derived, isfinite_predicate<typename traits<Derived>::Scalar>());
  }
};
#endif

}  // end namespace internal

/** \returns true if all coefficients are true
 *
 * Example: \include MatrixBase_all.cpp
 * Output: \verbinclude MatrixBase_all.out
 *
 * \sa any(), Cwise::operator<()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::all() const {
  return internal::all_of(derived(), internal::nonzero_predicate<Scalar>());
}

/** \returns true if at least one coefficient is true
 *
 * \sa all()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::any() const {
  return internal::any_of(derived(), internal::nonzero_predicate<Scalar>());
}

/** \returns the number of coefficients which evaluate to true
 *
 * \sa all(), any()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC Index DenseBase<Derived>::count() const {
  return internal::count_if(derived(), internal::nonzero_predicate<Scalar>());
}

template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::hasNaN() const {
  return derived().cwiseTypedNotEqual(derived()).any();
}

/** \returns true if \c *this contains only finite numbers, i.e., no NaN and no +/-INF values.
 *
 * \sa hasNaN()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC inline bool DenseBase<Derived>::allFinite() const {
  return internal::all_finite_impl<Derived>::run(derived());
}

/** \returns true if all coefficients in this matrix are approximately equal to \a val, to within precision \a prec */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool DenseBase<Derived>::isApproxToConstant(const Scalar& val,
                                                                                  const RealScalar& prec) const {
  typename internal::nested_eval<Derived, 1>::type self(derived());
  return internal::all_of(self, internal::fuzzy_constant_predicate<Scalar, true>{val, prec});
}

/** \returns true if *this is approximately equal to the zero matrix,
 *          within the precision given by \a prec.
 *
 * Example: \include MatrixBase_isZero.cpp
 * Output: \verbinclude MatrixBase_isZero.out
 *
 * \sa class CwiseNullaryOp, Zero()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool DenseBase<Derived>::isZero(const RealScalar& prec) const {
  typename internal::nested_eval<Derived, 1>::type self(derived());
  return internal::all_of(self, internal::fuzzy_constant_predicate<Scalar, false>{Scalar(0), prec});
}

}  // end namespace Eigen

#endif  // EIGEN_VISITOR_H
