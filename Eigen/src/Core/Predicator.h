// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_PREDICATOR_H
#define EIGEN_PREDICATOR_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Predicate reductions all_of and any_of. A predicate tests a scalar with operator() and a packet with
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
// Both searches look for a set lane: all_of negates each predicate mask, which compilers fold into the predicate's
// compare, so the traversal is one por reduction with a predux_any test and needs no pnot on the combined mask.
template <bool Target>
struct predicate_search {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool combine(bool a, bool b) { return a | b; }
  template <typename Predicate, typename Scalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool test(const Predicate& pred, const Scalar& x) {
    return pred(x) == Target;
  }
  template <typename Packet, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet test_packet(const Predicate& pred, const Packet& x) {
    Packet m = pred.packetOp(x);
    EIGEN_IF_CONSTEXPR (Target) {
      return m;
    } else {
      return pandnot(ptrue(m), m);
    }
  }
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
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool predicate_search_scalar(const Segment& seg, Index i, Predicate pred) {
  const Index size = seg.size();
  for (; i + 3 < size; i += 4) {
    bool b = Search::combine(Search::combine(Search::test(pred, seg.coeff(i)), Search::test(pred, seg.coeff(i + 1))),
                             Search::combine(Search::test(pred, seg.coeff(i + 2)), Search::test(pred, seg.coeff(i + 3))));
    if EIGEN_PREDICT_FALSE (b) return true;
  }
  // At most three coefficients remain.
  if (i == size) return false;
  bool b = Search::test(pred, seg.coeff(i));
  if (i + 1 < size) b = Search::combine(b, Search::test(pred, seg.coeff(i + 1)));
  if (i + 2 < size) b = Search::combine(b, Search::test(pred, seg.coeff(i + 2)));
  return b;
}

template <typename Search, typename Packet, bool Vectorize>
struct predicate_search_segment {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(const Segment& seg, Predicate pred) {
    return predicate_search_scalar<Search>(seg, 0, pred);
  }
};
template <typename Search, typename Packet>
struct predicate_search_segment<Search, Packet, true> {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet test(const Segment& seg, Index i, const Predicate& pred) {
    return Search::test_packet(pred, seg.template packet<Packet>(i));
  }
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run(const Segment& seg, Predicate pred) {
    constexpr Index PacketSize = unpacket_traits<Packet>::size;
    const Index size = seg.size();
    Index i = 0;
    if (PacketSize <= size) {
      // Short segments are the common call; the block loop is laid out off the fall-through path.
      if (EIGEN_PREDICT_FALSE(8 * PacketSize <= size)) {
        do {
          Packet m = por(por(por(test(seg, i + 0 * PacketSize, pred), test(seg, i + 1 * PacketSize, pred)),
                             por(test(seg, i + 2 * PacketSize, pred), test(seg, i + 3 * PacketSize, pred))),
                         por(por(test(seg, i + 4 * PacketSize, pred), test(seg, i + 5 * PacketSize, pred)),
                             por(test(seg, i + 6 * PacketSize, pred), test(seg, i + 7 * PacketSize, pred))));
          if EIGEN_PREDICT_FALSE (predux_any(m)) return true;
          i += 8 * PacketSize;
        } while (i + 8 * PacketSize <= size);
      }
      if (EIGEN_PREDICT_FALSE(i + 4 * PacketSize <= size)) {
        Packet m = por(por(test(seg, i + 0 * PacketSize, pred), test(seg, i + 1 * PacketSize, pred)),
                       por(test(seg, i + 2 * PacketSize, pred), test(seg, i + 3 * PacketSize, pred)));
        if (predux_any(m)) return true;
        i += 4 * PacketSize;
      }
      if (EIGEN_PREDICT_TRUE(i + PacketSize <= size)) {
        if (predux_any(test(seg, i, pred))) return true;
        i += PacketSize;
        if (EIGEN_PREDICT_TRUE(i + PacketSize <= size)) {
          if (predux_any(test(seg, i, pred))) return true;
          i += PacketSize;
          if (EIGEN_PREDICT_TRUE(i + PacketSize <= size)) {
            if (predux_any(test(seg, i, pred))) return true;
            i += PacketSize;
          }
        }
      }
      if (EIGEN_PREDICT_TRUE(i == size)) return false;
    }
    return predicate_search_scalar<Search>(seg, i, pred);
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
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool search(const Derived& xpr, Predicate pred) {
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
};

/** \internal \returns true if \a pred holds for every coefficient of \a xpr. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool all_of(const Derived& xpr, const Predicate& pred) {
  return !predicate_reduction<Derived, Predicate>::template search<false>(xpr, pred);
}

/** \internal \returns true if \a pred holds for some coefficient of \a xpr. */
template <typename Derived, typename Predicate>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool any_of(const Derived& xpr, const Predicate& pred) {
  return predicate_reduction<Derived, Predicate>::template search<true>(xpr, pred);
}

template <typename Derived, bool AlwaysTrue = NumTraits<typename traits<Derived>::Scalar>::IsInteger>
struct all_finite_impl {
  static EIGEN_DEVICE_FUNC inline bool run(const Derived& /*derived*/) { return true; }
};
#if !defined(__FINITE_MATH_ONLY__) || !(__FINITE_MATH_ONLY__)
template <typename Derived>
struct all_finite_impl<Derived, false> {
  static EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool run(const Derived& derived) {
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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::all() const {
  return internal::all_of(derived(), internal::nonzero_predicate<Scalar>());
}

/** \returns true if at least one coefficient is true
 *
 * \sa all()
 */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::any() const {
  return internal::any_of(derived(), internal::nonzero_predicate<Scalar>());
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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::allFinite() const {
  return internal::all_finite_impl<Derived>::run(derived());
}

/** \returns true if all coefficients in this matrix are approximately equal to \a val, to within precision \a prec */
template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::isApproxToConstant(const Scalar& val,
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
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool DenseBase<Derived>::isZero(const RealScalar& prec) const {
  typename internal::nested_eval<Derived, 1>::type self(derived());
  return internal::all_of(self, internal::fuzzy_constant_predicate<Scalar, false>{Scalar(0), prec});
}

}  // end namespace Eigen

#endif  // EIGEN_PREDICATOR_H
