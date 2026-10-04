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
// Integer packet with the lanes of Packet, or void. count_if subtracts truth masks from it: a true lane is -1.
template <typename Packet, typename = void>
struct count_lane_packet_impl {
  using type = void;
};
template <typename Packet>
struct count_lane_packet_impl<Packet, void_t<typename unpacket_traits<Packet>::integer_packet>> {
  using IntPacket = typename unpacket_traits<Packet>::integer_packet;
  using type = std::conditional_t<unpacket_traits<IntPacket>::vectorizable &&
                                      unpacket_traits<IntPacket>::size == unpacket_traits<Packet>::size,
                                  IntPacket, void>;
};
template <typename Packet, typename Scalar = typename unpacket_traits<Packet>::type>
using count_lane_packet =
    std::conditional_t<NumTraits<Scalar>::IsInteger && NumTraits<Scalar>::IsSigned && (sizeof(Scalar) >= 4), Packet,
                       typename count_lane_packet_impl<Packet>::type>;

template <typename Packet, typename IntPacket = count_lane_packet<Packet>>
struct predicate_count_packets {
  using IntScalar = typename unpacket_traits<IntPacket>::type;
  static constexpr Index PacketSize = unpacket_traits<Packet>::size;
  // A chunk's count, and so every lane and partial sum, fits in IntScalar.
  static constexpr Index Chunk = sizeof(IntScalar) < sizeof(Index)
                                     ? Index(NumTraits<IntScalar>::highest()) / (4 * PacketSize) * (4 * PacketSize)
                                     : NumTraits<Index>::highest();

  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IntPacket lanes(const Segment& seg, const Predicate& pred, Index i) {
    return preinterpret<IntPacket>(pred.packetOp(seg.template packet<Packet>(i)));
  }

  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index run(const Segment& seg, const Predicate& pred, Index& i) {
    const Index size = seg.size();
    const Index packetEnd = size - size % PacketSize;
    Index count = 0;
    while (i < packetEnd) {
      const Index end = i + numext::mini(packetEnd - i, Chunk);
      IntPacket c0 = pset1<IntPacket>(IntScalar(0)), c1 = c0, c2 = c0, c3 = c0;
      for (; i + 4 * PacketSize <= end; i += 4 * PacketSize) {
        c0 = psub(c0, lanes(seg, pred, i));
        c1 = psub(c1, lanes(seg, pred, i + PacketSize));
        c2 = psub(c2, lanes(seg, pred, i + 2 * PacketSize));
        c3 = psub(c3, lanes(seg, pred, i + 3 * PacketSize));
      }
      for (; i < end; i += PacketSize) c0 = psub(c0, lanes(seg, pred, i));
      count += static_cast<Index>(predux(padd(padd(c0, c1), padd(c2, c3))));
    }
    return count;
  }
};
// Without integer lanes, reduce each packet.
template <typename Packet>
struct predicate_count_packets<Packet, void> {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index run(const Segment& seg, const Predicate& pred, Index& i) {
    constexpr Index PacketSize = unpacket_traits<Packet>::size;
    const Index size = seg.size();
    Index count = 0;
    for (; i + PacketSize <= size; i += PacketSize) count += predux_count(pred.packetOp(seg.template packet<Packet>(i)));
    return count;
  }
};
template <typename Packet>
struct predicate_count_segment<Packet, true> {
  template <typename Segment, typename Predicate>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index run(const Segment& seg, const Predicate& pred) {
    Index i = 0;
    Index count = predicate_count_packets<Packet>::run(seg, pred, i);
    for (; i < seg.size(); ++i) count += pred(seg.coeff(i)) ? 1 : 0;
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

#endif  // EIGEN_PREDICATOR_H
