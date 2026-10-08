// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// all(), any(), allFinite(), isZero() and isApproxToConstant() over full scans: no coefficient decides the result
// early. Dynamic sizes hide the size and data pointer from the compiler on every iteration, as a caller whose sizes
// vary would; fixed sizes and a block of a matrix cover the other traversals.

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cstdint>

namespace Eigen {
namespace {

enum Op { All, Any, AllFinite, IsZero, ApproxConstant };

template <typename T, Op op>
T fill() {
  return op == Any || op == IsZero ? T(0) : op == ApproxConstant ? T(2) : T(1);
}

template <Op op, typename Xpr>
bool apply(const Xpr& x) {
  using Scalar = typename Xpr::Scalar;
  switch (op) {
    case All:
      return x.all();
    case Any:
      return x.any();
    case AllFinite:
      return x.allFinite();
    case IsZero:
      return x.isZero();
    default:
      return x.isApproxToConstant(Scalar(2));
  }
}

template <typename T, Op op>
void BM_Dynamic(benchmark::State& state) {
  const Index n = state.range(0);
  ArrayX<T> a = ArrayX<T>::Constant(n, fill<T, op>());
  for (auto _ : state) {
    Index m = n;
    benchmark::DoNotOptimize(m);
    const T* p = a.data();
    benchmark::DoNotOptimize(p);
    bool r = apply<op>(Map<const ArrayX<T>>(p, m));
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations() * n);
}

template <typename Xpr, Op op>
void BM_Fixed(benchmark::State& state) {
  Xpr a = Xpr::Constant(fill<typename Xpr::Scalar, op>());
  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    bool r = apply<op>(a);
    benchmark::DoNotOptimize(r);
  }
}

// isZero() on the top rows of a 64-column matrix: one inner vector per column, no linear access.
void BM_BlockIsZero(benchmark::State& state) {
  const Index n = state.range(0);
  MatrixXf m = MatrixXf::Zero(n + 3, 64);
  for (auto _ : state) {
    Index k = n;
    benchmark::DoNotOptimize(k);
    benchmark::DoNotOptimize(m.data());
    bool r = m.topRows(k).isZero();
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations() * n * 64);
}

template <typename T, Op op>
void registerDynamic(const char* name) {
  auto* b = benchmark::RegisterBenchmark(name, BM_Dynamic<T, op>);
  b->DenseRange(1, 16)->Arg(32)->Arg(64)->Arg(1024)->Arg(16384);
}

const bool registered = [] {
  registerDynamic<float, All>("all<float>");
  registerDynamic<float, Any>("any<float>");
  registerDynamic<float, AllFinite>("allFinite<float>");
  registerDynamic<float, IsZero>("isZero<float>");
  registerDynamic<float, ApproxConstant>("isApproxToConstant<float>");
  registerDynamic<double, All>("all<double>");
  registerDynamic<double, AllFinite>("allFinite<double>");
  registerDynamic<double, IsZero>("isZero<double>");
  registerDynamic<int, All>("all<int>");
  registerDynamic<int, Any>("any<int>");
  registerDynamic<std::int16_t, All>("all<int16_t>");
  registerDynamic<std::int16_t, Any>("any<int16_t>");
  registerDynamic<bool, All>("all<bool>");
  registerDynamic<bool, Any>("any<bool>");
  benchmark::RegisterBenchmark("any<Array4f>", BM_Fixed<Array4f, Any>);
  benchmark::RegisterBenchmark("allFinite<Array<float,16,1>>", BM_Fixed<Array<float, 16, 1>, AllFinite>);
  benchmark::RegisterBenchmark("isZero<Array<float,33,1>>", BM_Fixed<Array<float, 33, 1>, IsZero>);
  benchmark::RegisterBenchmark("isZero<Matrix4d>", BM_Fixed<Matrix4d, IsZero>);
  benchmark::RegisterBenchmark("all<Vector3f>", BM_Fixed<Vector3f, All>);
  benchmark::RegisterBenchmark("blockIsZero<float>", BM_BlockIsZero)->Arg(4)->Arg(16)->Arg(64)->Arg(256);
  return true;
}();

}  // namespace
}  // namespace Eigen
