// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of tensors on the default device. main.h adds instrumentation that breaks constant
// evaluation, so this test only compiles.

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/TensorSymmetry>

using namespace Eigen;

// Tensor evaluation in a constant expression needs constexpr destructors.
#if EIGEN_COMP_CXXVER >= 20

constexpr bool fixed_size() {
  TensorFixedSize<int, Sizes<2, 3>> a;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = 3 * i + j;
  TensorFixedSize<int, Sizes<2, 3>> b;
  b.setConstant(2);
  TensorFixedSize<int, Sizes<2, 3>> c;
  c = a * b + a;
  bool ok = c(1, 2) == 15 && c(0, 0) == 0;
  TensorFixedSize<int, Sizes<>> total;
  total = a.sum();
  ok = ok && total() == 15;
  TensorFixedSize<int, Sizes<>> largest;
  largest = a.maximum();
  return ok && largest() == 5;
}

static_assert(fixed_size());

#if defined(__cpp_lib_constexpr_dynamic_alloc)
struct CoordinateGenerator {
  constexpr int operator()(const array<Index, 2>& coordinates) const {
    return int(10 * coordinates[0] + coordinates[1]);
  }
};

struct Doubler {
  constexpr DSizes<Index, 2> dimensions(const Tensor<int, 2>& input) const { return input.dimensions(); }
  template <typename Output, typename Device>
  constexpr void eval(const Tensor<int, 2>& input, Output& output, const Device& device) const {
    output.device(device) = input + input;
  }
};

struct Adder {
  constexpr DSizes<Index, 2> dimensions(const Tensor<int, 2>& a, const Tensor<int, 2>&) const { return a.dimensions(); }
  template <typename Output, typename Device>
  constexpr void eval(const Tensor<int, 2>& a, const Tensor<int, 2>& b, Output& output, const Device& device) const {
    output.device(device) = a + b;
  }
};
constexpr bool dynamic_size() {
  Tensor<double, 2> a(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = 3 * i + j;
  Tensor<double, 2> b = a * 2.0 - a.constant(1.0);
  bool ok = b(1, 2) == 9 && b.dimension(0) == 2 && b.dimension(1) == 3;

  // Reductions over one dimension and over all of them.
  const array<Index, 1> rows{0};
  Tensor<double, 1> column_sums = a.sum(rows);
  ok = ok && column_sums.size() == 3 && column_sums(0) == 3 && column_sums(2) == 7;
  Tensor<double, 0> total = a.sum();
  Tensor<double, 0> largest = a.maximum();
  Tensor<double, 0> smallest = a.minimum();
  ok = ok && total() == 15 && largest() == 5 && smallest() == 0;

  // a a^T = [[5, 14], [14, 50]].
  const array<IndexPair<Index>, 1> dims{IndexPair<Index>(1, 1)};
  Tensor<double, 2> gram = a.contract(a, dims);
  ok = ok && gram(0, 0) == 5 && gram(0, 1) == 14 && gram(1, 1) == 50;

  const array<Index, 2> shape{3, 2};
  Tensor<double, 2> reshaped = a.reshape(shape);
  ok = ok && reshaped(1, 0) == 3 && reshaped(0, 1) == 4;
  const array<Index, 2> offsets{0, 1};
  const array<Index, 2> extents{2, 2};
  Tensor<double, 2> sliced = a.slice(offsets, extents);
  ok = ok && sliced(1, 1) == 5 && sliced(0, 0) == 1;
  const array<int, 2> swap{1, 0};
  Tensor<double, 2> shuffled = a.shuffle(swap);
  ok = ok && shuffled(2, 1) == 5 && shuffled.dimension(0) == 3;
  const array<Index, 2> shifts{1, 1};
  Tensor<double, 2> rolled = a.roll(shifts);
  ok = ok && rolled(0, 0) == a(1, 1) && rolled(1, 2) == a(0, 0);
  return ok;
}

static_assert(dynamic_size());

constexpr bool more_operations() {
  Tensor<int, 2> a(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = 3 * i + j;

  const double sum = Tensor<double, 2>(a.cast<double>()).sum();
  bool ok = sum == 15;
  Tensor<int, 2> joined = a.concatenate(a, 0);
  ok = ok && joined.dimension(0) == 4 && joined(3, 2) == 5;
  const array<bool, 2> flip{true, false};
  Tensor<int, 2> reversed = a.reverse(flip);
  ok = ok && reversed(0, 0) == 3 && reversed(1, 2) == 2;
  const array<std::pair<Index, Index>, 2> padding{std::make_pair(1, 0), std::make_pair(0, 1)};
  Tensor<int, 2> padded = a.pad(padding);
  ok = ok && padded.dimension(0) == 3 && padded.dimension(1) == 4 && padded(0, 0) == 0 && padded(2, 1) == 4;
  const array<Index, 2> strides{1, 2};
  Tensor<int, 2> strided = a.stride(strides);
  ok = ok && strided.dimension(1) == 2 && strided(1, 1) == 5;
  Tensor<int, 1> row = a.chip(1, 0);
  ok = ok && row.size() == 3 && row(2) == 5;
  const array<Index, 2> copies{2, 1};
  Tensor<int, 2> broadcast = a.broadcast(copies);
  ok = ok && broadcast.dimension(0) == 4 && broadcast(2, 1) == 1;
  Tensor<int, 2, RowMajor> swapped = a.swap_layout();
  ok = ok && swapped.dimension(0) == 3 && swapped(2, 1) == 5;
  Tensor<int, 2> selected = (a > a.constant(2)).select(a, a.constant(-1));
  ok = ok && selected(0, 2) == -1 && selected(1, 0) == 3;
  Tensor<Index, 0> largest = a.argmax();
  ok = ok && largest() == 5;

  // A seeded generator is a pure function of (seed, index).
  Tensor<double, 1> u(16);
  u = u.random(internal::UniformRandomGenerator<double>(42));
  Tensor<double, 1> v(16);
  v = v.random(internal::UniformRandomGenerator<double>(42));
  for (int i = 0; i < 16; ++i) ok = ok && u(i) >= 0 && u(i) < 1 && u(i) == v(i);
  return ok && u(0) != u(1);
}

static_assert(more_operations());

// A valid convolution: out(i, j) = sum_{p, q} in(i + p, j + q) k(p, q).
constexpr bool convolution() {
  Tensor<int, 2> in(3, 4), kernel(2, 2);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) in(i, j) = i * i + 3 * j;
  kernel(0, 0) = 1;
  kernel(1, 0) = -1;
  kernel(0, 1) = 2;
  kernel(1, 1) = 0;
  const array<Index, 2> dims{0, 1};
  Tensor<int, 2> out = in.convolve(kernel, dims);
  bool ok = out.dimension(0) == 2 && out.dimension(1) == 3;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      int expected = 0;
      for (int p = 0; p < 2; ++p)
        for (int q = 0; q < 2; ++q) expected += in(i + p, j + q) * kernel(p, q);
      ok = ok && out(i, j) == expected;
    }
  return ok;
}

static_assert(convolution());

// With SAME padding a 3x3 patch is centred on its pixel and reads zero outside the image; patch p = r + 3 c.
constexpr bool image_patches() {
  Tensor<int, 4> image(1, 3, 3, 1);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) image(0, r, c, 0) = 1 + r + 3 * c;
  Tensor<int, 5> patches = image.extract_image_patches(3, 3);
  bool ok = patches.dimension(1) == 3 && patches.dimension(2) == 3 && patches.dimension(3) == 9;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const int y = r + i - 1, x = c + j - 1;
          const int expected = y >= 0 && y < 3 && x >= 0 && x < 3 ? image(0, y, x, 0) : 0;
          ok = ok && patches(0, i, j, r + 3 * c, 0) == expected;
        }
  // Single-voxel volume patches enumerate the voxels in storage order.
  Tensor<int, 5> volume(1, 2, 2, 3, 1);
  for (int k = 0; k < 12; ++k) volume.data()[k] = 10 * k;
  Tensor<int, 6> voxels = volume.extract_volume_patches(1, 1, 1);
  ok = ok && voxels.dimension(4) == 12;
  for (int k = 0; k < 12; ++k) ok = ok && voxels(0, 0, 0, 0, k, 0) == 10 * k;
  return ok;
}

static_assert(image_patches());

// Patch p = r + 2 c of the 2x2 patches of a 3x4 tensor starts at (r, c).
constexpr bool patches() {
  Tensor<int, 2> t(3, 4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) t(i, j) = 4 * i + j;
  const array<Index, 2> patch_dims{2, 2};
  Tensor<int, 3> p = t.extract_patches(patch_dims);
  bool ok = p.dimension(2) == 6;
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 3; ++c)
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) ok = ok && p(i, j, r + 2 * c) == t(r + i, c + j);
  return ok;
}

static_assert(patches());

constexpr bool scans_traces_generators() {
  Tensor<int, 1> v(5);
  for (int i = 0; i < 5; ++i) v(i) = i + 1;
  Tensor<int, 1> inclusive = v.cumsum(0), exclusive = v.cumsum(0, true), product = v.cumprod(0);
  bool ok = true;
  for (int i = 0, sum = 0, prod = 1; i < 5; ++i) {
    ok = ok && exclusive(i) == sum;
    sum += i + 1;
    prod *= i + 1;
    ok = ok && inclusive(i) == sum && product(i) == prod;
  }
  // trace over dimensions 0 and 2 of a 3x2x3 tensor: out(j) = sum_i t(i, j, i).
  Tensor<int, 3> t(3, 2, 3);
  for (int k = 0; k < 18; ++k) t.data()[k] = k;
  const array<Index, 2> dims{0, 2};
  Tensor<int, 1> partial = t.trace(dims);
  ok = ok && partial(0) == t(0, 0, 0) + t(1, 0, 1) + t(2, 0, 2) && partial(1) == t(0, 1, 0) + t(1, 1, 1) + t(2, 1, 2);
  Tensor<int, 2> square(3, 3);
  square.setConstant(1);
  Tensor<int, 0> full = square.trace();
  ok = ok && full() == 3;
  // inflate inserts stride - 1 zeros between coefficients.
  const array<Index, 1> stride{2};
  Tensor<int, 1> inflated = v.inflate(stride);
  ok = ok && inflated.size() == 9 && inflated(4) == 3 && inflated(5) == 0 && inflated(8) == 5;
  Tensor<int, 2> generated = square.generate(CoordinateGenerator());
  return ok && generated(2, 1) == 21 && generated(0, 2) == 2;
}

static_assert(scans_traces_generators());

// The custom ops evaluate into the output through the device, as user code does.
constexpr bool custom_ops() {
  Tensor<int, 2> a(2, 3), b(2, 3);
  for (int k = 0; k < 6; ++k) {
    a.data()[k] = k;
    b.data()[k] = 10 - k;
  }
  Tensor<int, 2> doubled = a.customOp(Doubler());
  Tensor<int, 2> sum = a.customOp(b, Adder());
  bool ok = true;
  for (int k = 0; k < 6; ++k) ok = ok && doubled.data()[k] == 2 * k && sum.data()[k] == 10;
  return ok;
}

static_assert(custom_ops());

// A map writes through to the mapped storage, in column-major order.
constexpr bool tensor_map() {
  int data[6] = {};
  TensorMap<Tensor<int, 2>> map(data, 2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) map(i, j) = 3 * i + j;
  map = map * map.constant(2);
  TensorMap<const Tensor<int, 2>> view(data, 2, 3);
  Tensor<int, 0> total = view.sum();
  return data[1] == 6 && data[2] == 2 && data[5] == 10 && total() == 30;
}

static_assert(tensor_map());

// Static dimensions from IndexList select the same reductions and broadcasts as runtime ones.
constexpr bool index_lists() {
  Tensor<int, 2> a(2, 3);
  for (int k = 0; k < 6; ++k) a.data()[k] = k * k;
  IndexList<type2index<0>> rows;
  const array<Index, 1> runtime_rows{0};
  Tensor<int, 1> static_sum = a.sum(rows), dynamic_sum = a.sum(runtime_rows);
  IndexList<type2index<1>, int> copies;
  copies.set(1, 2);
  Tensor<int, 2> broadcast = a.broadcast(copies);
  bool ok = broadcast.dimension(1) == 6 && broadcast(1, 4) == a(1, 1);
  for (int j = 0; j < 3; ++j) ok = ok && static_sum(j) == dynamic_sum(j) && static_sum(j) == a(0, j) + a(1, j);
  return ok;
}

static_assert(index_lists());

// Setting one entry of a tensor with the symmetries of the Levi-Civita symbol fills in the other five nonzeros.
template <typename Group>
constexpr bool levi_civita(const Group& group) {
  Tensor<int, 3> epsilon(3, 3, 3);
  epsilon.setZero();
  group(epsilon, 0, 1, 2) = 1;
  bool ok = true;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) ok = ok && epsilon(i, j, k) == (j - i) * (k - j) * (k - i) / 2;
  return ok;
}

constexpr bool tensor_symmetry() {
  DynamicSGroup dynamic;
  dynamic.addAntiSymmetry(0, 1);
  dynamic.addAntiSymmetry(1, 2);
  return levi_civita(SGroup<AntiSymmetry<0, 1>, AntiSymmetry<1, 2>>()) && levi_civita(dynamic) && dynamic.size() == 6;
}

static_assert(tensor_symmetry());
#endif

#endif

int main() { return 0; }
