// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Helpers shared across GPU library tests:
//   * runtime probes that exit(77) (CI skip) when a library is unavailable
//   * a small make_test_value() that constructs a Scalar with an imaginary
//     component for complex types so the complex code paths are genuinely
//     exercised — without this, Scalar(real_value) silently zeros the imag.
//   * StreamGate and LegacyStreamSentinel for stream-ordering checks.

#ifndef EIGEN_UNSUPPORTED_TEST_GPU_TEST_HELPERS_H
#define EIGEN_UNSUPPORTED_TEST_GPU_TEST_HELPERS_H

#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <type_traits>

namespace gpu_test {

template <typename Scalar, typename RealScalar>
inline std::enable_if_t<Eigen::NumTraits<Scalar>::IsComplex, Scalar> make_test_value(RealScalar re, RealScalar im) {
  return Scalar(re, im);
}
template <typename Scalar, typename RealScalar>
inline std::enable_if_t<!Eigen::NumTraits<Scalar>::IsComplex, Scalar> make_test_value(RealScalar re,
                                                                                      RealScalar /*im*/) {
  return Scalar(re);
}

#ifdef CUDART_VERSION
// The CUDA runtime loads without a driver, so a GPU test binary starts happily on
// a machine with no device and only fails at its first allocation -- as an abort
// out of EIGEN_CUDA_RUNTIME_CHECK, not a skip. Probe the device count first so
// those runs report as skipped, the way the per-library probes below do.
inline void require_cuda_device() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    std::cout << "SKIP: GPU tests require a CUDA device. cudaGetDeviceCount reported " << cudaGetErrorString(status)
              << " with " << count << " device(s)." << std::endl;
    std::exit(77);
  }
}

// Parks a stream: work enqueued on it after construction does not start until
// the gate opens. Ordering tests park one context, enqueue the access under
// test on another, and check that the second access waited for the first.
class StreamGate {
 public:
  explicit StreamGate(cudaStream_t stream) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(stream, &StreamGate::hold, this));
  }

  // Waits for the host function to finish, so it never touches a dead gate.
  ~StreamGate() {
    open();
    if (opener_.joinable()) opener_.join();
    while (!exited_.load(std::memory_order_acquire)) std::this_thread::yield();
  }

  StreamGate(const StreamGate&) = delete;
  StreamGate& operator=(const StreamGate&) = delete;

  void open() { open_.store(true, std::memory_order_release); }

  // Opens the gate from another thread once the stream has been parked for
  // `delay`, so the calling thread may block on work queued behind the gate.
  void openAfter(std::chrono::milliseconds delay) {
    opener_ = std::thread([this, delay] {
      while (!entered_.load(std::memory_order_acquire)) std::this_thread::yield();
      std::this_thread::sleep_for(delay);
      open();
    });
  }

 private:
  static void CUDART_CB hold(void* data) {
    StreamGate* gate = static_cast<StreamGate*>(data);
    gate->entered_.store(true, std::memory_order_release);
    while (!gate->open_.load(std::memory_order_acquire)) std::this_thread::yield();
    gate->exited_.store(true, std::memory_order_release);
  }

  std::atomic<bool> entered_{false};
  std::atomic<bool> open_{false};
  std::atomic<bool> exited_{false};
  std::thread opener_;
};

// Holds a capture open on a blocking stream, so that the CUDA runtime rejects any
// use of the legacy default stream, from any thread, and invalidates the capture;
// end() reports whether that happened. Creating a library handle synchronizes
// the device, which the capture rejects too: create handles beforehand.
class LegacyStreamSentinel {
 public:
  LegacyStreamSentinel() {
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamCreate(&stream_));
    EIGEN_CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed));
  }

  ~LegacyStreamSentinel() {
    (void)end();
    (void)cudaStreamDestroy(stream_);
  }

  LegacyStreamSentinel(const LegacyStreamSentinel&) = delete;
  LegacyStreamSentinel& operator=(const LegacyStreamSentinel&) = delete;

  // True iff nothing used the legacy default stream since construction.
  bool end() {
    if (!ended_) {
      ended_ = true;
      cudaGraph_t graph = nullptr;
      const cudaError_t status = cudaStreamEndCapture(stream_, &graph);
      if (graph) (void)cudaGraphDestroy(graph);
      (void)cudaGetLastError();
      clean_ = status == cudaSuccess;
    }
    return clean_;
  }

 private:
  cudaStream_t stream_ = nullptr;
  bool ended_ = false;
  bool clean_ = false;
};

// Parks `stream` behind a host function that returns when the ParkedStream is
// destroyed or after `timeout`, whichever comes first. An operation that does
// not wait for `stream` returns while held() is still true; one that does
// returns only after the timeout, when held() is false.
class ParkedStream {
 public:
  explicit ParkedStream(cudaStream_t stream, std::chrono::milliseconds timeout = std::chrono::seconds(1))
      : stream_(stream), timeout_(timeout) {
    EIGEN_CUDA_RUNTIME_CHECK(cudaLaunchHostFunc(stream_, &ParkedStream::hold, this));
  }

  ~ParkedStream() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      unpark_ = true;
    }
    cv_.notify_one();
    (void)cudaStreamSynchronize(stream_);
  }

  ParkedStream(const ParkedStream&) = delete;
  ParkedStream& operator=(const ParkedStream&) = delete;

  bool held() const { return !released_.load(std::memory_order_acquire); }

 private:
  static void CUDART_CB hold(void* data) {
    ParkedStream* self = static_cast<ParkedStream*>(data);
    std::unique_lock<std::mutex> lock(self->mutex_);
    self->cv_.wait_for(lock, self->timeout_, [self] { return self->unpark_; });
    self->released_.store(true, std::memory_order_release);
  }

  cudaStream_t stream_;
  std::chrono::milliseconds timeout_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool unpark_ = false;
  std::atomic<bool> released_{false};
};
#endif

#ifdef CUSOLVER_VERSION
// cusolverDnXpotrf in cuSOLVER 11.4.1 (CUDA 11.8) and 11.4.2 (CUDA 12.0) reports
// success on a matrix that is not positive definite and returns a NaN factor;
// 11.4.4 (CUDA 12.1) and later set info > 0 again. Tests that expect
// NumericalIssue from such an LLT pass the reported info here: on an affected
// library this prints a warning and returns true, and the caller skips the check.
inline bool cusolver_potrf_missed_non_spd(Eigen::ComputationInfo info) {
  constexpr int kFirstVersionReportingNonSpd = 11404;
  int version = 0;
  if (cusolverGetVersion(&version) != CUSOLVER_STATUS_SUCCESS || version >= kFirstVersionReportingNonSpd ||
      info != Eigen::Success) {
    return false;
  }
  std::cout << "WARNING: cuSOLVER " << version / 1000 << '.' << version / 100 % 10 << '.' << version % 100
            << " reported success for the potrf of a matrix that is not positive definite (defect in 11.4.1 and "
               "11.4.2, fixed in 11.4.4); skipping the NumericalIssue check."
            << std::endl;
  return true;
}
#endif

#ifdef CUDSS_VERSION
inline void require_cudss_context() {
  cudssHandle_t handle = nullptr;
  const cudssStatus_t status = cudssCreate(&handle);
  if (status != CUDSS_STATUS_SUCCESS) {
    std::cout << "SKIP: cuDSS tests require an initialized cuDSS context. cudssCreate failed with status "
              << static_cast<int>(status) << std::endl;
    std::exit(77);
  }
  EIGEN_CUDSS_CHECK(cudssDestroy(handle));
}
#endif

#ifdef CUSPARSE_VERSION
inline void require_cusparse_context() {
  cusparseHandle_t handle = nullptr;
  const cusparseStatus_t status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cout << "SKIP: cuSPARSE tests require an initialized cuSPARSE context. cusparseCreate failed with status "
              << static_cast<int>(status) << std::endl;
    std::exit(77);
  }
  EIGEN_CUSPARSE_CHECK(cusparseDestroy(handle));
}
#endif

#ifdef CUFFT_VERSION
inline void require_cufft_context() {
  cufftHandle plan = 0;
  // cufftCreate allocates a plan handle without configuring it; succeeds only
  // when the cuFFT runtime is loadable.
  const cufftResult status = cufftCreate(&plan);
  if (status != CUFFT_SUCCESS) {
    std::cout << "SKIP: cuFFT tests require a working cuFFT runtime. cufftCreate failed with status "
              << static_cast<int>(status) << std::endl;
    std::exit(77);
  }
  EIGEN_CUFFT_CHECK(cufftDestroy(plan));
}
#endif

}  // namespace gpu_test

#endif  // EIGEN_UNSUPPORTED_TEST_GPU_TEST_HELPERS_H
