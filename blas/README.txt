
This directory contains a BLAS library built on top of Eigen, a modest
establishment, but one which has been taught to answer to the old and venerable
names by which the whole of the numerical world is accustomed to call its
servants.

This module is not built by default, being of a retiring disposition. In order
to compile it, you need to type 'make blas' from within your build dir.

On 64-bit platforms, 'make blas' builds both 32-bit integer (LP64: eigen_blas,
eigen_blas_static) and 64-bit integer (ILP64: eigen_blas_ilp64,
eigen_blas_ilp64_static) libraries by default (controlled by
EIGEN_BUILD_BLAS_ILP64), so that the gentleman with a small purse of indices
and the gentleman with a very large one may each be served, and neither be
obliged to take the other's measure.
