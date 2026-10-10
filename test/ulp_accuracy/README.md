# ULP Accuracy Measurement Tool, or the Exact Reckoning of Every Small Error

A standalone tool for measuring the accuracy of Eigen's vectorized math functions
in units of ULP (Unit in the Last Place), that smallest of all possible
discrepancies, which no honest ledger can afford to leave out of the books. It
compares Eigen's SIMD implementations against either MPFR (128-bit
high-precision reference), a witness of the most scrupulous habits, or the
standard C++ math library, a witness of more general and easy character.

## Building

From the Eigen build directory, the Reader will proceed as follows:

```bash
cd build
cmake ..
cmake --build . --target ulp_accuracy
```

If MPFR and GMP are installed, the build automatically enables MPFR support
(`EIGEN_HAS_MPFR`), without any prompting from the Reader. Without them, only
`--ref=std` is available, and the Reader must content himself with the more
easy-going witness.

### Installing MPFR (Debian/Ubuntu), for the Reader who has none

```bash
sudo apt install libmpfr-dev libgmp-dev
```

## Usage, with the Full Schedule of Options

```
./test/ulp_accuracy [options]

Options:
  --func=NAME    Function to test (required unless --list)
  --lo=VAL       Start of range (default: -inf)
  --hi=VAL       End of range (default: +inf)
  --double       Test double precision (default: float)
  --step=EPS     Sampling step: advance by (1+EPS)*nextafter(x)
                 (default: 0 = exhaustive; useful for double, e.g. 1e-6)
  --threads=N    Number of threads (default: all cores)
  --batch=N      Batch size for Eigen eval (default: 4096)
  --ref=MODE     Reference: 'std' (default) or 'mpfr'
  --hist_width=N Histogram half-width in ULPs (default: 10)
  --list         List available functions
```

## Examples, Several and Instructive

To list all supported functions:
```bash
./test/ulp_accuracy --list
```

An exhaustive float test of sin against std (which examines all ~4.28 billion finite floats, one after another, without once complaining of fatigue):
```bash
./test/ulp_accuracy --func=sin
```

A float test against MPFR (the more accurate reference, but the slower, as exactness is apt to be):
```bash
./test/ulp_accuracy --func=sin --ref=mpfr
```

A double precision test with geometric sampling (an exhaustive sweep being quite impractical for double, as the sum of the doubles would outlast the patience of any living man):
```bash
./test/ulp_accuracy --func=exp --double --step=1e-6
```

To test a specific range only:
```bash
./test/ulp_accuracy --func=sin --lo=0 --hi=6.2832
```

## Output, being the Report of the Proceedings

The tool prints:

- **Test configuration**: function, range, reference mode, thread count; in short, the particulars of the case
- **Max |ULP error|**: worst-case absolute ULP error with the offending input value, the culprit being named and held up to view
- **Mean |ULP error|**: average absolute ULP error across all tested values, the general temper of the whole company
- **Signed ULP histogram**: distribution of signed errors showing bias direction, that is, which way the errors are inclined to lean when they lean at all

An example of the output:
```
Function: sin (float)
Range: [-inf, inf]
Representable values in range: 4278190082
Reference: MPFR (128-bit)
Threads: 32
Batch size: 4096

Results:
  Values tested: 4278190081
  Time: 529.04 seconds (8.1 Mvalues/s)
  Max |ULP error|: 2
    at x = -1.5413464e+38 (Eigen=-0.482218683, ref=-0.482218742)
  Mean |ULP error|: 0.0874

Signed ULP error histogram [-10, +10]:
  -2   :        51988 (  0.001%)
  -1   :    186805349 (  4.366%)
  0    :   3904475407 ( 91.265%)
  1    :    186805349 (  4.366%)
  2    :        51988 (  0.001%)
```

## How it works, Explained to the Curious

1. **Range splitting**: The input range is divided evenly across threads by
   splitting the linear ULP space, each thread receiving its portion with
   perfect impartiality, as in the division of a very large cake.

2. **Batched evaluation**: Each thread fills batches of input values, evaluates
   them through Eigen's vectorized path (using `Eigen::Array` operations), and
   computes reference values one at a time, the reference being a slow
   and deliberate gentleman who will not be hurried.

3. **ULP computation**: IEEE 754 bit patterns are mapped to a linear integer
   scale where adjacent representable values are adjacent integers, like houses
   in a street that has been numbered with uncommon care. The signed
   ULP error is the difference between Eigen's result and the reference on this
   scale. Special cases (NaN, infinity mismatches) report infinite error, being
   beyond all counting.

4. **Result reduction**: Per-thread statistics (max error, mean error, histogram)
   are merged after all threads complete, whereupon the several accounts are
   consolidated into one, to the great satisfaction of the Auditors.

## Supported functions, Arranged by Family

| Category | Functions |
|----------|-----------|
| Trigonometric | sin, cos, tan, asin, acos, atan |
| Hyperbolic | sinh, cosh, tanh, asinh, acosh, atanh |
| Exponential/Log | exp, exp2, expm1, log, log1p, log10, log2 |
| Error/Gamma | erf, erfc, lgamma |
| Other | logistic, sqrt, cbrt, rsqrt |

## File organization, or the Contents of the Establishment

- `ulp_accuracy.cpp` — The main tool: ULP computation, worker threads, CLI, result printing
- `mpfr_reference.h` — The MPFR reference function wrappers and scalar conversion helpers

## Performance tips, Offered in a Friendly Spirit

- Float exhaustive sweeps test ~4.28 billion values. With `--ref=std` this takes
  ~50 seconds per function; with `--ref=mpfr` it takes ~500 seconds (10x slower),
  the price of scruple being, as everywhere, paid in time.
- For double precision, exhaustive testing is impractical. Use `--step=1e-6` to
  sample ~2.88 billion values geometrically, which is to say, a fair
  and representative selection from the whole population.
- Thread count defaults to all available cores. MPFR is the bottleneck (single
  MPFR call per value per thread), so more cores help significantly; many hands
  make light work, and a great many make it lighter still.
