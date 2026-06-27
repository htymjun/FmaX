# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OUxMIX implements **float-float (FF) precision arithmetic** for CUDA Fortran. The `fltflt` type represents a value as an unevaluated sum `(hi + lo)` of two `real(4)` components, yielding ~48 mantissa bits (equivalent to `real(8)` precision) while remaining callable inside CUDA kernel code without modification.

## Build and Run

```bash
# Full test suite (from ouxmix/)
cmake -B build && cmake --build build -j
cd build && ./a.out

# Generate accuracy + throughput plot (output: test/ffp_results.png)
cmake --build build --target plot

# Individual tutorials (from tutorial/)
cmake -B build && cmake --build build -j
cd build && ./tut_add && ./tut_sub && ./tut_mul && ./tut_div

# Override GPU compute capability (default: 89 for RTX 4060)
cmake -B build -DCASE_GPU_CC=86
```

The compiler is `nvfortran` (set via `CMAKE_Fortran_COMPILER`; falls back to `mpif90`). Critical flags: `-cuda -fast -Mpreprocess -Minline=... -gpu=ptxinfo,rdc,lto,ccXX`.

## Architecture

```
OUxMIX/
  ouxmix/
    mod_ffp_eft.f90  — fltflt type + EFT backend (two_sum, fast_two_sum, two_prod)
    mod_ffp.f90      — public API (init, operators, warp shuffles); uses mod_ffp_eft
    CMakeLists.txt   — Build config; defines a `plot` custom target
    build/           — CMake build directory (gitignored)
  test/
    test_ffp.f90     — exhaustive accuracy + benchmark tests; writes ffp_results.csv
    plot_ffp.py      — reads ffp_results.csv, writes ffp_results.png
    ffp_results.csv  — latest benchmark output
    ffp_results.png  — latest plot
  tutorial/
    CMakeLists.txt   — builds four executables (tut_add, tut_sub, tut_mul, tut_div)
    tut_add.f90      — addition:       1e8 + 1, shows exact lo component
    tut_sub.f90      — subtraction:    (1+2^-25) - 1, below real(4) epsilon
    tut_mul.f90      — multiplication: 1.1 * 1.1, FMA-based TwoProd
    tut_div.f90      — division:       1/3, Dekker one-Newton-step refinement
```

### ouxmix/mod_ffp_eft.f90 (backend)

Defines `fltflt` and the three EFT primitives used internally by `mod_ffp`:
- `two_sum` — exact sum, 6 flops, no precondition
- `fast_two_sum` — exact sum, 3 flops, requires `|a| >= |b|`
- `two_prod` — exact product: `hi = a*b`, `lo = fma(a, b, -hi)` (1 MUL + 1 FMAF.F32)

### ouxmix/mod_ffp.f90 (public API)

Does `use mod_ffp_eft, only: fltflt, two_sum, fast_two_sum, two_prod` and exports:
- `fltflt` — the `(hi, lo)` type
- `init(a)` — constructor from `real(4)` or `real(8)` (both device and host)
- Overloaded `+`, `-`, `*`, `/` for all combinations of `fltflt` and `real(4)`
- `shfl_down_ff`, `shfl_xor_ff` — warp shuffle for `fltflt` (wraps `__shfl_down`/`__shfl_xor`)

### test/test_ffp.f90

Contains two modules: `test_kernels` (CUDA `attributes(global)` kernels) and the `test_ffp` program. The program writes results to `ffp_results.csv` and calls:
- `run_accuracy_tests()` — scalar ff+ff, mixed ff+r4, and dot-product tests
- `run_benchmark()` — times `real(8)` vs `fltflt` on a chained MAD loop over 2²⁰ elements

## Key Constraints

- **No default initializers on `fltflt`**: nvfortran crashes when compiling device code for types with default-initialized components. Always use `init()` or set `%hi` and `%lo` explicitly before use.
- **`two_prod` requires hardware FMA**: the `-fast` flag (enables `-Mfma`) is mandatory; without it `fma(a,b,-hi)` degenerates and `two_prod` breaks silently.
- **Warp shuffles use nvfortran 25.x API**: `__shfl_down`/`__shfl_xor` (mask-implicit form). Older nvfortran versions may need `__shfl_down_sync`.
- **EFT inlining is required for GPU performance**: `two_sum`, `fast_two_sum`, `two_prod`, `shfl_down_ff`, `shfl_xor_ff` must appear in the `-Minline=name:...` list in `CMakeLists.txt`.
