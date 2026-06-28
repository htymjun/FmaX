# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OUxMIX implements **float-float (FF) precision arithmetic** for CUDA Fortran. The `fltflt` type represents a value as an unevaluated sum `(hi + lo)` of two `real(4)` components, yielding ~48 mantissa bits (equivalent to `real(8)` precision) while remaining callable inside CUDA kernel code without modification.

## Build and Run

```bash
# Full test suite (from ouxmix/)
cmake -B build && cmake --build build -j
cd build && ./test_add && ./test_sub && ./test_mul && ./test_div && ./test_dot && ./test_bench && ./test_extra

# Generate accuracy + throughput plot (output: test/ffp_results.png)
cmake --build build --target plot

# Individual tutorials (from tutorial/)
cmake -B build && cmake --build build -j
cd build && ./tut_add && ./tut_sub && ./tut_mul && ./tut_div && ./tut_dot3

# Override GPU compute capability (default: 89 for RTX 4060)
cmake -B build -DCASE_GPU_CC=86
```

The compiler is `nvfortran` (set via `CMAKE_Fortran_COMPILER`; falls back to `mpif90`). Critical flags: `-cuda -fast -Mpreprocess -Minline=... -gpu=ptxinfo,rdc,lto,ccXX`.

## Architecture

```
OUxMIX/
  ouxmix/
    fltflt.f90       — single source: fltflt type + EFT primitives + public API
    CMakeLists.txt   — builds 6 test executables + plot target
    build/           — CMake build directory (gitignored)
  test/
    test_add.f90     — addition accuracy + benchmark (r4, r8, fltflt)
    test_sub.f90     — subtraction accuracy + benchmark
    test_mul.f90     — multiplication accuracy + benchmark
    test_div.f90     — division accuracy + benchmark
    test_dot.f90     — compensated dot2/dot3 accuracy + benchmark
    test_bench.f90   — head-to-head r4 vs r8 vs fltflt throughput; writes ffp_results.csv
    plot_ffp.py      — reads ffp_results.csv, writes ffp_results.png
  tutorial/
    CMakeLists.txt   — builds 9 executables
    tut_add.f90      — addition: 1e8 + 1
    tut_sub.f90      — subtraction: (1+2^-25) - 1, below real(4) epsilon
    tut_mul.f90      — multiplication: 1.1 * 1.1, FMA-based TwoProd
    tut_div.f90      — division: 1/3, Newton-step refinement
    tut_add_r8.f90   — real(8)-input addition
    tut_sub_r8.f90   — real(8)-input subtraction
    tut_mul_r8.f90   — real(8)-input multiplication
    tut_div_r8.f90   — real(8)-input division
    tut_dot3.f90     — compensated dot product: a*b + c*d + e*f
```

### ouxmix/fltflt.f90

Single module `fltflt`. Private EFT primitives (always inlined, never in public API):
- `fltflt_two_sum` — exact sum, 6 flops, no precondition
- `fltflt_fast_two_sum` — exact sum, 3 flops, requires `|a| >= |b|`
- `fltflt_two_prod_fma` — exact product: `hi = a*b`, `lo = __fmaf_rn(a,b,-hi)` (1 MUL + 1 FMAF.F32)

Public API (all `pure attributes(device)` unless noted):
- `fltflt` — the `(hi, lo)` type
- `fltflt_init(a)` — constructor from `real(4)` or `real(8)` (`attributes(device,host)`)
- `+`, `-`, `*`, `/` — operators for all combinations of `fltflt`, `real(4)`, `real(8)`
- `==`, `/=`, `<`, `>`, `<=`, `>=` — comparison operators (lexicographic on hi then lo)
- `fltflt_add_same_sign(a,b)` — 11-flop add, valid only when `sign(a) == sign(b)`
- `fltflt_fma(a,b,c)` — fused multiply-add `a*b+c`; 7 overloads (ff/r4 combinations)
- `fltflt_fma_approx(a,b,c)` — like `fltflt_fma` but skips `a%lo*b%lo`; 7 overloads
- `fltflt_fmod(a,b)` — modulo, 2 overloads (ff/ff, ff/r4); not `pure` (uses `do while`)
- `fltflt_add3(a,b,c)`, `fltflt_add4`, `fltflt_add5` — multi-operand exact add
- `fltflt_square(a)` — `a*a` using symmetric TwoProd
- `fltflt_recip(a)` — `1/a` via Newton step
- `fltflt_dot2(a,b,c,d)` — exact `a*b + c*d`
- `fltflt_dot3(a,b,c,d,e,f)` — exact `a*b + c*d + e*f`
- `fltflt_dot4(a,b,c,d,e,f,g,h)` — exact 4-pair dot product
- `fltflt_abs(a)` — branchless absolute value
- `fltflt_sqrt(a)` — Newton-step square root
- `fltflt_sqrt_fast(a)` — ~7-flop rsqrt-based square root
- `fltflt_norm3d(dx,dy,dz)` — `sqrt(dx^2 + dy^2 + dz^2)`
- `fltflt_hypot(a,b)` — `sqrt(a^2 + b^2)` for `real(4)` inputs via exact dot2
- `fltflt_round_to_nearest(a)`, `fltflt_round_toward_zero(a)`, `fltflt_floor(a)`, `fltflt_ceil(a)` — rounding
- `fltflt_min(a,b)`, `fltflt_max(a,b)` — min/max; 3 overloads each (ff/ff, ff/r4, r4/ff)
- `fltflt_clamp(a,lo,hi)` — clamp to [lo,hi]
- `fltflt_sign(a,b)` — copysign: magnitude of a with sign of b
- `fltflt_lerp(a,b,t)` — linear interpolation `a + t*(b-a)`, `t :: real(4)`
- `fltflt_pow_int(a,n)` — `a^n` for non-negative integer n; not `pure`
- `fltflt_cross3d(cx,cy,cz, ax,ay,az, bx,by,bz)` — 3D cross product (subroutine); uses dot2 for exact cancellation
- `fltflt_shfl_down(a,delta)`, `fltflt_shfl_xor(a,mask)` — warp shuffles for `fltflt`
- `fltflt_warp_reduce_sum(val)` — XOR-butterfly all-reduce across 32 lanes; not `pure`

### test/

Programs (one per operation domain), each containing a `module test_*_kern` with CUDA `attributes(global)` kernels and a `program test_*` with accuracy cases and a benchmark. All write CSV results to `test/`.

## Key Constraints

- **No default initializers on `fltflt`**: nvfortran crashes when compiling device code for types with default-initialized components. Always use `fltflt_init()` or set `%hi` and `%lo` explicitly before use.

- **Hardware FMA via `__fmaf_rn`**: The Fortran 2023 `fma()` intrinsic triggers error 1253 in device code on nvfortran 24.7 (resolves as host C lib). A naive wrapper `ff_fma(a,b,c)=a*b+c` silently breaks `fltflt_two_prod_fma`: the GPU optimizer CSE-folds `a*b+(−a*b)=0` before `-Mfma` contraction runs, making `lo=0` and destroying accuracy. Solution: use `__fmaf_rn(a,b,c)` from `use cudadevice` directly — it is an opaque intrinsic that maps to hardware FMAF.F32 and cannot be simplified by the optimizer. Generates W-0473 warning when used inside `pure` functions; compile succeeds and works correctly. The `-fast` flag (enables `-Mfma`) is still required.

- **Warp shuffles use nvfortran 25.x API**: `__shfl_down`/`__shfl_xor` (mask-implicit form). Older nvfortran versions may need `__shfl_down_sync`.

- **EFT inlining required for GPU performance**: `fltflt_two_sum`, `fltflt_fast_two_sum`, `fltflt_two_prod_fma`, and all major named functions must appear in `-Minline=name:...` in `CMakeLists.txt`. Without inlining, the Fortran-level inliner refuses derived-type returns ("return type doesn't match"), but the GPU backend inliner handles it at link time via `-gpu=lto`.

- **Kernel module `private`**: Any module that does `use cudafor` (or `use fltflt`) and is itself used by a program that also does `use cudafor` must declare `private` with explicit `public` for its kernel names. Without `private`, the module re-exports all cudafor generic interfaces, causing nvfortran 24.7 to fail resolving `cudaEventRecord`/`cudaEventDestroy` in internal subroutines with "Could not resolve generic procedure".

- **cudaEventRecord stream argument**: Requires `integer(8)` (e.g. `0_8`), not default `integer`. Default int triggers "Could not resolve generic procedure" in nvfortran 24.7.

- **`.not.` operator broken on logical in device code**: In nvfortran, `.true.` is represented as the integer `+1` (not `-1` as in many Fortran implementations). Bitwise `.not.(+1)` = `0xFFFFFFFE` = `-2`, which is non-zero and therefore "true" in CUDA. Consequence: `.not. .true.` evaluates to "true" instead of "false". Affected functions: any `/=` implementation written as `.not. (==)`. Fix: implement `ne` directly using `/=` combined with `.or.`, never via `.not. eq_*()`. This is why `ne_ff_ff`, `ne_ff_r4`, and `ne_r4_ff` in `fltflt.f90` avoid `.not.`.
