# OUxMIX — Float-Float Precision for CUDA Fortran

A single-module CUDA Fortran library (`fltflt`) that delivers `real(8)`-equivalent
accuracy at near-`real(4)` throughput inside GPU kernels.

## Overview

The `fltflt` type represents a floating-point value as an unevaluated sum `(hi + lo)` of two `real(4)` components, 
giving ~48 mantissa bits — matching the precision of `real(8)`. 
Every routine carries `attributes(device)` and is callable inside `attributes(global)` / `!$cuf kernel do` kernels without modification. 
The one-file design (`ouxmix/fltflt.f90`) makes integration straightforward: copy the file and add `use fltflt` to your kernel module.

## Build & Run

Requires **nvfortran** (NVIDIA HPC SDK).

```bash
# Full test suite
cd ouxmix
cmake -B build && cmake --build build -j
cd build
./test_add && ./test_sub && ./test_mul && ./test_div \
  && ./test_dot && ./test_bench && ./test_extra

# Accuracy + throughput plot  →  test/ffp_results.png
cmake --build build --target plot

# Tutorials
cd tutorial
cmake -B build && cmake --build build -j
cd build
./tut_add && ./tut_sub && ./tut_mul && ./tut_div && ./tut_dot3

# Override GPU compute capability
cmake -B build -DCASE_GPU_CC=86
```

## Quick Start

```fortran
module my_kern
  use cudafor
  use fltflt          ! single-file module from ouxmix/fltflt.f90
  implicit none
contains

  attributes(global) subroutine kern(a_in, b_in, res_hi, res_lo)
    real(4), intent(in),  value  :: a_in, b_in
    real(4), intent(out), device :: res_hi(1), res_lo(1)
    type(fltflt) :: a, b, c
    a = fltflt_init(a_in)     ! construct from real(4)
    b = fltflt_init(b_in)
    c = a + b                 ! exact TwoSum addition
    res_hi(1) = c%hi
    res_lo(1) = c%lo
  end subroutine

end module my_kern

program main
  use cudafor
  use my_kern
  implicit none
  real(4), device :: rhi(1), rlo(1)
  real(4) :: hi, lo
  integer :: istat

  call kern<<<1,1>>>(1.0e8_4, 1.0_4, rhi, rlo)
  istat = cudaDeviceSynchronize()
  hi = rhi(1);  lo = rlo(1)

  ! Read the result as real(8)
  print *, "result =", real(hi, 8) + real(lo, 8)   ! 1.00000001E+08
end program
```

## Public API

All functions are `pure attributes(device)` unless noted.

### Type & Initialization

| Symbol | Description |
|--------|-------------|
| `type(fltflt)` | `(hi, lo)` pair of two `real(4)` |
| `fltflt_init(a)` | constructor from `real(4)` or `real(8)` (`attributes(device,host)`) |

### Arithmetic Operators

`+`, `-`, `*`, `/` — all combinations of `fltflt`, `real(4)`, `real(8)`.  
`==`, `/=`, `<`, `>`, `<=`, `>=` — lexicographic comparison on `(hi, lo)`.

### Core Functions

| Function | Description |
|----------|-------------|
| `fltflt_add_same_sign(a, b)` | 11-flop add valid only when `sign(a) == sign(b)` |
| `fltflt_fma(a, b, c)` | fused multiply-add `a*b+c`; 7 overloads (ff/r4 combinations) |
| `fltflt_fma_approx(a, b, c)` | like `fltflt_fma` but skips `a%lo*b%lo` |
| `fltflt_fmod(a, b)` | modulo; 2 overloads (ff/ff, ff/r4); not `pure` |

### Multi-Operand Addition

`fltflt_add3(a,b,c)`, `fltflt_add4`, `fltflt_add5` — exact multi-term addition.

### Algebraic

| Function | Description |
|----------|-------------|
| `fltflt_square(a)` | `a*a` via symmetric TwoProd |
| `fltflt_recip(a)` | `1/a` via Newton step |
| `fltflt_abs(a)` | branchless absolute value |
| `fltflt_sqrt(a)` | Newton-step square root |
| `fltflt_sqrt_fast(a)` | ~7-flop rsqrt-based square root |

### Geometry

| Function | Description |
|----------|-------------|
| `fltflt_norm3d(dx,dy,dz)` | `sqrt(dx²+dy²+dz²)` |
| `fltflt_hypot(a,b)` | `sqrt(a²+b²)` for `real(4)` inputs via exact dot2 |
| `fltflt_cross3d(cx,cy,cz, ax,ay,az, bx,by,bz)` | 3D cross product (subroutine); uses dot2 for exact cancellation |

### Exact Dot Products

Each function has three overloads dispatched by argument type:
`real(4)`, `real(8)`, or `type(fltflt)`.

| Function | Arguments | Description |
|----------|-----------|-------------|
| `fltflt_dot2(a,b,c,d)` | 4 args | exact `a*b + c*d` |
| `fltflt_dot3(a,b,c,d,e,f)` | 6 args | exact `a*b + c*d + e*f` |
| `fltflt_dot4(a,b,c,d,e,f,g,h)` | 8 args | exact 4-pair dot product |

### Rounding & Clamping

| Function | Description |
|----------|-------------|
| `fltflt_floor(a)` | floor |
| `fltflt_ceil(a)` | ceiling |
| `fltflt_round_to_nearest(a)` | round half-to-even |
| `fltflt_round_toward_zero(a)` | truncation |
| `fltflt_min(a,b)` | min; 3 overloads (ff/ff, ff/r4, r4/ff) |
| `fltflt_max(a,b)` | max; 3 overloads |
| `fltflt_clamp(a,lo,hi)` | clamp to [lo,hi] |

### Utility

| Function | Description |
|----------|-------------|
| `fltflt_sign(a,b)` | copysign: magnitude of `a` with sign of `b` |
| `fltflt_lerp(a,b,t)` | linear interpolation `a + t*(b-a)`, `t :: real(4)` |
| `fltflt_pow_int(a,n)` | `a^n` for non-negative integer `n`; not `pure` |

### Warp Primitives

| Function | Description |
|----------|-------------|
| `fltflt_shfl_down(a, delta)` | warp shuffle down for `fltflt` |
| `fltflt_shfl_xor(a, mask)` | warp shuffle XOR for `fltflt` |
| `fltflt_warp_reduce_sum(val)` | XOR-butterfly all-reduce across 32 lanes; not `pure` |

## Key Constraints

**No default initializers.** nvfortran crashes compiling device code for types
with default-initialized components. Always use `fltflt_init()` or set `%hi`
and `%lo` explicitly before use.

**Use `__fmaf_rn` for hardware FMA.** The standard `fma()` intrinsic triggers
error 1253 in device code on nvfortran 24.7. A naive `a*b+c` wrapper lets the
GPU optimizer fold `a*b+(-a*b)` to zero, breaking EFT accuracy. Import
`__fmaf_rn` via `use cudadevice`.

**Inline the EFT helpers.** Add every helper listed in `ouxmix/CMakeLists.txt`
under `FFP_INLINE` to `-Minline=name:...`. Without inlining, the GPU backend
cannot inline derived-type returns through Fortran's front-end inliner; LTO
(`-gpu=lto`) handles it at link time.

## License

This project is a CUDA Fortran translation of the C++ header `fltflt.h`
distributed with [NVIDIA MatX](https://github.com/NVIDIA/MatX).

| Work | Copyright |
|------|-----------|
| Original C++ `fltflt.h` | Copyright (c) 2026, NVIDIA Corporation |
| CUDA Fortran translation | Copyright (c) 2026, Jun Hatayama |

Both works are released under the **BSD 3-Clause License**.
The full license text is in the header of [`ouxmix/fltflt.f90`](ouxmix/fltflt.f90).

## References

- A. Thall (2006). "Extended-Precision Floating-Point Numbers for GPU Computation."
- Y. Zhang & J. Aiken (SC'25). "High-Performance Branch-Free Algorithms for Extended-Precision Floating-Point Arithmetic." *(FPAN addition)*
- T. Ogita, S. M. Rump & S. Oishi (2005). "Accurate Sum and Dot Product." *SIAM J. Sci. Comput.* *(compensated dot)*
