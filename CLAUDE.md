# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FmaX implements **float-float (FF) precision arithmetic** for CUDA Fortran. The `fltflt` type represents a value as an unevaluated sum `(hi + lo)` of two `real(4)` components, yielding ~48 mantissa bits (equivalent to `real(8)` precision) while remaining callable inside CUDA kernel code without modification.

A second, independent module, `vfma` (`FmaX/vfma.f90.fypp`), provides the `vfma` fypp macro: given array-slice/scalar operand strings, it parses which operands are stencil-style array slices, deduplicates overlapping loads across operands into single register reads, then unrolls `n` scalar `__fma_rn()` calls. It targets plain `real(8)` and is independent of `fltflt` (the module's own comment marks `fltflt`-style precision support as future/aspirational — currently `vfma`'s public interface is empty). The module is deliberately not named `fmax`: nvfortran's `libm` module (which it imports) exports a real device intrinsic named `fmax(a,b)`, and a same-named containing module risked a future collision if a public function were ever added under that name.

## Build and Run

**No top-level or `FmaX/CMakeLists.txt` build exists.** `ouxmix/CMakeLists.txt` and `test/CMakeLists.txt` were deleted during the `ouxmix` → `FmaX` rename and have not been replaced; `test/` was removed entirely.

`tutorial/` is split into one directory per case, each independently buildable via its own Makefile (`nvfortran` at `/opt/nvidia/hpc_sdk/.../compilers/bin/nvfortran`, RTX 4060 GPU, `cc89` default `CASE_GPU_CC` — override via `make CASE_GPU_CC=86`):

```bash
cd tutorial/add   # or sub / mul / div / dot3
make              # compiles ../../FmaX/fltflt.f90 + tut_<case>.f90 -> executable tut_<case>
make run          # build (if needed) then run it
make clean        # remove the executable, *.o, *.mod
```

All five pass cleanly and print a real(4)-vs-fltflt-vs-real(8)-reference comparison — zero warnings of any kind, only informational `ptxas info` output.

`tutorial/vfma/` (the `vfma` macro demo, unrelated to `fltflt`) instead preprocesses with fypp then compiles the result (no `program`, so no `run` target — it's a bare `attributes(device)` subroutine):

```bash
cd tutorial/vfma
fypp -M ../../FmaX -m vfma_helpers tut_vfma.f90.fypp tut_vfma.f90   # or: make preprocess
make preprocess   # fypp only -> tut_vfma.f90
make compile      # nvfortran -c tut_vfma.f90
make clean        # remove generated tut_vfma.f90, *.o, *.mod
make              # == make compile
```

`make compile`/`make`/`make all` pass cleanly (only informational `ptxas info` output, no errors).

## Architecture

```
FmaX/
  FmaX/
    fltflt.f90         — single source: fltflt type + EFT primitives + public API
    vfma.f90.fypp       — vfma module: vfma macro for vector-FMA codegen (see below)
    vfma_helpers.py      — Python helpers for vfma.f90.fypp, loaded via fypp -m
  tutorial/
    add/    tut_add.f90    Makefile   — addition: 1.0d8 + 1.0d0 (dble-initialized fltflt)
    sub/    tut_sub.f90    Makefile   — subtraction: (1+2^-25) - 1, below real(4) epsilon
    mul/    tut_mul.f90    Makefile   — multiplication: 1.1d0 * 1.1d0, FMA-based TwoProd
    div/    tut_div.f90    Makefile   — division: 1/3, Newton-step refinement
    dot3/   tut_dot3.f90   Makefile   — compensated dot product: a*b + c*d + e*f
    vfma/   tut_vfma.f90.fypp   tut_vfma.f90   Makefile   — vfma macro demo (bare device subroutine, no host driver)
```

Each of `add`/`sub`/`mul`/`div`/`dot3` is a standalone fltflt tutorial: it initializes `fltflt` from a `real(8)`/dble value via `fltflt_init()` (an exact 2-component split, not a lossy real(4) truncation), computes the operation as `fltflt`, and compares the result against a `real(8), parameter` reference computed in pure double precision — alongside a naive real(4)-truncate-then-compute comparison for contrast. These were previously split across 8 files (a real(4)-init variant and a `_r8` real(8)-init variant per operation, e.g. `tut_add.f90`/`tut_add_r8.f90`); they're now consolidated to one dble-based case per operation, one directory each.

`test/` (test_add/sub/mul/div/dot/bench/extra + plot_ffp.py) was removed entirely in the `ouxmix` → `FmaX` rename and has no replacement.

### FmaX/fltflt.f90

Single module `fltflt`. Private EFT primitives (always inlined, never in public API):
- `fltflt_two_sum` — exact sum, 6 flops, no precondition
- `fltflt_fast_two_sum` — exact sum, 3 flops, requires `|a| >= |b|`
- `fltflt_two_prod_fma` — exact product: `hi = a*b`, `lo = __fmaf_rn(a,b,-hi)` (1 MUL + 1 FMAF.F32)

Public API (`attributes(device)`; `pure` unless noted — see the "not pure" note below on why FMA-based functions lost `pure`):
- `fltflt` — the `(hi, lo)` type
- `fltflt_init(a)` — constructor from `real(4)` or `real(8)` (`attributes(device,host)`)
- `+`, `-` — add/sub operators for all combinations of `fltflt`, `real(4)`, `real(8)`; **pure** (two_sum-based, no FMA)
- `*`, `/` — mul/div operators for all combinations; **not pure** (FMA-based)
- `==`, `/=`, `<`, `>`, `<=`, `>=` — comparison operators (lexicographic on hi then lo); pure
- `fltflt_add_same_sign(a,b)` — 11-flop add, valid only when `sign(a) == sign(b)`; pure
- `fltflt_fma(a,b,c)` — fused multiply-add `a*b+c`; 7 overloads (ff/r4 combinations); **not pure**
- `fltflt_fma_approx(a,b,c)` — like `fltflt_fma` but skips `a%lo*b%lo`; 7 overloads; **not pure**
- `fltflt_fmod(a,b)` — modulo, 2 overloads (ff/ff, ff/r4); not `pure` (uses `do while`)
- `fltflt_add3(a,b,c)`, `fltflt_add4`, `fltflt_add5` — multi-operand exact add; pure
- `fltflt_square(a)` — `a*a` using symmetric TwoProd; **not pure**
- `fltflt_recip(a)` — `1/a` via Newton step; **not pure**
- `fltflt_dot2(a,b,c,d)` — exact `a*b + c*d`; **not pure**
- `fltflt_dot3(a,b,c,d,e,f)` — exact `a*b + c*d + e*f`; **not pure**
- `fltflt_dot4(a,b,c,d,e,f,g,h)` — exact 4-pair dot product; **not pure**
- `fltflt_abs(a)` — branchless absolute value; pure
- `fltflt_sqrt(a)` — Newton-step square root; returns `(0,0)` for zero input (guard required because `0/0 = NaN`); **not pure**
- `fltflt_sqrt_fast(a)` — ~7-flop rsqrt-based square root; also guards zero input; **not pure**
- `fltflt_norm3d(dx,dy,dz)` — `sqrt(dx^2 + dy^2 + dz^2)`; **not pure**
- `fltflt_hypot(a,b)` — `sqrt(a^2 + b^2)` for `real(4)` inputs via exact dot2; **not pure**
- `fltflt_round_to_nearest(a)`, `fltflt_round_toward_zero(a)`, `fltflt_floor(a)`, `fltflt_ceil(a)` — rounding; pure
- `fltflt_min(a,b)`, `fltflt_max(a,b)` — min/max; 3 overloads each (ff/ff, ff/r4, r4/ff); pure
- `fltflt_clamp(a,lo,hi)` — clamp to [lo,hi]; pure
- `fltflt_sign(a,b)` — copysign: magnitude of a with sign of b; pure
- `fltflt_lerp(a,b,t)` — linear interpolation `a + t*(b-a)`, `t :: real(4)`; **not pure**
- `fltflt_pow_int(a,n)` — `a^n` for non-negative integer n; not `pure`
- `fltflt_cross3d(cx,cy,cz, ax,ay,az, bx,by,bz)` — 3D cross product (subroutine); uses dot2 for exact cancellation; **not pure**
- `fltflt_shfl_down(a,delta)`, `fltflt_shfl_xor(a,mask)` — warp shuffles for `fltflt`
- `fltflt_warp_reduce_sum(val)` — XOR-butterfly all-reduce across 32 lanes; not `pure`

**"not pure"** above means `pure` was deliberately removed from that function (and every FMA-based function beneath it in the call chain, 42 functions total, including internal r4/r8 overload wrappers not listed above) to eliminate `NVFORTRAN-W-0473`. See Key Constraints for why this was the only way to actually remove the warning rather than just relocate it.

### FmaX/vfma.f90.fypp

`module vfma` (`use libm`) is currently just a placeholder host for the `vfma(n, a_str, b_str, c_str, res_str)` fypp macro — its public interface is empty (`fltflt`-style precision support is commented out as future work). `vfma` is invoked as `#:call vfma(...) ... #:endcall` (not `@:vfma(...)` — direct-call `@:` syntax passes arguments as raw unevaluated text, so `n` would arrive as the string `"3"` and quoted args would keep their literal quote characters; `#:call`'s header, by contrast, is evaluated as real Python via `__getargvalues`). Given operand strings like `'a(i-1:i+1,j,k)'`, `'a(i:i+2,j,k)'`, `'b(1:3)'`, `'c(1:3)'`:
1. Each operand is parsed (`vfma_helpers.parse_arg`) as either an array slice (base name + list of `n` element indices, preserving any trailing dimensions like `,j,k` verbatim on every generated index) or a scalar expression.
2. Overlapping array reads across the three input operands are deduplicated (`vfma_helpers.compute_loads`) into a single `real(8)` register load per distinct element (e.g. `a(i-1,j,k)`, `a(i,j,k)`, `a(i+1,j,k)`, `a(i+2,j,k)` load once each, even though `a(i-1:i+1,j,k)` and `a(i:i+2,j,k)` overlap at `i` and `i+1`). All register declarations are emitted before any register-load assignment, since Fortran requires specification statements to precede executable statements.
3. `n` unrolled `__fma_rn(a, b, c)` calls are emitted (the `real(8)` CUDA Fortran device intrinsic — plain `fma()` is a host call and is rejected inside `attributes(device)` code), each referencing register variables (via `vfma_helpers.clean_var_name`) instead of re-reading arrays.

The whole expansion (all three steps) is wrapped in `block ... end block`. Register names are only unique *within one macro invocation* (e.g. `reg_a_i_j_k`) — without the `block`, two `#:call vfma(...)` invocations in the same subroutine over the same array/overlapping indices would redeclare the same names in the subroutine's single specification part, an illegal duplicate declaration. `block` gives each invocation its own local scope for its registers while the caller's own variables (`a`, `b`, `c`, `i`, `j`, `k`, ...) remain visible via host association. Verified: nvfortran accepts `block` inside `attributes(device)` code, and a two-`vfma`-call reproduction over the same array/indices compiles cleanly with the wrapping (it does not without it).

### FmaX/vfma_helpers.py

Holds the macro's Python logic (`parse_arg`, `clean_var_name`, `compute_loads`, `resolve_terms`) as a real importable module, loaded via `fypp -m vfma_helpers` (see Build and Run). This exists because **fypp has no directive for embedding multi-statement Python code** — see Key Constraints.

## Key Constraints

- **No default initializers on `fltflt`**: nvfortran crashes when compiling device code for types with default-initialized components. Always use `fltflt_init()` or set `%hi` and `%lo` explicitly before use.

- **Hardware FMA via `__fmaf_rn`**: The Fortran 2023 `fma()` intrinsic triggers error 1253 in device code on nvfortran 24.7 (resolves as host C lib). A naive wrapper `ff_fma(a,b,c)=a*b+c` silently breaks `fltflt_two_prod_fma`: the GPU optimizer CSE-folds `a*b+(−a*b)=0` before `-Mfma` contraction runs, making `lo=0` and destroying accuracy. Solution: use `__fmaf_rn(a,b,c)` from `use cudadevice` directly — it is an opaque intrinsic that maps to hardware FMAF.F32 and cannot be simplified by the optimizer. The `-fast` flag (enables `-Mfma`) is still required.

- **`__fmaf_rn` and `pure`: W-0473, and why `pure` was removed instead of suppressed**: nvfortran flags `__fmaf_rn must have the PURE attribute` (`NVFORTRAN-W-0473`) whenever it's called inside a `pure` function — confirmed experimentally that this is tied to the intrinsic itself (renaming the import or giving it an explicit `pure` reinterface makes no difference), and that removing `pure` from just the immediate call site doesn't remove the warning, it relocates an identical one to that function's own caller (`<name> must have the PURE attribute`), and so on up the call chain. Since there's no per-message suppression flag in nvfortran (only the blanket `-Minform=severe`/`-w`, which would also hide genuinely new warnings), the only way to actually eliminate W-0473 — as opposed to hiding it — was to follow the cascade to its fixed point: `pure` was removed from all 42 functions that transitively reach `__fmaf_rn` (found by iterating "recompile → strip `pure` from every function named in a new W-0473 → repeat" until a clean build), covering every FMA-based operation (`*`, `/`, `fltflt_fma[_approx]`, `fltflt_square`, `fltflt_recip`, `fltflt_dot2/3/4`, `fltflt_sqrt[_fast]`, `fltflt_norm3d`, `fltflt_hypot`, `fltflt_lerp`, `fltflt_cross3d`, and their r4/r8 overload wrappers) — see the Public API list above for exactly which functions this touched. `+`/`-` and everything else that never reaches `__fmaf_rn` (two_sum-based add, comparisons, min/max/clamp, rounding, etc.) kept `pure`. Verified: identical numeric output before/after across all 5 tutorial cases, and zero warnings of any kind on a clean rebuild.

- **`0.0 / 0.0` as a literal produces W-0132, not just a runtime NaN**: nvfortran's front end constant-folds a literal-vs-literal division like `0.0 / 0.0` at compile time and flags it (`Floating pt. invalid oprnd. Check constants and constant expressions`), even though the intent (`fltflt_fmod`'s `b == 0` guard, both overloads) is a deliberate hardware NaN. Fixed by dividing a runtime variable already known to be zero by itself (e.g. `b_in%hi / b_in%hi` instead of `0.0 / 0.0`) — the front end only folds literal constant expressions, not variable reads, so this produces the identical NaN at runtime with no warning.

- **Linker `missing .note.GNU-stack section` warning is a toolchain artifact, not our code**: every `nvfortran -cuda` link emits this for its internal `pgcudafat*.o` temporary object (reproduced even in a two-line throwaway program with no `fltflt` involvement at all). Silenced by adding `-Wl,-z,noexecstack` to the link flags (see the per-case `tutorial/*/Makefile`s) — a standard, safe linker flag; nothing in the Fortran source can address it since the offending object is compiler-generated.

- **Warp shuffles use nvfortran 25.x API**: `__shfl_down`/`__shfl_xor` (mask-implicit form). Older nvfortran versions may need `__shfl_down_sync`.

- **EFT inlining required for GPU performance**: `fltflt_two_sum`, `fltflt_fast_two_sum`, `fltflt_two_prod_fma`, and all major named functions must appear in `-Minline=name:...` in each `tutorial/*/Makefile`'s `FCFLAGS`. The Fortran-source-level inliner these names request always refuses derived-type returns (`subprogram not inlined -- return type doesn't match: fltflt_two_sum`, printed under `-Minfo=inline`) — that's a structural nvfortran limitation, not fixable by requesting it differently, since `type(fltflt)` is exactly the kind of return the source-level inliner can't handle; actual inlining still happens at link time via `-gpu=lto` regardless of whether the source-level attempt "succeeds". The per-case Makefiles use `-Minfo=accel` (not `-Minfo=accel,inline`) specifically to not print that structurally-unfixable inform message — it's not a defect, just noise from a diagnostic we don't need, and `-Minfo=accel` still surfaces the useful ptxas register/spill info.

- **Kernel module `private`**: Any module that does `use cudafor` (or `use fltflt`) and is itself used by a program that also does `use cudafor` must declare `private` with explicit `public` for its kernel names. Without `private`, the module re-exports all cudafor generic interfaces, causing nvfortran 24.7 to fail resolving `cudaEventRecord`/`cudaEventDestroy` in internal subroutines with "Could not resolve generic procedure".

- **cudaEventRecord stream argument**: Requires `integer(8)` (e.g. `0_8`), not default `integer`. Default int triggers "Could not resolve generic procedure" in nvfortran 24.7.

- **Fortran `mod` sign follows dividend**: `mod(-3.0, 2.0) = -1.0` (not `+1.0`). Even-parity checks must use `mod(x, 2.0) == 0.0`, not `mod(a, 2.0) == mod(b, 2.0)` — the latter fails when `a` and `b` have opposite signs and are both odd. `fltflt_round_to_nearest` uses `mod(a%hi + r_lo, 2.0) /= 0.0` for exactly this reason.

- **`.not.` operator broken on logical in device code**: In nvfortran, `.true.` is represented as the integer `+1` (not `-1` as in many Fortran implementations). Bitwise `.not.(+1)` = `0xFFFFFFFE` = `-2`, which is non-zero and therefore "true" in CUDA. Consequence: `.not. .true.` evaluates to "true" instead of "false". Affected functions: any `/=` implementation written as `.not. (==)`. Fix: implement `ne` directly using `/=` combined with `.or.`, never via `.not. eq_*()`. This is why `ne_ff_ff`, `ne_ff_r4`, and `ne_r4_ff` in `fltflt.f90` avoid `.not.`.

- **fypp has no `#:python`/`#:endpython` directive**: Confirmed absent through the latest release (3.2 — checked both the installed version and PyPI's version list). fypp's directive set is `if/elif/else/endif`, `def/enddef`, `set`, `del`, `for/endfor`, `call/block/endcall/endblock/nextarg/contains`, `include`, `mute/endmute`, `stop`, `assert`, `global` — nothing for embedding multi-statement Python (`def` functions, loops, etc.) inline. The supported mechanism is the `-m MODULE` CLI flag, which imports a real `.py` module and binds it to a variable of its own name (not its members — reference helpers as `modname.func(...)`, not bare `func(...)`). This is why `FmaX/vfma_helpers.py` exists as a companion to `FmaX/vfma.f90.fypp`, loaded via `fypp -M FmaX -m vfma_helpers`.

- **fypp `@:name(...)` direct-call args are raw text, not Python values**: Unlike `#:call name(...) ... #:endcall`, which evaluates its argument list as a genuine Python expression (via `__getargvalues`), the `@:name(...)` direct-call form re-parses each argument as literal template text. An integer literal like `3` arrives as the string `"3"` (breaking anything that does `range(n)`), and a quoted string literal like `'mu(i)'` keeps its literal quote characters. Use `#:call`/`#:endcall` whenever a macro needs real Python types for its arguments — see `tutorial/vfma/tut_vfma.f90.fypp`'s use of `vfma`.

- **`vfma` codegen fixes (found via `tutorial/vfma/Makefile`'s `compile` target)**: three bugs had to be fixed before the generated code would compile. (1) The "Bulk Register Loading" step originally emitted `real(8) :: reg_x` declarations interleaved one-by-one with `reg_x = ...` executable assignments — invalid Fortran, since all specification statements must precede executable statements in a scoping unit (`NVFORTRAN-S-0070`); fixed by splitting into two passes over `loads` — declare all registers, then load all registers. (2) For a multi-dimensional slice operand like `a(i-1:i+1,j,k)`, `vfma_helpers.parse_arg` originally tracked only the sliced dimension's index and reloaded as `a(i-1)`, dropping the trailing `,j,k` and producing a rank-1 reference into a rank-3 array (`NVFORTRAN-W-0155`); fixed by splitting the parenthesized content on top-level commas and appending the trailing dimensions verbatim to every generated index (`clean_var_name` sanitizes the resulting commas into `_`). (3) The macro originally emitted the `fma()` intrinsic inside `attributes(device)` code, which nvfortran rejects as a host call (`NVFORTRAN-S-1253`) — same class of issue as `fltflt.f90`'s `__fmaf_rn`; fixed by emitting `__fma_rn(...)` (the `real(8)` analogue) instead, which requires `use cudadevice` in the calling subroutine (see `tutorial/vfma/tut_vfma.f90.fypp`).

- **`module vfma`, not `module fmax`**: the file/module was originally named `fmax`, but nvfortran's `libm` module (imported via `use libm`) exports a real device intrinsic `fmax(a,b)` (`accelmath.h`: `#pragma libm(fabs, floor, fma, fmin, fmax, fmod)`). Verified this doesn't error *today* — `module fmax`'s public interface is currently empty, and Fortran module names and use-associated entity names are different namespaces, so `use fmax; use libm; ... = fmax(a,b)` compiles fine as-is — but the module's own header comment plans real public functions (`! public :: float_float, double_double`), and naming one `fmax` inside `module fmax` would immediately collide with the `libm` intrinsic. Renamed to `vfma` (matching the macro's own name and `tutorial/vfma/`) before that could happen.
