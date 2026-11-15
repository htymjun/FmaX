!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! CUDA Fortran Implementation of fltflt
!
! Copyright (c) 2026, Jun Hatayama
! This is a Fortran translation/derivative work of the original C++ fltflt.h.
!
! Original C++ implementation:
! Copyright (c) 2026, NVIDIA Corporation
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! 3. Neither the name of the copyright holder nor the names of its
!    contributors may be used to endorse or promote products derived from
!    this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Float-float (FF) precision arithmetic — public API
!
! Represents a value as an unevaluated sum (hi + lo) of two real(4) numbers,
! giving ~48 mantissa bits — equivalent to real(8) precision.
! All procedures carry attributes(device) and are callable inside CUDA Fortran
! kernels (!$cuf kernel do / attributes(global)) without change.
!
! Companion C++ header: fltflt.h  (NVIDIA MatX, BSD-3)
! References:
!   Thall 2006 "Extended-Precision Floating-Point Numbers for GPU Computation"
!   Zhang & Aiken SC'25 "High-Performance Branch-Free Algorithms for
!     Extended-Precision Floating-Point Arithmetic" (FPAN addition)
!   Ogita, Rump, Oishi 2005 "Accurate Sum and Dot Product" (compensated dot)
!
! Critical EFT helpers to add to -Minline= in CMakeLists.txt for GPU perf:
!   name:fltflt_two_sum, name:fltflt_fast_two_sum, name:fltflt_two_prod_fma,
!   name:fltflt_add_same_sign, name:fltflt_fma, name:fltflt_fma_approx,
!   name:fltflt_abs, name:fltflt_sqrt, name:fltflt_sqrt_fast, name:fltflt_norm3d,
!   name:fltflt_dot2, name:fltflt_dot3,
!   name:fltflt_add3, name:fltflt_add4, name:fltflt_add5,
!   name:fltflt_square, name:fltflt_recip,
!   name:fltflt_shfl_down, name:fltflt_shfl_xor
module fltflt
  use cudafor
  use cudadevice
  implicit none
  private

  ! ================================================================
  ! The fltflt type: an unevaluated sum hi + lo where |lo| <= 0.5*ulp(hi).
  ! No default initializers — nvfortran crashes with default-init on device types.
  ! Always use fltflt_init() or assign %hi/%lo explicitly before first use.
  ! ================================================================
  type, public :: fltflt
    real(4) :: hi, lo
  end type fltflt

  ! Quiet NaN from its bit pattern (fltflt_fmod with b == 0). A runtime 0/0
  ! is foldable to 1.0 under finite-math assumptions; a constant is not.
  ! (A transfer() inside a device function instead crashes nvfortran 26.5's
  ! fort2 when the function is not pure.)
  real(4), parameter :: ff_qnan = transfer(int(z'7FC00000', 4), 1.0_4)

  ! ================================================================
  ! Constructors
  ! ================================================================
  interface fltflt_init
    module procedure init_from_r4
    module procedure init_from_r8
  end interface fltflt_init
  public :: fltflt_init

  ! ================================================================
  ! Arithmetic operators
  ! ================================================================
  interface operator(+)
    module procedure add_ff_ff, add_ff_r4, add_r4_ff, add_ff_r8, add_r8_ff
  end interface
  public :: operator(+)

  interface operator(-)
    module procedure neg_ff, sub_ff_ff, sub_ff_r4, sub_r4_ff, sub_ff_r8, sub_r8_ff
  end interface
  public :: operator(-)

  interface operator(*)
    module procedure mul_ff_ff, mul_ff_r4, mul_r4_ff, mul_ff_r8, mul_r8_ff
  end interface
  public :: operator(*)

  interface operator(/)
    module procedure div_ff_ff, div_ff_r4, div_r4_ff, div_ff_r8, div_r8_ff
  end interface
  public :: operator(/)

  ! ================================================================
  ! Comparison operators (match fltflt.h semantics)
  ! ================================================================
  interface operator(==)
    module procedure eq_ff_ff, eq_ff_r4, eq_r4_ff
  end interface
  public :: operator(==)

  interface operator(/=)
    module procedure ne_ff_ff, ne_ff_r4, ne_r4_ff
  end interface
  public :: operator(/=)

  interface operator(<)
    module procedure lt_ff_ff, lt_ff_r4, lt_r4_ff
  end interface
  public :: operator(<)

  interface operator(>)
    module procedure gt_ff_ff, gt_ff_r4, gt_r4_ff
  end interface
  public :: operator(>)

  interface operator(<=)
    module procedure le_ff_ff, le_ff_r4, le_r4_ff
  end interface
  public :: operator(<=)

  interface operator(>=)
    module procedure ge_ff_ff, ge_ff_r4, ge_r4_ff
  end interface
  public :: operator(>=)

  ! ================================================================
  ! Named function interfaces
  ! ================================================================

  ! fltflt_add_same_sign: 11 flops; valid only when a and b share the same sign.
  interface fltflt_add_same_sign
    module procedure add_same_sign_ff_ff, add_same_sign_ff_r4, add_same_sign_r4_ff
  end interface fltflt_add_same_sign
  public :: fltflt_add_same_sign

  ! fltflt_fma: a*b + c — more accurate and efficient than fltflt_add(fltflt_mul(a,b),c).
  interface fltflt_fma
    module procedure fma_ff_ff_ff, fma_ff_ff_r4, fma_r4_ff_ff, fma_ff_r4_ff, &
                     fma_ff_r4_r4, fma_r4_ff_r4, fma_r4_r4_ff
  end interface fltflt_fma
  public :: fltflt_fma

  ! fltflt_fma_approx: a*b + c omitting the a%lo*b%lo term (~1 ULP less precise,
  ! but faster in throughput-bound kernels).
  interface fltflt_fma_approx
    module procedure fma_approx_ff_ff_ff, fma_approx_ff_ff_r4, &
                     fma_approx_r4_ff_ff, fma_approx_ff_r4_ff, &
                     fma_approx_ff_r4_r4, fma_approx_r4_ff_r4, &
                     fma_approx_r4_r4_ff
  end interface fltflt_fma_approx
  public :: fltflt_fma_approx

  ! fltflt_fmod: floating-point remainder a - trunc(a/b)*b (2 overloads: ff/ff and ff/r4).
  interface fltflt_fmod
    module procedure fmod_ff_ff, fmod_ff_r4
  end interface fltflt_fmod
  public :: fltflt_fmod

  public :: fltflt_shfl_down, fltflt_shfl_xor
  public :: fltflt_add3, fltflt_add4, fltflt_add5
  public :: fltflt_square, fltflt_recip
  ! fltflt_dot2/3/4: compensated dot products. 3 overloads each (r4 / r8 / fltflt).
  interface fltflt_dot2
    module procedure dot2_r4, dot2_r8, dot2_ff
  end interface fltflt_dot2
  public :: fltflt_dot2

  interface fltflt_dot3
    module procedure dot3_r4, dot3_r8, dot3_ff
  end interface fltflt_dot3
  public :: fltflt_dot3

  interface fltflt_dot4
    module procedure dot4_r4, dot4_r8, dot4_ff
  end interface fltflt_dot4
  public :: fltflt_dot4
  public :: fltflt_abs, fltflt_sqrt, fltflt_sqrt_fast
  public :: fltflt_norm3d
  public :: fltflt_round_to_nearest, fltflt_round_toward_zero, fltflt_floor
  ! fltflt_min / fltflt_max: branchless on GPU (predicated select). 3 overloads each.
  interface fltflt_min
    module procedure fltflt_min_ff_ff, fltflt_min_ff_r4, fltflt_min_r4_ff
  end interface fltflt_min
  public :: fltflt_min

  interface fltflt_max
    module procedure fltflt_max_ff_ff, fltflt_max_ff_r4, fltflt_max_r4_ff
  end interface fltflt_max
  public :: fltflt_max

  public :: fltflt_clamp, fltflt_ceil, fltflt_hypot
  public :: fltflt_cross3d
  public :: fltflt_pow_int, fltflt_sign, fltflt_lerp
  public :: fltflt_warp_reduce_sum

contains

  ! ================================================================
  ! EFT Backend (private) — names match fltflt.h counterparts
  ! ================================================================

  ! Every floating-point step of an error-free transformation (and of every
  ! renormalisation built on one) goes through __fadd_rn / __fmul_rn /
  ! __fmaf_rn, exactly as fltflt.h does with detail::fadd_rn/fsub_rn/fmul_rn.
  ! These intrinsics are never contracted into an FMA. With plain `+ - *`,
  ! `-fast` (FMA contraction on) is free to fuse a product into a following
  ! add once the routines are inlined -- e.g. r%hi = a*b of two_prod into the
  ! adds of a following fast_two_sum -- and the transformation is no longer
  ! exact (measured: 7e-8 instead of 1e-16 relative error). test/ builds the
  ! suite with and without FMA contraction and requires bit-identical results.
  ! nvfortran has no __fsub_rn: a - b is written __fadd_rn(a, -b) (negation
  ! is exact, so this is the same operation).

  ! fltflt_two_sum: Knuth 1969 exact split, 6 flops, no precondition.
  attributes(device) function fltflt_two_sum(a, b) result(r)
    real(4), intent(in) :: a, b
    type(fltflt) :: r
    real(4) :: v
    r%hi = __fadd_rn(a, b)
    v    = __fadd_rn(r%hi, -a)
    r%lo = __fadd_rn(__fadd_rn(a, -__fadd_rn(r%hi, -v)), __fadd_rn(b, -v))
  end function fltflt_two_sum

  ! fltflt_fast_two_sum: Dekker 1971, 3 flops, requires |a| >= |b|.
  attributes(device) function fltflt_fast_two_sum(a, b) result(r)
    real(4), intent(in) :: a, b
    type(fltflt) :: r
    r%hi = __fadd_rn(a, b)
    r%lo = __fadd_rn(b, -__fadd_rn(r%hi, -a))
  end function fltflt_fast_two_sum

  ! fltflt_two_prod_fma: exact product via FMA. 1 MUL + 1 FMAF.F32.
  ! __fmul_rn keeps hi = RN(a*b) a separate rounding (a plain a*b could be
  ! fused into a later add); __fmaf_rn computes a*b - hi exactly.
  attributes(device) function fltflt_two_prod_fma(a, b) result(r)
    real(4), intent(in) :: a, b
    type(fltflt) :: r
    r%hi = __fmul_rn(a, b)
    r%lo = __fmaf_rn(a, b, -r%hi)
  end function fltflt_two_prod_fma

  ! ================================================================
  ! Constructors
  ! ================================================================

  pure attributes(device, host) function init_from_r4(a) result(r)
    real(4), intent(in) :: a
    type(fltflt) :: r
    r%hi = a
    r%lo = 0.0
  end function init_from_r4

  ! Splits a double into two non-overlapping single-precision halves:
  ! hi = RN(a), lo = RN(a - hi), then one FastTwoSum so that fl(hi + lo) == hi
  ! holds even when lo rounds to exactly half an ulp of hi (otherwise the pair
  ! is not canonical and the lexicographic comparisons misjudge it).
  ! |a| > huge(1.0_4), Inf and NaN give (RN(a), 0) -- never (Inf, -Inf).
  ! Same contract as fltflt.h's fltflt(double) host path; the device path of
  ! fltflt.h extracts the halves from the bit pattern instead (no FP64
  ! instruction), which can differ from this one in the last bit of lo.
  ! Plain arithmetic is safe here (no product to contract) and keeps the
  ! routine callable on the host.
  pure attributes(device, host) function init_from_r8(a) result(r)
    real(8), intent(in) :: a
    type(fltflt) :: r
    real(4) :: s
    r%hi = real(a, 4)
    if (abs(r%hi) <= huge(1.0_4)) then
      r%lo = real(a - real(r%hi, 8), 4)
      s    = r%hi + r%lo
      r%lo = r%lo - (s - r%hi)
      r%hi = s
    else
      r%lo = 0.0
    end if
  end function init_from_r8

  ! ================================================================
  ! Negation
  ! ================================================================

  pure attributes(device) function neg_ff(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    c%hi = -a%hi
    c%lo = -a%lo
  end function neg_ff

  ! ================================================================
  ! Addition — FPAN algorithm (Zhang & Aiken SC'25, Figure 2)
  !
  ! ff+ff: critical-path depth 10 ops vs 13 for Thall, same 20 flops.
  ! Two TwoSums on hi paths run in parallel; their error words join
  ! the lo accumulation before the final FastTwoSum.
  ! ================================================================

  attributes(device) function add_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, s, t, q
    real(4) :: st_lo
    s     = fltflt_two_sum(a%hi, b%hi)
    t     = fltflt_two_sum(a%lo, b%lo)
    q     = fltflt_fast_two_sum(s%hi, t%hi)
    st_lo = __fadd_rn(s%lo, t%lo)
    c     = fltflt_fast_two_sum(q%hi, __fadd_rn(st_lo, q%lo))
  end function add_ff_ff

  ! ff + scalar: b%lo = 0 inlined, saves one TwoSum. ~9 flops.
  attributes(device) function add_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, s
    s    = fltflt_two_sum(a%hi, b)
    s%lo = __fadd_rn(s%lo, a%lo)
    c    = fltflt_fast_two_sum(s%hi, s%lo)
  end function add_ff_r4

  attributes(device) function add_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r4(b, a)
  end function add_r4_ff

  attributes(device) function add_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c
    c = add_ff_ff(a, init_from_r8(b))
  end function add_ff_r8

  attributes(device) function add_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r8(b, a)
  end function add_r8_ff

  ! ================================================================
  ! Subtraction
  ! ================================================================

  attributes(device) function sub_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c
    c = add_ff_ff(a, neg_ff(b))
  end function sub_ff_ff

  attributes(device) function sub_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r4(a, -b)
  end function sub_ff_r4

  attributes(device) function sub_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r4(neg_ff(b), a)
  end function sub_r4_ff

  attributes(device) function sub_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r8(a, -b)
  end function sub_ff_r8

  attributes(device) function sub_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r8(neg_ff(b), a)
  end function sub_r8_ff

  ! ================================================================
  ! Multiplication (Thall / Hida: TwoProdFMA + cross terms)
  ! ================================================================

  ! ff*ff: TwoProdFMA + cross terms. ~8 flops + 1 FMA.
  attributes(device) function mul_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, p
    p    = fltflt_two_prod_fma(a%hi, b%hi)
    p%lo = __fmaf_rn(a%hi, b%lo, p%lo)
    p%lo = __fmaf_rn(a%lo, b%hi, p%lo)
    c    = fltflt_fast_two_sum(p%hi, p%lo)
  end function mul_ff_ff

  ! ff*scalar: b%lo = 0 inlined. ~5 flops + 1 FMA.
  attributes(device) function mul_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, p
    p    = fltflt_two_prod_fma(a%hi, b)
    p%lo = __fmaf_rn(a%lo, b, p%lo)
    c    = fltflt_fast_two_sum(p%hi, p%lo)
  end function mul_ff_r4

  attributes(device) function mul_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = mul_ff_r4(b, a)
  end function mul_r4_ff

  attributes(device) function mul_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c
    c = mul_ff_ff(a, init_from_r8(b))
  end function mul_ff_r8

  attributes(device) function mul_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = mul_ff_r8(b, a)
  end function mul_r8_ff

  ! ================================================================
  ! Division: Karp's algorithm (Thall 2006, Alg. 6), as fltflt.h's fltflt_div
  !
  ! xn = 1/b%hi (IEEE, __fdiv_rn)      reciprocal of the leading part
  ! yn = a%hi*xn                       first quotient approximation
  ! diff = a - yn*b                    exact-product residual via fltflt_fma
  ! q = TwoProd(xn, diff%hi) + yn      Newton correction, one renormalisation
  !
  ! b == 0 gives xn = 0 and therefore a/0 = 0 (not Inf/NaN) -- fltflt.h's
  ! semantics, kept so that the two libraries agree bit for bit.
  ! ================================================================

  attributes(device) function div_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, diff, prod
    real(4) :: xn, yn
    xn   = 0.0
    if (b%hi /= 0.0) xn = __fdiv_rn(1.0, b%hi)
    yn   = __fmul_rn(a%hi, xn)
    diff = fma_r4_ff_ff(-yn, b, a)
    prod = fltflt_two_prod_fma(xn, diff%hi)
    c    = add_ff_r4(prod, yn)
  end function div_ff_ff

  ! ff/scalar: b%lo = 0.
  attributes(device) function div_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, diff, prod
    real(4) :: xn, yn
    xn   = 0.0
    if (b /= 0.0) xn = __fdiv_rn(1.0, b)
    yn   = __fmul_rn(a%hi, xn)
    diff = fma_r4_r4_ff(-yn, b, a)
    prod = fltflt_two_prod_fma(xn, diff%hi)
    c    = add_ff_r4(prod, yn)
  end function div_ff_r4

  ! scalar/ff: a%lo = 0.
  attributes(device) function div_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c, diff, prod
    real(4) :: xn, yn
    xn   = 0.0
    if (b%hi /= 0.0) xn = __fdiv_rn(1.0, b%hi)
    yn   = __fmul_rn(a, xn)
    diff = fma_r4_ff_r4(-yn, b, a)
    prod = fltflt_two_prod_fma(xn, diff%hi)
    c    = add_ff_r4(prod, yn)
  end function div_r4_ff

  attributes(device) function div_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c
    c = div_ff_ff(a, init_from_r8(b))
  end function div_ff_r8

  attributes(device) function div_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = div_ff_ff(init_from_r8(a), b)
  end function div_r8_ff

  ! ================================================================
  ! Warp shuffle — NOT pure (hardware warp communication)
  ! Both components shuffled independently.
  ! Uses __shfl_down/__shfl_xor (nvfortran 25.x mask-implicit API).
  ! ================================================================

  attributes(device) function fltflt_shfl_down(val, delta) result(r)
    type(fltflt), intent(in) :: val
    integer(4),   intent(in), value :: delta
    type(fltflt) :: r
    r%hi = __shfl_down(val%hi, delta)
    r%lo = __shfl_down(val%lo, delta)
  end function fltflt_shfl_down

  attributes(device) function fltflt_shfl_xor(val, lane_mask) result(r)
    type(fltflt), intent(in) :: val
    integer(4),   intent(in), value :: lane_mask
    type(fltflt) :: r
    r%hi = __shfl_xor(val%hi, lane_mask)
    r%lo = __shfl_xor(val%lo, lane_mask)
  end function fltflt_shfl_xor

  ! ================================================================
  ! Comparison operators (match fltflt.h semantics exactly)
  ! For normalized fltflt, sign(value) == sign(hi).
  ! ================================================================

  pure attributes(device) function eq_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    logical :: r
    r = a%hi == b%hi .and. a%lo == b%lo
  end function eq_ff_ff

  pure attributes(device) function eq_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    logical :: r
    r = a%hi == b .and. a%lo == 0.0
  end function eq_ff_r4

  pure attributes(device) function eq_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    logical :: r
    r = eq_ff_r4(b, a)
  end function eq_r4_ff

  ! ne_*: avoid .not. — in nvfortran device code .true.=+1, so bitwise
  ! .not.(+1) = -2 (non-zero = "true"), making .not. .true. still "true".
  pure attributes(device) function ne_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    logical :: r
    r = a%hi /= b%hi .or. a%lo /= b%lo
  end function ne_ff_ff

  pure attributes(device) function ne_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    logical :: r
    r = a%hi /= b .or. a%lo /= 0.0
  end function ne_ff_r4

  pure attributes(device) function ne_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    logical :: r
    r = a /= b%hi .or. b%lo /= 0.0
  end function ne_r4_ff

  pure attributes(device) function lt_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    logical :: r
    r = a%hi < b%hi .or. (a%hi == b%hi .and. a%lo < b%lo)
  end function lt_ff_ff

  pure attributes(device) function lt_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    logical :: r
    r = a%hi < b .or. (a%hi == b .and. a%lo < 0.0)
  end function lt_ff_r4

  pure attributes(device) function lt_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    logical :: r
    r = a < b%hi .or. (a == b%hi .and. b%lo > 0.0)
  end function lt_r4_ff

  pure attributes(device) function gt_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    logical :: r
    r = a%hi > b%hi .or. (a%hi == b%hi .and. a%lo > b%lo)
  end function gt_ff_ff

  pure attributes(device) function gt_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    logical :: r
    r = a%hi > b .or. (a%hi == b .and. a%lo > 0.0)
  end function gt_ff_r4

  pure attributes(device) function gt_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    logical :: r
    r = a > b%hi .or. (a == b%hi .and. b%lo < 0.0)
  end function gt_r4_ff

  pure attributes(device) function le_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    logical :: r
    r = lt_ff_ff(a, b) .or. eq_ff_ff(a, b)
  end function le_ff_ff

  pure attributes(device) function le_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    logical :: r
    r = lt_ff_r4(a, b) .or. eq_ff_r4(a, b)
  end function le_ff_r4

  pure attributes(device) function le_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    logical :: r
    r = lt_r4_ff(a, b) .or. eq_r4_ff(a, b)
  end function le_r4_ff

  pure attributes(device) function ge_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    logical :: r
    r = gt_ff_ff(a, b) .or. eq_ff_ff(a, b)
  end function ge_ff_ff

  pure attributes(device) function ge_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    logical :: r
    r = gt_ff_r4(a, b) .or. eq_ff_r4(a, b)
  end function ge_ff_r4

  pure attributes(device) function ge_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    logical :: r
    r = gt_r4_ff(a, b) .or. eq_r4_ff(a, b)
  end function ge_r4_ff

  ! ================================================================
  ! fltflt_add_same_sign: 11 flops; only valid when a and b share
  ! the same sign (Handbook of FP Arithmetic, Muller et al., Alg 14.1).
  ! Saves 9 flops vs fltflt_add_ff_ff for sign-guaranteed inputs.
  ! ================================================================

  attributes(device) function add_same_sign_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, r
    real(4) :: s
    r = fltflt_two_sum(a%hi, b%hi)
    s = __fadd_rn(__fadd_rn(r%lo, b%lo), a%lo)
    c = fltflt_fast_two_sum(r%hi, s)
  end function add_same_sign_ff_ff

  ! ff + r4 variant (b%lo = 0).
  attributes(device) function add_same_sign_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, r
    real(4) :: s
    r = fltflt_two_sum(a%hi, b)
    s = __fadd_rn(r%lo, a%lo)
    c = fltflt_fast_two_sum(r%hi, s)
  end function add_same_sign_ff_r4

  ! r4 + ff variant: commutative, delegates to ff+r4.
  attributes(device) function add_same_sign_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_same_sign_ff_r4(b, a)
  end function add_same_sign_r4_ff

  ! ================================================================
  ! fltflt_fma: a*b + c (7 overloads)
  !
  ! More accurate and efficient than fltflt_add(fltflt_mul(a,b), c)
  ! because it skips one normalization step and preserves the exact
  ! product from TwoProdFMA all the way into the accumulation.
  ! ================================================================

  ! Full ff*ff+ff.
  attributes(device) function fma_ff_ff_ff(a, b, c) result(r)
    type(fltflt), intent(in) :: a, b, c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a%hi, b%hi)
    p%lo = __fmaf_rn(a%hi, b%lo, p%lo)
    p%lo = __fmaf_rn(a%lo, b%hi, p%lo)
    s    = fltflt_two_sum(p%hi, c%hi)
    s%lo = __fadd_rn(s%lo, p%lo)
    s    = fltflt_fast_two_sum(s%hi, s%lo)
    s%lo = __fadd_rn(s%lo, c%lo)
    s%lo = __fmaf_rn(a%lo, b%lo, s%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_ff_ff_ff

  ! ff*ff + r4: c%lo = 0. Keeps the a%lo*b%lo term, which carries the
  ! answer when a*b cancels against c (only fma_approx drops it).
  attributes(device) function fma_ff_ff_r4(a, b, c) result(r)
    type(fltflt), intent(in) :: a, b
    real(4),      intent(in) :: c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a%hi, b%hi)
    p%lo = __fmaf_rn(a%hi, b%lo, p%lo)
    p%lo = __fmaf_rn(a%lo, b%hi, p%lo)
    s    = fltflt_two_sum(p%hi, c)
    s%lo = __fadd_rn(s%lo, p%lo)
    s%lo = __fmaf_rn(a%lo, b%lo, s%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_ff_ff_r4

  ! r4*ff + ff: a is scalar, no a%lo cross terms.
  attributes(device) function fma_r4_ff_ff(a, b, c) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b, c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a, b%hi)
    p%lo = __fmaf_rn(a, b%lo, p%lo)
    s    = fltflt_two_sum(p%hi, c%hi)
    s%lo = __fadd_rn(s%lo, p%lo)
    s    = fltflt_fast_two_sum(s%hi, s%lo)
    s%lo = __fadd_rn(s%lo, c%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_r4_ff_ff

  ! ff*r4 + ff: b is scalar, commutative with fma_r4_ff_ff.
  attributes(device) function fma_ff_r4_ff(a, b, c) result(r)
    type(fltflt), intent(in) :: a, c
    real(4),      intent(in) :: b
    type(fltflt) :: r
    r = fma_r4_ff_ff(b, a, c)
  end function fma_ff_r4_ff

  ! ff*r4 + r4: both b and c are scalars.
  attributes(device) function fma_ff_r4_r4(a, b, c) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b, c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a%hi, b)
    p%lo = __fmaf_rn(a%lo, b, p%lo)
    s    = fltflt_two_sum(p%hi, c)
    s%lo = __fadd_rn(s%lo, p%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_ff_r4_r4

  ! r4*ff + r4: commutative with fma_ff_r4_r4.
  attributes(device) function fma_r4_ff_r4(a, b, c) result(r)
    real(4),      intent(in) :: a, c
    type(fltflt), intent(in) :: b
    type(fltflt) :: r
    r = fma_ff_r4_r4(b, a, c)
  end function fma_r4_ff_r4

  ! r4*r4 + ff: both a and b are scalars.
  attributes(device) function fma_r4_r4_ff(a, b, c) result(r)
    real(4),      intent(in) :: a, b
    type(fltflt), intent(in) :: c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a, b)
    s    = fltflt_two_sum(p%hi, c%hi)
    s%lo = __fadd_rn(s%lo, p%lo)
    s    = fltflt_fast_two_sum(s%hi, s%lo)
    s%lo = __fadd_rn(s%lo, c%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_r4_r4_ff

  ! ================================================================
  ! fltflt_fma_approx: a*b + c omitting a%lo*b%lo term (~1 ULP less
  ! accurate than fltflt_fma, but saves one FMA in hot loops).
  ! Only the two ff*ff overloads differ from fltflt_fma; the rest
  ! are identical and delegate to fltflt_fma implementations.
  ! ================================================================

  ! ff*ff + ff (approx): no a%lo*b%lo correction.
  attributes(device) function fma_approx_ff_ff_ff(a, b, c) result(r)
    type(fltflt), intent(in) :: a, b, c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a%hi, b%hi)
    p%lo = __fmaf_rn(a%hi, b%lo, p%lo)
    p%lo = __fmaf_rn(a%lo, b%hi, p%lo)
    s    = fltflt_two_sum(p%hi, c%hi)
    s%lo = __fadd_rn(s%lo, p%lo)
    s    = fltflt_fast_two_sum(s%hi, s%lo)
    s%lo = __fadd_rn(s%lo, c%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_approx_ff_ff_ff

  ! ff*ff + r4 (approx): no a%lo*b%lo correction.
  attributes(device) function fma_approx_ff_ff_r4(a, b, c) result(r)
    type(fltflt), intent(in) :: a, b
    real(4),      intent(in) :: c
    type(fltflt) :: r, p, s
    p    = fltflt_two_prod_fma(a%hi, b%hi)
    p%lo = __fmaf_rn(a%hi, b%lo, p%lo)
    p%lo = __fmaf_rn(a%lo, b%hi, p%lo)
    s    = fltflt_two_sum(p%hi, c)
    s%lo = __fadd_rn(s%lo, p%lo)
    r    = fltflt_fast_two_sum(s%hi, s%lo)
  end function fma_approx_ff_ff_r4

  ! Remaining overloads: identical to fltflt_fma (scalar args have no lo term).
  attributes(device) function fma_approx_r4_ff_ff(a, b, c) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b, c
    type(fltflt) :: r
    r = fma_r4_ff_ff(a, b, c)
  end function fma_approx_r4_ff_ff

  attributes(device) function fma_approx_ff_r4_ff(a, b, c) result(r)
    type(fltflt), intent(in) :: a, c
    real(4),      intent(in) :: b
    type(fltflt) :: r
    r = fma_ff_r4_ff(a, b, c)
  end function fma_approx_ff_r4_ff

  attributes(device) function fma_approx_ff_r4_r4(a, b, c) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b, c
    type(fltflt) :: r
    r = fma_ff_r4_r4(a, b, c)
  end function fma_approx_ff_r4_r4

  attributes(device) function fma_approx_r4_ff_r4(a, b, c) result(r)
    real(4),      intent(in) :: a, c
    type(fltflt), intent(in) :: b
    type(fltflt) :: r
    r = fma_r4_ff_r4(a, b, c)
  end function fma_approx_r4_ff_r4

  attributes(device) function fma_approx_r4_r4_ff(a, b, c) result(r)
    real(4),      intent(in) :: a, b
    type(fltflt), intent(in) :: c
    type(fltflt) :: r
    r = fma_r4_r4_ff(a, b, c)
  end function fma_approx_r4_r4_ff

  ! ================================================================
  ! Multi-operand addition (Fortran-specific; no C++ equivalent)
  ! ================================================================

  ! fltflt_add3: 2 TwoSums + lo accumulation. ~19 flops vs 28 sequential.
  attributes(device) function fltflt_add3(a, b, c) result(r)
    type(fltflt), intent(in) :: a, b, c
    type(fltflt) :: r, s1, s2
    real(4) :: lo
    s1 = fltflt_two_sum(a%hi, b%hi)
    s2 = fltflt_two_sum(s1%hi, c%hi)
    lo = __fadd_rn(__fadd_rn(__fadd_rn(a%lo, b%lo), c%lo), __fadd_rn(s1%lo, s2%lo))
    r  = fltflt_fast_two_sum(s2%hi, lo)
  end function fltflt_add3

  ! fltflt_add4: balanced tree of 3 TwoSums. ~27 flops vs 42 sequential.
  attributes(device) function fltflt_add4(a, b, c, d) result(r)
    type(fltflt), intent(in) :: a, b, c, d
    type(fltflt) :: r, s1, s2, s3
    real(4) :: lo
    s1 = fltflt_two_sum(a%hi, b%hi)
    s2 = fltflt_two_sum(c%hi, d%hi)
    s3 = fltflt_two_sum(s1%hi, s2%hi)
    lo = __fadd_rn(__fadd_rn(__fadd_rn(a%lo, b%lo), __fadd_rn(c%lo, d%lo)), &
                   __fadd_rn(__fadd_rn(s1%lo, s2%lo), s3%lo))
    r  = fltflt_fast_two_sum(s3%hi, lo)
  end function fltflt_add4

  ! fltflt_add5: 4 TwoSums (balanced + 1). ~35 flops vs 56 sequential.
  attributes(device) function fltflt_add5(a, b, c, d, e) result(r)
    type(fltflt), intent(in) :: a, b, c, d, e
    type(fltflt) :: r, s1, s2, s3, s4
    real(4) :: lo
    s1 = fltflt_two_sum(a%hi, b%hi)
    s2 = fltflt_two_sum(c%hi, d%hi)
    s3 = fltflt_two_sum(s1%hi, s2%hi)
    s4 = fltflt_two_sum(s3%hi, e%hi)
    lo = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(a%lo, b%lo), __fadd_rn(c%lo, d%lo)), e%lo), &
                   __fadd_rn(__fadd_rn(__fadd_rn(s1%lo, s2%lo), s3%lo), s4%lo))
    r  = fltflt_fast_two_sum(s4%hi, lo)
  end function fltflt_add5

  ! ================================================================
  ! Squaring and reciprocal (Fortran-specific; no C++ equivalent)
  ! ================================================================

  ! fltflt_square: symmetric cross-term; saves one FMA vs mul_ff_ff. ~6 ops + 2 FMAs.
  attributes(device) function fltflt_square(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c, p
    p    = fltflt_two_prod_fma(a%hi, a%hi)
    p%lo = __fmaf_rn(__fadd_rn(a%hi, a%hi), a%lo, p%lo)
    c    = fltflt_fast_two_sum(p%hi, p%lo)
  end function fltflt_square

  ! fltflt_recip: 1/a, the division algorithm with numerator 1 (a/0 = 0).
  attributes(device) function fltflt_recip(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    c = div_r4_ff(1.0, a)
  end function fltflt_recip

  ! ================================================================
  ! Compensated dot products (Ogita-Rump-Oishi 2005)
  ! All inputs are real(4).  TwoProdFMA gives the exact product;
  ! accumulating all error words before the final fltflt_fast_two_sum
  ! yields a result faithful to the true dot product.
  ! ================================================================

  ! ---- dot2: a*b + c*d ----

  ! dot2_r4: real(4) inputs. ~8 flops + 2 FMAs.
  attributes(device) function dot2_r4(a, b, c, d) result(r)
    real(4), intent(in) :: a, b, c, d
    type(fltflt) :: r, p1, p2, s
    p1 = fltflt_two_prod_fma(a, b)
    p2 = fltflt_two_prod_fma(c, d)
    s  = fltflt_two_sum(p1%hi, p2%hi)
    r  = fltflt_fast_two_sum(s%hi, __fadd_rn(__fadd_rn(s%lo, p1%lo), p2%lo))
  end function dot2_r4

  ! dot2_r8: real(8) inputs — each converted to fltflt, then delegates to dot2_ff.
  attributes(device) function dot2_r8(a, b, c, d) result(r)
    real(8), intent(in) :: a, b, c, d
    type(fltflt) :: r
    r = dot2_ff(init_from_r8(a), init_from_r8(b), init_from_r8(c), init_from_r8(d))
  end function dot2_r8

  ! dot2_ff: fltflt inputs. TwoProd(hi,hi) + 2 cross FMAs per product. ~12 flops + 6 FMAs.
  ! a%lo*b%lo is omitted — beyond ~48-bit budget, same convention as mul_ff_ff.
  attributes(device) function dot2_ff(a, b, c, d) result(r)
    type(fltflt), intent(in) :: a, b, c, d
    type(fltflt) :: r, p1, p2, s
    p1    = fltflt_two_prod_fma(a%hi, b%hi)
    p2    = fltflt_two_prod_fma(c%hi, d%hi)
    p1%lo = __fmaf_rn(a%hi, b%lo, p1%lo)
    p1%lo = __fmaf_rn(a%lo, b%hi, p1%lo)
    p2%lo = __fmaf_rn(c%hi, d%lo, p2%lo)
    p2%lo = __fmaf_rn(c%lo, d%hi, p2%lo)
    s     = fltflt_two_sum(p1%hi, p2%hi)
    r     = fltflt_fast_two_sum(s%hi, __fadd_rn(__fadd_rn(s%lo, p1%lo), p2%lo))
  end function dot2_ff

  ! ---- dot3: a*b + c*d + e*f ----

  ! dot3_r4: real(4) inputs. ~13 flops + 3 FMAs.
  attributes(device) function dot3_r4(a, b, c, d, e, f) result(r)
    real(4), intent(in) :: a, b, c, d, e, f
    type(fltflt) :: r, p1, p2, p3, s1, s2
    real(4) :: lo
    p1 = fltflt_two_prod_fma(a, b)
    p2 = fltflt_two_prod_fma(c, d)
    p3 = fltflt_two_prod_fma(e, f)
    s1 = fltflt_two_sum(p1%hi, p2%hi)
    s2 = fltflt_two_sum(s1%hi, p3%hi)
    lo = __fadd_rn(__fadd_rn(s1%lo, s2%lo), __fadd_rn(__fadd_rn(p1%lo, p2%lo), p3%lo))
    r  = fltflt_fast_two_sum(s2%hi, lo)
  end function dot3_r4

  ! dot3_r8: real(8) inputs.
  attributes(device) function dot3_r8(a, b, c, d, e, f) result(r)
    real(8), intent(in) :: a, b, c, d, e, f
    type(fltflt) :: r
    r = dot3_ff(init_from_r8(a), init_from_r8(b), init_from_r8(c), &
                init_from_r8(d), init_from_r8(e), init_from_r8(f))
  end function dot3_r8

  ! dot3_ff: fltflt inputs. ~19 flops + 9 FMAs.
  attributes(device) function dot3_ff(a, b, c, d, e, f) result(r)
    type(fltflt), intent(in) :: a, b, c, d, e, f
    type(fltflt) :: r, p1, p2, p3, s1, s2
    real(4) :: lo
    p1    = fltflt_two_prod_fma(a%hi, b%hi)
    p2    = fltflt_two_prod_fma(c%hi, d%hi)
    p3    = fltflt_two_prod_fma(e%hi, f%hi)
    p1%lo = __fmaf_rn(a%hi, b%lo, p1%lo)
    p1%lo = __fmaf_rn(a%lo, b%hi, p1%lo)
    p2%lo = __fmaf_rn(c%hi, d%lo, p2%lo)
    p2%lo = __fmaf_rn(c%lo, d%hi, p2%lo)
    p3%lo = __fmaf_rn(e%hi, f%lo, p3%lo)
    p3%lo = __fmaf_rn(e%lo, f%hi, p3%lo)
    s1    = fltflt_two_sum(p1%hi, p2%hi)
    s2    = fltflt_two_sum(s1%hi, p3%hi)
    lo    = __fadd_rn(__fadd_rn(s1%lo, s2%lo), __fadd_rn(__fadd_rn(p1%lo, p2%lo), p3%lo))
    r     = fltflt_fast_two_sum(s2%hi, lo)
  end function dot3_ff

  ! ---- dot4: a*b + c*d + e*f + g*h ----

  ! dot4_r4: real(4) inputs. ~23 flops + 4 FMAs.
  attributes(device) function dot4_r4(a, b, c, d, e, f, g, h) result(r)
    real(4), intent(in) :: a, b, c, d, e, f, g, h
    type(fltflt) :: r, p1, p2, p3, p4, s1, s2, s3
    real(4) :: lo
    p1 = fltflt_two_prod_fma(a, b)
    p2 = fltflt_two_prod_fma(c, d)
    p3 = fltflt_two_prod_fma(e, f)
    p4 = fltflt_two_prod_fma(g, h)
    s1 = fltflt_two_sum(p1%hi, p2%hi)
    s2 = fltflt_two_sum(s1%hi, p3%hi)
    s3 = fltflt_two_sum(s2%hi, p4%hi)
    lo = __fadd_rn(__fadd_rn(__fadd_rn(s1%lo, s2%lo), s3%lo), &
               __fadd_rn(__fadd_rn(__fadd_rn(p1%lo, p2%lo), p3%lo), p4%lo))
    r  = fltflt_fast_two_sum(s3%hi, lo)
  end function dot4_r4

  ! dot4_r8: real(8) inputs.
  attributes(device) function dot4_r8(a, b, c, d, e, f, g, h) result(r)
    real(8), intent(in) :: a, b, c, d, e, f, g, h
    type(fltflt) :: r
    r = dot4_ff(init_from_r8(a), init_from_r8(b), init_from_r8(c), init_from_r8(d), &
                init_from_r8(e), init_from_r8(f), init_from_r8(g), init_from_r8(h))
  end function dot4_r8

  ! dot4_ff: fltflt inputs. ~26 flops + 12 FMAs.
  attributes(device) function dot4_ff(a, b, c, d, e, f, g, h) result(r)
    type(fltflt), intent(in) :: a, b, c, d, e, f, g, h
    type(fltflt) :: r, p1, p2, p3, p4, s1, s2, s3
    real(4) :: lo
    p1    = fltflt_two_prod_fma(a%hi, b%hi)
    p2    = fltflt_two_prod_fma(c%hi, d%hi)
    p3    = fltflt_two_prod_fma(e%hi, f%hi)
    p4    = fltflt_two_prod_fma(g%hi, h%hi)
    p1%lo = __fmaf_rn(a%hi, b%lo, p1%lo)
    p1%lo = __fmaf_rn(a%lo, b%hi, p1%lo)
    p2%lo = __fmaf_rn(c%hi, d%lo, p2%lo)
    p2%lo = __fmaf_rn(c%lo, d%hi, p2%lo)
    p3%lo = __fmaf_rn(e%hi, f%lo, p3%lo)
    p3%lo = __fmaf_rn(e%lo, f%hi, p3%lo)
    p4%lo = __fmaf_rn(g%hi, h%lo, p4%lo)
    p4%lo = __fmaf_rn(g%lo, h%hi, p4%lo)
    s1    = fltflt_two_sum(p1%hi, p2%hi)
    s2    = fltflt_two_sum(s1%hi, p3%hi)
    s3    = fltflt_two_sum(s2%hi, p4%hi)
    lo    = __fadd_rn(__fadd_rn(__fadd_rn(s1%lo, s2%lo), s3%lo), &
               __fadd_rn(__fadd_rn(__fadd_rn(p1%lo, p2%lo), p3%lo), p4%lo))
    r     = fltflt_fast_two_sum(s3%hi, lo)
  end function dot4_ff

  ! ================================================================
  ! Absolute value
  ! ================================================================

  ! fltflt_abs: negate both components iff hi < 0 (fltflt.h semantics;
  ! predicated select, no branch). Assumes normalized input (sign(value) ==
  ! sign(hi)); -0.0 is returned unchanged.
  pure attributes(device) function fltflt_abs(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    c%hi = merge(-a%hi, a%hi, a%hi < 0.0)
    c%lo = merge(-a%lo, a%lo, a%hi < 0.0)
  end function fltflt_abs

  ! ================================================================
  ! Square root (fltflt.h algorithms)
  ! xn = rsqrt(a%hi) (MUFU.RSQ, <= 2 ulp, as rsqrtf in fltflt.h), yn = a%hi*xn
  ! fltflt_sqrt:      Karp (Thall 2006, Alg. 7): full fltflt residual a - yn^2
  ! fltflt_sqrt_fast: residual a%hi - yn^2 by one FMA, + a%lo (~1/5 the cost,
  !                   ~45 bits instead of ~48 for some inputs)
  ! sqrt(0) = 0 (xn is forced to 0), sqrt(x < 0) = NaN.
  ! ================================================================

  attributes(device) function fltflt_sqrt(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c, ynsqr, diff, prod
    real(4) :: xn, yn
    xn    = 0.0
    if (a%hi /= 0.0) xn = rsqrt(a%hi)
    yn    = __fmul_rn(a%hi, xn)
    ynsqr = fltflt_two_prod_fma(yn, yn)
    diff  = sub_ff_ff(a, ynsqr)
    prod  = fltflt_two_prod_fma(xn, 0.5 * diff%hi)
    c     = add_ff_r4(prod, yn)
  end function fltflt_sqrt

  attributes(device) function fltflt_sqrt_fast(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    real(4) :: xn, yn, residual, correction
    xn         = 0.0
    if (a%hi /= 0.0) xn = rsqrt(a%hi)
    yn         = __fmul_rn(a%hi, xn)
    residual   = __fadd_rn(__fmaf_rn(-yn, yn, a%hi), a%lo)
    correction = __fmul_rn(__fmul_rn(xn, 0.5), residual)
    c          = fltflt_fast_two_sum(yn, correction)
  end function fltflt_sqrt_fast

  ! ================================================================
  ! fltflt_norm3d: sqrt(dx^2 + dy^2 + dz^2) (fltflt.h)
  !
  ! Three exact squares accumulated with one normalization, then
  ! fltflt_sqrt_fast. Cross-terms (2*hi*lo) from each square are
  ! folded into the lo accumulation via FMA. ~39 flops.
  ! ================================================================

  attributes(device) function fltflt_norm3d(dx, dy, dz) result(c)
    type(fltflt), intent(in) :: dx, dy, dz
    type(fltflt) :: c, px, py, pz, s, t, sum_sq
    real(4) :: lo
    px  = fltflt_two_prod_fma(dx%hi, dx%hi)
    py  = fltflt_two_prod_fma(dy%hi, dy%hi)
    pz  = fltflt_two_prod_fma(dz%hi, dz%hi)
    s   = fltflt_two_sum(px%hi, py%hi)
    t   = fltflt_two_sum(s%hi, pz%hi)
    lo  = __fadd_rn(t%lo, s%lo)
    lo  = __fadd_rn(lo, px%lo)
    lo  = __fadd_rn(lo, py%lo)
    lo  = __fadd_rn(lo, pz%lo)
    lo  = __fmaf_rn(__fadd_rn(dx%hi, dx%hi), dx%lo, lo)
    lo  = __fmaf_rn(__fadd_rn(dy%hi, dy%hi), dy%lo, lo)
    lo  = __fmaf_rn(__fadd_rn(dz%hi, dz%hi), dz%lo, lo)
    sum_sq = fltflt_fast_two_sum(t%hi, lo)
    c   = fltflt_sqrt_fast(sum_sq)
  end function fltflt_norm3d

  ! ================================================================
  ! Rounding functions (branchful but correct)
  ! Uses Fortran intrinsics: aint (truncate toward zero), anint
  ! (round nearest, ties away from zero), floor, mod, abs, sign.
  ! ================================================================

  ! fltflt_round_toward_zero: largest-magnitude integer with |result| <= |a|.
  attributes(device) function fltflt_round_toward_zero(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    real(4) :: hi_trunc, lo_trunc
    logical :: opp_sign
    if (abs(a%hi) < 8388608.0) then  ! 2^23: a%hi not yet an integer
      hi_trunc = aint(a%hi)
      opp_sign = (a%hi > 0.0 .and. a%lo < 0.0) .or. &
                 (a%hi < 0.0 .and. a%lo > 0.0)
      if (hi_trunc == a%hi .and. opp_sign) then
        c%hi = a%hi + merge(-1.0, 1.0, a%hi > 0.0)
        c%lo = 0.0
      else
        c%hi = hi_trunc
        c%lo = 0.0
      end if
    else  ! a%hi is already an integer; refine with lo
      lo_trunc = aint(a%lo)
      opp_sign = (a%hi > 0.0 .and. a%lo < 0.0) .or. &
                 (a%hi < 0.0 .and. a%lo > 0.0)
      if (lo_trunc /= a%lo .and. opp_sign) then
        lo_trunc = lo_trunc + merge(-1.0, 1.0, a%hi > 0.0)
      end if
      c = fltflt_fast_two_sum(a%hi, lo_trunc)
    end if
  end function fltflt_round_toward_zero

  ! floorf: floor of a real(4) as a real(4). The intrinsic floor() returns a
  ! default (32-bit) integer and overflows once |x| >= 2^31 -- which lo does
  ! for |hi| >= ~2^56. aint() is exact at any magnitude (|x| >= 2^23 is an
  ! integer already).
  pure attributes(device) function floorf(x) result(f)
    real(4), intent(in) :: x
    real(4) :: f
    f = aint(x)
    if (f > x) f = f - 1.0
  end function floorf

  ! fltflt_floor: largest integer not greater than a.
  attributes(device) function fltflt_floor(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    real(4) :: hi_floor
    if (abs(a%hi) < 8388608.0) then  ! 2^23
      hi_floor = floorf(a%hi)
      if (hi_floor == a%hi .and. a%lo < 0.0) then
        c%hi = a%hi - 1.0
        c%lo = 0.0
      else
        c%hi = hi_floor
        c%lo = 0.0
      end if
    else  ! a%hi is already an integer; refine with lo
      c = fltflt_fast_two_sum(a%hi, floorf(a%lo))
    end if
  end function fltflt_floor

  ! fltflt_round_to_nearest: round to nearest integer, ties to even.
  attributes(device) function fltflt_round_to_nearest(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c
    real(4) :: candidate, err, r_lo, frac
    if (abs(a%hi) < 8388608.0) then  ! 2^23
      candidate = anint(a%hi)
      err       = a%hi - candidate
      if (abs(err) /= 0.5) then
        c%hi = candidate
        c%lo = 0.0
      else if (a%lo == 0.0) then
        ! Exact tie at 0.5: round to even
        if (mod(candidate, 2.0) == 0.0) then
          c%hi = candidate
        else
          c%hi = candidate + sign(1.0, err)
        end if
        c%lo = 0.0
      else if ((err > 0.0 .and. a%lo > 0.0) .or. &
               (err < 0.0 .and. a%lo < 0.0)) then
        c%hi = candidate + sign(1.0, err)
        c%lo = 0.0
      else
        c%hi = candidate
        c%lo = 0.0
      end if
    else  ! a%hi is already an integer; round the lo part
      r_lo = anint(a%lo)
      frac = a%lo - r_lo
      if (abs(frac) > 0.5) then
        r_lo = r_lo + sign(1.0, frac)
      else if (abs(frac) == 0.5) then
        ! hi + r_lo is odd iff exactly one of them is odd. Their FP32 sum
        ! cannot be used: it is rounded once |hi| >= 2^24 and loses the bit.
        ! Integer flags, not .neqv. of logicals (nvfortran device logicals,
        ! see the .not. note in CLAUDE.md).
        if (merge(1, 0, mod(a%hi, 2.0) /= 0.0) /= merge(1, 0, mod(r_lo, 2.0) /= 0.0)) then
          r_lo = r_lo + sign(1.0, frac)
        end if
      end if
      c = fltflt_fast_two_sum(a%hi, r_lo)
    end if
  end function fltflt_round_to_nearest

  ! ================================================================
  ! fltflt_fmod: a - trunc(a/b)*b  (floating-point remainder)
  ! Returns {NaN, NaN} if b is zero.
  ! ================================================================

  ! ff/ff overload.
  attributes(device) function fmod_ff_ff(a_in, b_in) result(c)
    type(fltflt), intent(in) :: a_in, b_in
    type(fltflt) :: c
    type(fltflt) :: a, b, q, trunc_q, rem
    real(4) :: sgn
    if (b_in%hi == 0.0 .and. b_in%lo == 0.0) then
      c%hi = ff_qnan
      c%lo = c%hi
      return
    end if
    sgn = 1.0
    a   = a_in
    if (lt_ff_r4(a, 0.0)) then
      sgn = -1.0
      a   = neg_ff(a)
    end if
    b       = fltflt_abs(b_in)
    q       = div_ff_ff(a, b)
    trunc_q = fltflt_round_toward_zero(q)
    rem     = neg_ff(fma_ff_ff_ff(trunc_q, b, neg_ff(a)))  ! a - trunc_q*b
    do while (ge_ff_ff(rem, b))
      rem = sub_ff_ff(rem, b)
    end do
    do while (lt_ff_r4(rem, 0.0))
      rem = add_ff_ff(rem, b)
    end do
    c%hi = sgn * rem%hi
    c%lo = sgn * rem%lo
  end function fmod_ff_ff

  ! ff/r4 overload.
  attributes(device) function fmod_ff_r4(a_in, b_in) result(c)
    type(fltflt), intent(in) :: a_in
    real(4),      intent(in) :: b_in
    type(fltflt) :: c
    type(fltflt) :: a, q, trunc_q, rem
    real(4) :: sgn, b
    if (b_in == 0.0) then
      c%hi = ff_qnan
      c%lo = c%hi
      return
    end if
    sgn = 1.0
    a   = a_in
    if (lt_ff_r4(a, 0.0)) then
      sgn = -1.0
      a   = neg_ff(a)
    end if
    b       = abs(b_in)
    q       = div_ff_r4(a, b)
    trunc_q = fltflt_round_toward_zero(q)
    rem     = neg_ff(fma_ff_r4_ff(trunc_q, b, neg_ff(a)))  ! a - trunc_q*b
    do while (ge_ff_r4(rem, b))
      rem = sub_ff_r4(rem, b)
    end do
    do while (lt_ff_r4(rem, 0.0))
      rem = add_ff_r4(rem, b)
    end do
    c%hi = sgn * rem%hi
    c%lo = sgn * rem%lo
  end function fmod_ff_r4


  ! ================================================================
  ! min / max / clamp
  ! ================================================================

  pure attributes(device) function fltflt_min_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: r
    if (a <= b) then;  r = a;  else;  r = b;  end if
  end function fltflt_min_ff_ff

  pure attributes(device) function fltflt_min_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: r, bf
    bf = init_from_r4(b)
    if (a <= bf) then;  r = a;  else;  r = bf;  end if
  end function fltflt_min_ff_r4

  pure attributes(device) function fltflt_min_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: r, af
    af = init_from_r4(a)
    if (af <= b) then;  r = af;  else;  r = b;  end if
  end function fltflt_min_r4_ff

  pure attributes(device) function fltflt_max_ff_ff(a, b) result(r)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: r
    if (a >= b) then;  r = a;  else;  r = b;  end if
  end function fltflt_max_ff_ff

  pure attributes(device) function fltflt_max_ff_r4(a, b) result(r)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: r, bf
    bf = init_from_r4(b)
    if (a >= bf) then;  r = a;  else;  r = bf;  end if
  end function fltflt_max_ff_r4

  pure attributes(device) function fltflt_max_r4_ff(a, b) result(r)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: r, af
    af = init_from_r4(a)
    if (af >= b) then;  r = af;  else;  r = b;  end if
  end function fltflt_max_r4_ff

  ! fltflt_clamp: clamp a to [lo, hi].
  pure attributes(device) function fltflt_clamp(a, lo, hi) result(r)
    type(fltflt), intent(in) :: a, lo, hi
    type(fltflt) :: r
    r = fltflt_min_ff_ff(fltflt_max_ff_ff(a, lo), hi)
  end function fltflt_clamp

  ! ================================================================
  ! ceil
  ! ================================================================

  ! fltflt_ceil: smallest integer not less than a. Uses identity ceil(x) = -floor(-x).
  attributes(device) function fltflt_ceil(a) result(c)
    type(fltflt), intent(in) :: a
    type(fltflt) :: c, neg
    neg%hi = -a%hi
    neg%lo = -a%lo
    c = fltflt_floor(neg)
    c%hi = -c%hi
    c%lo = -c%lo
  end function fltflt_ceil

  ! ================================================================
  ! hypot
  ! ================================================================

  ! fltflt_hypot: sqrt(a^2 + b^2) for real(4) inputs. Uses exact dot2 for the sum of squares.
  ! No scaling: the squares overflow FP32 for max(|a|,|b|) > ~1.8e19.
  attributes(device) function fltflt_hypot(a, b) result(c)
    real(4), intent(in) :: a, b
    type(fltflt) :: c
    c = fltflt_sqrt(fltflt_dot2(a, a, b, b))
  end function fltflt_hypot

  ! ================================================================
  ! cross product
  ! ================================================================

  ! fltflt_cross3d: a x b = (ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx).
  ! Uses fltflt_dot2 for exact cancellation in each component. 3 x (8 flops + 2 FMAs).
  attributes(device) subroutine fltflt_cross3d(cx, cy, cz, ax, ay, az, bx, by, bz)
    type(fltflt), intent(out) :: cx, cy, cz
    real(4),      intent(in)  :: ax, ay, az, bx, by, bz
    cx = fltflt_dot2( ay, bz, -az, by)
    cy = fltflt_dot2( az, bx, -ax, bz)
    cz = fltflt_dot2( ax, by, -ay, bx)
  end subroutine fltflt_cross3d

  ! ================================================================
  ! integer power, sign, lerp, warp reduction
  ! ================================================================

  ! fltflt_pow_int: a^n via square-and-multiply; n < 0 gives 1/a^|n|.
  ! Error grows like ~log2(|n|) roundings (a few units of 2^-44 for |n| <= 8).
  attributes(device) function fltflt_pow_int(a, n) result(r)
    type(fltflt), intent(in) :: a
    integer,      intent(in) :: n
    type(fltflt) :: r, base
    integer :: m
    r%hi = 1.0;  r%lo = 0.0
    base = a;    m = abs(n)
    do while (m > 0)
      if (mod(m, 2) == 1) r = r * base
      base = fltflt_square(base)
      m = m / 2
    end do
    if (n < 0) r = fltflt_recip(r)
  end function fltflt_pow_int

  ! fltflt_sign: copysign — magnitude of a with sign of b.
  pure attributes(device) function fltflt_sign(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, abs_a
    real(4) :: s
    abs_a = fltflt_abs(a)
    s     = sign(1.0, b%hi)
    c%hi  = abs_a%hi * s
    c%lo  = abs_a%lo * s
  end function fltflt_sign

  ! fltflt_lerp: linear interpolation a + t*(b-a) for real(4) t in [0,1].
  attributes(device) function fltflt_lerp(a, b, t) result(c)
    type(fltflt), intent(in) :: a, b
    real(4),      intent(in) :: t
    type(fltflt) :: c
    c = a + t * (b - a)
  end function fltflt_lerp

  ! fltflt_warp_reduce_sum: XOR-butterfly all-reduce across 32 lanes.
  ! After the call every lane holds the sum of all 32 input values.
  attributes(device) function fltflt_warp_reduce_sum(val) result(r)
    type(fltflt), intent(in) :: val
    type(fltflt) :: r
    integer :: mask
    r    = val
    mask = 16
    do while (mask > 0)
      r    = r + fltflt_shfl_xor(r, mask)
      mask = mask / 2
    end do
  end function fltflt_warp_reduce_sum

end module fltflt
