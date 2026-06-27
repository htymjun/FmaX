! Float-float (FF) precision arithmetic — public API
!
! Represents a value as an unevaluated sum (hi + lo) of two real(4) numbers,
! giving ~48 mantissa bits — equivalent to real(8) precision.
! All procedures carry attributes(device) and are callable inside CUDA Fortran
! kernels (!$cuf kernel do / attributes(global)) without change.
!
! EFT primitives (two_sum, fast_two_sum, two_prod) and the fltflt type
! are defined in mod_ffp_eft.
!
! Inlining: add  name:two_sum,name:fast_two_sum,name:two_prod,
!                name:shfl_down_ff,name:shfl_xor_ff
! to the case CMakeLists.txt -Minline= list for optimal GPU performance.
module mod_ffp
  use cudafor
  use cudadevice
  use mod_ffp_eft, only: fltflt, two_sum, fast_two_sum, two_prod
  implicit none
  private

  public :: fltflt

  interface init
    module procedure init_from_r4
    module procedure init_from_r8
  end interface init
  public :: init

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

  public :: shfl_down_ff, shfl_xor_ff

contains

  ! ================================================================
  ! Constructors
  ! ================================================================

  pure attributes(device, host) function init_from_r4(a) result(r)
    real(4), intent(in) :: a
    type(fltflt) :: r
    r%hi = a
    r%lo = 0.0
  end function init_from_r4

  ! Splits a double value exactly into two single-precision halves.
  pure attributes(device, host) function init_from_r8(a) result(r)
    real(8), intent(in) :: a
    type(fltflt) :: r
    r%hi = real(a, 4)
    r%lo = real(a - real(r%hi, 8), 4)
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
  ! Addition
  ! ================================================================

  ! Full ff+ff: two TwoSums + renorm. ~14 flops.
  pure attributes(device) function add_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, s, t
    s    = two_sum(a%hi, b%hi)
    t    = two_sum(a%lo, b%lo)
    s%lo = s%lo + t%hi
    c    = fast_two_sum(s%hi, s%lo)
    c%lo = c%lo + t%lo
  end function add_ff_ff

  ! ff + scalar: b%lo = 0 inlined, saves one TwoSum. ~9 flops.
  pure attributes(device) function add_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, s
    s    = two_sum(a%hi, b)
    s%lo = s%lo + a%lo
    c    = fast_two_sum(s%hi, s%lo)
  end function add_ff_r4

  pure attributes(device) function add_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r4(b, a)
  end function add_r4_ff

  pure attributes(device) function add_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c, b_ff
    b_ff%hi = real(b, 4)
    b_ff%lo = real(b - real(b_ff%hi, 8), 4)
    c = add_ff_ff(a, b_ff)
  end function add_ff_r8

  pure attributes(device) function add_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r8(b, a)
  end function add_r8_ff
  
  ! ================================================================
  ! Subtraction
  ! ================================================================

  pure attributes(device) function sub_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c
    c = add_ff_ff(a, neg_ff(b))
  end function sub_ff_ff

  pure attributes(device) function sub_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r4(a, -b)
  end function sub_ff_r4

  pure attributes(device) function sub_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r4(neg_ff(b), a)
  end function sub_r4_ff

  pure attributes(device) function sub_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r8(a, -b)
  end function sub_ff_r8
  
  pure attributes(device) function sub_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = add_ff_r8(neg_ff(b), a)
  end function sub_r8_ff

  ! ================================================================
  ! Multiplication
  ! ================================================================

  ! ff*ff: TwoProd for leading term + cross terms. ~8 flops + 1 FMA.
  pure attributes(device) function mul_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, p
    p    = two_prod(a%hi, b%hi)
    p%lo = fma(a%hi, b%lo, fma(a%lo, b%hi, p%lo))
    c    = fast_two_sum(p%hi, p%lo)
  end function mul_ff_ff

  ! ff*scalar: b%lo = 0 inlined. ~5 flops + 1 FMA.
  pure attributes(device) function mul_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, p
    p    = two_prod(a%hi, b)
    p%lo = fma(a%lo, b, p%lo)
    c    = fast_two_sum(p%hi, p%lo)
  end function mul_ff_r4

  pure attributes(device) function mul_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = mul_ff_r4(b, a)
  end function mul_r4_ff

  pure attributes(device) function mul_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c, b_ff
    b_ff%hi = real(b, 4)
    b_ff%lo = real(b - real(b_ff%hi, 8), 4)
    c = mul_ff_ff(a, b_ff)
  end function mul_ff_r8

  pure attributes(device) function mul_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = mul_ff_r8(b, a)
  end function mul_r8_ff

  ! ================================================================
  ! Division  (Dekker 1971, one Newton-step refinement)
  !
  ! q1 = a%hi / b%hi                  first quotient approximation
  ! (p%hi, p%lo) = TwoProd(q1, b%hi)  exact q1*b%hi
  ! s = a%hi - p%hi                   residual, high part (includes p%lo)
  ! e = s - p%lo + a%lo - q1*b%lo     full residual ≈ a - q1*b
  ! q2 = e / b%hi                     correction
  ! ================================================================

  pure attributes(device) function div_ff_ff(a, b) result(c)
    type(fltflt), intent(in) :: a, b
    type(fltflt) :: c, p
    real(4) :: q1, q2, s, e
    q1   = a%hi / b%hi
    p    = two_prod(q1, b%hi)
    s    = a%hi - p%hi
    e    = fma(-q1, b%lo, s - p%lo + a%lo)
    q2   = e / b%hi
    c    = fast_two_sum(q1, q2)
  end function div_ff_ff

  ! ff/scalar: b%lo = 0 inlined. ~9 flops + 1 FMA.
  pure attributes(device) function div_ff_r4(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(4),      intent(in) :: b
    type(fltflt) :: c, p
    real(4) :: q1, q2, s, e
    q1   = a%hi / b
    p    = two_prod(q1, b)
    s    = a%hi - p%hi
    e    = s - p%lo + a%lo
    q2   = e / b
    c    = fast_two_sum(q1, q2)
  end function div_ff_r4

  pure attributes(device) function div_r4_ff(a, b) result(c)
    real(4),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c
    c = div_ff_ff(init_from_r4(a), b)
  end function div_r4_ff

  pure attributes(device) function div_ff_r8(a, b) result(c)
    type(fltflt), intent(in) :: a
    real(8),      intent(in) :: b
    type(fltflt) :: c, b_ff
    b_ff%hi = real(b, 4)
    b_ff%lo = real(b - real(b_ff%hi, 8), 4)
    c = div_ff_ff(a, b_ff)
  end function div_ff_r8

  pure attributes(device) function div_r8_ff(a, b) result(c)
    real(8),      intent(in) :: a
    type(fltflt), intent(in) :: b
    type(fltflt) :: c, a_ff
    a_ff%hi = real(a, 4)
    a_ff%lo = real(a - real(a_ff%hi, 8), 4)
    c = div_ff_ff(a_ff, b)
  end function div_r8_ff

  ! ================================================================
  ! Warp shuffle for fltflt — NOT pure (hardware warp communication)
  ! Both components (hi, lo) are shuffled independently.
  ! Uses __shfl_down / __shfl_xor (nvfortran 25.x API).
  ! WARP_MASK = 0xFFFFFFFF activates all 32 lanes.
  ! ================================================================

  attributes(device) function shfl_down_ff(val, delta) result(r)
    type(fltflt), intent(in) :: val
    integer(4),   intent(in), value :: delta
    type(fltflt) :: r
    r%hi = __shfl_down(val%hi, delta)
    r%lo = __shfl_down(val%lo, delta)
  end function shfl_down_ff

  attributes(device) function shfl_xor_ff(val, lane_mask) result(r)
    type(fltflt), intent(in) :: val
    integer(4),   intent(in), value :: lane_mask
    type(fltflt) :: r
    r%hi = __shfl_xor(val%hi, lane_mask)
    r%lo = __shfl_xor(val%lo, lane_mask)
  end function shfl_xor_ff

end module mod_ffp
