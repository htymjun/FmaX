! Float-float (FF) precision arithmetic — EFT backend
!
! Defines the fltflt type and the three error-free transformation primitives
! used internally by mod_ffp.  Not part of the public API.
!
! FMA requirement: two_prod uses fma(a, b, -hi) which maps to a single
! FMAF.F32 instruction.  Hardware FMA is mandatory; compile with -fast
! (enables -Mfma) on nvfortran.  Without it the error term is wrong.
!
! Inlining: add  name:two_sum,name:fast_two_sum,name:two_prod
! to the case CMakeLists.txt -Minline= list for optimal GPU performance.
module mod_ffp_eft
  use cudafor
  use cudadevice
  implicit none
  private

  ! hi + lo = exact value; |lo| <= 0.5 ulp(hi)
  ! No default initializers: nvfortran crashes compiling device code for types
  ! with default-initialized components.  Callers must use init() or set both
  ! components explicitly before use.
  type, public :: fltflt
    real(4) :: hi, lo
  end type fltflt

  public :: two_sum, fast_two_sum, two_prod

contains

  ! TwoSum (Knuth 1969): exact split, no precondition, 6 flops.
  pure attributes(device) function two_sum(a, b) result(r)
    real(4), intent(in) :: a, b
    type(fltflt) :: r
    real(4) :: v
    r%hi = a + b
    v    = r%hi - a
    r%lo = (a - (r%hi - v)) + (b - v)
  end function two_sum

  ! FastTwoSum (Dekker 1971): 3 flops; requires |a| >= |b|.
  pure attributes(device) function fast_two_sum(a, b) result(r)
    real(4), intent(in) :: a, b
    type(fltflt) :: r
    r%hi = a + b
    r%lo = b - (r%hi - a)
  end function fast_two_sum

  ! TwoProd: exact product using FMA.  fma(a, b, -hi) = a*b - hi exactly
  ! in a single FMAF.F32 instruction.  Replaces 17-op Veltkamp split.
  ! Requires hardware FMA; compile with -fast (-Mfma) on nvfortran.
  pure attributes(device) function two_prod(a, b) result(r)
    real(4), intent(in) :: a, b
    type(fltflt) :: r
    r%hi = a * b
    r%lo = fma(a, b, -r%hi)
  end function two_prod

end module mod_ffp_eft
