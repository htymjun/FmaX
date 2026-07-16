! kernel.fypp
!===============================================================================
! FmaX: Performance-Maximizing Utility Library for CUDA Fortran
! File: vfma.f90 (to be preprocessed with 'fypp')
!===============================================================================
module vfma
  use libm
  implicit none
  private

  ! Public interfaces and placeholders for future FmaX precision models
  ! (e.g., float_float / double_double from MatX fltflt.h)
  ! public :: float_float, double_double

contains

! clean_var_name, parse_arg, compute_loads, resolve_terms are Python helpers
! from vfma_helpers.py, loaded via: fypp -M FmaX -m vfma_helpers ...
! (fypp has no #:python/#:endpython directive for inline multi-statement code).
! fypp's `-m` binds the imported module to a variable of its own name, not its
! members, so calls below are qualified as vfma_helpers.<name>(...).


end module vfma
attributes(device) subroutine stencil_kernel(a, b, c, i, j, k)
  use vfma
  use cudadevice
  implicit none
  real(8), intent(in)  :: a(:,:,:)
  real(8), intent(in)  :: b(:)
  real(8), intent(out) :: c(:)
  integer, value       :: i, j, k

  ! Call the FmaX core macro!
  ! (`#:call`/`#:endcall`, not `@:vfma(...)`, because `@:` direct-call args are
  ! passed as raw, unevaluated text -- n would arrive as the string "3" and the
  ! quoted array-slice strings would keep their literal quote characters.
  ! `#:call` evaluates its header as real Python via __getargvalues, so n is
  ! an int and the quoted args become plain strings, as vfma expects.)

  block
  ! [1a] Register declarations (all specification statements before any executable ones)
  real(8) :: reg_a_i_p1_j_k
  real(8) :: reg_a_i_p2_j_k
  real(8) :: reg_a_i_j_k
  real(8) :: reg_a_i_m1_j_k
  real(8) :: reg_b_1
  real(8) :: reg_b_2
  real(8) :: reg_b_3

  ! [1b] Bulk Register Loading (completely deduplicated, forcing exactly 1 load per element)
  reg_a_i_p1_j_k = a(i+1,j,k)
  reg_a_i_p2_j_k = a(i+2,j,k)
  reg_a_i_j_k = a(i,j,k)
  reg_a_i_m1_j_k = a(i-1,j,k)
  reg_b_1 = b(1)
  reg_b_2 = b(2)
  reg_b_3 = b(3)

  ! [2] Inline FMA Expansion (zero-overhead execution, device-safe double-precision FMA)
  c(1) = __fma_rn(reg_a_i_m1_j_k, reg_a_i_j_k, reg_b_1)
  c(2) = __fma_rn(reg_a_i_j_k, reg_a_i_p1_j_k, reg_b_2)
  c(3) = __fma_rn(reg_a_i_p1_j_k, reg_a_i_p2_j_k, reg_b_3)
  end block
end subroutine
