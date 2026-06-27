! tut_sub.f90  —  fltflt subtraction
!
! Build:  cd tutorial && cmake -B build && cmake --build build -j
! Run:    cd build && ./tut_sub
!
! Demonstrates ff-r4 subtraction (negation + add_ff_r4, same cost as addition).
! Input a = 1 + 2^{-25} stored as fltflt via init_from_r8; b = 1.0_4.
! 2^{-25} < eps(1.0_4) = 2^{-23}, so real(4) subtraction gives 0 (100% error).
! fltflt carries the tiny difference in lo and recovers the exact result.

module tut_sub_kern
  use cudafor
  use mod_ffp
  implicit none
contains

  ! Compute a-b where a is a real(8) split into fltflt on-device.  Launch: <<<1,1>>>
  attributes(global) subroutine kern_sub(a_r8, b_r4, res_r4, res_hi, res_lo)
    real(8), intent(in),  value  :: a_r8
    real(4), intent(in),  value  :: b_r4
    real(4), intent(out), device :: res_r4(1), res_hi(1), res_lo(1)
    type(fltflt) :: a, c
    a = init(a_r8)                   ! exact 2-component split of the double
    res_r4(1) = real(a_r8, 4) - b_r4 ! naive: truncates a to r4 first
    c = a - b_r4
    res_hi(1) = c%hi;  res_lo(1) = c%lo
  end subroutine kern_sub

end module tut_sub_kern


program tut_sub
  use cudafor
  use mod_ffp
  use tut_sub_kern
  implicit none

  real(8), parameter :: A_R8 = 1.0d0 + 2.0d0**(-25)
  real(4), parameter :: B_R4 = 1.0_4
  real(8), parameter :: REF  = A_R8 - real(B_R4, 8)

  real(4), device :: res_r4(1), res_hi(1), res_lo(1)
  real(4) :: h_r4, h_hi, h_lo
  real(8) :: ff_val, err_r4, err_ff
  integer :: istat

  call kern_sub<<<1,1>>>(A_R8, B_R4, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()

  h_r4 = res_r4(1);  h_hi = res_hi(1);  h_lo = res_lo(1)
  ff_val = real(h_hi, 8) + real(h_lo, 8)
  err_r4 = abs((real(h_r4, 8) - REF) / REF)
  err_ff = abs((ff_val         - REF) / REF)

  write(*,*) "=== fltflt subtraction: (1 + 2^-25) - 1.0 ==="
  write(*,*)
  write(*,'("  real(4)  result : ", ES15.8, "  [2^-25 below r4 eps: absorbed]")') h_r4
  write(*,'("  fltflt   result : hi=", ES15.8, "  lo=", ES12.5)') h_hi, h_lo
  write(*,'("  reference (r8)  : ", ES15.8)') REF
  write(*,*)
  write(*,'("  real(4)  error  : ", ES9.2)') err_r4
  if (err_ff == 0.0d0) then
    write(*,'("  fltflt   error  : 0.00E+00  [exact]")')
  else
    write(*,'("  fltflt   error  : ", ES9.2)') err_ff
  end if

end program tut_sub
