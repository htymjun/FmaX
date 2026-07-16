! tut_sub.f90  —  fltflt subtraction (dble input)
!
! Build:  cd tutorial/sub && make && make run
!
! Demonstrates init_from_r8 under catastrophic cancellation.
! Input: a = 1 + 2^-25, b = 1.  The difference 2^-25 is below eps(real(4)(1.0))
! so truncating a to real(4) gives 1.0 and the subtraction yields 0 (100% error).
! fltflt splits a exactly into hi=1.0, lo=2^-25 and recovers the true difference.

module tut_sub_kern
  use cudafor
  use fltflt
  implicit none
contains

  ! Compute a-b via real(4) truncation and via fltflt.  Launch: <<<1,1>>>
  attributes(global) subroutine kern_sub(a_in, b_in, res_r4, res_hi, res_lo)
    real(8), intent(in),  value  :: a_in, b_in
    real(4), intent(out), device :: res_r4(1), res_hi(1), res_lo(1)
    type(fltflt) :: a, b, c
    a = fltflt_init(a_in);  b = fltflt_init(b_in)           ! exact 2-component splits
    res_r4(1) = real(a_in, 4) - real(b_in, 4) ! naive: truncates 2^-25 away
    c = a - b
    res_hi(1) = c%hi;  res_lo(1) = c%lo
  end subroutine kern_sub

end module tut_sub_kern


program tut_sub
  use cudafor
  use fltflt
  use tut_sub_kern
  implicit none

  real(8), parameter :: A = 1.0d0 + 2.0d0**(-25), B = 1.0d0
  real(8), parameter :: REF = A - B   ! = 2^-25 ~ 2.98e-8

  real(4), device :: res_r4(1), res_hi(1), res_lo(1)
  real(4) :: h_r4, h_hi, h_lo
  real(8) :: ff_val, err_r4, err_ff
  integer :: istat

  call kern_sub<<<1,1>>>(A, B, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()

  h_r4 = res_r4(1);  h_hi = res_hi(1);  h_lo = res_lo(1)
  ff_val = real(h_hi, 8) + real(h_lo, 8)
  err_r4 = abs((real(h_r4, 8) - REF) / REF)
  err_ff = abs((ff_val         - REF) / REF)

  write(*,*) "=== fltflt subtraction (dble input): (1 + 2^-25) - 1.0 ==="
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
