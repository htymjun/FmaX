! tut_mul.f90  —  fltflt multiplication (dble input)
!
! Build:  cd tutorial/mul && make && make run
!
! Demonstrates init_from_r8 for multiplication.
! Input: a = b = 1.1d0.  The exact product 1.21 requires >24 mantissa bits.
! Truncating 1.1d0 to real(4) and multiplying gives ~3e-8 relative error.
! fltflt splits both inputs via init_from_r8 and uses FMA-based TwoProd
! to compute the product to near-real(8) accuracy.

module tut_mul_kern
  use cudafor
  use fltflt
  implicit none
contains

  ! Compute a*b via real(4) truncation and via fltflt.  Launch: <<<1,1>>>
  attributes(global) subroutine kern_mul(a_in, b_in, res_r4, res_hi, res_lo)
    real(8), intent(in),  value  :: a_in, b_in
    real(4), intent(out), device :: res_r4(1), res_hi(1), res_lo(1)
    type(fltflt) :: a, b, c
    a = fltflt_init(a_in);  b = fltflt_init(b_in)           ! exact 2-component splits
    res_r4(1) = real(a_in, 4) * real(b_in, 4) ! naive: truncate first, then mul
    c = a * b
    res_hi(1) = c%hi;  res_lo(1) = c%lo
  end subroutine kern_mul

end module tut_mul_kern


program tut_mul
  use cudafor
  use fltflt
  use tut_mul_kern
  implicit none

  real(8), parameter :: A = 1.1d0, B = 1.1d0
  real(8), parameter :: REF = A * B   ! = 1.21 (needs >24 bits)

  real(4), device :: res_r4(1), res_hi(1), res_lo(1)
  real(4) :: h_r4, h_hi, h_lo
  real(8) :: ff_val, err_r4, err_ff
  integer :: istat

  call kern_mul<<<1,1>>>(A, B, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()

  h_r4 = res_r4(1);  h_hi = res_hi(1);  h_lo = res_lo(1)
  ff_val = real(h_hi, 8) + real(h_lo, 8)
  err_r4 = abs((real(h_r4, 8) - REF) / REF)
  err_ff = abs((ff_val         - REF) / REF)

  write(*,*) "=== fltflt multiplication (dble input): 1.1d0 * 1.1d0 ==="
  write(*,*)
  write(*,'("  real(4)  result : ", ES15.8)') h_r4
  write(*,'("  fltflt   result : hi=", ES15.8, "  lo=", ES12.5)') h_hi, h_lo
  write(*,'("  reference (r8)  : ", ES15.8)') REF
  write(*,*)
  write(*,'("  real(4)  error  : ", ES9.2)') err_r4
  if (err_ff == 0.0d0) then
    write(*,'("  fltflt   error  : 0.00E+00  [exact]")')
  else
    write(*,'("  fltflt   error  : ", ES9.2)') err_ff
  end if

end program tut_mul
