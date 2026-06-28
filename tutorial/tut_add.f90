! tut_add.f90  —  fltflt addition
!
! Build:  cd tutorial && cmake -B build && cmake --build build -j
! Run:    cd build && ./tut_add
!
! Demonstrates ff+ff addition via Knuth TwoSum + FastTwoSum (~14 flops).
! Input a=1e8, b=1: ULP(1e8)=8 so b is absorbed by real(4) rounding.
! fltflt stores the lost '1' in the lo component and recovers exact result.

module tut_add_kern
  use cudafor
  use fltflt
  implicit none
contains

  ! Compute a+b in real(4) and fltflt side by side.  Launch: <<<1,1>>>
  attributes(global) subroutine kern_add(a_in, b_in, res_r4, res_hi, res_lo)
    real(4), intent(in),  value  :: a_in, b_in
    real(4), intent(out), device :: res_r4(1), res_hi(1), res_lo(1)
    type(fltflt) :: a, b, c
    a = fltflt_init(a_in);  b = fltflt_init(b_in)
    res_r4(1) = a_in + b_in
    c = a + b
    res_hi(1) = c%hi;  res_lo(1) = c%lo
  end subroutine kern_add

end module tut_add_kern


program tut_add
  use cudafor
  use fltflt
  use tut_add_kern
  implicit none

  real(4), parameter :: A = 1.0e8_4, B = 1.0_4
  real(8), parameter :: REF = real(A, 8) + real(B, 8)

  real(4), device :: res_r4(1), res_hi(1), res_lo(1)
  real(4) :: h_r4, h_hi, h_lo
  real(8) :: ff_val, err_r4, err_ff
  integer :: istat

  call kern_add<<<1,1>>>(A, B, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()

  h_r4 = res_r4(1);  h_hi = res_hi(1);  h_lo = res_lo(1)
  ff_val = real(h_hi, 8) + real(h_lo, 8)
  err_r4 = abs((real(h_r4, 8) - REF) / REF)
  err_ff = abs((ff_val         - REF) / REF)

  write(*,*) "=== fltflt addition: 1.0E+08 + 1.0 ==="
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

end program tut_add
