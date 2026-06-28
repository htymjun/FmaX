! tut_dot3.f90  —  compensated dot3: a*b + c*d + e*f with fltflt precision
!
! Demonstrates that three inexact real(4) products accumulated naively
! lose precision, while fltflt_dot3() retains ~48 mantissa bits by preserving
! all error words from TwoProd before the final summation.
!
! Inputs: a=1.1, b=1.1, c=2.2, d=2.2, e=3.3, f=3.3
! Exact:  1.1^2 + 2.2^2 + 3.3^2 = 1.21 + 4.84 + 10.89 = 16.94
!
! Build:  cd tutorial && cmake -B build && cmake --build build -j
! Run:    cd build && ./tut_dot3

module tut_dot3_mod
  use cudafor
  use fltflt
  implicit none
contains

  attributes(global) subroutine kern_dot3(a_in, b_in, c_in, d_in, e_in, f_in, &
                                           r4_out, hi_out, lo_out)
    real(8), intent(in),  value  :: a_in, b_in, c_in, d_in, e_in, f_in
    real(4), intent(out), device :: r4_out, hi_out, lo_out
    real(4) :: ar, br, cr, dr, er, fr
    type(fltflt) :: rf
    ar = real(a_in, 4);  br = real(b_in, 4)
    cr = real(c_in, 4);  dr = real(d_in, 4)
    er = real(e_in, 4);  fr = real(f_in, 4)
    r4_out = ar*br + cr*dr + er*fr
    rf     = fltflt_dot3(ar, br, cr, dr, er, fr)
    hi_out = rf%hi
    lo_out = rf%lo
  end subroutine kern_dot3

end module tut_dot3_mod


program tut_dot3
  use cudafor
  use fltflt
  use tut_dot3_mod
  implicit none

  real(8), parameter :: a_r8 = 1.1d0, b_r8 = 1.1d0
  real(8), parameter :: c_r8 = 2.2d0, d_r8 = 2.2d0
  real(8), parameter :: e_r8 = 3.3d0, f_r8 = 3.3d0
  real(8), parameter :: ref  = a_r8*b_r8 + c_r8*d_r8 + e_r8*f_r8

  real(4), device :: d_r4, d_hi, d_lo
  real(4) :: h_r4, h_hi, h_lo
  real(8) :: ff_val, err_r4, err_ff
  integer :: istat

  call kern_dot3<<<1,1>>>(a_r8, b_r8, c_r8, d_r8, e_r8, f_r8, d_r4, d_hi, d_lo)
  istat = cudaDeviceSynchronize()
  h_r4 = d_r4;  h_hi = d_hi;  h_lo = d_lo

  ff_val = real(h_hi, 8) + real(h_lo, 8)
  err_r4 = abs((real(h_r4, 8) - ref) / ref)
  err_ff = abs((ff_val        - ref) / ref)

  write(*,'(A)')       "=== tut_dot3: a*b + c*d + e*f ==="
  write(*,'(A,F20.15)') "  a*b = 1.1 * 1.1 = ", a_r8*b_r8
  write(*,'(A,F20.15)') "  c*d = 2.2 * 2.2 = ", c_r8*d_r8
  write(*,'(A,F20.15)') "  e*f = 3.3 * 3.3 = ", e_r8*f_r8
  write(*,'(A,F20.15)') "  reference (r8)  = ", ref
  write(*,'(A)')       ""
  write(*,'(A,F20.15)') "  real(4) result   = ", real(h_r4, 8)
  write(*,'(A,F20.15,A,ES9.2,A)') &
    "  fltflt result   = ", ff_val, "  (hi=", real(h_hi,8), ")"
  write(*,'(A,F20.15)') "  lo component    = ", real(h_lo, 8)
  write(*,'(A)')       ""
  if (err_r4 == 0.0d0) then
    write(*,'(A)') "  r4 error  = 0 (exact)"
  else
    write(*,'(A,ES9.2)') "  r4 error  = ", err_r4
  end if
  if (err_ff == 0.0d0) then
    write(*,'(A)') "  ff error  = 0 (exact)"
  else
    write(*,'(A,ES9.2)') "  ff error  = ", err_ff
  end if

end program tut_dot3
