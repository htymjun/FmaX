! test_dot.f90  —  fltflt compensated dot products: accuracy + throughput
!
! Build:  cd ouxmix && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_dot
!        (writes test/dot_results.csv)
!
! Accuracy: NCASE scalar cases testing dot2 and dot3, real(8) inputs.
! Benchmark: chained x = fltflt_dot3(a,b,c,d,e,f) vs real(4) naive for real(4), fltflt.

module test_dot_kern
  use cudafor
  use fltflt
  implicit none

  integer, parameter :: NCASE   = 4
  integer, parameter :: N_BENCH = 2**20
  integer, parameter :: NITER   = 100
  integer, parameter :: NBLK    = 256

contains

  ! ---- accuracy kernel for dot3 ----------------------------------------
  ! Takes 6 real(8) inputs per element; computes naive r4 and fltflt dot3.
  attributes(global) subroutine kern_dot3(a_d, b_d, c_d, d_d, e_d, f_d, n, &
                                           res_r4, res_hi, res_lo)
    integer,  intent(in),  value  :: n
    real(8),  intent(in),  device :: a_d(n), b_d(n), c_d(n), d_d(n), e_d(n), f_d(n)
    real(4),  intent(out), device :: res_r4(n), res_hi(n), res_lo(n)
    real(4) :: ar, br, cr, dr, er, fr
    type(fltflt) :: rf
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    ar = real(a_d(i), 4);  br = real(b_d(i), 4)
    cr = real(c_d(i), 4);  dr = real(d_d(i), 4)
    er = real(e_d(i), 4);  fr = real(f_d(i), 4)
    res_r4(i) = ar*br + cr*dr + er*fr
    rf = fltflt_dot3(ar, br, cr, dr, er, fr)
    res_hi(i) = rf%hi;  res_lo(i) = rf%lo
  end subroutine kern_dot3

  ! ---- benchmark kernels: chained x = fltflt_dot3(a,b,c,d,e,f) ---------------
  attributes(global) subroutine kern_bench_dot3_r4(n, niter, a_d, b_d, c_d, d_d, e_d, f_d, out_d)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n), c_d(n), d_d(n), e_d(n), f_d(n)
    real(4), intent(out), device  :: out_d(n)
    real(4) :: x, a, b, c, d, e, f
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    a = real(a_d(i), 4);  b = real(b_d(i), 4)
    c = real(c_d(i), 4);  d = real(d_d(i), 4)
    e = real(e_d(i), 4);  f = real(f_d(i), 4)
    x = 0.0
    do j = 1, niter
      x = x + (a*b + c*d + e*f)
    end do
    out_d(i) = x
  end subroutine kern_bench_dot3_r4

  attributes(global) subroutine kern_bench_dot3_ff(n, niter, a_d, b_d, c_d, d_d, e_d, f_d, out_hi, out_lo)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n), c_d(n), d_d(n), e_d(n), f_d(n)
    real(4), intent(out), device  :: out_hi(n), out_lo(n)
    real(4) :: a, b, c, d, e, f
    type(fltflt) :: xf
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    a = real(a_d(i), 4);  b = real(b_d(i), 4)
    c = real(c_d(i), 4);  d = real(d_d(i), 4)
    e = real(e_d(i), 4);  f = real(f_d(i), 4)
    xf%hi = 0.0;  xf%lo = 0.0
    do j = 1, niter
      xf = xf + fltflt_dot3(a, b, c, d, e, f)
    end do
    out_hi(i) = xf%hi;  out_lo(i) = xf%lo
  end subroutine kern_bench_dot3_ff

end module test_dot_kern


program test_dot
  use cudafor
  use fltflt
  use test_dot_kern
  implicit none

  integer, parameter :: N = NCASE
  ! fltflt_dot3 computes the exact sum of float32 products; the threshold is
  ! 1e-6 (not 1e-12) because float32 inputs like 1.1 round differently from
  ! the float64 reference, giving irreducible ~1e-8 input-precision error.
  real(8), parameter :: FF_THRESH = 1.0d-6
  character(len=20), parameter :: CLABELS(N) = &
    [ character(len=20) :: &
      "dot3[1.1^2+...]", &
      "dot3[cancel]", &
      "dot3[pi*e+...]", &
      "dot3[catastrophic]" ]

  real(8) :: a_r8(N), b_r8(N), c_r8(N), d_r8(N), e_r8(N), f_r8(N), ref(N)
  real(8), device, allocatable :: ad(:), bd(:), cd(:), dd(:), ed(:), fd(:)
  real(4), device, allocatable :: res_r4(:), res_hi(:), res_lo(:)
  real(4) :: h_r4(N), h_hi(N), h_lo(N)
  real(8) :: ff_val, err_r4, err_ff
  integer :: i, istat, nfail, csv_unit

  csv_unit = 11
  open(unit=csv_unit, file='dot_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,val1,val2'

  ! (1) 1.1*1.1 + 2.2*2.2 + 3.3*3.3 — three inexact products accumulate error
  a_r8(1) = 1.1d0;  b_r8(1) = 1.1d0
  c_r8(1) = 2.2d0;  d_r8(1) = 2.2d0
  e_r8(1) = 3.3d0;  f_r8(1) = 3.3d0

  ! (2) Partial cancellation: (1+2^-13)^2 + (-(1+2^-12))*1.0 + 1.5*1.0
  !     = 1+2^-12+2^-26 - 1 - 2^-12 + 1.5 ≈ 1.5 + 2^-26 (tiny residual lost by r4)
  a_r8(2) = 1.0d0 + 2.0d0**(-13);  b_r8(2) = 1.0d0 + 2.0d0**(-13)
  c_r8(2) = -(1.0d0 + 2.0d0**(-12));  d_r8(2) = 1.0d0
  e_r8(2) = 1.5d0;  f_r8(2) = 1.0d0

  ! (3) Transcendentals: pi*e + sqrt(2)*sqrt(3) + (1/3)*3
  a_r8(3) = 3.14159265358979d0;  b_r8(3) = 2.71828182845904d0
  c_r8(3) = 1.41421356237310d0;  d_r8(3) = 1.73205080756888d0
  e_r8(3) = 1.0d0/3.0d0;         f_r8(3) = 3.0d0

  ! (4) Float32-exact catastrophic cancellation:
  !     (1+2^-13)^2 + (-1)*(1+2^-12) + 0 = 2^-26 ≈ 1.49e-8
  !     All inputs are exactly representable in float32.
  !     Naive r4: float32((1+2^-13)^2) = 1+2^-12, so result = 0  (100% error).
  !     fltflt_dot3: recovers the exact 2^-26 rounding error via two_prod_fma.
  a_r8(4) = 1.0d0 + 2.0d0**(-13);  b_r8(4) = 1.0d0 + 2.0d0**(-13)
  c_r8(4) = -1.0d0;                  d_r8(4) = 1.0d0 + 2.0d0**(-12)
  e_r8(4) = 0.0d0;                   f_r8(4) = 1.0d0

  ref = a_r8*b_r8 + c_r8*d_r8 + e_r8*f_r8

  allocate(ad(N), bd(N), cd(N), dd(N), ed(N), fd(N))
  allocate(res_r4(N), res_hi(N), res_lo(N))
  ad = a_r8;  bd = b_r8;  cd = c_r8;  dd = d_r8;  ed = e_r8;  fd = f_r8
  call kern_dot3<<<1, N>>>(ad, bd, cd, dd, ed, fd, N, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()
  h_r4 = res_r4;  h_hi = res_hi;  h_lo = res_lo
  deallocate(ad, bd, cd, dd, ed, fd, res_r4, res_hi, res_lo)

  write(*,*) "=== test_dot: fltflt compensated dot3 (real(8) inputs) ==="
  nfail = 0
  do i = 1, N
    ff_val = real(h_hi(i), 8) + real(h_lo(i), 8)
    if (ref(i) /= 0.0d0) then
      err_r4 = abs((real(h_r4(i), 8) - ref(i)) / ref(i))
      err_ff = abs((ff_val            - ref(i)) / ref(i))
    else
      err_r4 = abs(real(h_r4(i), 8) - ref(i))
      err_ff = abs(ff_val            - ref(i))
    end if
    if (err_ff > FF_THRESH) nfail = nfail + 1
    write(*,'("  case",I2,": r4_err=",ES9.2,"  ff_err=",ES9.2,"  ",A)') &
      i, err_r4, err_ff, merge("PASS", "FAIL", err_ff <= FF_THRESH)
    write(csv_unit, '("acc,",A,",",ES12.5,",",ES12.5)') &
      trim(adjustl(CLABELS(i))), err_r4, err_ff
  end do

  call run_bench()

  close(csv_unit)

  if (nfail == 0) then
    write(*,*) "ALL PASS"
  else
    write(*,'("  FAILED: ",I0," case(s)")') nfail
    stop 1
  end if

contains

  subroutine run_bench()
    integer, parameter :: NB = N_BENCH, NI = NITER
    real(8), allocatable :: ah(:), bh(:), ch(:), dh(:), eh(:), fh(:)
    real(8), device, allocatable :: ad2(:), bd2(:), cd2(:), dd2(:), ed2(:), fd2(:)
    real(4), device, allocatable :: out_r4(:), out_hi(:), out_lo(:)
    real(8) :: ms_r4, ms_ff, mops_r4, mops_ff
    real(4) :: elapsed
    real(4) :: g_r4, g_hi, g_lo
    type(cudaEvent) :: t0, t1
    type(dim3) :: blk, thr
    integer :: nb_blk, is

    thr    = dim3(NBLK, 1, 1)
    nb_blk = (NB + NBLK - 1) / NBLK
    blk    = dim3(nb_blk, 1, 1)

    allocate(ah(NB), bh(NB), ch(NB), dh(NB), eh(NB), fh(NB))
    ah = 1.1d0;  bh = 1.1d0
    ch = 2.2d0;  dh = 2.2d0
    eh = 3.3d0;  fh = 3.3d0

    allocate(ad2(NB), bd2(NB), cd2(NB), dd2(NB), ed2(NB), fd2(NB))
    allocate(out_r4(NB), out_hi(NB), out_lo(NB))
    ad2 = ah;  bd2 = bh;  cd2 = ch;  dd2 = dh;  ed2 = eh;  fd2 = fh

    is = cudaEventCreate(t0);  is = cudaEventCreate(t1)

    ! real(4) naive
    call kern_bench_dot3_r4<<<blk,thr>>>(NB, NI, ad2, bd2, cd2, dd2, ed2, fd2, out_r4)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_dot3_r4<<<blk,thr>>>(NB, NI, ad2, bd2, cd2, dd2, ed2, fd2, out_r4)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r4 = dble(elapsed)

    ! fltflt dot3
    call kern_bench_dot3_ff<<<blk,thr>>>(NB, NI, ad2, bd2, cd2, dd2, ed2, fd2, out_hi, out_lo)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_dot3_ff<<<blk,thr>>>(NB, NI, ad2, bd2, cd2, dd2, ed2, fd2, out_hi, out_lo)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_ff = dble(elapsed)

    is = cudaEventDestroy(t0);  is = cudaEventDestroy(t1)

    g_r4 = out_r4(1);  g_hi = out_hi(1);  g_lo = out_lo(1)
    mops_r4 = dble(NB)*dble(NI) / (ms_r4 * 1.0d3)
    mops_ff = dble(NB)*dble(NI) / (ms_ff * 1.0d3)

    write(*,'("--- benchmark (N=2^20, NITER=100, chained dot3) ---")')
    write(*,'("  (guard r4=",F8.4,"  ff=",F8.4,"+",E9.2,")")') g_r4, g_hi, g_lo
    write(*,'("  real(4) : ",F8.2," ms  | ",F8.0," MOPS")') ms_r4, mops_r4
    write(*,'("  fltflt  : ",F8.2," ms  | ",F8.0," MOPS")') ms_ff, mops_ff
    write(*,'("  ff/r4 slowdown : ",F5.2,"x")') ms_ff / ms_r4

    write(csv_unit, '("bench,real(4),",F12.4,",",F12.1)') ms_r4, mops_r4
    write(csv_unit, '("bench,fltflt,",F12.4,",",F12.1)')  ms_ff, mops_ff

    deallocate(ah, bh, ch, dh, eh, fh)
    deallocate(ad2, bd2, cd2, dd2, ed2, fd2, out_r4, out_hi, out_lo)
  end subroutine run_bench

end program test_dot
