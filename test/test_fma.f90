! test_fma.f90  —  fltflt_fma, fltflt_fma_approx, fltflt_add_same_sign,
!                   fltflt_add3, fltflt_add4, fltflt_add5: accuracy + throughput
!
! Build:  cd ouxmix && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_fma
!        (writes test/fma_results.csv)

module test_fma_kern
  use cudafor
  use cudadevice
  use fltflt
  implicit none
  private
  public :: NCASE, N_BENCH, NITER, NBLK
  public :: kern_fma, kern_bench_fma_r4, kern_bench_fma_ff

  integer, parameter :: NCASE   = 8
  integer, parameter :: N_BENCH = 2**20
  integer, parameter :: NITER   = 100
  integer, parameter :: NBLK    = 256

contains

  ! ---- accuracy kernel: compute all fma variants + add_same_sign + addN ---
  ! Input: 5 real(8) scalars (a,b,c,d,e) per element.
  ! Output: res_hi/res_lo for 8 result slots.
  attributes(global) subroutine kern_fma(a_d, b_d, c_d, d_d, e_d, n, &
                                          res_r4, res_hi, res_lo)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n), b_d(n), c_d(n), d_d(n), e_d(n)
    real(4), intent(out), device :: res_r4(n,8), res_hi(n,8), res_lo(n,8)
    real(4) :: ar, br, cr, dr, er
    type(fltflt) :: af, bf, cf, df, ef, rf
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    ar = real(a_d(i), 4);  br = real(b_d(i), 4)
    cr = real(c_d(i), 4);  dr = real(d_d(i), 4)
    er = real(e_d(i), 4)
    af = fltflt_init(a_d(i));  bf = fltflt_init(b_d(i))
    cf = fltflt_init(c_d(i));  df = fltflt_init(d_d(i))
    ef = fltflt_init(e_d(i))

    ! (1) fltflt_fma(ff, ff, ff): a*b + c
    res_r4(i,1) = ar*br + cr
    rf = fltflt_fma(af, bf, cf)
    res_hi(i,1) = rf%hi;  res_lo(i,1) = rf%lo

    ! (2) fltflt_fma(ff, r4, ff): a*b + c  (b as scalar)
    res_r4(i,2) = ar*br + cr
    rf = fltflt_fma(af, br, cf)
    res_hi(i,2) = rf%hi;  res_lo(i,2) = rf%lo

    ! (3) fltflt_fma(r4, ff, ff): a*b + c  (a as scalar)
    res_r4(i,3) = ar*br + cr
    rf = fltflt_fma(ar, bf, cf)
    res_hi(i,3) = rf%hi;  res_lo(i,3) = rf%lo

    ! (4) fltflt_fma_approx(ff, ff, ff): a*b + c  (approx, <=1 ULP worse)
    res_r4(i,4) = ar*br + cr
    rf = fltflt_fma_approx(af, bf, cf)
    res_hi(i,4) = rf%hi;  res_lo(i,4) = rf%lo

    ! (5) fltflt_add_same_sign(ff, ff): a + b, both same sign
    res_r4(i,5) = ar + br
    rf = fltflt_add_same_sign(af, bf)
    res_hi(i,5) = rf%hi;  res_lo(i,5) = rf%lo

    ! (6) fltflt_add3(ff, ff, ff): a + b + c
    res_r4(i,6) = ar + br + cr
    rf = fltflt_add3(af, bf, cf)
    res_hi(i,6) = rf%hi;  res_lo(i,6) = rf%lo

    ! (7) fltflt_add4(ff, ff, ff, ff): a + b + c + d
    res_r4(i,7) = ar + br + cr + dr
    rf = fltflt_add4(af, bf, cf, df)
    res_hi(i,7) = rf%hi;  res_lo(i,7) = rf%lo

    ! (8) fltflt_add5(ff, ff, ff, ff, ff): a + b + c + d + e
    res_r4(i,8) = ar + br + cr + dr + er
    rf = fltflt_add5(af, bf, cf, df, ef)
    res_hi(i,8) = rf%hi;  res_lo(i,8) = rf%lo
  end subroutine kern_fma

  ! ---- benchmark: chained x = fltflt_fma(a, b, x) vs x = x + a*b ----------
  attributes(global) subroutine kern_bench_fma_r4(n, niter, a_d, b_d, out_d)
    integer, intent(in), value   :: n, niter
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: out_d(n)
    real(4) :: x, a, b
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = 0.0;  a = real(a_d(i), 4);  b = real(b_d(i), 4)
    do j = 1, niter
      x = x + a * b
    end do
    out_d(i) = x
  end subroutine kern_bench_fma_r4

  attributes(global) subroutine kern_bench_fma_ff(n, niter, a_d, b_d, out_hi, out_lo)
    integer, intent(in), value   :: n, niter
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: out_hi(n), out_lo(n)
    type(fltflt) :: x, af, bf
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x%hi = 0.0;  x%lo = 0.0
    af = fltflt_init(a_d(i));  bf = fltflt_init(b_d(i))
    do j = 1, niter
      x = fltflt_fma(af, bf, x)
    end do
    out_hi(i) = x%hi;  out_lo(i) = x%lo
  end subroutine kern_bench_fma_ff

end module test_fma_kern


program test_fma
  use cudafor
  use fltflt
  use test_fma_kern
  implicit none

  integer, parameter :: N = NCASE
  real(8), parameter :: FF_THRESH  = 1.0d-12
  real(8), parameter :: FF_THRESH_APPROX = 1.0d-6  ! fma_approx is ~1 ULP less precise
  character(len=24), parameter :: CLABELS(N) = [ character(len=24) :: &
    "fma[ff*ff+ff cancel]",  &
    "fma[ff*r4+ff]",         &
    "fma[r4*ff+ff]",         &
    "fma_approx[ff*ff+ff]",  &
    "add_same_sign[ff+ff]",  &
    "add3[ff+ff+ff]",        &
    "add4[ff+ff+ff+ff]",     &
    "add5[ff*5]"             ]

  real(8) :: a_r8(N), b_r8(N), c_r8(N), d_r8(N), e_r8(N), ref(N)
  real(8) :: thresh(N)
  real(8), device, allocatable :: ad(:), bd(:), cd(:), dd(:), ed(:)
  real(4), device, allocatable :: res_r4(:,:), res_hi(:,:), res_lo(:,:)
  real(4) :: h_r4(N,8), h_hi(N,8), h_lo(N,8)
  real(8) :: ff_val, err_r4, err_ff
  integer :: i, istat, nfail, csv_unit

  csv_unit = 11
  open(unit=csv_unit, file='fma_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,val1,val2'

  ! (1) fma(ff*ff+ff): (1+2^-13)^2 - (1+2^-12) = 2^-26
  !     Float32: (1+2^-13)^2 rounds to 1+2^-12 (dropping 2^-26 below float32 ULP),
  !     so r4 gives 0 (100% error). fltflt_fma recovers the exact 2^-26 via TwoProdFMA.
  a_r8(1) = 1.0d0 + 2.0d0**(-13)  ! exactly representable in float32
  b_r8(1) = 1.0d0 + 2.0d0**(-13)
  c_r8(1) = -(1.0d0 + 2.0d0**(-12))
  d_r8(1) = 0.0d0;  e_r8(1) = 0.0d0

  ! (2) fma(ff*r4+ff): b used as r4 scalar; same cancellation as (1).
  !     All inputs exactly representable in float32 → float64 ref is exact.
  a_r8(2) = 1.0d0 + 2.0d0**(-13);  b_r8(2) = 1.0d0 + 2.0d0**(-13)
  c_r8(2) = -(1.0d0 + 2.0d0**(-12));  d_r8(2) = 0.0d0;  e_r8(2) = 0.0d0

  ! (3) fma(r4*ff+ff): a used as r4 scalar; same cancellation, product commutes.
  a_r8(3) = 1.0d0 + 2.0d0**(-13);  b_r8(3) = 1.0d0 + 2.0d0**(-13)
  c_r8(3) = -(1.0d0 + 2.0d0**(-12));  d_r8(3) = 0.0d0;  e_r8(3) = 0.0d0

  ! (4) fma_approx(ff*ff+ff): well-conditioned input, should match fma to ~1e-6
  a_r8(4) = 1.1d0;  b_r8(4) = 1.1d0;  c_r8(4) = 2.2d0
  d_r8(4) = 0.0d0;  e_r8(4) = 0.0d0

  ! (5) add_same_sign(ff,ff): both positive, result matches fltflt_add
  a_r8(5) = 1.0d0 + 2.0d0**(-13)
  b_r8(5) = 2.0d0**(-14)
  c_r8(5) = 0.0d0;  d_r8(5) = 0.0d0;  e_r8(5) = 0.0d0

  ! (6) add3: three values with small increments that don't accumulate in r4
  a_r8(6) = 1.0d0;  b_r8(6) = 2.0d0**(-25);  c_r8(6) = 2.0d0**(-26)
  d_r8(6) = 0.0d0;  e_r8(6) = 0.0d0

  ! (7) add4: four values
  a_r8(7) = 1.0d0;  b_r8(7) = 2.0d0**(-25)
  c_r8(7) = 2.0d0**(-26);  d_r8(7) = 2.0d0**(-27)
  e_r8(7) = 0.0d0

  ! (8) add5: five values
  a_r8(8) = 1.0d0;  b_r8(8) = 2.0d0**(-25)
  c_r8(8) = 2.0d0**(-26);  d_r8(8) = 2.0d0**(-27);  e_r8(8) = 2.0d0**(-28)

  ! Reference for each case: the appropriate operation in r8
  ref(1) = a_r8(1)*b_r8(1) + c_r8(1)        ! fma
  ref(2) = a_r8(2)*b_r8(2) + c_r8(2)        ! fma (ff*r4)
  ref(3) = a_r8(3)*b_r8(3) + c_r8(3)        ! fma (r4*ff)
  ref(4) = a_r8(4)*b_r8(4) + c_r8(4)        ! fma_approx
  ref(5) = a_r8(5) + b_r8(5)                 ! add_same_sign
  ref(6) = a_r8(6) + b_r8(6) + c_r8(6)      ! add3
  ref(7) = a_r8(7) + b_r8(7) + c_r8(7) + d_r8(7)   ! add4
  ref(8) = a_r8(8) + b_r8(8) + c_r8(8) + d_r8(8) + e_r8(8)  ! add5

  thresh(1:3) = FF_THRESH;  thresh(4) = FF_THRESH_APPROX
  thresh(5:8) = FF_THRESH

  allocate(ad(N), bd(N), cd(N), dd(N), ed(N))
  allocate(res_r4(N,8), res_hi(N,8), res_lo(N,8))
  ad = a_r8;  bd = b_r8;  cd = c_r8;  dd = d_r8;  ed = e_r8
  call kern_fma<<<1, N>>>(ad, bd, cd, dd, ed, N, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()
  h_r4 = res_r4;  h_hi = res_hi;  h_lo = res_lo
  deallocate(ad, bd, cd, dd, ed, res_r4, res_hi, res_lo)

  write(*,*) "=== test_fma: fltflt_fma + fltflt_fma_approx + add_same_sign + addN ==="
  nfail = 0
  do i = 1, N
    ff_val = real(h_hi(i,i), 8) + real(h_lo(i,i), 8)
    if (abs(ref(i)) > 0.0d0) then
      err_r4 = abs((real(h_r4(i,i), 8) - ref(i)) / ref(i))
      err_ff = abs((ff_val             - ref(i)) / ref(i))
    else
      err_r4 = abs(real(h_r4(i,i), 8))
      err_ff = abs(ff_val)
    end if
    if (err_ff > thresh(i)) nfail = nfail + 1
    write(*,'("  case",I2,": r4_err=",ES9.2,"  ff_err=",ES9.2,"  ",A)') &
      i, err_r4, err_ff, merge("PASS", "FAIL", err_ff <= thresh(i))
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
    real(8), allocatable :: ah(:), bh(:)
    real(8), device, allocatable :: ad2(:), bd2(:)
    real(4), device, allocatable :: out_r4(:), out_hi(:), out_lo(:)
    real(8) :: ms_r4, ms_ff, mops_r4, mops_ff
    real(4) :: elapsed, g_r4, g_hi, g_lo
    type(cudaEvent) :: t0, t1
    type(dim3) :: blk, thr
    integer :: nb_blk, is

    thr    = dim3(NBLK, 1, 1)
    nb_blk = (NB + NBLK - 1) / NBLK
    blk    = dim3(nb_blk, 1, 1)

    allocate(ah(NB), bh(NB))
    ah = 1.23456789d0;  bh = 9.87654321d-1
    allocate(ad2(NB), bd2(NB), out_r4(NB), out_hi(NB), out_lo(NB))
    ad2 = ah;  bd2 = bh

    is = cudaEventCreate(t0);  is = cudaEventCreate(t1)

    call kern_bench_fma_r4<<<blk,thr>>>(NB, NI, ad2, bd2, out_r4)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_fma_r4<<<blk,thr>>>(NB, NI, ad2, bd2, out_r4)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r4 = dble(elapsed)

    call kern_bench_fma_ff<<<blk,thr>>>(NB, NI, ad2, bd2, out_hi, out_lo)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_fma_ff<<<blk,thr>>>(NB, NI, ad2, bd2, out_hi, out_lo)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_ff = dble(elapsed)

    is = cudaEventDestroy(t0);  is = cudaEventDestroy(t1)

    g_r4 = out_r4(1);  g_hi = out_hi(1);  g_lo = out_lo(1)
    mops_r4 = dble(NB)*dble(NI) / (ms_r4 * 1.0d3)
    mops_ff = dble(NB)*dble(NI) / (ms_ff * 1.0d3)

    write(*,'("--- benchmark (N=2^20, NITER=100, chained fma) ---")')
    write(*,'("  (guard r4=",F8.4,"  ff=",F8.4,"+",E9.2,")")') g_r4, g_hi, g_lo
    write(*,'("  real(4) naive : ",F8.2," ms  | ",F8.0," MOPS")') ms_r4, mops_r4
    write(*,'("  fltflt_fma    : ",F8.2," ms  | ",F8.0," MOPS")') ms_ff, mops_ff
    write(*,'("  ff/r4 slowdown : ",F5.2,"x")') ms_ff / ms_r4

    write(csv_unit, '("bench,real(4),",F12.4,",",F12.1)') ms_r4, mops_r4
    write(csv_unit, '("bench,fltflt,",F12.4,",",F12.1)')  ms_ff, mops_ff

    deallocate(ah, bh, ad2, bd2, out_r4, out_hi, out_lo)
  end subroutine run_bench

end program test_fma
