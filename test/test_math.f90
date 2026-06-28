! test_math.f90  —  fltflt math functions: sqrt, norm3d, square, recip,
!                    rounding, fmod, shfl_xor: accuracy + throughput
!
! Build:  cd ouxmix && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_math
!        (writes test/math_results.csv)

module test_math_kern
  use cudafor
  use cudadevice
  use fltflt
  implicit none
  private
  public :: NCASE, N_BENCH, NITER, NBLK, WARP_SIZE
  public :: kern_math, kern_shfl_xor, kern_bench_sqrt_r4, kern_bench_sqrt_ff

  integer, parameter :: NCASE     = 8
  integer, parameter :: N_BENCH   = 2**20
  integer, parameter :: NITER     = 100
  integer, parameter :: NBLK      = 256
  integer, parameter :: WARP_SIZE = 32

contains

  ! ---- accuracy kernel for cases 1-7 (scalar math functions) -------------
  attributes(global) subroutine kern_math(a_d, b_d, n, res_hi, res_lo, res_r4)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: res_hi(n,7), res_lo(n,7), res_r4(n,7)
    real(4) :: ar, br
    type(fltflt) :: af, bf, rf
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    ar = real(a_d(i), 4);  br = real(b_d(i), 4)
    af = fltflt_init(a_d(i));  bf = fltflt_init(b_d(i))

    ! (1) fltflt_sqrt
    res_r4(i,1) = sqrt(ar)
    rf = fltflt_sqrt(af)
    res_hi(i,1) = rf%hi;  res_lo(i,1) = rf%lo

    ! (2) fltflt_sqrt_fast
    res_r4(i,2) = sqrt(ar)
    rf = fltflt_sqrt_fast(af)
    res_hi(i,2) = rf%hi;  res_lo(i,2) = rf%lo

    ! (3) fltflt_norm3d: a is dx, b is dy, dz=0 (3-4-5 right triangle uses different a/b)
    res_r4(i,3) = sqrt(ar*ar + br*br)
    rf = fltflt_norm3d(af, bf, fltflt_init(0.0_4))
    res_hi(i,3) = rf%hi;  res_lo(i,3) = rf%lo

    ! (4) fltflt_square(a) vs a*a
    res_r4(i,4) = ar * ar
    rf = fltflt_square(af)
    res_hi(i,4) = rf%hi;  res_lo(i,4) = rf%lo

    ! (5) fltflt_recip(a) vs 1/a
    res_r4(i,5) = 1.0 / ar
    rf = fltflt_recip(af)
    res_hi(i,5) = rf%hi;  res_lo(i,5) = rf%lo

    ! (6) fltflt_floor / round_toward_zero / round_to_nearest: use a%hi=br, a%lo=ar-br
    !     Encodes three rounding ops for one element; use a as the value to round.
    res_r4(i,6) = real(floor(ar), 4)
    rf = fltflt_floor(af)
    res_hi(i,6) = rf%hi;  res_lo(i,6) = rf%lo

    ! (7) fltflt_fmod(a, b) vs mod(a, b) in r4
    res_r4(i,7) = mod(ar, br)
    rf = fltflt_fmod(af, br)
    res_hi(i,7) = rf%hi;  res_lo(i,7) = rf%lo
  end subroutine kern_math

  ! ---- case 8: shfl_xor butterfly reduction within one warp ---------------
  ! Each lane holds its index value; reduce to the sum in lane 0 using XOR.
  attributes(global) subroutine kern_shfl_xor(out_hi, out_lo)
    real(4), intent(out), device :: out_hi(1), out_lo(1)
    type(fltflt) :: val, other
    integer :: lane, mask
    lane = mod(threadIdx%x - 1, WARP_SIZE)
    val = fltflt_init(real(lane + 1, 4))  ! lanes hold 1..32
    mask = WARP_SIZE / 2
    do while (mask >= 1)
      other = fltflt_shfl_xor(val, mask)
      val   = val + other
      mask  = mask / 2
    end do
    if (lane == 0) then
      out_hi(1) = val%hi;  out_lo(1) = val%lo
    end if
  end subroutine kern_shfl_xor

  ! ---- benchmark: chained x = fltflt_sqrt(x + delta) ----------------------
  attributes(global) subroutine kern_bench_sqrt_r4(n, niter, a_d, out_d)
    integer, intent(in), value   :: n, niter
    real(8), intent(in),  device :: a_d(n)
    real(4), intent(out), device :: out_d(n)
    real(4) :: x
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = real(a_d(i), 4)
    do j = 1, niter
      x = sqrt(x)
    end do
    out_d(i) = x
  end subroutine kern_bench_sqrt_r4

  attributes(global) subroutine kern_bench_sqrt_ff(n, niter, a_d, out_hi, out_lo)
    integer, intent(in), value   :: n, niter
    real(8), intent(in),  device :: a_d(n)
    real(4), intent(out), device :: out_hi(n), out_lo(n)
    type(fltflt) :: x
    real(4) :: delta
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = fltflt_init(a_d(i))
    delta = 1.0e-6
    do j = 1, niter
      x = fltflt_sqrt(x + delta)
    end do
    out_hi(i) = x%hi;  out_lo(i) = x%lo
  end subroutine kern_bench_sqrt_ff

end module test_math_kern


program test_math
  use cudafor
  use fltflt
  use test_math_kern
  implicit none

  integer, parameter :: N = NCASE
  ! Thresholds per case:
  ! sqrt/norm3d/square/recip: ~1e-14 (Newton-step full precision)
  ! sqrt_fast: ~1e-6 (approximate, ~45 bits)
  ! floor: exact integer result → 0 error
  ! fmod: ~1e-12 (exact quotient residual)
  ! shfl_xor: ~1e-12 (exact warp sum)
  real(8), parameter :: FF_THRESH    = 1.0d-12
  real(8), parameter :: FF_FAST_THRESH = 1.0d-6
  character(len=24), parameter :: CLABELS(N) = [ character(len=24) :: &
    "sqrt[newton]",       &
    "sqrt_fast[rsqrt]",   &
    "norm3d[3-4-5]",      &
    "square[1.1^2]",      &
    "recip[1/3]",         &
    "floor[2.0-eps]",     &
    "fmod[10.5 mod 3]",   &
    "shfl_xor[warp_sum]"  ]

  integer, parameter :: N7 = 7  ! cases 1-7 use scalar kernel
  real(8) :: a_r8(N7), b_r8(N7), ref(N)
  real(8) :: thresh(N)
  real(8), device, allocatable :: ad(:), bd(:)
  real(4), device, allocatable :: res_hi(:,:), res_lo(:,:), res_r4(:,:)
  real(4), device, allocatable :: shfl_hi(:), shfl_lo(:)
  real(4) :: h_hi(N7,7), h_lo(N7,7), h_r4(N7,7)
  real(4) :: shfl_h, shfl_l
  real(8) :: ff_val, err_r4, err_ff
  integer :: i, istat, nfail, csv_unit

  csv_unit = 11
  open(unit=csv_unit, file='math_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,val1,val2'

  ! Case 1: sqrt(2)
  a_r8(1) = 2.0d0;   b_r8(1) = 0.0d0
  ! Case 2: sqrt_fast(2) — same input, looser threshold
  a_r8(2) = 2.0d0;   b_r8(2) = 0.0d0
  ! Case 3: norm3d(3, 4, 0) → 5
  a_r8(3) = 3.0d0;   b_r8(3) = 4.0d0
  ! Case 4: square(1.1)
  a_r8(4) = 1.1d0;   b_r8(4) = 0.0d0
  ! Case 5: recip(3) = 1/3
  a_r8(5) = 3.0d0;   b_r8(5) = 0.0d0
  ! Case 6: floor(2.0 - eps): a%hi=2.0, a%lo=-eps → floor should be 1.0
  !         Use fltflt_init(2.0-eps_r8) = {2.0, -eps}
  a_r8(6) = 2.0d0 - dble(2.0**(-24))  ! 2.0 minus one r4 ULP
  b_r8(6) = 0.0d0
  ! Case 7: fmod(10.5, 3.0) = 1.5
  a_r8(7) = 10.5d0;  b_r8(7) = 3.0d0

  ref(1) = sqrt(2.0d0)
  ref(2) = sqrt(2.0d0)
  ref(3) = 5.0d0
  ref(4) = 1.1d0 * 1.1d0
  ref(5) = 1.0d0 / 3.0d0
  ref(6) = 1.0d0   ! floor(2.0-eps) = 1
  ref(7) = mod(10.5d0, 3.0d0)
  ref(8) = 528.0d0  ! sum 1+2+...+32 = 32*33/2 = 528

  thresh(1) = FF_THRESH;   thresh(2) = FF_FAST_THRESH
  thresh(3) = FF_THRESH;   thresh(4) = FF_THRESH
  thresh(5) = FF_THRESH;   thresh(6) = FF_THRESH
  thresh(7) = FF_THRESH;   thresh(8) = FF_THRESH

  allocate(ad(N7), bd(N7))
  allocate(res_hi(N7,7), res_lo(N7,7), res_r4(N7,7))
  allocate(shfl_hi(1), shfl_lo(1))
  ad = a_r8;  bd = b_r8
  call kern_math<<<1, N7>>>(ad, bd, N7, res_hi, res_lo, res_r4)
  call kern_shfl_xor<<<1, WARP_SIZE>>>(shfl_hi, shfl_lo)
  istat = cudaDeviceSynchronize()
  h_hi = res_hi;  h_lo = res_lo;  h_r4 = res_r4
  shfl_h = shfl_hi(1);  shfl_l = shfl_lo(1)
  deallocate(ad, bd, res_hi, res_lo, res_r4, shfl_hi, shfl_lo)

  write(*,*) "=== test_math: sqrt, norm3d, square, recip, floor, fmod, shfl_xor ==="
  nfail = 0

  ! Cases 1-7: scalar math
  do i = 1, 7
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

  ! Case 8: shfl_xor warp reduction
  ff_val = real(shfl_h, 8) + real(shfl_l, 8)
  err_r4 = abs((real(shfl_h, 8) - ref(8)) / ref(8))
  err_ff = abs((ff_val           - ref(8)) / ref(8))
  if (err_ff > thresh(8)) nfail = nfail + 1
  write(*,'("  case 8: r4_err=",ES9.2,"  ff_err=",ES9.2,"  ",A)') &
    err_r4, err_ff, merge("PASS", "FAIL", err_ff <= thresh(8))
  write(csv_unit, '("acc,",A,",",ES12.5,",",ES12.5)') &
    trim(adjustl(CLABELS(8))), err_r4, err_ff

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
    real(8), allocatable :: ah(:)
    real(8), device, allocatable :: ad2(:)
    real(4), device, allocatable :: out_r4(:), out_hi(:), out_lo(:)
    real(8) :: ms_r4, ms_ff, mops_r4, mops_ff
    real(4) :: elapsed, g_r4, g_hi, g_lo
    type(cudaEvent) :: t0, t1
    type(dim3) :: blk, thr
    integer :: nb_blk, is

    thr    = dim3(NBLK, 1, 1)
    nb_blk = (NB + NBLK - 1) / NBLK
    blk    = dim3(nb_blk, 1, 1)

    allocate(ah(NB))
    ah = 1.23456789d0
    allocate(ad2(NB), out_r4(NB), out_hi(NB), out_lo(NB))
    ad2 = ah

    is = cudaEventCreate(t0);  is = cudaEventCreate(t1)

    call kern_bench_sqrt_r4<<<blk,thr>>>(NB, NI, ad2, out_r4)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_sqrt_r4<<<blk,thr>>>(NB, NI, ad2, out_r4)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r4 = dble(elapsed)

    call kern_bench_sqrt_ff<<<blk,thr>>>(NB, NI, ad2, out_hi, out_lo)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_sqrt_ff<<<blk,thr>>>(NB, NI, ad2, out_hi, out_lo)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_ff = dble(elapsed)

    is = cudaEventDestroy(t0);  is = cudaEventDestroy(t1)

    g_r4 = out_r4(1);  g_hi = out_hi(1);  g_lo = out_lo(1)
    mops_r4 = dble(NB)*dble(NI) / (ms_r4 * 1.0d3)
    mops_ff = dble(NB)*dble(NI) / (ms_ff * 1.0d3)

    write(*,'("--- benchmark (N=2^20, NITER=100, chained sqrt) ---")')
    write(*,'("  (guard r4=",F8.6,"  ff=",F8.6,"+",E9.2,")")') g_r4, g_hi, g_lo
    write(*,'("  real(4) sqrt : ",F8.2," ms  | ",F8.0," MOPS")') ms_r4, mops_r4
    write(*,'("  fltflt_sqrt  : ",F8.2," ms  | ",F8.0," MOPS")') ms_ff, mops_ff
    write(*,'("  ff/r4 slowdown : ",F5.2,"x")') ms_ff / ms_r4

    write(csv_unit, '("bench,real(4),",F12.4,",",F12.1)') ms_r4, mops_r4
    write(csv_unit, '("bench,fltflt,",F12.4,",",F12.1)')  ms_ff, mops_ff

    deallocate(ah, ad2, out_r4, out_hi, out_lo)
  end subroutine run_bench

end program test_math
