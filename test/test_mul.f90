! test_mul.f90  —  fltflt multiplication: accuracy + throughput
!
! Build:  cd test && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_mul
!        (writes test/mul_results.csv; plot: cmake --build . --target plot_mul)
!
! Accuracy: NCASE scalar cases, real(8) inputs.
! Benchmark: chained x = x * b for real(4), real(8), fltflt.

module test_mul_kern
  use cudafor
  use fltflt
  implicit none

  integer, parameter :: NCASE   = 4
  integer, parameter :: N_BENCH = 2**20
  integer, parameter :: NITER   = 100
  integer, parameter :: NBLK    = 256

contains

  ! ---- accuracy kernel ------------------------------------------------
  attributes(global) subroutine kern_mul(a_d, b_d, n, res_r4, res_hi, res_lo)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: res_r4(n), res_hi(n), res_lo(n)
    type(fltflt) :: af, bf, cf
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    af = fltflt_init(a_d(i));  bf = fltflt_init(b_d(i))
    res_r4(i) = real(a_d(i), 4) * real(b_d(i), 4)
    cf = af * bf
    res_hi(i) = cf%hi;  res_lo(i) = cf%lo
  end subroutine kern_mul

  ! ---- benchmark kernels: chained x = x * b --------------------------
  attributes(global) subroutine kern_bench_mul_r4(n, niter, a_d, b_d, out_d)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n)
    real(4), intent(out), device  :: out_d(n)
    real(4) :: x, b
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = real(a_d(i), 4);  b = real(b_d(i), 4)
    do j = 1, niter
      x = x * b
    end do
    out_d(i) = x
  end subroutine kern_bench_mul_r4

  attributes(global) subroutine kern_bench_mul_r8(n, niter, a_d, b_d, out_d)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n)
    real(8), intent(out), device  :: out_d(n)
    real(8) :: x, b
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = a_d(i);  b = b_d(i)
    do j = 1, niter
      x = x * b
    end do
    out_d(i) = x
  end subroutine kern_bench_mul_r8

  attributes(global) subroutine kern_bench_mul_ff(n, niter, a_d, b_d, out_hi, out_lo)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n)
    real(4), intent(out), device  :: out_hi(n), out_lo(n)
    type(fltflt) :: xf, bf
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    xf = fltflt_init(a_d(i));  bf = fltflt_init(b_d(i))
    do j = 1, niter
      xf = xf * bf
    end do
    out_hi(i) = xf%hi;  out_lo(i) = xf%lo
  end subroutine kern_bench_mul_ff

end module test_mul_kern


program test_mul
  use cudafor
  use fltflt
  use test_mul_kern
  implicit none

  integer, parameter :: N = NCASE
  real(8), parameter :: FF_THRESH = 1.0d-12
  character(len=16), parameter :: CLABELS(N) = &
    [ character(len=16) :: "mul[1.1*1.1]", "mul[bench]", "mul[near1]", "mul[pi*e]" ]

  real(8) :: a_r8(N), b_r8(N), ref(N)
  real(8), device, allocatable :: a_d(:), b_d(:)
  real(4), device, allocatable :: res_r4(:), res_hi(:), res_lo(:)
  real(4) :: h_r4(N), h_hi(N), h_lo(N)
  real(8) :: ff_val, err_r4, err_ff
  integer :: i, istat, nfail, csv_unit

  csv_unit = 11
  open(unit=csv_unit, file='mul_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,val1,val2'

  ! (1) 1.1 * 1.1 = 1.21 — exact product needs >24 mantissa bits
  a_r8(1) = 1.1d0;                  b_r8(1) = 1.1d0
  ! (2) general: benchmark operands
  a_r8(2) = 1.23456789d0;           b_r8(2) = 9.99999870d-1
  ! (3) near-unity: (1+2^-24)^2 = 1 + 2^-23 + 2^-48
  a_r8(3) = 1.0d0 + 2.0d0**(-24);  b_r8(3) = 1.0d0 + 2.0d0**(-24)
  ! (4) transcendental approximations: pi * e
  a_r8(4) = 3.14159265358979d0;     b_r8(4) = 2.71828182845904d0
  ref = a_r8 * b_r8

  allocate(a_d(N), b_d(N), res_r4(N), res_hi(N), res_lo(N))
  a_d = a_r8;  b_d = b_r8
  call kern_mul<<<1, N>>>(a_d, b_d, N, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()
  h_r4 = res_r4;  h_hi = res_hi;  h_lo = res_lo
  deallocate(a_d, b_d, res_r4, res_hi, res_lo)

  write(*,*) "=== test_mul: fltflt multiplication (real(8) inputs) ==="
  nfail = 0
  do i = 1, N
    ff_val = real(h_hi(i), 8) + real(h_lo(i), 8)
    err_r4 = abs((real(h_r4(i), 8) - ref(i)) / ref(i))
    err_ff = abs((ff_val            - ref(i)) / ref(i))
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
    real(8), allocatable :: a_h(:), b_h(:)
    real(8), device, allocatable :: ad(:), bd(:), out_r8(:)
    real(4), device, allocatable :: out_r4(:), out_hi(:), out_lo(:)
    real(8) :: ms_r4, ms_r8, ms_ff, mops_r4, mops_r8, mops_ff
    real(4) :: elapsed
    real(4) :: g_r4, g_r8_r4, g_hi, g_lo
    real(8) :: g_r8
    type(cudaEvent) :: t0, t1
    type(dim3) :: blk, thr
    integer :: nb_blk, is

    thr    = dim3(NBLK, 1, 1)
    nb_blk = (NB + NBLK - 1) / NBLK
    blk    = dim3(nb_blk, 1, 1)

    allocate(a_h(NB), b_h(NB))
    a_h = 1.23456789d0;  b_h = 0.99999987d0
    allocate(ad(NB), bd(NB), out_r4(NB), out_r8(NB), out_hi(NB), out_lo(NB))
    ad = a_h;  bd = b_h

    is = cudaEventCreate(t0);  is = cudaEventCreate(t1)

    ! real(4)
    call kern_bench_mul_r4<<<blk,thr>>>(NB, NI, ad, bd, out_r4)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_mul_r4<<<blk,thr>>>(NB, NI, ad, bd, out_r4)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r4 = dble(elapsed)

    ! real(8)
    call kern_bench_mul_r8<<<blk,thr>>>(NB, NI, ad, bd, out_r8)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_mul_r8<<<blk,thr>>>(NB, NI, ad, bd, out_r8)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r8 = dble(elapsed)

    ! fltflt
    call kern_bench_mul_ff<<<blk,thr>>>(NB, NI, ad, bd, out_hi, out_lo)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_mul_ff<<<blk,thr>>>(NB, NI, ad, bd, out_hi, out_lo)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_ff = dble(elapsed)

    is = cudaEventDestroy(t0);  is = cudaEventDestroy(t1)

    g_r4 = out_r4(1);  g_r8 = out_r8(1);  g_hi = out_hi(1);  g_lo = out_lo(1)
    g_r8_r4 = real(g_r8, 4)
    mops_r4 = dble(NB)*dble(NI) / (ms_r4 * 1.0d3)
    mops_r8 = dble(NB)*dble(NI) / (ms_r8 * 1.0d3)
    mops_ff = dble(NB)*dble(NI) / (ms_ff * 1.0d3)

    write(*,'("--- benchmark (N=2^20, NITER=100, chained mul) ---")')
    write(*,'("  (guard r4=",F8.4,"  r8=",F8.4,"  ff=",F8.4,"+",E9.2,")")') &
      g_r4, g_r8_r4, g_hi, g_lo
    write(*,'("  real(4) : ",F8.2," ms  | ",F8.0," MOPS")') ms_r4, mops_r4
    write(*,'("  real(8) : ",F8.2," ms  | ",F8.0," MOPS")') ms_r8, mops_r8
    write(*,'("  fltflt  : ",F8.2," ms  | ",F8.0," MOPS")') ms_ff, mops_ff
    write(*,'("  ff/r4 slowdown : ",F5.2,"x")') ms_ff / ms_r4

    write(csv_unit, '("bench,real(4),",F12.4,",",F12.1)') ms_r4, mops_r4
    write(csv_unit, '("bench,real(8),",F12.4,",",F12.1)') ms_r8, mops_r8
    write(csv_unit, '("bench,fltflt,",F12.4,",",F12.1)')  ms_ff, mops_ff

    deallocate(a_h, b_h, ad, bd, out_r4, out_r8, out_hi, out_lo)
  end subroutine run_bench

end program test_mul
