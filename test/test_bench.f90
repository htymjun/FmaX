! test_bench.f90  —  fltflt GPU throughput benchmark
!
! Build:  cd test && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_bench
!        (writes ffp_results.csv in test/; then: cmake --build . --target plot)
!
! Times a chained multiply-add loop (real(8) vs fltflt) using cudaEvent timers.
! N=2^20 elements, NITER=100 inner iterations per thread.

module test_bench_kern
  use cudafor
  use fltflt
  implicit none

  integer, parameter :: N_BENCH = 2**20
  integer, parameter :: NITER   = 100
  integer, parameter :: NBLK    = 256

contains

  attributes(global) subroutine kern_bench_r8(n, niter, a_d, b_d, out_d)
    integer, intent(in), value   :: n, niter
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(8), intent(out), device :: out_d(n)
    integer :: i, j
    real(8) :: x
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = a_d(i)
    do j = 1, niter
      x = x * b_d(i) + a_d(i)
    end do
    out_d(i) = x
  end subroutine kern_bench_r8

  attributes(global) subroutine kern_bench_ff(n, niter, a_d, b_d, out_hi, out_lo)
    integer, intent(in), value   :: n, niter
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: out_hi(n), out_lo(n)
    type(fltflt) :: xf, af, bf
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    af = fltflt_init(a_d(i));  bf = fltflt_init(b_d(i));  xf = af
    do j = 1, niter
      xf = xf * bf + af
    end do
    out_hi(i) = xf%hi;  out_lo(i) = xf%lo
  end subroutine kern_bench_ff

end module test_bench_kern


program test_bench
  use cudafor
  use fltflt
  use test_bench_kern
  implicit none

  integer, parameter :: N = N_BENCH, NI = NITER

  real(8), allocatable :: a_h(:), b_h(:)
  real(8), device, allocatable :: a_d(:), b_d(:), out_r8(:)
  real(4), device, allocatable :: out_hi(:), out_lo(:)
  real(8) :: guard_r8, ms_r8, ms_ff, mops_r8, mops_ff
  real(4) :: guard_hi, guard_lo, elapsed
  type(cudaEvent) :: t_start, t_stop
  type(dim3) :: blk, thr
  integer :: nb, istat, csv_unit

  thr = dim3(NBLK, 1, 1)
  nb  = (N + NBLK - 1) / NBLK
  blk = dim3(nb, 1, 1)

  allocate(a_h(N), b_h(N))
  a_h = 1.23456789d0;  b_h = 0.99999987d0
  allocate(a_d(N), b_d(N), out_r8(N), out_hi(N), out_lo(N))
  a_d = a_h;  b_d = b_h

  istat = cudaEventCreate(t_start)
  istat = cudaEventCreate(t_stop)

  ! Warmup + time real(8)
  call kern_bench_r8<<<blk, thr>>>(N, NI, a_d, b_d, out_r8)
  istat = cudaDeviceSynchronize()
  istat = cudaEventRecord(t_start, 0_8)
  call kern_bench_r8<<<blk, thr>>>(N, NI, a_d, b_d, out_r8)
  istat = cudaEventRecord(t_stop, 0_8)
  istat = cudaEventSynchronize(t_stop)
  istat = cudaEventElapsedTime(elapsed, t_start, t_stop)
  ms_r8 = dble(elapsed)

  ! Warmup + time fltflt
  call kern_bench_ff<<<blk, thr>>>(N, NI, a_d, b_d, out_hi, out_lo)
  istat = cudaDeviceSynchronize()
  istat = cudaEventRecord(t_start, 0_8)
  call kern_bench_ff<<<blk, thr>>>(N, NI, a_d, b_d, out_hi, out_lo)
  istat = cudaEventRecord(t_stop, 0_8)
  istat = cudaEventSynchronize(t_stop)
  istat = cudaEventElapsedTime(elapsed, t_start, t_stop)
  ms_ff = dble(elapsed)

  istat = cudaEventDestroy(t_start)
  istat = cudaEventDestroy(t_stop)

  guard_r8 = out_r8(1);  guard_hi = out_hi(1);  guard_lo = out_lo(1)
  mops_r8 = dble(N)*dble(NI) / (ms_r8 * 1.0d3)
  mops_ff = dble(N)*dble(NI) / (ms_ff * 1.0d3)

  write(*,*) "=== fltflt benchmark (real(8) inputs, N=2^20, NITER=100) ==="
  write(*,'("  (guard: r8=",F10.6,"  ff=",F10.6,"+",E10.3,")")') &
    guard_r8, guard_hi, guard_lo
  write(*,'("  real(8)  : ", F8.2, " ms  | ", F8.0, " MOPS")') ms_r8, mops_r8
  write(*,'("  fltflt   : ", F8.2, " ms  | ", F8.0, " MOPS")') ms_ff, mops_ff
  write(*,'("  slowdown : ", F6.2, "x")') ms_ff / ms_r8

  csv_unit = 11
  open(unit=csv_unit, file='ffp_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,val1,val2'
  write(csv_unit, '("bench,real(8),",F12.4,",",F12.1)') ms_r8, mops_r8
  write(csv_unit, '("bench,fltflt,",F12.4,",",F12.1)')  ms_ff, mops_ff
  close(csv_unit)

  deallocate(a_h, b_h, a_d, b_d, out_r8, out_hi, out_lo)

end program test_bench
