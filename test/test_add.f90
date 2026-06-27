! test_add.f90  —  fltflt addition: accuracy + throughput
!
! Build:  cd test && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_add
!        (writes test/add_results.csv; plot: cmake --build . --target plot_add)
!
! Accuracy: NCASE scalar cases + dotprod reduction, real(8) inputs.
! Benchmark: chained x = x + b for real(4), real(8), fltflt.

module test_add_kern
  use cudafor
  use cudadevice
  use mod_ffp
  implicit none

  integer, parameter :: NCASE   = 4
  integer, parameter :: N_DOT   = 1024
  integer, parameter :: N_BENCH = 2**20
  integer, parameter :: NITER   = 100
  integer, parameter :: NBLK    = 256

contains

  ! ---- accuracy kernel ------------------------------------------------
  attributes(global) subroutine kern_add(a_d, b_d, n, res_r4, res_hi, res_lo)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: res_r4(n), res_hi(n), res_lo(n)
    type(fltflt) :: af, bf, cf
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    af = init(a_d(i));  bf = init(b_d(i))
    res_r4(i) = real(a_d(i), 4) + real(b_d(i), 4)
    cf = af + bf
    res_hi(i) = cf%hi;  res_lo(i) = cf%lo
  end subroutine kern_add

  ! ---- dotprod: warp + block reduction --------------------------------
  attributes(global) subroutine kern_dotprod(n, a_d, b_d, res_r4, res_hi, res_lo)
    integer, intent(in), value   :: n
    real(8), intent(in),  device :: a_d(n), b_d(n)
    real(4), intent(out), device :: res_r4(1), res_hi(1), res_lo(1)
    type(fltflt) :: acc, other
    real(4) :: acc_r4
    real(4), shared :: smem_hi(0:31), smem_lo(0:31), smem_r4(0:31)
    integer :: i, lane, warp_id, delta

    i       = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    lane    = mod(threadIdx%x - 1, 32)
    warp_id = (threadIdx%x - 1) / 32

    if (i <= n) then
      acc    = init(a_d(i)) * init(b_d(i))
      acc_r4 = real(a_d(i), 4) * real(b_d(i), 4)
    else
      acc%hi = 0.0_4;  acc%lo = 0.0_4;  acc_r4 = 0.0_4
    end if

    delta = 16
    do while (delta >= 1)
      other  = shfl_down_ff(acc, delta)
      acc    = acc + other
      acc_r4 = acc_r4 + __shfl_down(acc_r4, delta)
      delta  = delta / 2
    end do

    if (lane == 0) then
      smem_hi(warp_id) = acc%hi;  smem_lo(warp_id) = acc%lo
      smem_r4(warp_id) = acc_r4
    end if
    call syncthreads()

    if (warp_id == 0) then
      acc%hi = smem_hi(lane);  acc%lo = smem_lo(lane);  acc_r4 = smem_r4(lane)
      delta  = 16
      do while (delta >= 1)
        other  = shfl_down_ff(acc, delta)
        acc    = acc + other
        acc_r4 = acc_r4 + __shfl_down(acc_r4, delta)
        delta  = delta / 2
      end do
      if (lane == 0) then
        res_hi(1) = acc%hi;  res_lo(1) = acc%lo;  res_r4(1) = acc_r4
      end if
    end if
  end subroutine kern_dotprod

  ! ---- benchmark kernels: chained x = x + b --------------------------
  attributes(global) subroutine kern_bench_add_r4(n, niter, a_d, b_d, out_d)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n)
    real(4), intent(out), device  :: out_d(n)
    real(4) :: x, b
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = real(a_d(i), 4);  b = real(b_d(i), 4)
    do j = 1, niter
      x = x + b
    end do
    out_d(i) = x
  end subroutine kern_bench_add_r4

  attributes(global) subroutine kern_bench_add_r8(n, niter, a_d, b_d, out_d)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n)
    real(8), intent(out), device  :: out_d(n)
    real(8) :: x, b
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = a_d(i);  b = b_d(i)
    do j = 1, niter
      x = x + b
    end do
    out_d(i) = x
  end subroutine kern_bench_add_r8

  attributes(global) subroutine kern_bench_add_ff(n, niter, a_d, b_d, out_hi, out_lo)
    integer, intent(in), value    :: n, niter
    real(8), intent(in),  device  :: a_d(n), b_d(n)
    real(4), intent(out), device  :: out_hi(n), out_lo(n)
    type(fltflt) :: xf, bf
    integer :: i, j
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    xf = init(a_d(i));  bf = init(b_d(i))
    do j = 1, niter
      xf = xf + bf
    end do
    out_hi(i) = xf%hi;  out_lo(i) = xf%lo
  end subroutine kern_bench_add_ff

end module test_add_kern


program test_add
  use cudafor
  use mod_ffp
  use test_add_kern
  implicit none

  integer, parameter :: N = NCASE, ND = N_DOT
  real(8), parameter :: FF_THRESH = 1.0d-12
  character(len=16), parameter :: CLABELS(N) = &
    [ character(len=16) :: "add[1e8+1]", "add[~r4eps]", "add[cancel]", "add[general]" ]

  real(8) :: a_r8(N), b_r8(N), ref(N)
  real(8), device, allocatable :: a_d(:), b_d(:)
  real(4), device, allocatable :: res_r4(:), res_hi(:), res_lo(:)
  real(4) :: h_r4(N), h_hi(N), h_lo(N)
  real(8) :: ff_val, err_r4, err_ff
  integer :: i, istat, nfail, csv_unit

  csv_unit = 11
  open(unit=csv_unit, file='add_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,val1,val2'

  a_r8(1) = 1.0d8;                 b_r8(1) = 1.0d0
  a_r8(2) = 1.0d0 + 2.0d0**(-25); b_r8(2) = 2.0d0**(-25)
  a_r8(3) = 1.0d0 + 2.0d0**(-25); b_r8(3) = -1.0d0
  a_r8(4) = 1.23456789d0;          b_r8(4) = 9.87654321d-1
  ref = a_r8 + b_r8

  allocate(a_d(N), b_d(N), res_r4(N), res_hi(N), res_lo(N))
  a_d = a_r8;  b_d = b_r8
  call kern_add<<<1, N>>>(a_d, b_d, N, res_r4, res_hi, res_lo)
  istat = cudaDeviceSynchronize()
  h_r4 = res_r4;  h_hi = res_hi;  h_lo = res_lo
  deallocate(a_d, b_d, res_r4, res_hi, res_lo)

  write(*,*) "=== test_add: fltflt addition (real(8) inputs) ==="
  nfail = 0
  do i = 1, N
    ff_val = real(h_hi(i), 8) + real(h_lo(i), 8)
    if (abs(ref(i)) > 0.0d0) then
      err_r4 = abs((real(h_r4(i), 8) - ref(i)) / ref(i))
      err_ff = abs((ff_val            - ref(i)) / ref(i))
    else
      err_r4 = abs(real(h_r4(i), 8))
      err_ff = abs(ff_val)
    end if
    if (err_ff > FF_THRESH) nfail = nfail + 1
    write(*,'("  case",I2,": r4_err=",ES9.2,"  ff_err=",ES9.2,"  ",A)') &
      i, err_r4, err_ff, merge("PASS", "FAIL", err_ff <= FF_THRESH)
    write(csv_unit, '("acc,",A,",",ES12.5,",",ES12.5)') &
      trim(adjustl(CLABELS(i))), err_r4, err_ff
  end do

  call run_dotprod()
  call run_bench()

  close(csv_unit)

  if (nfail == 0) then
    write(*,*) "ALL PASS"
  else
    write(*,'("  FAILED: ",I0," case(s)")') nfail
    stop 1
  end if

contains

  subroutine run_dotprod()
    real(8), allocatable :: ah(:), bh(:)
    real(8), device, allocatable :: ad(:), bd(:)
    real(4), device, allocatable :: rdot(:), rhi(:), rlo(:)
    real(8) :: ref_dot, ff_as_r8, err_r4d, err_ffd
    real(4) :: h_rd, h_rhi, h_rlo
    integer :: ii, is

    allocate(ah(ND), bh(ND))
    ref_dot = 0.0d0
    do ii = 1, ND
      ah(ii) = 1.0d0 + dble(ii) * 2.0d0**(-15)
      bh(ii) = 1.0d0 - dble(ii) * 2.0d0**(-15)
      ref_dot = ref_dot + ah(ii) * bh(ii)
    end do
    allocate(ad(ND), bd(ND), rdot(1), rhi(1), rlo(1))
    ad = ah;  bd = bh
    call kern_dotprod<<<1, ND>>>(ND, ad, bd, rdot, rhi, rlo)
    is = cudaDeviceSynchronize()
    h_rd = rdot(1);  h_rhi = rhi(1);  h_rlo = rlo(1)
    ff_as_r8 = real(h_rhi, 8) + real(h_rlo, 8)
    err_r4d  = abs((real(h_rd, 8)  - ref_dot) / ref_dot)
    err_ffd  = abs((ff_as_r8       - ref_dot) / ref_dot)
    if (err_ffd > FF_THRESH) nfail = nfail + 1
    write(*,'("  dotprod : r4_err=",ES9.2,"  ff_err=",ES9.2,"  ",A)') &
      err_r4d, err_ffd, merge("PASS", "FAIL", err_ffd <= FF_THRESH)
    write(csv_unit, '("acc,dotprod,",ES12.5,",",ES12.5)') err_r4d, err_ffd
    deallocate(ah, bh, ad, bd, rdot, rhi, rlo)
  end subroutine run_dotprod

  subroutine run_bench()
    integer, parameter :: NB = N_BENCH, NI = NITER
    real(8), allocatable :: a_h(:), b_h(:)
    real(8), device, allocatable :: ad(:), bd(:), out_r8(:)
    real(4), device, allocatable :: out_r4(:), out_hi(:), out_lo(:)
    real(8) :: ms_r4, ms_r8, ms_ff, mops_r4, mops_r8, mops_ff
    real(4) :: elapsed
    real(4) :: g_r4, g_r8_r4, g_hi, g_lo  ! guards
    real(8) :: g_r8
    type(cudaEvent) :: t0, t1
    type(dim3) :: blk, thr
    integer :: nb_blk, is

    thr    = dim3(NBLK, 1, 1)
    nb_blk = (NB + NBLK - 1) / NBLK
    blk    = dim3(nb_blk, 1, 1)

    allocate(a_h(NB), b_h(NB))
    a_h = 1.23456789d0;  b_h = 1.23456789d-2
    allocate(ad(NB), bd(NB), out_r4(NB), out_r8(NB), out_hi(NB), out_lo(NB))
    ad = a_h;  bd = b_h

    is = cudaEventCreate(t0);  is = cudaEventCreate(t1)

    ! real(4)
    call kern_bench_add_r4<<<blk,thr>>>(NB, NI, ad, bd, out_r4)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0)
    call kern_bench_add_r4<<<blk,thr>>>(NB, NI, ad, bd, out_r4)
    is = cudaEventRecord(t1, 0);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r4 = dble(elapsed)

    ! real(8)
    call kern_bench_add_r8<<<blk,thr>>>(NB, NI, ad, bd, out_r8)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0)
    call kern_bench_add_r8<<<blk,thr>>>(NB, NI, ad, bd, out_r8)
    is = cudaEventRecord(t1, 0);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r8 = dble(elapsed)

    ! fltflt
    call kern_bench_add_ff<<<blk,thr>>>(NB, NI, ad, bd, out_hi, out_lo)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0)
    call kern_bench_add_ff<<<blk,thr>>>(NB, NI, ad, bd, out_hi, out_lo)
    is = cudaEventRecord(t1, 0);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_ff = dble(elapsed)

    is = cudaEventDestroy(t0);  is = cudaEventDestroy(t1)

    g_r4 = out_r4(1);  g_r8 = out_r8(1);  g_hi = out_hi(1);  g_lo = out_lo(1)
    g_r8_r4 = real(g_r8, 4)
    mops_r4 = dble(NB)*dble(NI) / (ms_r4 * 1.0d3)
    mops_r8 = dble(NB)*dble(NI) / (ms_r8 * 1.0d3)
    mops_ff = dble(NB)*dble(NI) / (ms_ff * 1.0d3)

    write(*,'("--- benchmark (N=2^20, NITER=100, chained add) ---")')
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

end program test_add
