! test_cmp.f90  —  fltflt comparison operators: correctness + throughput
!
! Build:  cd ouxmix && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_cmp
!        (writes test/cmp_results.csv)
!
! Tests all 6 operators (==, /=, <, >, <=, >=) for ff/ff, ff/r4, r4/ff overloads.
! Benchmark: count-in-range kernel using <= and >= on fltflt vs real(4).

module test_cmp_kern
  use cudafor
  use fltflt
  implicit none
  private
  public :: NCASE, N_BENCH, NITER, NBLK
  public :: kern_cmp, kern_bench_cmp_r4, kern_bench_cmp_ff

  integer, parameter :: NCASE   = 12
  integer, parameter :: N_BENCH = 2**20
  integer, parameter :: NITER   = 100
  integer, parameter :: NBLK    = 256

contains

  ! ---- correctness kernel -----------------------------------------------
  ! Each thread tests one (a, b) pair through all 6 operators.
  ! Results packed as integer 0/1 in res(12): ==,/=,<,>,<=,>= for ff/ff,
  ! then the same pair as ff == r4.
  attributes(global) subroutine kern_cmp(a_hi_d, a_lo_d, b_hi_d, b_lo_d, n, res_d)
    integer,  intent(in),  value  :: n
    real(4),  intent(in),  device :: a_hi_d(n), a_lo_d(n), b_hi_d(n), b_lo_d(n)
    integer,  intent(out), device :: res_d(n, 12)
    type(fltflt) :: a, b
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    a%hi = a_hi_d(i);  a%lo = a_lo_d(i)
    b%hi = b_hi_d(i);  b%lo = b_lo_d(i)
    ! ff/ff
    res_d(i,  1) = merge(1, 0, a == b)
    res_d(i,  2) = merge(1, 0, a /= b)
    res_d(i,  3) = merge(1, 0, a < b)
    res_d(i,  4) = merge(1, 0, a > b)
    res_d(i,  5) = merge(1, 0, a <= b)
    res_d(i,  6) = merge(1, 0, a >= b)
    ! ff/r4 (compare a to b%hi, b%lo should be 0 for these cases)
    res_d(i,  7) = merge(1, 0, a == b%hi)
    res_d(i,  8) = merge(1, 0, a /= b%hi)
    res_d(i,  9) = merge(1, 0, a < b%hi)
    res_d(i, 10) = merge(1, 0, a > b%hi)
    res_d(i, 11) = merge(1, 0, a <= b%hi)
    res_d(i, 12) = merge(1, 0, a >= b%hi)
  end subroutine kern_cmp

  ! ---- benchmark: count-in-range using comparisons ----------------------
  attributes(global) subroutine kern_bench_cmp_r4(n, niter, x_d, lo_d, hi_d, cnt_d)
    integer, intent(in),  value   :: n, niter
    real(4), intent(in),  device  :: x_d(n), lo_d(n), hi_d(n)
    integer, intent(out), device  :: cnt_d(n)
    real(4) :: x, lo, hi
    integer :: i, j, s
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x = x_d(i);  lo = lo_d(i);  hi = hi_d(i)
    s = 0
    do j = 1, niter
      if (x >= lo .and. x <= hi) s = s + 1
    end do
    cnt_d(i) = s
  end subroutine kern_bench_cmp_r4

  attributes(global) subroutine kern_bench_cmp_ff(n, niter, x_hi_d, x_lo_d, lo_hi_d, lo_lo_d, &
                                                    hi_hi_d, hi_lo_d, cnt_d)
    integer, intent(in),  value   :: n, niter
    real(4), intent(in),  device  :: x_hi_d(n), x_lo_d(n)
    real(4), intent(in),  device  :: lo_hi_d(n), lo_lo_d(n)
    real(4), intent(in),  device  :: hi_hi_d(n), hi_lo_d(n)
    integer, intent(out), device  :: cnt_d(n)
    type(fltflt) :: x, lo, hi
    integer :: i, j, s
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    x%hi  = x_hi_d(i);   x%lo  = x_lo_d(i)
    lo%hi = lo_hi_d(i);  lo%lo = lo_lo_d(i)
    hi%hi = hi_hi_d(i);  hi%lo = hi_lo_d(i)
    s = 0
    do j = 1, niter
      if (x >= lo .and. x <= hi) s = s + 1
    end do
    cnt_d(i) = s
  end subroutine kern_bench_cmp_ff

end module test_cmp_kern


program test_cmp
  use cudafor
  use fltflt
  use test_cmp_kern
  implicit none

  ! -----------------------------------------------------------------------
  ! Test cases: each row is (a_hi, a_lo, b_hi, b_lo, expected results)
  ! Expected bit-vector: ==, /=, <, >, <=, >= for ff/ff
  !                      ==, /=, <, >, <=, >= for ff/r4 (b%lo must be 0)
  ! -----------------------------------------------------------------------
  integer, parameter :: N = NCASE
  character(len=24), parameter :: CLABELS(N) = [ character(len=24) :: &
    "eq[same_hi_lo]",   &   ! (1) a==b: same hi and lo → all eq ops true
    "eq[same_hi_neg_lo]",&  ! (2) a/=b: same hi, lo differs
    "lt[hi_differs]",   &   ! (3) a < b: hi differs
    "gt[hi_differs]",   &   ! (4) a > b: hi differs
    "lt[same_hi_lo]",   &   ! (5) a < b by lo only
    "gt[same_hi_lo]",   &   ! (6) a > b by lo only
    "le[equal]",        &   ! (7) a <= b, a==b
    "ge[equal]",        &   ! (8) a >= b, a==b
    "eq_r4[lo_zero]",   &   ! (9) ff==r4, a%lo==0, b%lo==0
    "ne_r4[lo_nonzero]",&   !(10) ff/=r4, a%lo!=0
    "lt_r4[lo_neg]",    &   !(11) a < r4: a%hi==b, a%lo<0
    "neg[cross_zero]"   ]   !(12) negative a < positive b

  real(4) :: a_hi(N), a_lo(N), b_hi(N), b_lo(N)
  real(4), device :: ah_d(N), alo_d(N), bh_d(N), blo_d(N)
  integer, device :: res_d(N, 12)
  integer :: res(N, 12)
  integer :: expected(N, 12)
  logical :: row_pass
  integer :: i, j, istat, nfail, csv_unit

  ! (1) Identical: a == b, lo == lo
  a_hi(1) = 1.0;   a_lo(1) = 1.0e-7;  b_hi(1) = 1.0;   b_lo(1) = 1.0e-7
  ! (2) Same hi, different lo → a /= b, a > b (a%lo > b%lo)
  a_hi(2) = 1.0;   a_lo(2) = 1.0e-7;  b_hi(2) = 1.0;   b_lo(2) = -1.0e-7
  ! (3) hi differs: a < b
  a_hi(3) = 1.0;   a_lo(3) = 0.0;     b_hi(3) = 2.0;   b_lo(3) = 0.0
  ! (4) hi differs: a > b
  a_hi(4) = 2.0;   a_lo(4) = 0.0;     b_hi(4) = 1.0;   b_lo(4) = 0.0
  ! (5) Same hi, a%lo < b%lo → a < b
  a_hi(5) = 1.0;   a_lo(5) = -1.0e-7; b_hi(5) = 1.0;   b_lo(5) = 1.0e-7
  ! (6) Same hi, a%lo > b%lo → a > b
  a_hi(6) = 1.0;   a_lo(6) = 1.0e-7;  b_hi(6) = 1.0;   b_lo(6) = -1.0e-7
  ! (7) Equal values: a <= b and a >= b both true
  a_hi(7) = 3.0;   a_lo(7) = 0.0;     b_hi(7) = 3.0;   b_lo(7) = 0.0
  ! (8) Same as (7) for ge test
  a_hi(8) = 3.0;   a_lo(8) = 0.0;     b_hi(8) = 3.0;   b_lo(8) = 0.0
  ! (9) ff == r4: a%lo == 0 and a%hi == b%hi (b%lo must be 0 for ff/r4 test)
  a_hi(9) = 5.0;   a_lo(9) = 0.0;     b_hi(9) = 5.0;   b_lo(9) = 0.0
  ! (10) ff /= r4: a%lo != 0 so a != b%hi
  a_hi(10) = 5.0;  a_lo(10) = 1.0e-7; b_hi(10) = 5.0;  b_lo(10) = 0.0
  ! (11) ff < r4: a%hi == b but a%lo < 0 → a < b%hi
  a_hi(11) = 5.0;  a_lo(11) = -1.0e-7; b_hi(11) = 5.0; b_lo(11) = 0.0
  ! (12) Negative a < positive b (cross-zero)
  a_hi(12) = -1.0; a_lo(12) = 0.0;    b_hi(12) = 1.0;  b_lo(12) = 0.0

  ! Expected results [==, /=, <, >, <=, >=,   ==r4, /=r4, <r4, >r4, <=r4, >=r4]
  ! ff/r4 ops compare a (fltflt) to b%hi (scalar r4), not to b as a pair.
  ! a == b%hi only when a%hi==b%hi AND a%lo==0.
  expected( 1,:) = [1, 0, 0, 0, 1, 1,   0, 1, 0, 1, 0, 1]  ! ff: a==b; r4: a>b%hi (a%lo>0)
  expected( 2,:) = [0, 1, 0, 1, 0, 1,   0, 1, 0, 1, 0, 1]  ! ff: a>b;  r4: a>b%hi (a%lo>0)
  expected( 3,:) = [0, 1, 1, 0, 1, 0,   0, 1, 1, 0, 1, 0]  ! ff: a<b;  r4: a<b%hi (hi differs)
  expected( 4,:) = [0, 1, 0, 1, 0, 1,   0, 1, 0, 1, 0, 1]  ! ff: a>b;  r4: a>b%hi (hi differs)
  expected( 5,:) = [0, 1, 1, 0, 1, 0,   0, 1, 1, 0, 1, 0]  ! ff: a<b by lo; r4: a<b%hi (a%lo<0)
  expected( 6,:) = [0, 1, 0, 1, 0, 1,   0, 1, 0, 1, 0, 1]  ! ff: a>b by lo; r4: a>b%hi (a%lo>0)
  expected( 7,:) = [1, 0, 0, 0, 1, 1,   1, 0, 0, 0, 1, 1]  ! ff: a==b; r4: a==b%hi (a%lo==0)
  expected( 8,:) = [1, 0, 0, 0, 1, 1,   1, 0, 0, 0, 1, 1]  ! same
  expected( 9,:) = [1, 0, 0, 0, 1, 1,   1, 0, 0, 0, 1, 1]  ! ff: a==b; r4: a==b%hi (a%lo==0)
  expected(10,:) = [0, 1, 0, 1, 0, 1,   0, 1, 0, 1, 0, 1]  ! ff: a>b;  r4: a>b%hi (a%lo>0)
  expected(11,:) = [0, 1, 1, 0, 1, 0,   0, 1, 1, 0, 1, 0]  ! ff: a<b;  r4: a<b%hi (a%lo<0)
  expected(12,:) = [0, 1, 1, 0, 1, 0,   0, 1, 1, 0, 1, 0]  ! ff: a<b;  r4: a<b%hi (hi differs)

  csv_unit = 11
  open(unit=csv_unit, file='cmp_results.csv', status='replace', action='write')
  write(csv_unit, '(A)') 'type,label,pass'

  ah_d = a_hi;  alo_d = a_lo;  bh_d = b_hi;  blo_d = b_lo
  call kern_cmp<<<1, N>>>(ah_d, alo_d, bh_d, blo_d, N, res_d)
  istat = cudaDeviceSynchronize()
  res = res_d

  write(*,*) "=== test_cmp: comparison operators ==="
  nfail = 0
  do i = 1, N
    row_pass = .true.
    do j = 1, 12
      if (res(i,j) /= expected(i,j)) then
        row_pass = .false.
        nfail = nfail + 1
        write(*,'("  case",I2,", op",I2,": got ",I0," expected ",I0,"  FAIL  ",A)') &
          i, j, res(i,j), expected(i,j), trim(adjustl(CLABELS(i)))
      end if
    end do
    if (row_pass) &
      write(*,'("  case",I2,": PASS  ",A)') i, trim(adjustl(CLABELS(i)))
    write(csv_unit, '("acc,",A,",",I0)') trim(adjustl(CLABELS(i))), merge(1, 0, row_pass)
  end do

  call run_bench()
  close(csv_unit)

  if (nfail == 0) then
    write(*,*) "ALL PASS"
  else
    write(*,'("  FAILED: ",I0," mismatch(es)")') nfail
    stop 1
  end if

contains

  subroutine run_bench()
    integer, parameter :: NB = N_BENCH, NI = NITER
    real(4), allocatable :: xh(:), loh(:), hih(:)
    real(4), device, allocatable :: x_d(:), lo_d(:), hi_d(:)
    real(4), device, allocatable :: xhi_d(:), xlo_d(:)
    real(4), device, allocatable :: lohi_d(:), lolo_d(:), hihi_d(:), hilo_d(:)
    integer, device, allocatable :: cnt_r4(:), cnt_ff(:)
    real(8) :: ms_r4, ms_ff, mops_r4, mops_ff
    real(4) :: elapsed
    integer :: g_r4, g_ff, is
    type(cudaEvent) :: t0, t1
    type(dim3) :: blk, thr
    integer :: nb_blk

    thr    = dim3(NBLK, 1, 1)
    nb_blk = (NB + NBLK - 1) / NBLK
    blk    = dim3(nb_blk, 1, 1)

    allocate(xh(NB), loh(NB), hih(NB))
    xh  = 0.5;  loh = 0.0;  hih = 1.0
    allocate(x_d(NB), lo_d(NB), hi_d(NB))
    allocate(xhi_d(NB), xlo_d(NB))
    allocate(lohi_d(NB), lolo_d(NB), hihi_d(NB), hilo_d(NB))
    allocate(cnt_r4(NB), cnt_ff(NB))
    x_d = xh;  lo_d = loh;  hi_d = hih
    xhi_d = xh;   xlo_d  = 0.0
    lohi_d = loh; lolo_d = 0.0
    hihi_d = hih; hilo_d = 0.0

    is = cudaEventCreate(t0);  is = cudaEventCreate(t1)

    call kern_bench_cmp_r4<<<blk,thr>>>(NB, NI, x_d, lo_d, hi_d, cnt_r4)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_cmp_r4<<<blk,thr>>>(NB, NI, x_d, lo_d, hi_d, cnt_r4)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_r4 = dble(elapsed)

    call kern_bench_cmp_ff<<<blk,thr>>>(NB, NI, xhi_d, xlo_d, lohi_d, lolo_d, &
                                         hihi_d, hilo_d, cnt_ff)
    is = cudaDeviceSynchronize()
    is = cudaEventRecord(t0, 0_8)
    call kern_bench_cmp_ff<<<blk,thr>>>(NB, NI, xhi_d, xlo_d, lohi_d, lolo_d, &
                                         hihi_d, hilo_d, cnt_ff)
    is = cudaEventRecord(t1, 0_8);  is = cudaEventSynchronize(t1)
    is = cudaEventElapsedTime(elapsed, t0, t1);  ms_ff = dble(elapsed)

    is = cudaEventDestroy(t0);  is = cudaEventDestroy(t1)

    g_r4 = cnt_r4(1);  g_ff = cnt_ff(1)
    mops_r4 = dble(NB)*dble(NI) / (ms_r4 * 1.0d3)
    mops_ff = dble(NB)*dble(NI) / (ms_ff * 1.0d3)

    write(*,'("--- benchmark (N=2^20, NITER=100, count-in-range) ---")')
    write(*,'("  (guard r4=",I0,"  ff=",I0,")")') g_r4, g_ff
    write(*,'("  real(4) : ",F8.2," ms  | ",F8.0," MOPS")') ms_r4, mops_r4
    write(*,'("  fltflt  : ",F8.2," ms  | ",F8.0," MOPS")') ms_ff, mops_ff
    write(*,'("  ff/r4 slowdown : ",F5.2,"x")') ms_ff / ms_r4

    write(csv_unit, '("bench,real(4),",F12.4,",",F12.1)') ms_r4, mops_r4
    write(csv_unit, '("bench,fltflt,",F12.4,",",F12.1)')  ms_ff, mops_ff

    deallocate(xh, loh, hih, x_d, lo_d, hi_d)
    deallocate(xhi_d, xlo_d, lohi_d, lolo_d, hihi_d, hilo_d)
    deallocate(cnt_r4, cnt_ff)
  end subroutine run_bench

end program test_cmp
