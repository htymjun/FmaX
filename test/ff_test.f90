! Test driver: runs one FmaX/fltflt.f90 operation over an input file and writes
! the result. test/matx/ff_matx.cu does the same with NVIDIA MatX's fltflt.h;
! test/run_tests.py generates the inputs and compares both against each other
! and against an extended-precision reference.
!
!   usage: ff_test <op> <in.bin> <out.bin>
!
! Binary layout (stream, little endian):
!   in : int32 n | 8 x real(4)(n): ah al bh bl ch cl dh dl | real(8)(n) x8 | int32(n) k
!   out: real(4)(n) rh | real(4)(n) rl | int32(n) ri
! Op codes 1..58 exist in MatX as well; 60.. are FmaX-only extensions.
module ff_test_kern
  use cudafor
  use fltflt
  implicit none
  private
  public :: kern
contains
  attributes(global) subroutine kern(op, n, ah, al, bh, bl, ch, cl, dh, dl, x8, k, rh, rl, ri)
    integer, value :: op, n
    real(4), device, intent(in) :: ah(n), al(n), bh(n), bl(n), ch(n), cl(n), dh(n), dl(n)
    real(8), device, intent(in) :: x8(n)
    integer, device, intent(in) :: k(n)
    real(4), device, intent(out) :: rh(n), rl(n)
    integer, device, intent(out) :: ri(n)
    type(fltflt) :: a, b, c, d, r
    real(4) :: a4, b4, c4, d4
    integer :: i, iv
    logical :: t
    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (i > n) return
    a%hi = ah(i); a%lo = al(i)
    b%hi = bh(i); b%lo = bl(i)
    c%hi = ch(i); c%lo = cl(i)
    d%hi = dh(i); d%lo = dl(i)
    a4 = ah(i); b4 = bh(i); c4 = ch(i); d4 = dh(i)
    r%hi = 0.0; r%lo = 0.0
    iv = 0
    t = .false.
    select case (op)
    case (1);  r = a + b
    case (2);  r = a + b4
    case (3);  r = a4 + b
    case (4);  r = a - b
    case (5);  r = a - b4
    case (6);  r = a4 - b
    case (7);  r = a * b
    case (8);  r = a * b4
    case (9);  r = a4 * b
    case (10); r = a / b
    case (11); r = a / b4
    case (12); r = a4 / b
    case (13); r = fltflt_fma(a, b, c)
    case (14); r = fltflt_fma(a, b, c4)
    case (15); r = fltflt_fma(a4, b, c)
    case (16); r = fltflt_fma(a, b4, c)
    case (17); r = fltflt_fma(a, b4, c4)
    case (18); r = fltflt_fma(a4, b, c4)
    case (19); r = fltflt_fma(a4, b4, c)
    case (20); r = fltflt_fma_approx(a, b, c)
    case (21); r = fltflt_fma_approx(a, b, c4)
    case (22); r = fltflt_fma_approx(a4, b, c)
    case (23); r = fltflt_fma_approx(a, b4, c)
    case (24); r = fltflt_fma_approx(a, b4, c4)
    case (25); r = fltflt_fma_approx(a4, b, c4)
    case (26); r = fltflt_fma_approx(a4, b4, c)
    case (27); r = fltflt_add_same_sign(a, b)
    case (28); r = fltflt_add_same_sign(a, b4)
    case (29); r = fltflt_add_same_sign(a4, b)
    case (30); r = fltflt_sqrt(a)
    case (31); r = fltflt_sqrt_fast(a)
    case (32); r = fltflt_norm3d(a, b, c)
    case (33); r = fltflt_abs(a)
    case (34); r = fltflt_round_to_nearest(a)
    case (35); r = fltflt_round_toward_zero(a)
    case (36); r = fltflt_floor(a)
    case (37); r = fltflt_fmod(a, b)
    case (38); r = fltflt_fmod(a, b4)
    case (39); r = fltflt_init(x8(i))
    case (40); r = -a
    ! comparisons -> ri (0/1); logicals are turned into integers with merge,
    ! never with .not. (see CLAUDE.md, nvfortran device logicals)
    case (41); t = (a == b)
    case (42); t = (a /= b)
    case (43); t = (a <  b)
    case (44); t = (a >  b)
    case (45); t = (a <= b)
    case (46); t = (a >= b)
    case (47); t = (a == b4)
    case (48); t = (a /= b4)
    case (49); t = (a <  b4)
    case (50); t = (a >  b4)
    case (51); t = (a <= b4)
    case (52); t = (a >= b4)
    case (53); t = (a4 == b)
    case (54); t = (a4 /= b)
    case (55); t = (a4 <  b)
    case (56); t = (a4 >  b)
    case (57); t = (a4 <= b)
    case (58); t = (a4 >= b)
    ! ---- FmaX-only extensions ----
    case (60); r = fltflt_recip(a)
    case (61); r = fltflt_square(a)
    case (62); r = fltflt_dot2(a4, b4, c4, d4)
    case (63); r = fltflt_dot2(a, b, c, d)
    case (64); r = fltflt_dot3(a, b, c, d, b, c)
    case (65); r = fltflt_dot4(a, b, c, d, a, d, b, c)
    case (66); r = fltflt_add3(a, b, c)
    case (67); r = fltflt_add4(a, b, c, d)
    case (68); r = fltflt_add5(a, b, c, d, a)
    case (69); r = fltflt_ceil(a)
    case (70); r = fltflt_hypot(a4, b4)
    case (71); r = fltflt_lerp(a, b, c4)
    case (72); r = fltflt_pow_int(a, k(i))
    case (73); r = fltflt_min(a, b)
    case (74); r = fltflt_max(a, b)
    case (75); r = fltflt_sign(a, b)
    case (76); r = a + x8(i)
    case (77); r = a * x8(i)
    case (78); r = a / x8(i)
    case (79); r = x8(i) / b
    case (80); r = fltflt_dot3(a4, b4, c4, d4, a4, d4)
    case (81); r = fltflt_warp_reduce_sum(a)
    case (82); r = fltflt_dot4(a4, b4, c4, d4, a4, d4, b4, c4)
    case (83); r = a - x8(i)
    case default
      iv = -999
    end select
    if (op >= 41 .and. op <= 58) iv = merge(1, 0, t)
    rh(i) = r%hi
    rl(i) = r%lo
    ri(i) = iv
  end subroutine kern
end module ff_test_kern

program ff_test
  use cudafor
  use ff_test_kern
  implicit none
  integer :: op, n, u, istat
  character(len=512) :: arg, fin, fout
  real(4), allocatable :: v(:,:), hrh(:), hrl(:)
  real(8), allocatable :: hx8(:)
  integer, allocatable :: hk(:), hri(:)
  real(4), device, allocatable :: dah(:), dal(:), dbh(:), dbl(:), dch(:), dcl(:), ddh(:), ddl(:)
  real(4), device, allocatable :: drh(:), drl(:)
  real(8), device, allocatable :: dx8(:)
  integer, device, allocatable :: dk(:), dri(:)
  if (command_argument_count() /= 3) then
    print *, 'usage: ff_test op in.bin out.bin'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) op
  call get_command_argument(2, fin)
  call get_command_argument(3, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) n
  allocate(v(n, 8), hx8(n), hk(n), hrh(n), hrl(n), hri(n))
  read(u) v(:,1), v(:,2), v(:,3), v(:,4), v(:,5), v(:,6), v(:,7), v(:,8)
  read(u) hx8
  read(u) hk
  close(u)
  allocate(dah(n), dal(n), dbh(n), dbl(n), dch(n), dcl(n), ddh(n), ddl(n), dx8(n), dk(n))
  allocate(drh(n), drl(n), dri(n))
  dah = v(:,1); dal = v(:,2); dbh = v(:,3); dbl = v(:,4)
  dch = v(:,5); dcl = v(:,6); ddh = v(:,7); ddl = v(:,8)
  dx8 = hx8; dk = hk
  call kern<<<(n + 127) / 128, 128>>>(op, n, dah, dal, dbh, dbl, dch, dcl, ddh, ddl, dx8, dk, drh, drl, dri)
  istat = cudaDeviceSynchronize()
  if (istat /= cudaSuccess) then
    print *, 'CUDA error: ', cudaGetErrorString(istat)
    stop 2
  end if
  hrh = drh; hrl = drl; hri = dri
  if (hri(1) == -999) stop 3
  open(newunit=u, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(u) hrh, hrl, hri
  close(u)
end program ff_test
