! test_extra.f90  —  fltflt CFD extensions: accuracy checks
!
! Build:  cd ouxmix && cmake -B build && cmake --build build -j
! Run:    cd build && ./test_extra
!
! Covers: dot2/dot3/dot4 (r4/r8/fltflt overloads),
!         min/max/clamp, ceil, hypot, cross3d,
!         pow_int, sign, lerp, warp_reduce_sum.

module test_extra_kern
  use cudafor
  use fltflt
  implicit none
  private
  public :: NCASE, NBLK
  public :: kern_dot2_all, kern_dot3_all, kern_dot4_all
  public :: kern_util, kern_warp_reduce

  integer, parameter :: NCASE = 8
  integer, parameter :: NBLK  = 256

  real(8), parameter :: FF_THRESH = 1.0d-12

contains

  ! ----------------------------------------------------------------
  ! One kernel per dot-N family: tests all three overloads (r4/r8/ff)
  ! from the same real(8) device inputs.
  ! ----------------------------------------------------------------

  attributes(global) subroutine kern_dot2_all(a_d,b_d,c_d,d_d,n, &
      r4_hi,r4_lo, r8_hi,r8_lo, ff_hi,ff_lo)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n),b_d(n),c_d(n),d_d(n)
    real(4), intent(out), device :: r4_hi(n),r4_lo(n)
    real(4), intent(out), device :: r8_hi(n),r8_lo(n)
    real(4), intent(out), device :: ff_hi(n),ff_lo(n)
    real(4) :: a4,b4,c4,d4
    type(fltflt) :: rr
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    a4 = real(a_d(i),4);  b4 = real(b_d(i),4)
    c4 = real(c_d(i),4);  d4 = real(d_d(i),4)
    ! r4 overload
    rr = fltflt_dot2(a4, b4, c4, d4)
    r4_hi(i) = rr%hi;  r4_lo(i) = rr%lo
    ! r8 overload
    rr = fltflt_dot2(a_d(i), b_d(i), c_d(i), d_d(i))
    r8_hi(i) = rr%hi;  r8_lo(i) = rr%lo
    ! ff overload: construct fltflt from r4 cast (non-trivial lo from fltflt_init(r4) is 0,
    ! so use r8-init to get meaningful lo components)
    rr = fltflt_dot2(fltflt_init(a_d(i)), fltflt_init(b_d(i)), &
                     fltflt_init(c_d(i)), fltflt_init(d_d(i)))
    ff_hi(i) = rr%hi;  ff_lo(i) = rr%lo
  end subroutine kern_dot2_all

  attributes(global) subroutine kern_dot3_all(a_d,b_d,c_d,d_d,e_d,f_d,n, &
      r4_hi,r4_lo, r8_hi,r8_lo, ff_hi,ff_lo)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n),b_d(n),c_d(n),d_d(n),e_d(n),f_d(n)
    real(4), intent(out), device :: r4_hi(n),r4_lo(n)
    real(4), intent(out), device :: r8_hi(n),r8_lo(n)
    real(4), intent(out), device :: ff_hi(n),ff_lo(n)
    real(4) :: a4,b4,c4,d4,e4,f4
    type(fltflt) :: rr
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    a4 = real(a_d(i),4);  b4 = real(b_d(i),4)
    c4 = real(c_d(i),4);  d4 = real(d_d(i),4)
    e4 = real(e_d(i),4);  f4 = real(f_d(i),4)
    rr = fltflt_dot3(a4, b4, c4, d4, e4, f4)
    r4_hi(i) = rr%hi;  r4_lo(i) = rr%lo
    rr = fltflt_dot3(a_d(i), b_d(i), c_d(i), d_d(i), e_d(i), f_d(i))
    r8_hi(i) = rr%hi;  r8_lo(i) = rr%lo
    rr = fltflt_dot3(fltflt_init(a_d(i)), fltflt_init(b_d(i)), &
                     fltflt_init(c_d(i)), fltflt_init(d_d(i)), &
                     fltflt_init(e_d(i)), fltflt_init(f_d(i)))
    ff_hi(i) = rr%hi;  ff_lo(i) = rr%lo
  end subroutine kern_dot3_all

  attributes(global) subroutine kern_dot4_all(a_d,b_d,c_d,d_d,e_d,f_d,g_d,h_d,n, &
      r4_hi,r4_lo, r8_hi,r8_lo, ff_hi,ff_lo)
    integer, intent(in),  value  :: n
    real(8), intent(in),  device :: a_d(n),b_d(n),c_d(n),d_d(n), &
                                    e_d(n),f_d(n),g_d(n),h_d(n)
    real(4), intent(out), device :: r4_hi(n),r4_lo(n)
    real(4), intent(out), device :: r8_hi(n),r8_lo(n)
    real(4), intent(out), device :: ff_hi(n),ff_lo(n)
    real(4) :: a4,b4,c4,d4,e4,f4,g4,h4
    type(fltflt) :: rr
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    a4 = real(a_d(i),4);  b4 = real(b_d(i),4)
    c4 = real(c_d(i),4);  d4 = real(d_d(i),4)
    e4 = real(e_d(i),4);  f4 = real(f_d(i),4)
    g4 = real(g_d(i),4);  h4 = real(h_d(i),4)
    rr = fltflt_dot4(a4, b4, c4, d4, e4, f4, g4, h4)
    r4_hi(i) = rr%hi;  r4_lo(i) = rr%lo
    rr = fltflt_dot4(a_d(i), b_d(i), c_d(i), d_d(i), a_d(i), b_d(i), c_d(i), d_d(i))
    r8_hi(i) = rr%hi;  r8_lo(i) = rr%lo
    rr = fltflt_dot4(fltflt_init(a_d(i)), fltflt_init(b_d(i)), &
                     fltflt_init(c_d(i)), fltflt_init(d_d(i)), &
                     fltflt_init(e_d(i)), fltflt_init(f_d(i)), &
                     fltflt_init(g_d(i)), fltflt_init(h_d(i)))
    ff_hi(i) = rr%hi;  ff_lo(i) = rr%lo
  end subroutine kern_dot4_all

  ! ----------------------------------------------------------------
  ! Utility kernels: min, max, clamp, ceil, hypot, cross3d, pow_int,
  !                  sign, lerp
  ! ----------------------------------------------------------------

  attributes(global) subroutine kern_util(n, &
      a_d, b_d, c_d, lo_d, hi_d, t_d, &
      min_hi, min_lo, max_hi, max_lo, clamp_hi, clamp_lo, &
      ceil_hi, ceil_lo, hypot_hi, hypot_lo, &
      cxhi, cxlo, cyhi, cylo, czhi, czlo, &
      pow_hi, pow_lo, sign_hi, sign_lo, lerp_hi, lerp_lo)
    integer, intent(in),  value  :: n
    real(4), intent(in),  device :: a_d(n), b_d(n), c_d(n)
    real(4), intent(in),  device :: lo_d(n), hi_d(n), t_d(n)
    real(4), intent(out), device :: min_hi(n),   min_lo(n)
    real(4), intent(out), device :: max_hi(n),   max_lo(n)
    real(4), intent(out), device :: clamp_hi(n), clamp_lo(n)
    real(4), intent(out), device :: ceil_hi(n),  ceil_lo(n)
    real(4), intent(out), device :: hypot_hi(n), hypot_lo(n)
    real(4), intent(out), device :: cxhi(n), cxlo(n)
    real(4), intent(out), device :: cyhi(n), cylo(n)
    real(4), intent(out), device :: czhi(n), czlo(n)
    real(4), intent(out), device :: pow_hi(n), pow_lo(n)
    real(4), intent(out), device :: sign_hi(n), sign_lo(n)
    real(4), intent(out), device :: lerp_hi(n), lerp_lo(n)
    type(fltflt) :: fa, fb, flo, fhi, r, cx, cy, cz
    integer :: i
    i = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    if (i > n) return
    fa  = fltflt_init(a_d(i))
    fb  = fltflt_init(b_d(i))
    flo = fltflt_init(lo_d(i))
    fhi = fltflt_init(hi_d(i))

    r = fltflt_min(fa, fb);       min_hi(i) = r%hi;   min_lo(i) = r%lo
    r = fltflt_max(fa, fb);       max_hi(i) = r%hi;   max_lo(i) = r%lo
    r = fltflt_clamp(fa,flo,fhi); clamp_hi(i) = r%hi; clamp_lo(i) = r%lo
    r = fltflt_ceil(fa);          ceil_hi(i) = r%hi;  ceil_lo(i) = r%lo
    r = fltflt_hypot(a_d(i), b_d(i)); hypot_hi(i) = r%hi; hypot_lo(i) = r%lo

    call fltflt_cross3d(cx, cy, cz, a_d(i), b_d(i), c_d(i), &
                        b_d(i), c_d(i), a_d(i))
    cxhi(i) = cx%hi;  cxlo(i) = cx%lo
    cyhi(i) = cy%hi;  cylo(i) = cy%lo
    czhi(i) = cz%hi;  czlo(i) = cz%lo

    r = fltflt_pow_int(fa, 5);    pow_hi(i) = r%hi;   pow_lo(i) = r%lo
    r = fltflt_sign(fa, fb);      sign_hi(i) = r%hi;  sign_lo(i) = r%lo
    r = fltflt_lerp(fa, fb, t_d(i)); lerp_hi(i) = r%hi; lerp_lo(i) = r%lo
  end subroutine kern_util

  ! warp_reduce_sum: each thread contributes its lane index (0-31);
  ! every lane should see sum = 0+1+...+31 = 496.
  attributes(global) subroutine kern_warp_reduce(out_hi, out_lo)
    real(4), intent(out), device :: out_hi(*), out_lo(*)
    type(fltflt) :: val, r
    integer :: lane, i
    lane  = mod(threadIdx%x - 1, 32)
    i     = (blockIdx%x - 1)*blockDim%x + threadIdx%x
    val%hi = real(lane, 4);  val%lo = 0.0
    r = fltflt_warp_reduce_sum(val)
    out_hi(i) = r%hi;  out_lo(i) = r%lo
  end subroutine kern_warp_reduce

end module test_extra_kern

! ==============================================================
program test_extra
  use cudafor
  use fltflt
  use test_extra_kern
  implicit none

  integer, parameter :: NC  = NCASE
  real(8), parameter :: TOL = 1.0d-12

  logical :: all_pass
  integer :: nfail, istat

  all_pass = .true.
  nfail    = 0

  ! ================================================================
  ! Helper: catastrophic-cancellation input set
  !   v = 1 + 2^{-13}, neg = -(1 + 2^{-12})
  !   pair (v,v): product = v^2 = 1 + 2^{-12} + 2^{-26}
  !   pair (neg,1): product = neg = -(1 + 2^{-12})
  !   sum of two such pairs ≈ 2^{-26}  (massive cancellation)
  ! ================================================================

  ! ================================================================
  ! 1. dot2: all three overloads
  ! ================================================================
  block
    integer, parameter :: N = NC
    real(8) :: v, neg
    real(8), allocatable :: a8(:),b8(:),c8(:),d8(:)
    real(8), allocatable, device :: a_d(:),b_d(:),c_d(:),d_d(:)
    real(4), allocatable, device :: r4hi_d(:),r4lo_d(:),r8hi_d(:),r8lo_d(:), &
                                    ffhi_d(:),fflo_d(:)
    real(4), allocatable :: r4hi(:),r4lo(:),r8hi(:),r8lo(:),ffhi(:),fflo(:)
    real(8) :: ref8, ff_val, ff_err
    integer :: i, nb
    logical :: pass

    allocate(a8(N),b8(N),c8(N),d8(N))
    allocate(a_d(N),b_d(N),c_d(N),d_d(N))
    allocate(r4hi_d(N),r4lo_d(N),r8hi_d(N),r8lo_d(N),ffhi_d(N),fflo_d(N))
    allocate(r4hi(N),r4lo(N),r8hi(N),r8lo(N),ffhi(N),fflo(N))

    v   = 1.0d0 + 2.0d0**(-13)
    neg = -(1.0d0 + 2.0d0**(-12))
    do i = 1, N
      a8(i)=v; b8(i)=v; c8(i)=neg; d8(i)=1.0d0
    end do
    a_d=a8; b_d=b8; c_d=c8; d_d=d8

    nb = (N + NBLK - 1) / NBLK
    call kern_dot2_all<<<nb,NBLK>>>(a_d,b_d,c_d,d_d,N, &
        r4hi_d,r4lo_d, r8hi_d,r8lo_d, ffhi_d,fflo_d)
    istat = cudaDeviceSynchronize()
    r4hi=r4hi_d; r4lo=r4lo_d
    r8hi=r8hi_d; r8lo=r8lo_d
    ffhi=ffhi_d; fflo=fflo_d

    pass = .true.
    do i = 1, N
      ref8 = a8(i)*b8(i) + c8(i)*d8(i)
      ! r4 overload
      ff_val = real(r4hi(i),8) + real(r4lo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val - ref8) / ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot2_r4 case ',i,' ff_err=',ff_err
        pass = .false.
      end if
      ! r8 overload
      ff_val = real(r8hi(i),8) + real(r8lo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val - ref8) / ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot2_r8 case ',i,' ff_err=',ff_err
        pass = .false.
      end if
      ! ff overload
      ff_val = real(ffhi(i),8) + real(fflo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val - ref8) / ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot2_ff case ',i,' ff_err=',ff_err
        pass = .false.
      end if
    end do
    if (pass) then
      write(*,'(A)') 'dot2 (r4/r8/ff): PASS'
    else
      all_pass = .false.; nfail = nfail + 1
    end if
    deallocate(a8,b8,c8,d8,a_d,b_d,c_d,d_d)
    deallocate(r4hi_d,r4lo_d,r8hi_d,r8lo_d,ffhi_d,fflo_d)
    deallocate(r4hi,r4lo,r8hi,r8lo,ffhi,fflo)
  end block

  ! ================================================================
  ! 2. dot3: all three overloads
  ! ================================================================
  block
    integer, parameter :: N = NC
    real(8) :: v, neg
    real(8), allocatable :: a8(:),b8(:),c8(:),d8(:),e8(:),f8(:)
    real(8), allocatable, device :: a_d(:),b_d(:),c_d(:),d_d(:),e_d(:),f_d(:)
    real(4), allocatable, device :: r4hi_d(:),r4lo_d(:),r8hi_d(:),r8lo_d(:), &
                                    ffhi_d(:),fflo_d(:)
    real(4), allocatable :: r4hi(:),r4lo(:),r8hi(:),r8lo(:),ffhi(:),fflo(:)
    real(8) :: ref8, ff_val, ff_err
    integer :: i, nb
    logical :: pass

    allocate(a8(N),b8(N),c8(N),d8(N),e8(N),f8(N))
    allocate(a_d(N),b_d(N),c_d(N),d_d(N),e_d(N),f_d(N))
    allocate(r4hi_d(N),r4lo_d(N),r8hi_d(N),r8lo_d(N),ffhi_d(N),fflo_d(N))
    allocate(r4hi(N),r4lo(N),r8hi(N),r8lo(N),ffhi(N),fflo(N))

    v   = 1.0d0 + 2.0d0**(-13)
    neg = -(1.0d0 + 2.0d0**(-12))
    do i = 1, N
      a8(i)=v; b8(i)=v; c8(i)=neg; d8(i)=1.0d0; e8(i)=v; f8(i)=v
    end do
    a_d=a8; b_d=b8; c_d=c8; d_d=d8; e_d=e8; f_d=f8

    nb = (N + NBLK - 1) / NBLK
    call kern_dot3_all<<<nb,NBLK>>>(a_d,b_d,c_d,d_d,e_d,f_d,N, &
        r4hi_d,r4lo_d, r8hi_d,r8lo_d, ffhi_d,fflo_d)
    istat = cudaDeviceSynchronize()
    r4hi=r4hi_d; r4lo=r4lo_d
    r8hi=r8hi_d; r8lo=r8lo_d
    ffhi=ffhi_d; fflo=fflo_d

    pass = .true.
    do i = 1, N
      ref8 = a8(i)*b8(i) + c8(i)*d8(i) + e8(i)*f8(i)
      ff_val = real(r4hi(i),8)+real(r4lo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val-ref8)/ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot3_r4 case ',i,' ff_err=',ff_err
        pass = .false.
      end if
      ff_val = real(r8hi(i),8)+real(r8lo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val-ref8)/ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot3_r8 case ',i,' ff_err=',ff_err
        pass = .false.
      end if
      ff_val = real(ffhi(i),8)+real(fflo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val-ref8)/ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot3_ff case ',i,' ff_err=',ff_err
        pass = .false.
      end if
    end do
    if (pass) then
      write(*,'(A)') 'dot3 (r4/r8/ff): PASS'
    else
      all_pass = .false.; nfail = nfail + 1
    end if
    deallocate(a8,b8,c8,d8,e8,f8,a_d,b_d,c_d,d_d,e_d,f_d)
    deallocate(r4hi_d,r4lo_d,r8hi_d,r8lo_d,ffhi_d,fflo_d)
    deallocate(r4hi,r4lo,r8hi,r8lo,ffhi,fflo)
  end block

  ! ================================================================
  ! 3. dot4: all three overloads
  ! ================================================================
  block
    integer, parameter :: N = NC
    real(8) :: v, neg
    real(8), allocatable :: a8(:),b8(:),c8(:),d8(:),e8(:),f8(:),g8(:),h8(:)
    real(8), allocatable, device :: a_d(:),b_d(:),c_d(:),d_d(:), &
                                    e_d(:),f_d(:),g_d(:),h_d(:)
    real(4), allocatable, device :: r4hi_d(:),r4lo_d(:),r8hi_d(:),r8lo_d(:), &
                                    ffhi_d(:),fflo_d(:)
    real(4), allocatable :: r4hi(:),r4lo(:),r8hi(:),r8lo(:),ffhi(:),fflo(:)
    real(8) :: ref8, ff_val, ff_err
    integer :: i, nb
    logical :: pass

    allocate(a8(N),b8(N),c8(N),d8(N),e8(N),f8(N),g8(N),h8(N))
    allocate(a_d(N),b_d(N),c_d(N),d_d(N),e_d(N),f_d(N),g_d(N),h_d(N))
    allocate(r4hi_d(N),r4lo_d(N),r8hi_d(N),r8lo_d(N),ffhi_d(N),fflo_d(N))
    allocate(r4hi(N),r4lo(N),r8hi(N),r8lo(N),ffhi(N),fflo(N))

    v   = 1.0d0 + 2.0d0**(-13)
    neg = -(1.0d0 + 2.0d0**(-12))
    do i = 1, N
      a8(i)=v;   b8(i)=v;   c8(i)=neg; d8(i)=1.0d0
      e8(i)=v;   f8(i)=v;   g8(i)=neg; h8(i)=1.0d0
    end do
    a_d=a8; b_d=b8; c_d=c8; d_d=d8; e_d=e8; f_d=f8; g_d=g8; h_d=h8

    nb = (N + NBLK - 1) / NBLK
    call kern_dot4_all<<<nb,NBLK>>>(a_d,b_d,c_d,d_d,e_d,f_d,g_d,h_d,N, &
        r4hi_d,r4lo_d, r8hi_d,r8lo_d, ffhi_d,fflo_d)
    istat = cudaDeviceSynchronize()
    r4hi=r4hi_d; r4lo=r4lo_d
    r8hi=r8hi_d; r8lo=r8lo_d
    ffhi=ffhi_d; fflo=fflo_d

    pass = .true.
    do i = 1, N
      ref8 = a8(i)*b8(i) + c8(i)*d8(i) + e8(i)*f8(i) + g8(i)*h8(i)
      ff_val = real(r4hi(i),8)+real(r4lo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val-ref8)/ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot4_r4 case ',i,' ff_err=',ff_err
        pass = .false.
      end if
      ! r8 overload: note kern uses (a,b,c,d,a,b,c,d) for the 8 args
      ref8 = a8(i)*b8(i) + c8(i)*d8(i) + a8(i)*b8(i) + c8(i)*d8(i)
      ff_val = real(r8hi(i),8)+real(r8lo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val-ref8)/ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot4_r8 case ',i,' ff_err=',ff_err
        pass = .false.
      end if
      ! ff overload: same (a,b,c,d,e,f,g,h) inputs as r4
      ref8 = a8(i)*b8(i) + c8(i)*d8(i) + e8(i)*f8(i) + g8(i)*h8(i)
      ff_val = real(ffhi(i),8)+real(fflo(i),8)
      ff_err = 0.0d0
      if (ref8 /= 0.0d0) ff_err = abs((ff_val-ref8)/ref8)
      if (ff_err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL dot4_ff case ',i,' ff_err=',ff_err
        pass = .false.
      end if
    end do
    if (pass) then
      write(*,'(A)') 'dot4 (r4/r8/ff): PASS'
    else
      all_pass = .false.; nfail = nfail + 1
    end if
    deallocate(a8,b8,c8,d8,e8,f8,g8,h8,a_d,b_d,c_d,d_d,e_d,f_d,g_d,h_d)
    deallocate(r4hi_d,r4lo_d,r8hi_d,r8lo_d,ffhi_d,fflo_d)
    deallocate(r4hi,r4lo,r8hi,r8lo,ffhi,fflo)
  end block

  ! ================================================================
  ! 4. Utility functions
  ! ================================================================
  block
    integer, parameter :: N = 4
    real(4) :: a(N), b(N), c(N), lo(N), hi(N), t(N)
    real(4), device :: a_d(N), b_d(N), c_d(N), lo_d(N), hi_d(N), t_d(N)
    real(4) :: min_hi(N),min_lo(N),max_hi(N),max_lo(N)
    real(4) :: clamp_hi(N),clamp_lo(N),ceil_hi(N),ceil_lo(N)
    real(4) :: hypot_hi(N),hypot_lo(N)
    real(4) :: cxhi(N),cxlo(N),cyhi(N),cylo(N),czhi(N),czlo(N)
    real(4) :: pow_hi(N),pow_lo(N),sign_hi(N),sign_lo(N),lerp_hi(N),lerp_lo(N)
    real(4), device :: dmin_hi(N),dmin_lo(N),dmax_hi(N),dmax_lo(N)
    real(4), device :: dclamp_hi(N),dclamp_lo(N),dceil_hi(N),dceil_lo(N)
    real(4), device :: dhypot_hi(N),dhypot_lo(N)
    real(4), device :: dcxhi(N),dcxlo(N),dcyhi(N),dcylo(N),dczhi(N),dczlo(N)
    real(4), device :: dpow_hi(N),dpow_lo(N),dsign_hi(N),dsign_lo(N)
    real(4), device :: dlerp_hi(N),dlerp_lo(N)
    real(8) :: ref8, ff_val, err
    integer :: i, nb
    logical :: pass

    a  = (/  3.0, -2.5,  0.0,  1.25 /)
    b  = (/  1.0, -3.0,  4.0, -0.5  /)
    c  = (/  2.0,  1.0, -1.0,  0.3  /)
    lo = (/ -1.0, -4.0, -2.0,  0.0  /)
    hi = (/  2.0,  0.0,  2.0,  1.0  /)
    t  = (/  0.0,  0.5,  1.0,  0.25 /)

    a_d=a; b_d=b; c_d=c; lo_d=lo; hi_d=hi; t_d=t

    nb = (N + NBLK - 1) / NBLK
    call kern_util<<<nb,NBLK>>>(N, a_d,b_d,c_d,lo_d,hi_d,t_d, &
      dmin_hi,dmin_lo,dmax_hi,dmax_lo,dclamp_hi,dclamp_lo, &
      dceil_hi,dceil_lo,dhypot_hi,dhypot_lo, &
      dcxhi,dcxlo,dcyhi,dcylo,dczhi,dczlo, &
      dpow_hi,dpow_lo,dsign_hi,dsign_lo,dlerp_hi,dlerp_lo)
    istat = cudaDeviceSynchronize()
    min_hi=dmin_hi;   min_lo=dmin_lo
    max_hi=dmax_hi;   max_lo=dmax_lo
    clamp_hi=dclamp_hi; clamp_lo=dclamp_lo
    ceil_hi=dceil_hi;   ceil_lo=dceil_lo
    hypot_hi=dhypot_hi; hypot_lo=dhypot_lo
    cxhi=dcxhi; cxlo=dcxlo; cyhi=dcyhi; cylo=dcylo; czhi=dczhi; czlo=dczlo
    pow_hi=dpow_hi; pow_lo=dpow_lo
    sign_hi=dsign_hi; sign_lo=dsign_lo
    lerp_hi=dlerp_hi; lerp_lo=dlerp_lo

    pass = .true.
    do i = 1, N
      ref8 = real(min(a(i),b(i)),8)
      ff_val = real(min_hi(i),8)+real(min_lo(i),8)
      if (abs(ff_val-ref8)>0.0d0) then
        write(*,'(A,I0,2(A,F8.3))') '  FAIL min case ',i,' got=',ff_val,' ref=',ref8
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = real(max(a(i),b(i)),8)
      ff_val = real(max_hi(i),8)+real(max_lo(i),8)
      if (abs(ff_val-ref8)>0.0d0) then
        write(*,'(A,I0,2(A,F8.3))') '  FAIL max case ',i,' got=',ff_val,' ref=',ref8
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = real(max(lo(i),min(a(i),hi(i))),8)
      ff_val = real(clamp_hi(i),8)+real(clamp_lo(i),8)
      if (abs(ff_val-ref8)>0.0d0) then
        write(*,'(A,I0,2(A,F8.3))') '  FAIL clamp case ',i,' got=',ff_val,' ref=',ref8
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = real(ceiling(a(i)),8)
      ff_val = real(ceil_hi(i),8)+real(ceil_lo(i),8)
      if (abs(ff_val-ref8)>0.0d0) then
        write(*,'(A,I0,2(A,F8.3))') '  FAIL ceil case ',i,' got=',ff_val,' ref=',ref8
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = sqrt(real(a(i),8)**2+real(b(i),8)**2)
      ff_val = real(hypot_hi(i),8)+real(hypot_lo(i),8)
      err = 0.0d0
      if (ref8 /= 0.0d0) err = abs((ff_val-ref8)/ref8)
      if (err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL hypot case ',i,' rel_err=',err
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = real(b(i),8)*real(a(i),8) - real(c(i),8)*real(c(i),8)
      ff_val = real(cxhi(i),8)+real(cxlo(i),8)
      err = 0.0d0
      if (ref8 /= 0.0d0) err = abs((ff_val-ref8)/ref8)
      if (err > TOL .and. abs(ff_val-ref8)>1.0d-30) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL cross3d cx case ',i,' rel_err=',err
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = real(a(i),8)**5
      ff_val = real(pow_hi(i),8)+real(pow_lo(i),8)
      err = 0.0d0
      if (ref8 /= 0.0d0) err = abs((ff_val-ref8)/ref8)
      if (err > TOL) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL pow_int case ',i,' rel_err=',err
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = abs(real(a(i),8))*sign(1.0d0,real(b(i),8))
      ff_val = real(sign_hi(i),8)+real(sign_lo(i),8)
      if (abs(ff_val-ref8)>1.0d-30) then
        write(*,'(A,I0,2(A,F8.3))') '  FAIL sign case ',i,' got=',ff_val,' ref=',ref8
        pass = .false.
      end if
    end do
    do i = 1, N
      ref8 = real(a(i),8)+real(t(i),8)*(real(b(i),8)-real(a(i),8))
      ff_val = real(lerp_hi(i),8)+real(lerp_lo(i),8)
      err = 0.0d0
      if (ref8 /= 0.0d0) err = abs((ff_val-ref8)/ref8)
      if (err > 1.0d-6) then
        write(*,'(A,I0,A,ES10.2)') '  FAIL lerp case ',i,' rel_err=',err
        pass = .false.
      end if
    end do

    if (pass) then
      write(*,'(A)') 'min/max/clamp/ceil/hypot/cross3d/pow_int/sign/lerp: PASS'
    else
      all_pass = .false.; nfail = nfail + 1
    end if
  end block

  ! ================================================================
  ! 5. warp_reduce_sum
  ! ================================================================
  block
    integer, parameter :: NTHR = 64
    real(4) :: out_hi(NTHR), out_lo(NTHR)
    real(4), device :: out_hi_d(NTHR), out_lo_d(NTHR)
    real(8) :: ff_val
    integer :: i
    logical :: pass
    real(8), parameter :: WSUM = 496.0d0

    call kern_warp_reduce<<<1,NTHR>>>(out_hi_d, out_lo_d)
    istat = cudaDeviceSynchronize()
    out_hi = out_hi_d;  out_lo = out_lo_d

    pass = .true.
    do i = 1, NTHR
      ff_val = real(out_hi(i),8)+real(out_lo(i),8)
      if (abs(ff_val-WSUM)>0.0d0) then
        write(*,'(A,I0,A,F10.1)') '  FAIL warp_reduce lane ',i,' sum=',ff_val
        pass = .false.
      end if
    end do
    if (pass) then
      write(*,'(A)') 'warp_reduce_sum: PASS'
    else
      all_pass = .false.; nfail = nfail + 1
    end if
  end block

  ! ================================================================
  ! Summary
  ! ================================================================
  write(*,*)
  if (all_pass) then
    write(*,'(A)') 'ALL PASS'
  else
    write(*,'(A,I0,A)') 'FAIL: ', nfail, ' test(s) failed'
    stop 1
  end if

end program test_extra
