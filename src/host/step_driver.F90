module step_driver
  ! Host time loop. Named steps match roms_bugfix/src/main.F90.
  ! nfast stays on the host. Each fast step is a CUDA graph of
  ! {step2d interiors, halo_fill} with host OBC between the sections.
  ! Physics kernels that are not written yet are explicit stubs.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use geom
  use halo_mod
  use step2d_mod, only: step2d_with_halo
  use prsgrd_mod, only: prsgrd, dump_pgf_col
  use advect, only: tracer_advect_fluxes
  use rho_eos_mod, only: rho_eos
  use pre_step3d_mod, only: pre_step3d, ncol_probe, icol_probe, jcol_probe
  use step3d_uv_mod, only: step3d_uv1, step3d_uv2
  use marbl_gpu, only: marbl_gpu_step
  use visc3d_mod, only: visc3d
  use step3d_t_mod, only: step3d_t
  use t3dmix_mod, only: t3dmix
  use passive_mod, only: ntp, passive_t3dmix
  use lmd_mod, only: lmd_vmix
  use frc_io, only: set_forces
  use bry_io, only: set_bry_all, dump_bry_north
  use tide_io, only: set_tides
  use arch_mod, only: sync_arch
  implicit none
  private

  public :: run_geom_only
  public :: run_baroclinic_step
  public :: run_barotropic_only
  public :: run_prsgrd_only
  public :: run_advect_only
  public :: run_lmd_only
  public :: not_implemented

  integer, save :: clock_rate = 0
  integer, save :: clock_start = 0
  integer, save :: clock_last = 0
contains

  subroutine run_geom_only()
    real(rp) :: umis, vmis, vol

    iic = 0
    knew = 1
    nrhs = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call set_HUV()
    call omega()
    call check_mode_coupling(umis, vmis, vol)
    write(*,'(a,es14.6)') 'geom_only: volume m^3 = ', vol
    write(*,'(a,es14.6)') 'geom_only: max |Hz|   = ', maxval(abs(Hz))
    write(*,'(a,es14.6)') 'geom_only: max |We|   = ', maxval(abs(We))
  end subroutine run_geom_only

  subroutine run_baroclinic_step()
    integer(ip) :: n
    real(rp) :: umis, vmis, vol, umax, akvmax, hblmax
    logical :: info_step

    if (iic < ntstart) iic = ntstart
    call update_clock()
    call set_probe_windows()
    probe_on = (iic - ntstart < probe_steps)
    fast_probe_on = (iic - ntstart >= fast_probe_from .and. &
         iic - ntstart < fast_probe_to)
    call begin_predictor_indices()
    call step_timer_init()

    ! --- predictor (nrhs = n, nnew = 3) ---
    frc_slot = 0
    call set_forces()
    call dump_swflx_b('after pred set_forces')
    call step_mark('set_forces')
    frc_slot = 1
    call set_bry_all()
    call step_mark('set_bry')
    call set_tides()
    call dump_bry_north('after tide', frc_modtime())
    call step_mark('set_tides')
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call step_mark('halo zeta')
    call halo_fill_3d_s(u, STAGGER_U, nrhs)
    call step_mark('halo u')
    call halo_fill_3d_s(v, STAGGER_V, nrhs)
    call step_mark('halo v')
    call dump_vic('after halo v')
    call halo_fill_3d_t(t, STAGGER_RHO, nrhs, 1_ip)
    call step_mark('halo t')
    call set_depth()
    call step_mark('set_depth')
    call dump_pgf_col('after set_depth')
    call dump_rim('after set_depth')
    call rho_eos(nrhs)
    call step_mark('rho_eos')
    call dump_pgf_col('after rho_eos')
    call set_HUV()
    call step_mark('set_HUV')
    call dump_huv_probe('after set_HUV')
    call dump_p1_fun('set_HUV')
    call dump_rim('set_HUV')
    call omega()
    call dump_we_b('after pred omega')
    call step_mark('omega')
    call lmd_vmix(nstp)
    call step_mark('lmd_vmix')
    call prsgrd()
    call step_mark('prsgrd')
    call dump_col_probe('start', nrhs)
    call pre_step3d()
    call step_mark('pre_step3d')
    call dump_col_probe('after pre_step3d', nnew)
    call dump_uvt('after pre_step3d', nnew)
    call dump_vic('after pre_step3d')

    call set_HUV1()
    call step_mark('set_HUV1')
    call dump_huv_probe('after set_HUV1')
    call dump_p1_fun('set_HUV1')
    call dump_rim('set_HUV1')
    call dump_uvt('after set_HUV1', nnew)
    call dump_vic('after set_HUV1')
    call begin_corrector_indices()
    ! CPU: corr omega uses the still-current (time-n) swflx, then
    ! set_forces interpolates n+1/2, then bry/tides, then prsgrd/uv1.
    call dump_swflx_b('before corr omega')
    call omega()
    call dump_we_b('after corr omega')
    call dump_swflx_b('after corr omega')
    call rho_eos(nrhs)
    frc_slot = 1
    call set_forces()
    call dump_swflx_b('after corr set_forces')
    call lmd_vmix(nrhs)
    frc_slot = 3
    call set_bry_all()
    call set_tides()
    call dump_bry_north('after tide', frc_modtime())
    call step_mark('corrector forces')
    call dump_vic('before uv1')
    call dump_p1_fun('before_uv1')
    call dump_rim('before_uv1')
    call prsgrd()
    call step_mark('corrector rhs')
    call step3d_uv1()
    call visc3d()
    call step_mark('step3d_uv1+visc')
    call dump_uvt('after step3d_uv1', nnew)
    call dump_bt2d_probe('after step3d_uv1+visc')
    call dump_rvS('after step3d_uv1+visc')
    call dump_rufrc_build('after visc')
    call dump_ztd_pt('after step3d_uv1+visc')
    call dump_rim('after step3d_uv1+visc')

    do n = 1, nfast
      iif = n
      call cycle_fast_index()
      call step2d_with_halo(iif)
      if (n == 1) call dump_bt2d_probe('after iif=1')
      if (n == nfast) then
        call dump_bt2d_probe('after last fast')
        call dump_ztd_pt('after last fast')
        call dump_rim('after last fast')
      endif
      call dump_fast_probe()
    enddo
    call step_mark('step2d')
    call dump_ic_jet_2d('after step2d')

    call step3d_uv2()
    call step_mark('step3d_uv2')
    call dump_huv_probe('after step3d_uv2')
    call dump_p1_fun('after step3d_uv2')
    call dump_uvt('after step3d_uv2', nnew)
    call dump_rim('after step3d_uv2')
    call omega()
    call dump_we_b('after uv2 omega')
    call dump_swflx_b('after uv2 omega')
    call step3d_t()
    call marbl_gpu_step()
    call dump_max_t('after step3d_t')
    call dump_col_probe('after step3d_t', nnew)
    call dump_rim('after step3d_t')
    call t3dmix()
    if (ntp > 0) call passive_t3dmix()
    call dump_col_probe('after t3dmix', nnew)
    call rho_eos(nnew)
    call step_mark('step3d_t')
    ! Step diagnostics are device reductions, every ninfo steps.
    info_step = (mod(iic - ntstart, ninfo_steps) == 0)
    if (info_step) then
      call check_mode_coupling(umis, vmis, vol)
      write(*,'(a,3es14.6)') 'coupling: u_mis, v_mis, vol = ', umis, vmis, vol
      call step_maxima(umax, akvmax, hblmax)
      write(*,'(a,es14.6)') 'step: max |u| = ', umax
    endif
    call dump_ztd_pt('end of step')
    call dump_rim('end of step')
    call dump_ztd_stats()
    call dump_max_t('end of step')
    if (info_step) then
      write(*,'(a,es14.6)') 'step: max Akv = ', akvmax
      write(*,'(a,es14.6)') 'step: max hbls = ', hblmax
    endif
    call step_mark('baroclinic step')
  end subroutine run_baroclinic_step

  subroutine run_barotropic_only()
    integer(ip) :: n
    real(rp) :: vol0, vol1

    iic = 1
    knew = 1
    kstp = 1
    call hg_begin()
    call hg_add(F_ZETA, 1_ip)
    call hg_add(F_UBAR, 1_ip)
    call hg_add(F_VBAR, 1_ip)
    call hg_end()
    call set_depth()

    vol0 = sum((h(1:decomp%nx, 1:decomp%ny) + zeta(1:decomp%nx, 1:decomp%ny, 1)) / &
               (pm(1:decomp%nx, 1:decomp%ny) * pn(1:decomp%nx, 1:decomp%ny)))

    do n = 1, nfast
      iif = n
      call cycle_fast_index()
      call step2d_with_halo(iif)
    enddo

    vol1 = sum((h(1:decomp%nx, 1:decomp%ny) + zeta(1:decomp%nx, 1:decomp%ny, knew)) / &
               (pm(1:decomp%nx, 1:decomp%ny) * pn(1:decomp%nx, 1:decomp%ny)))
    write(*,'(a,i4)') 'barotropic: nfast = ', nfast
    write(*,'(a,2es14.6)') 'barotropic: vol0, vol1 = ', vol0, vol1
    write(*,'(a,es14.6)') 'barotropic: max |zeta| = ', maxval(abs(zeta(1:decomp%nx,1:decomp%ny,knew)))
    write(*,'(a,es14.6)') 'barotropic: max |ubar| = ', maxval(abs(ubar(1:decomp%nx+1,1:decomp%ny,knew)))
  end subroutine run_barotropic_only

  subroutine run_advect_only()
    integer(ip) :: tidx, itrc

    tidx = 1
    itrc = 1
    iic = 0
    knew = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call set_HUV()
    call omega()
    ! Ideal box is at rest, so volume fluxes are 0. Impose a uniform
    ! transport so the kernel has something to interpolate.
    FlxU = 1.0_rp
    FlxV = 1.0_rp
    We = 1.0_rp
    call tracer_advect_fluxes(tidx, itrc, .true.)
    write(*,'(a,es14.6)') 'advect C4: max |FX| = ', maxval(abs(wrk3a(1:decomp%nx+1,1:decomp%ny,:)))
    write(*,'(a,es14.6)') 'advect C4: max |FE| = ', maxval(abs(wrk3b(1:decomp%nx,1:decomp%ny+1,:)))
    write(*,'(a,es14.6)') 'advect C4: max |Fz| = ', maxval(abs(col_wc(1:decomp%nx,1:decomp%ny,1:decomp%nz-1)))
    call tracer_advect_fluxes(tidx, itrc, .false.)
    write(*,'(a,es14.6)') 'advect U3: max |Fz| = ', maxval(abs(col_wc(1:decomp%nx,1:decomp%ny,1:decomp%nz-1)))
  end subroutine run_advect_only

  subroutine run_lmd_only()
    iic = 1
    knew = 1
    nstp = 1
    nrhs = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call rho_eos(nstp)
    call lmd_vmix(nstp)
    write(*,'(a,es14.6)') 'lmd: max Akv  = ', maxval(Akv(1:decomp%nx,1:decomp%ny,:))
    write(*,'(a,es14.6)') 'lmd: max AktT = ', maxval(Akt(1:decomp%nx,1:decomp%ny,:,itemp))
    write(*,'(a,es14.6)') 'lmd: max hbls = ', maxval(hbls(1:decomp%nx,1:decomp%ny))
    write(*,'(a,es14.6)') 'lmd: max hbbl = ', maxval(hbbl(1:decomp%nx,1:decomp%ny))
    write(*,'(a,es14.6)') 'lmd: min swr  = ', minval(swr_frac(1:decomp%nx,1:decomp%ny,0))
  end subroutine run_lmd_only

  subroutine run_prsgrd_only()
    iic = 0
    knew = 1
    call halo_fill_2d_s(zeta, STAGGER_RHO, knew)
    call set_depth()
    call prsgrd()
    write(*,'(a,es14.6)') 'prsgrd: max |ru| = ', maxval(abs(ru(1:decomp%nx+1,1:decomp%ny,:)))
    write(*,'(a,es14.6)') 'prsgrd: max |rv| = ', maxval(abs(rv(1:decomp%nx,1:decomp%ny+1,:)))
    write(*,'(a,es14.6)') 'prsgrd: max |P|  = ', maxval(abs(P(1:decomp%nx,1:decomp%ny,:)))
  end subroutine run_prsgrd_only

  subroutine dump_max_t(tag)
    ! After Thomas, t(:,:,:,nnew) is temperature. Print the hot cell
    ! together with the 1/Hz flux pieces that can heat a 3 cm box.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, im, jm, km, ih, jh, kh
    real(rp) :: tmax, hmin

    if (.not. probe_on) return
    call sync_arch()
    tmax = 0.0_rp
    hmin = 1.0e30_rp
    im = 0; jm = 0; km = 0
    ih = 0; jh = 0; kh = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (Hz(i, j, k) < hmin) then
            hmin = Hz(i, j, k)
            ih = i; jh = j; kh = k
          endif
          if (abs(t(i, j, k, nnew, itemp)) > tmax) then
            tmax = abs(t(i, j, k, nnew, itemp))
            im = i; jm = j; km = k
          endif
        enddo
      enddo
    enddo
    write(*,'(a,a,a,es14.6,a,3i6)') 'diag: ', trim(tag), '  max |t| = ', tmax, &
         ' at i,j,k =', im, jm, km
    if (im > 0) call dump_t_cell(tag, 'hot', im, jm, km)
    if (ih > 0 .and. (ih /= im .or. jh /= jm .or. kh /= km)) then
      write(*,'(a,a,a,es14.6,a,3i6)') 'diag: ', trim(tag), '  min Hz = ', hmin, &
           ' at i,j,k =', ih, jh, kh
      call dump_t_cell(tag, 'thin', ih, jh, kh)
    endif
    flush(6)
  end subroutine dump_max_t

  subroutine dump_t_cell(tag, role, i, j, k)
    character(len=*), intent(in) :: tag, role
    integer(ip), intent(in) :: i, j, k
    integer(ip) :: kg
    real(rp) :: Tval, hzv, Dcol, dTst, dTrain, dTswr

    if (.not. probe_on) return
    Tval = t(i, j, k, nnew, itemp)
    hzv = max(Hz(i, j, k), 1.0e-30_rp)
    Dcol = h(i, j) + zeta(i, j, knew)
    dTst = dt * stflx(i, j, itemp) / hzv
    if (have_bulk) then
      dTrain = dt * swflx(i, j) * tair(i, j) / hzv
    else
      dTrain = dt * swflx(i, j) * Tval / hzv
    endif
    dTswr = dt * srflx(i, j) * swr_frac(i, j, 0) / hzv
    kg = max(k - 1, 0)
    write(*,'(a,a,a,a,a,3i6)') 'diag: ', trim(tag), '  ', trim(role), ' i,j,k=', i, j, k
    write(*,'(a,a,a,a,a,5es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' T,Hz,h,zeta,D=', Tval, Hz(i, j, k), h(i, j), zeta(i, j, knew), Dcol
    write(*,'(a,a,a,a,a,5es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' stflx,srflx,swflx,swr0,tair=', stflx(i, j, itemp), srflx(i, j), &
         swflx(i, j), swr_frac(i, j, 0), tair(i, j)
    write(*,'(a,a,a,a,a,4es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' dT(stflx,rain,swr0)/step,ghat=', dTst, dTrain, dTswr, ghat(i, j, kg)
    write(*,'(a,a,a,a,a,4es14.6)') 'diag: ', trim(tag), '  ', trim(role), &
         ' Tbot,Tsrf,hbls,Akt=', t(i, j, 1, nnew, itemp), &
         t(i, j, decomp%nz, nnew, itemp), hbls(i, j), Akt(i, j, max(k - 1, 1), itemp)
  end subroutine dump_t_cell

  subroutine dump_uvt(tag, slot)
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: slot
    integer(ip) :: n, i, j, k, im, jm, km, iu, ju, ku
    real(rp) :: tmax, umax, hzu
    if (.not. probe_on) return
    n = slot
    call sync_arch()
    write(*,'(a,a,a,i0,a,es14.6,a,es14.6)') 'diag: ', trim(tag), '  slot=', n, &
         '  max |u| = ', maxval(abs(u(1:decomp%nx+1,1:decomp%ny,:,n)), &
         mask=spread(umask(1:decomp%nx+1,1:decomp%ny) > 0.5_rp, 3, decomp%nz)), &
         '  max |t| = ', maxval(abs(t(1:decomp%nx,1:decomp%ny,:,n,1)), &
         mask=spread(rmask(1:decomp%nx,1:decomp%ny) > 0.5_rp, 3, decomp%nz))
    umax = 0.0_rp
    iu = 0; ju = 0; ku = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx + 1
        if (umask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (abs(u(i, j, k, n)) > umax) then
            umax = abs(u(i, j, k, n))
            iu = i; ju = j; ku = k
          endif
        enddo
      enddo
    enddo
    if (iu > 0) then
      hzu = 0.5_rp * (Hz(iu, ju, ku) + Hz(iu - 1, ju, ku))
      write(*,'(a,a,a,3i6,a,3es14.6)') 'diag: ', trim(tag), '  u at i,j,k=', iu, ju, ku, &
           '  Hz_u,u,u/Hz_u=', hzu, u(iu, ju, ku, n), &
           u(iu, ju, ku, n) / max(hzu, 1.0e-30_rp)
    endif
    if (ic_u_i > 0) then
      hzu = 0.5_rp * (Hz(ic_u_i, ic_u_j, ic_u_k) + Hz(ic_u_i - 1, ic_u_j, ic_u_k))
      write(*,'(a,a,a,3i6,a,3es14.6)') 'diag: ', trim(tag), '  IC-jet u at i,j,k=', &
           ic_u_i, ic_u_j, ic_u_k, '  Hz_u,u,u/Hz_u=', hzu, &
           u(ic_u_i, ic_u_j, ic_u_k, n), &
           u(ic_u_i, ic_u_j, ic_u_k, n) / max(hzu, 1.0e-30_rp)
    endif
    tmax = 0.0_rp
    im = 0; jm = 0; km = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (abs(t(i, j, k, n, 1)) > tmax) then
            tmax = abs(t(i, j, k, n, 1))
            im = i; jm = j; km = k
          endif
        enddo
      enddo
    enddo
    if (im > 0) then
      write(*,'(a,a,a,3i6,a,3es14.6)') 'diag: ', trim(tag), '  t at i,j,k=', im, jm, km, &
           '  Hz,Hz_fwd,t/Hz=', Hz(im, jm, km), Hz_fwd(im, jm, km), &
           t(im, jm, km, n, 1) / max(Hz(im, jm, km), 1.0e-30_rp)
    endif
    flush(6)
    call step_timer_lap()
  end subroutine dump_uvt

  subroutine dump_col_probe(tag, slot)
    ! Same "diag: col" lines as roms_bugfix smoke_dump_col.
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: slot
    integer(ip) :: n, i, j, k
    if (.not. probe_on) return
    call sync_arch()
    do n = 1, ncol_probe
      i = icol_probe(n)
      j = jcol_probe(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(*,'(a,a,a,i1,a,i6,8es22.14)') 'diag: colx ', trim(tag), ' P', n, &
           ' iic,hbls,hbbl,r_D,u1W,u1E,v1S,v1N(slot),u1W(nstp)=', iic, &
           hbls(i, j), hbbl(i, j), r_D(i, j), u(i, j, 1, slot), &
           u(i + 1, j, 1, slot), v(i, j, 1, slot), v(i, j + 1, 1, slot), &
           u(i, j, 1, nstp)
      do k = 1, decomp%nz
        write(*,'(a,a,a,i1,a,i6,i4,4es22.14)') 'diag: col ', trim(tag), ' P', n, &
             ' iic,k,T,S,AktT,Akv=', iic, k, t(i, j, k, slot, itemp), &
             t(i, j, k, slot, isalt), Akt(i, j, k, itemp), Akv(i, j, k)
      enddo
      do k = 1, decomp%nz
        if (k /= 1 .and. k /= 2 .and. k /= 3 .and. k /= 5 .and. k /= 10) cycle
        write(*,'(a,a,a,i1,a,i6,i4,4es22.14)') 'diag: cold ', trim(tag), ' P', n, &
             ' iic,k,Hz,z_r,We,Wi=', iic, k, Hz(i, j, k), z_r(i, j, k), &
             We(i, j, k), Wi(i, j, k)
      enddo
    enddo
    flush(6)
  end subroutine dump_col_probe

  subroutine dump_ic_jet_2d(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k
    real(rp) :: Dcol, dn, du_bar
    if (.not. probe_on) return
    if (ic_u_i <= 0) return
    call sync_arch()
    i = ic_u_i
    j = ic_u_j
    Dcol = 0.0_rp
    do k = 1, decomp%nz
      Dcol = Dcol + 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k))
    enddo
    dn = dn_u(i, j)
    du_bar = DU_avg1(i, j) / max(Dcol * dn, 1.0e-30_rp)
    write(*,'(a,a,a,2i6,a,4es14.6)') 'diag: ', trim(tag), '  IC-jet 2D i,j=', i, j, &
         '  Dcol,ubar,DU/D,DU_avg1=', Dcol, ubar(i, j, knew), du_bar, DU_avg1(i, j)
    flush(6)
    call step_timer_lap()
  end subroutine dump_ic_jet_2d

  subroutine dump_ztd_pt(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: n, i, j
    if (.not. probe_on) return
    call sync_arch()
    do n = 1, nprb_pt
      i = iprb_pt(n)
      j = jprb_pt(n)
      if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) cycle
      write(*,'(a,i1,a,a,a,i8,a,6es22.14)') 'diag: ztdP', n, '  ', trim(tag), &
           '  iic=', iic, '  zeta,Zt_avg1,Hz1,zr1,T1,We1=', &
           zeta(i, j, knew), zt_avg1(i, j), Hz(i, j, 1), z_r(i, j, 1), &
           t(i, j, 1, nnew, itemp), We(i, j, 1)
    enddo
    flush(6)
  end subroutine dump_ztd_pt

  subroutine dump_ztd_stats()
    integer(ip) :: i, j, iz, jz, iT, jT, ir, jr
    real(rp) :: zabs, tabs, rabs, zmax, Tmax, rmax, z2, nwet
    if (.not. probe_on) return
    call sync_arch()
    zmax = 0.0_rp; Tmax = 0.0_rp; rmax = 0.0_rp; z2 = 0.0_rp; nwet = 0.0_rp
    iz = 0; jz = 0; iT = 0; jT = 0; ir = 0; jr = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) <= 0.5_rp) cycle
        zabs = abs(zeta(i, j, knew))
        if (zabs > zmax) then
          zmax = zabs
          iz = i; jz = j
        endif
        z2 = z2 + zeta(i, j, knew)**2
        nwet = nwet + 1.0_rp
        if (h(i, j) > 2000.0_rp) then
          tabs = abs(t(i, j, 1, nnew, itemp))
          if (tabs > Tmax) then
            Tmax = tabs
            iT = i; jT = j
          endif
        endif
        rabs = abs(rufrc(i, j))
        if (rabs > rmax) then
          rmax = rabs
          ir = i; jr = j
        endif
      enddo
    enddo
    write(*,'(a,i8,a,es26.17,a,2i6,a,es26.17,a,es26.17,a,2i6,a,es26.17,a,2i6)') &
         'diag: ztd  iic=', iic, '  max|zeta|=', zmax, '  at i,j=', iz, jz, &
         '  rms=', sqrt(z2 / max(nwet, 1.0_rp)), &
         '  max|Tdeep|=', Tmax, '  at i,j=', iT, jT, &
         '  max|rufrc|=', rmax, '  at i,j=', ir, jr
    flush(6)
  end subroutine dump_ztd_stats

  subroutine set_probe_windows()
    ! ROMS_GPU_PROBES=0 disables every diagnostic probe window, e.g. for
    ! timing and profiling runs. Unset or nonzero keeps the defaults.
    logical, save :: checked = .false.
    character(len=16) :: val
    integer :: stat
    if (checked) return
    checked = .true.
    call get_environment_variable('ROMS_GPU_PROBES', val, status=stat)
    if (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '0') then
      probe_steps = 0
      fast_probe_from = 0
      fast_probe_to = 0
      kpp_probe_from = 0
      kpp_probe_to = -1
      write(*,'(a)') 'step_driver: diagnostic probes disabled (ROMS_GPU_PROBES=0)'
    endif
  end subroutine set_probe_windows

  subroutine step_timer_init()
    call system_clock(count=clock_start, count_rate=clock_rate)
    if (clock_rate <= 0) clock_rate = 1
    clock_last = clock_start
  end subroutine step_timer_init

  subroutine step_timer_lap()
    if (clock_rate <= 0) return
    call system_clock(count=clock_last)
  end subroutine step_timer_lap

  subroutine step_mark(msg)
    character(len=*), intent(in) :: msg
    integer :: now
    real(rp) :: dt_sec, tot_sec

    call sync_arch()
    if (clock_rate <= 0) then
      write(*,'(a,a)') 'step: ', trim(msg)
    else
      call system_clock(count=now)
      dt_sec = real(now - clock_last, rp) / real(clock_rate, rp)
      tot_sec = real(now - clock_start, rp) / real(clock_rate, rp)
      write(*,'(a,a,a,f10.4,a,f10.4,a)') 'step: ', trim(msg), &
           '  dt =', dt_sec, ' s  cumul =', tot_sec, ' s'
      clock_last = now
    endif
    flush(6)
  end subroutine step_mark

  subroutine not_implemented(name)
    character(len=*), intent(in) :: name
    write(*,'(a,a)') 'stub (not yet implemented): ', trim(name)
    error stop 'step_driver: refused to run an unimplemented kernel'
  end subroutine not_implemented

end module step_driver
