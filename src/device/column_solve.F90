module column_solve
  ! Implicit vertical diffusion/advection of a tracer (Thomas solve), one
  ! device thread per (i,j) column. The k-recurrence stays serial inside
  ! the thread. Only CF and DC are stored (the back-substitution needs
  ! them); FC/WC of the previous level stay in registers. (A CUF call into a device routine that uses managed
  ! arrays via USE is CUDA 700, so the solve is written inline here.)
  ! Arithmetic is the same as the former host implicit_tracer_thomas.
  ! On entry t(:,:,:,n_new) is Hz*t; on exit it is t, masked by rmask.
  use kinds
  use gpu_state
  use formulas, only: apply_mask
  implicit none
  private
  public :: implicit_tracer_columns
  public :: implicit_tracer_columns_frc

contains

  subroutine implicit_tracer_columns(nx, ny, nz, n_new, itrc, iAkt, dtau, use_fwd)
    integer(ip), intent(in) :: nx, ny, nz, n_new, itrc, iAkt
    real(rp), intent(in) :: dtau
    logical, intent(in) :: use_fwd
    integer(ip) :: i, j, k, lim
    real(rp) :: cff, dc0, hz_k, hz_kp
    real(rp) :: fck, wck, cfk, dck, fcm, wcm, cfm, dcm

    lim = dry_lim(G_TRC)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * pm(i, j) * pn(i, j)
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, 1)
          hz_kp = Hz_fwd(i, j, 2)
        else
          hz_k = Hz(i, j, 1)
          hz_kp = Hz(i, j, 2)
        endif
        fck = 2.0_rp * dtau * Akt(i, j, 1, iAkt) / (hz_kp + hz_k)
        wck = dc0 * Wi(i, j, 1)
        cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp))
        cfk = cff * (fck - min(wck, 0.0_rp))
        dck = cff * t(i, j, 1, n_new, itrc)
        col_cf(i, j, 1) = cfk
        col_dc(i, j, 1) = dck

        do k = 2, nz - 1
          fcm = fck; wcm = wck; cfm = cfk; dcm = dck
          if (use_fwd) then
            hz_k = Hz_fwd(i, j, k)
            hz_kp = Hz_fwd(i, j, k + 1)
          else
            hz_k = Hz(i, j, k)
            hz_kp = Hz(i, j, k + 1)
          endif
          fck = 2.0_rp * dtau * Akt(i, j, k, iAkt) / (hz_k + hz_kp)
          wck = dc0 * Wi(i, j, k)
          cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp) + &
                          fcm - min(wcm, 0.0_rp) - &
                          cfm * (fcm + max(wcm, 0.0_rp)))
          cfk = cff * (fck - min(wck, 0.0_rp))
          dck = cff * (t(i, j, k, n_new, itrc) + dcm * &
               (fcm + max(wcm, 0.0_rp)))
          col_cf(i, j, k) = cfk
          col_dc(i, j, k) = dck
        enddo

        if (use_fwd) then
          hz_k = Hz_fwd(i, j, nz)
        else
          hz_k = Hz(i, j, nz)
        endif
        t(i, j, nz, n_new, itrc) = apply_mask((t(i, j, nz, n_new, itrc) + dck * &
             (fck + max(wck, 0.0_rp))) / &
             (hz_k + fck - min(wck, 0.0_rp) - &
              cfk * (fck + max(wck, 0.0_rp))), rmask(i, j))
        do k = nz - 1, 1, -1
          t(i, j, k, n_new, itrc) = apply_mask(col_dc(i, j, k) + col_cf(i, j, k) * t(i, j, k + 1, n_new, itrc), &
               rmask(i, j))
        enddo
      enddo
    enddo
  end subroutine implicit_tracer_columns

  subroutine implicit_tracer_columns_frc(nx, ny, nz, n_new, itrc, iAkt, dtau)
    ! step3d_t's surface/KPP forcing fused into implicit_tracer_columns
    ! (use_fwd = .false.): the forward sweep applies, as it reads each
    ! level, the same updates the separate forcing kernel made, in the same
    ! order (t(nz) + dt*stflx first; level m gets + dt*cff(m), then
    ! - dt*cff(m-1); salt: + cff(m), then - cff(m-1)).
    integer(ip), intent(in) :: nx, ny, nz, n_new, itrc, iAkt
    real(rp), intent(in) :: dtau
    integer(ip) :: i, j, k, lim, itr
    real(rp) :: cff, dc0, hz_k, hz_kp, dt_loc, tk, cfrc, cfrc_m
    real(rp) :: fck, wck, cfk, dck, fcm, wcm, cfm, dcm

    lim = dry_lim(G_TRC)
    itr = itrc
    dt_loc = dtau
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        dc0 = dtau * pm(i, j) * pn(i, j)
        hz_k = Hz(i, j, 1)
        hz_kp = Hz(i, j, 2)
        ! forcing of level 1: + dt*cff(1) (temperature) or + cff(1) (salt)
        tk = t(i, j, 1, n_new, itr)
        cfrc_m = 0.0_rp
        if (itr == itemp) then
          cfrc_m = srflx(i, j) * swr_frac(i, j, 1) - &
                   ghat(i, j, 1) * (stflx(i, j, itemp) - srflx(i, j))
          tk = tk + dt_loc * cfrc_m
        elseif (itr == isalt) then
          cfrc_m = -dt_loc * ghat(i, j, 1) * stflx(i, j, isalt)
          tk = tk + cfrc_m
        endif
        fck = 2.0_rp * dtau * Akt(i, j, 1, iAkt) / (hz_kp + hz_k)
        wck = dc0 * Wi(i, j, 1)
        cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp))
        cfk = cff * (fck - min(wck, 0.0_rp))
        dck = cff * tk
        col_cf(i, j, 1) = cfk
        col_dc(i, j, 1) = dck

        do k = 2, nz - 1
          fcm = fck; wcm = wck; cfm = cfk; dcm = dck
          hz_k = Hz(i, j, k)
          hz_kp = Hz(i, j, k + 1)
          tk = t(i, j, k, n_new, itr)
          if (itr == itemp) then
            cfrc = srflx(i, j) * swr_frac(i, j, k) - &
                   ghat(i, j, k) * (stflx(i, j, itemp) - srflx(i, j))
            tk = tk + dt_loc * cfrc
            tk = tk - dt_loc * cfrc_m
            cfrc_m = cfrc
          elseif (itr == isalt) then
            cfrc = -dt_loc * ghat(i, j, k) * stflx(i, j, isalt)
            tk = tk + cfrc
            tk = tk - cfrc_m
            cfrc_m = cfrc
          endif
          fck = 2.0_rp * dtau * Akt(i, j, k, iAkt) / (hz_k + hz_kp)
          wck = dc0 * Wi(i, j, k)
          cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp) + &
                          fcm - min(wcm, 0.0_rp) - &
                          cfm * (fcm + max(wcm, 0.0_rp)))
          cfk = cff * (fck - min(wck, 0.0_rp))
          dck = cff * (tk + dcm * &
               (fcm + max(wcm, 0.0_rp)))
          col_cf(i, j, k) = cfk
          col_dc(i, j, k) = dck
        enddo

        hz_k = Hz(i, j, nz)
        ! level nz: + dt*stflx, then - dt*cff(nz-1) (or - cff(nz-1))
        tk = t(i, j, nz, n_new, itr) + dt_loc * stflx(i, j, itr)
        if (itr == itemp) then
          tk = tk - dt_loc * cfrc_m
        elseif (itr == isalt) then
          tk = tk - cfrc_m
        endif
        t(i, j, nz, n_new, itr) = apply_mask((tk + dck * &
             (fck + max(wck, 0.0_rp))) / &
             (hz_k + fck - min(wck, 0.0_rp) - &
              cfk * (fck + max(wck, 0.0_rp))), rmask(i, j))
        do k = nz - 1, 1, -1
          t(i, j, k, n_new, itr) = apply_mask(col_dc(i, j, k) + col_cf(i, j, k) * t(i, j, k + 1, n_new, itr), &
               rmask(i, j))
        enddo
      enddo
    enddo
  end subroutine implicit_tracer_columns_frc

end module column_solve
