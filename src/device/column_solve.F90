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

contains

  subroutine implicit_tracer_columns(nx, ny, nz, n_new, itrc, iAkt, dtau, use_fwd)
    integer(ip), intent(in) :: nx, ny, nz, n_new, itrc, iAkt
    real(rp), intent(in) :: dtau
    logical, intent(in) :: use_fwd
    integer(ip) :: i, j, k
    real(rp) :: cff, dc0, hz_k, hz_kp
    real(rp) :: fck, wck, cfk, dck, fcm, wcm, cfm, dcm

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
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

end module column_solve
