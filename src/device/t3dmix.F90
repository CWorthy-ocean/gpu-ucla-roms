module t3dmix_mod
  ! Harmonic tracer diffusion along s-surfaces (TS_DIF2).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use halo_mod
  implicit none
  private
  public :: t3dmix

contains

  subroutine t3dmix()
    integer(ip) :: i, j, k, itrc, nx, ny, nz, n_rhs, n_new
    real(rp) :: dt_loc, fxw, fxe, fes, fen

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_rhs = nrhs
    n_new = nnew
    dt_loc = dt

    do itrc = 1, nt
      ! slot nrhs (= 3) halos: exchanged by pre_step3d's group after t3dbc and
      ! again by the corrector's rho_eos; unchanged since. Slot nnew is
      ! written at the point only and exchanged by rho_eos(nnew) afterwards.
      ! The four faces of each cell are evaluated inline (the former wrk3a /
      ! wrk3b expressions) and summed in the same order: bit-identical to the
      ! three-kernel version without the two flux temporaries.
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            fxw = apply_mask(0.25_rp * (diff2(i, j, itrc) + diff2(i - 1, j, itrc)) * &
                 pmon_u(i, j) * (Hz(i, j, k) + Hz(i - 1, j, k)) * &
                 (t(i, j, k, n_rhs, itrc) - t(i - 1, j, k, n_rhs, itrc)), umask(i, j))
            fxe = apply_mask(0.25_rp * (diff2(i + 1, j, itrc) + diff2(i, j, itrc)) * &
                 pmon_u(i + 1, j) * (Hz(i + 1, j, k) + Hz(i, j, k)) * &
                 (t(i + 1, j, k, n_rhs, itrc) - t(i, j, k, n_rhs, itrc)), umask(i + 1, j))
            fes = apply_mask(0.25_rp * (diff2(i, j, itrc) + diff2(i, j - 1, itrc)) * &
                 pnom_v(i, j) * (Hz(i, j, k) + Hz(i, j - 1, k)) * &
                 (t(i, j, k, n_rhs, itrc) - t(i, j - 1, k, n_rhs, itrc)), vmask(i, j))
            fen = apply_mask(0.25_rp * (diff2(i, j + 1, itrc) + diff2(i, j, itrc)) * &
                 pnom_v(i, j + 1) * (Hz(i, j + 1, k) + Hz(i, j, k)) * &
                 (t(i, j + 1, k, n_rhs, itrc) - t(i, j, k, n_rhs, itrc)), vmask(i, j + 1))
            t(i, j, k, n_new, itrc) = t(i, j, k, n_new, itrc) + dt_loc * pm(i, j) * pn(i, j) * &
                 (fxe - fxw + fen - fes) / max(Hz(i, j, k), 1.0e-30_rp)
          enddo
        enddo
      enddo
    enddo
  end subroutine t3dmix

end module t3dmix_mod
