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
    real(rp) :: dt_loc

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    n_rhs = nrhs
    n_new = nnew
    dt_loc = dt
    call ensure_wrk3()

    do itrc = 1, nt
      call halo_fill_3d_t(t, STAGGER_RHO, nrhs, itrc)
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx + 1
            wrk3a(i, j, k) = apply_mask(0.25_rp * (diff2(i, j, itrc) + diff2(i - 1, j, itrc)) * &
                 pmon_u(i, j) * (Hz(i, j, k) + Hz(i - 1, j, k)) * &
                 (t(i, j, k, n_rhs, itrc) - t(i - 1, j, k, n_rhs, itrc)), umask(i, j))
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny + 1
          do i = 1, nx
            wrk3b(i, j, k) = apply_mask(0.25_rp * (diff2(i, j, itrc) + diff2(i, j - 1, itrc)) * &
                 pnom_v(i, j) * (Hz(i, j, k) + Hz(i, j - 1, k)) * &
                 (t(i, j, k, n_rhs, itrc) - t(i, j - 1, k, n_rhs, itrc)), vmask(i, j))
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            t(i, j, k, n_new, itrc) = t(i, j, k, n_new, itrc) + dt_loc * pm(i, j) * pn(i, j) * &
                 (wrk3a(i + 1, j, k) - wrk3a(i, j, k) + wrk3b(i, j + 1, k) - wrk3b(i, j, k)) / &
                 max(Hz(i, j, k), 1.0e-30_rp)
          enddo
        enddo
      enddo
      call halo_fill_3d_t(t, STAGGER_RHO, nnew, itrc)
    enddo
  end subroutine t3dmix

end module t3dmix_mod
