#include "mem_attr.inc"
module cdr_lite_io
  ! CDR_LITE surface forcing, host side: &cdr_lite_settings, the DIC
  ! tracers and their ALK pairs, the forcing series (frc_io), and the
  ! surface fluxes at the corrector's forcing time (cdr_lite_mod). As the
  ! CPU, the carbonate sensitivities ddic_dco2/ddic_dalk must be in the
  ! forcing files; <name>_flx of a DIC tracer may be absent (0).
  ! Not on the GPU yet: cdr_online_carbonate_sensitivity (stops at start-up).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use comm_mod, only: comm_rank
  use passive_mod, only: ntp, tp, n_oae, n_dor, ncdr, cdr0, passive_name
  use cdr_lite_mod
  use frc_io, only: init_cdr_lite_forcing, set_cdr_lite_forces, uwnd, vwnd
  implicit none
  private
  public :: read_cdr_lite_namelist, init_cdr_lite, set_cdr_lite

  logical :: cdr_online_carbonate_sensitivity = .false.
  namelist /cdr_lite_settings/ cdr_online_carbonate_sensitivity

contains

  subroutine read_cdr_lite_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    rewind(unit)
    read(unit, nml=cdr_lite_settings, iostat=ios)
    if (ios /= 0) cdr_online_carbonate_sensitivity = .false.
  end subroutine read_cdr_lite_namelist

  subroutine init_cdr_lite()
    ! After the passive tracer list and the wet columns are final.
    integer(ip) :: nd, m
    integer(ip), allocatable :: dic(:), alk(:)
    character(len=32), allocatable :: names(:)
    logical :: hd, ha
    logical, allocatable :: hf(:)
    if (ncdr == 0) then
      call cdrl_ensure()
      return
    endif
    if (cdr_online_carbonate_sensitivity) &
      error stop 'init_cdr_lite: cdr_online_carbonate_sensitivity is not on the GPU model yet'
    nd = n_oae + n_dor
    allocate(dic(max(nd, 1)), alk(max(nd, 1)), names(max(nd, 1)), hf(max(nd, 1)))
    do m = 1, n_oae
      dic(m) = cdr0 + 2 * m; alk(m) = cdr0 + 2 * m - 1
    enddo
    do m = 1, n_dor
      dic(n_oae + m) = cdr0 + 2 * n_oae + m; alk(n_oae + m) = 0
    enddo
    call cdrl_setup(nd, dic, alk, ntp)
    do m = 1, nd
      names(m) = passive_name(dic(m))
    enddo
    call init_cdr_lite_forcing(int(nd), names, hd, ha, hf)
    if (.not. (hd .and. ha)) then
      write(*,'(a,2l2)') 'init_cdr_lite: ddic_dco2, ddic_dalk found =', hd, ha
      error stop 'init_cdr_lite: CDR_LITE needs ddic_dco2 and ddic_dalk in the forcing files'
    endif
    if (comm_rank() == 0) then
      do m = 1, nd
        write(*,'(a,a,a,i0,a,i0,a,l2)') 'cdr_lite: ', trim(names(m)), ' (tracer ', dic(m), ', ALK ', alk(m), &
             ')  _flx found =', hf(m)
      enddo
      if (.not. allocated(uwnd)) write(*,'(a)') 'cdr_lite: no bulk winds, k_gas = 0'
    endif
    deallocate(dic, alk, names, hf)
  end subroutine init_cdr_lite

  subroutine set_cdr_lite()
    ! After the corrector's set_forces (frc_slot 1, n+1/2): the forcing at
    ! n+1/2 and the surface fluxes from the predicted state (slot nrhs).
    real(rp) :: tneed
    if (.not. cdrl_on) return
    tneed = frc_modtime()
    call set_cdr_lite_forces(tneed, int(cdrl_nd), cdrl_ddco2, cdrl_ddalk, cdrl_flx)
    if (allocated(uwnd)) then
      call cdrl_sflx(tp, ntp, nrhs, uwnd, vwnd, .true.)
    else
      call cdrl_sflx(tp, ntp, nrhs, cdrl_ddco2, cdrl_ddco2, .false.)
    endif
  end subroutine set_cdr_lite

end module cdr_lite_io
