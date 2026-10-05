module kinds
  implicit none
  private
  integer, parameter, public :: rp = selected_real_kind(15, 307)
  integer, parameter, public :: ip = selected_int_kind(9)
  ! Storage kind of the passive (anomaly) tracers: float32. Set to rp for
  ! an FP64 passive build (validation of the storage error).
  integer, parameter, public :: pp = selected_real_kind(6, 37)

  ! Neighbor sentinels. Periodic wrap on this rank is NBR_SELF.
  ! Physical edges are NBR_NONE. A real MPI neighbor is the rank (>= 0).
  integer, parameter, public :: NBR_NONE = -1
  integer, parameter, public :: NBR_SELF = -2

  integer, parameter, public :: STAGGER_RHO = 0
  integer, parameter, public :: STAGGER_U   = 1
  integer, parameter, public :: STAGGER_V   = 2

  integer, parameter, public :: itemp = 1
  integer, parameter, public :: isalt = 2
  ! Thread-private column scratch for CUF implicit solves (MiniPac nz=100).
  integer, parameter, public :: COL_KMAX = 256

  public :: is_nan

contains
  ! Host-safe NaN test. nvfortran -cuda maps ieee_is_nan to a device
  ! symbol, which cannot be called from host I/O.
  elemental logical function is_nan(x)
    real(rp), intent(in) :: x
    is_nan = (x /= x)
  end function is_nan
end module kinds
