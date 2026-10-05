module host_buf
  ! Contiguous host scratch for NetCDF. Working fields may be managed;
  ! nf90_get/put_var always sees these buffers, never a work array.
  use kinds
  use nc_io
  implicit none
  private

  public :: host_read_2d
  public :: host_read_3d

contains

  subroutine host_read_2d(ncid, name, dest, nxi, neta, rec)
    integer,          intent(in)    :: ncid, nxi, neta
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(nxi, neta)
    integer,          intent(in), optional :: rec
    real(rp), allocatable :: buf(:,:)
    allocate(buf(nxi, neta))
    call nc_read_2d(ncid, name, buf, nxi, neta, rec)
    dest = buf
    deallocate(buf)
  end subroutine host_read_2d

  subroutine host_read_3d(ncid, name, dest, nxi, neta, nz, rec)
    integer,          intent(in)    :: ncid, nxi, neta, nz
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(nxi, neta, nz)
    integer,          intent(in), optional :: rec
    real(rp), allocatable :: buf(:,:,:)
    allocate(buf(nxi, neta, nz))
    call nc_read_3d(ncid, name, buf, nxi, neta, nz, rec)
    dest = buf
    deallocate(buf)
  end subroutine host_read_3d

end module host_buf
