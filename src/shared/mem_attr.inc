#ifndef ROMS_MEM_ATTR_INC
#define ROMS_MEM_ATTR_INC
#ifdef USE_CUDA
#define WORK_ATTR managed,
#else
#define WORK_ATTR
#endif
! Device helper routines: device-only in the CUDA build, plain host
! routines otherwise (host-only builds are used to check ports on a login
! node). nvfortran 25.9 ICEs on attributes(host,device) subroutines.
#ifdef USE_CUDA
#define DEVICE_ATTR attributes(device) &
#else
#define DEVICE_ATTR
#endif
#endif
