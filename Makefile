# CPU backend: gfortran. GPU-first build: make gpu (nvfortran + CUDA).
# Working state is managed under USE_CUDA. One rank / one GPU / one tile.

FC_HOST   ?= gfortran
FFLAGS_HOST ?= -O2 -g -cpp -ffree-line-length-none -fcheck=bounds -Wall -Wextra -Wno-unused-dummy-argument -Wno-unused-parameter -Isrc/shared

FC_GPU    ?= nvfortran
FFLAGS_GPU ?= -O2 -g -cuda -gpu=cc80 -DUSE_CUDA -Mpreprocess -Mfree -Isrc/shared
FFLAGS_MPI ?= -DUSE_MPI

# conda nf-config --prefix is empty; take the first -L from --flibs.
NETCDF_FFLAGS ?= $(shell nf-config --fflags 2>/dev/null)
NETCDF_FLIBS  ?= $(shell nf-config --flibs 2>/dev/null)
NETCDF_LIBDIR := $(firstword $(patsubst -L%,%,$(filter -L%,$(NETCDF_FLIBS))))
ifneq ($(NETCDF_LIBDIR),)
  NETCDF_FLIBS += -Wl,-rpath,$(NETCDF_LIBDIR)
endif
# Parallel HDF5 NetCDF needs these even for serial Fortran I/O
# (--no-copy-dt-needed-entries does not pull them in via libnetcdf).
ifneq ($(wildcard $(NETCDF_LIBDIR)/libhdf5.so),)
  NETCDF_FLIBS += -lhdf5_hl -lhdf5
endif
ifneq ($(wildcard $(NETCDF_LIBDIR)/libmpi.so),)
  NETCDF_FLIBS += -lmpi
endif
# Fortran MPI bindings live in libmpifort (MPICH / Intel MPI).
ifneq ($(wildcard $(NETCDF_LIBDIR)/libmpifort.so),)
  NETCDF_FLIBS += -lmpifort
endif

# comm.F90 is always built with USE_MPI so every target shares one comm.mod.
# 'override' so the MPI/netCDF flags are appended even when FFLAGS_* is
# given on the make command line (a plain += is ignored in that case).
override FFLAGS_HOST += $(FFLAGS_MPI) $(NETCDF_FFLAGS)
override FFLAGS_GPU  += $(FFLAGS_MPI) $(NETCDF_FFLAGS)
LIBS        ?= $(NETCDF_FLIBS)

SRC_SHARED = src/shared/kinds.F90 src/shared/decomp.F90 src/shared/arch.F90 \
             src/shared/comm.F90 src/shared/formulas.F90 src/shared/time_state.F90
SRC_DEV    = src/device/gpu_state.F90 src/device/column_solve.F90 src/device/halo.F90 src/device/geom.F90 \
             src/device/obc.F90 src/device/step2d.F90 src/device/prsgrd.F90 \
             src/device/advect.F90 src/device/rho_eos.F90 src/device/uv_rhs.F90 \
             src/device/passive.F90 \
             src/device/pre_step3d.F90 src/device/step3d_uv.F90 src/device/visc3d.F90 \
             src/device/step3d_t.F90 src/device/t3dmix.F90 src/device/lmd.F90 \
             src/device/bulk_coare.F90
SRC_HOST   = src/host/step_driver.F90
SRC_IO     = src/host/nc_io.F90 src/host/host_buf.F90 src/host/frc_io.F90 \
             src/host/bry_io.F90 src/host/tide_io.F90 src/host/io_server.F90 src/host/his_io.F90 \
             src/host/case_io.F90
MODDIR     = build
GPUDIR     = build_gpu

.PHONY: all host tests gpu clean run

all: host

host: tests $(MODDIR)/roms_gpu

$(MODDIR):
	mkdir -p $(MODDIR)

$(GPUDIR):
	mkdir -p $(GPUDIR)

$(MODDIR)/roms_gpu: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) src/host/main.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_formulas: $(SRC_SHARED) tests/test_formulas.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_halo: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 tests/test_halo.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_mpi_halo: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 tests/test_mpi_halo.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_set_depth: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 tests/test_set_depth.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_step2d: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 src/device/obc.F90 src/device/step2d.F90 tests/test_step2d.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_prsgrd: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 src/device/prsgrd.F90 tests/test_prsgrd.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_advect: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/advect.F90 tests/test_advect.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_rho_eos: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 src/device/geom.F90 src/device/rho_eos.F90 tests/test_rho_eos.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_dyn_core: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_dyn_core.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_obc: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_obc.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_bulk: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_bulk.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_tides: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_tides.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_his: $(SRC_SHARED) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) tests/test_his.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_lmd: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 \
                    src/device/geom.F90 src/device/rho_eos.F90 src/device/lmd.F90 \
                    tests/test_lmd.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

$(MODDIR)/test_io: $(SRC_SHARED) src/device/gpu_state.F90 src/device/halo.F90 \
                   src/device/geom.F90 src/device/bulk_coare.F90 $(SRC_IO) tests/test_io.F90 | $(MODDIR)
	$(FC_HOST) $(FFLAGS_HOST) -J$(MODDIR) -o $@ $^ $(LIBS)

tests: $(MODDIR)/test_formulas $(MODDIR)/test_halo $(MODDIR)/test_mpi_halo $(MODDIR)/test_set_depth $(MODDIR)/test_step2d $(MODDIR)/test_prsgrd $(MODDIR)/test_advect $(MODDIR)/test_rho_eos $(MODDIR)/test_lmd $(MODDIR)/test_io $(MODDIR)/test_obc $(MODDIR)/test_bulk $(MODDIR)/test_tides $(MODDIR)/test_his $(MODDIR)/test_dyn_core
	$(MODDIR)/test_formulas
	$(MODDIR)/test_halo
	mpiexec -n 2 $(MODDIR)/test_mpi_halo
	$(MODDIR)/test_set_depth
	$(MODDIR)/test_step2d
	$(MODDIR)/test_prsgrd
	$(MODDIR)/test_advect
	$(MODDIR)/test_rho_eos
	$(MODDIR)/test_lmd
	$(MODDIR)/test_io
	$(MODDIR)/test_obc
	$(MODDIR)/test_bulk
	$(MODDIR)/test_tides
	$(MODDIR)/test_his
	$(MODDIR)/test_dyn_core

run: $(MODDIR)/roms_gpu
	$(MODDIR)/roms_gpu geom

# comm.F90 is compiled on its own: with USE_MPI its external MPI calls
# stop nvfortran's cross-file inlining for every file that follows it in
# the same command, and the device kernels then lose the inlined
# formulas functions (rho_eos went from 96 registers / no stack to 154 /
# 72-byte stack and a 1x1x1 launch grid, 265x slower). Its object is
# linked in; the main command sees only comm_mod.mod.
SRC_SHARED_NOCOMM = $(filter-out src/shared/comm.F90,$(SRC_SHARED))

gpu: | $(GPUDIR)
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -c src/shared/comm.F90 -o $(GPUDIR)/comm.o
	$(FC_GPU) $(FFLAGS_GPU) -module $(GPUDIR) -o $(GPUDIR)/roms_gpu_cuda \
	  $(SRC_SHARED_NOCOMM) $(SRC_DEV) $(SRC_IO) $(SRC_HOST) src/host/main.F90 $(GPUDIR)/comm.o $(LIBS)

clean:
	rm -rf $(MODDIR) $(GPUDIR)
