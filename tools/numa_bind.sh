#!/bin/bash
# numa_bind.sh <cmd...>: run <cmd> on the CPU cores of the NUMA domain that
# holds this rank's GPU. Launch with "mpiexec --cpu-bind none ... numa_bind.sh exe args".
# GPU = world rank mod the number of GPUs, as roms_gpu's select_device does for
# compute ranks. CUDA ordinals are put in PCI bus order, so ordinal g is
# nvidia-smi index g. Everything is read from the node (nvidia-smi, /sys), so no
# node layout is assumed. If any lookup fails, the command runs unbound.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
r=${PALS_RANKID:-${PMI_RANK:-0}}
# Only the first NUMA_BIND_NRANKS ranks (the GPU compute ranks; submit_gpu.sh
# passes their count) are bound; the writer ranks after them run unbound
# (binding them next to the GPU ranks slowed their file writes ~20%).
if [ -n "$NUMA_BIND_NRANKS" ] && [ "$r" -ge "$NUMA_BIND_NRANKS" ]; then exec "$@"; fi
ng=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ng" -ge 1 ]; then
  g=$(( r % ng ))
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$g" 2>/dev/null | tr 'A-Z' 'a-z')
  dev=${bus:4}                      # 00000000:03:00.0 -> 0000:03:00.0 (sysfs form)
  node=$(cat "/sys/bus/pci/devices/$dev/numa_node" 2>/dev/null)
  if [ -n "$node" ] && [ "$node" -ge 0 ] 2>/dev/null && [ -r "/sys/devices/system/node/node$node/cpulist" ]; then
    cpus=$(cat "/sys/devices/system/node/node$node/cpulist")
    [ "${NUMA_BIND_VERBOSE:-0}" = 1 ] && echo "numa_bind: rank $r -> GPU $g ($dev) NUMA $node cpus $cpus" >&2
    exec taskset -c "$cpus" "$@"
  fi
fi
exec "$@"
