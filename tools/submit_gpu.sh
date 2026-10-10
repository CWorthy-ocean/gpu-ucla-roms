#!/bin/bash
# Submit a roms_gpu case with the rank layout derived from its namelist:
# compute ranks = np_xi x np_eta, writer ranks from the history record size
# (tools/gpu_layout.py), PBS select line and mpiexec count set accordingly.
#   tools/submit_gpu.sh <submit_script> [NWRITERS=n] [extra qsub -v settings, e.g. EXE=...]
# The submit script must name its namelist on an NML= line (it may use
# ${CASE} from a CASE= line) and launch with "mpiexec -n ${NRANKS:-...}".
# PBS (qsub) version; for Slurm, use the printed counts with sbatch/srun.
set -euo pipefail
# ---- settings: edit here, or set the environment variable ----
ROOT=${ROMS_GPU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # roms_gpu checkout
PYTHON=${PYTHON:-python3}                                                 # gpu_layout.py needs only the standard library
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SCRIPT=$1; shift
CASE=$(grep -E '^CASE=' "$SCRIPT" | head -1 | sed -E 's/^CASE=//; s/^"//; s/"$//' || true)
NML=$(grep -E '^NML=' "$SCRIPT" | head -1 | sed -E 's/^NML=//; s/^"//; s/"$//')
NML=${NML//'${CASE}'/$CASE}
EXTRA=""
NWOPT=""
for a in "$@"; do
  case $a in
    NWRITERS=*) NWOPT="--max-writers ${a#NWRITERS=} --target-seconds 0.001" ;;
    *) EXTRA="${EXTRA},${a}" ;;
  esac
done
read NTILE NW NRANKS SELECT <<< "$($PYTHON "$ROOT/tools/gpu_layout.py" "$NML" --gpus-per-node $GPUS_PER_NODE $NWOPT)"
echo "layout: $NTILE compute ranks + $NW writer(s) = $NRANKS ranks; select=$SELECT"
qsub -l select=$SELECT -v "ROMS_GPU_PROBES=0,NRANKS=${NRANKS},NUMA_BIND_NRANKS=${NTILE}${EXTRA}" "$SCRIPT"
