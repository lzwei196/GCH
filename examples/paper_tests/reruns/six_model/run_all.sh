#!/bin/bash
# Six-model reruns, ONE MODEL AT A TIME (fleet rule). Seeds 0-2 of a model in parallel, except VIC (one at a time:
# shared /dev/shm caches). Each run: python run_model.py <MODEL> <seed>.
cd "$(dirname "$0")"
PY=/mnt/disk1/Hydrocraft_server/python_env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for M in WOFOST HBV MODFLOW6 CRHM SUMMA; do
  echo "START $M $(date)"
  for s in 0 1 2; do env -u KDT_SERIES_DUMP_DIR $PY run_model.py $M $s > logs/${M}_s$s.log 2>&1 & done
  wait
  echo "END $M $(date)"
done
echo "START VIC $(date)"
for s in 0 1 2; do env -u KDT_SERIES_DUMP_DIR $PY run_model.py VIC $s > logs/VIC_s$s.log 2>&1; done
echo "END VIC $(date)"
echo ALL_DONE $(date)
