#!/bin/bash
# runs one whole batch file of "<kind> <seed>" lines, 16 at a time
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -a "$1" -P 16 -L 1 sh -c '/mnt/disk1/Hydrocraft_server/python_env/bin/python run_native.py "$0" "$1" > logs/"$0"_s"$1".log 2>&1'
echo BATCH_DONE "$1" $(date)
