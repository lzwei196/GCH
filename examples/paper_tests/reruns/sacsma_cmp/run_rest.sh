#!/bin/bash
cd /mnt/disk1/Hydrocraft_server/agent_calibration_study/RERUN_FULL_HISTORY_2026-09-27/paper_replay/reruns/sacsma_cmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -a jobs.txt -P 9 -L 1 sh -c '/mnt/disk1/Hydrocraft_server/python_env/bin/python rec_cell.py "$0" "$1" "$2" > logs/"$0"_b"$1"_s"$2".log 2>&1'
echo ALL_DONE $(date)
