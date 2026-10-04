#!/bin/bash
# the other 20 GR4J cells, 8 at a time, each through the paper's sealed launcher
cd /mnt/disk1/Hydrocraft_server/agent_calibration_study/RERUN_FULL_HISTORY_2026-09-27/paper_replay/reruns/gr4j
xargs -a jobs.txt -P 8 -I{} sh -c '/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/02_tests/C_stability_gr4j/full/sealed_exec/sealed_python.sh /mnt/disk1/Hydrocraft_server/agent_calibration_study/RERUN_FULL_HISTORY_2026-09-27/paper_replay/reruns/gr4j/gr4j_cell.py {} > logs/{}.log 2>&1'
echo ALL_DONE $(date)
