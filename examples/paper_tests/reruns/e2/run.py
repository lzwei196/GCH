"""E2 SAC-SMA / CAMELS rerun (SCE-UA, seed 0) with the recorder: the September full-history driver, copied, with only
recording lines added (the scored series and the new kit's score panel of every call). Must reproduce the September
recorded rerun call for call."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/mnt/disk1/Hydrocraft_server/agent_calibration_study/GRL_PAPER_RECORD_2026-09-07/01_framework/framework_code")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "recorder"))
import frozen_recorder as R
R.install(os.path.join(HERE, "rec"))
runpy.run_path(os.path.join(HERE, "run_e2_recorded.py"), run_name="__main__")
