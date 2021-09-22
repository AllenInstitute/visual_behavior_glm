#!/bin/bash
#Make sure you run conda activate visbeh first

python deploy_glm_fits.py --version 18_dff_all_L2_optimize_by_session_with_change --env-path /home/alex.piet/codebase/miniconda3/envs/visbeh --src-path /home/alex.piet/codebase/GLM/visual_behavior_glm --job-end-fraction 1
