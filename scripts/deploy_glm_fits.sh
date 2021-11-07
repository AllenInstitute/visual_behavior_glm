#!/bin/bash
# Make sure you run conda activate <env> first
# to run this from an environment where the allenSDK is installed

python deploy_glm_fits.py --version 21_dff_all_L2_optimize_by_session --env-path /home/alex.piet/codebase/miniconda3/envs/visbeh --src-path /home/alex.piet/codebase/GLM/visual_behavior_glm --job-end-fraction 1 
