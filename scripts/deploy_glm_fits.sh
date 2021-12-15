#!/bin/bash
# Make sure you run conda activate <env> first
# to run this from an environment where the allenSDK is installed

python deploy_glm_fits.py --version 22_events_all_L1_optimize_by_cell_2_images --env-path /home/alex.piet/codebase/miniconda3/envs/visbeh --src-path /home/alex.piet/codebase/GLM/visual_behavior_glm --job-end-fraction 1 
