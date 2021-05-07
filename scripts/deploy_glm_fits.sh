#!/bin/bash
source vba

python deploy_glm_fits.py --version 12_dff_L2_optimize_by_session --env-path /home/dougo/anaconda3/envs/vba --src-path /home/dougo/code/visual_behavior_glm --job-start-fraction 0.0  --job-end-fraction 0.01
