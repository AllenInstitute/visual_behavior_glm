#!/bin/bash
source activate visual_behavior
python deploy_glm_fits.py --version 11b_L2_optimize_by_session --env visual_behavior --src-path /home/dougo/code/visual_behavior_glm
