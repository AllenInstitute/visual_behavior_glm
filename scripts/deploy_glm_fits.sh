#!/bin/bash
# Make sure you run conda activate <env> first
# to run this from an environment where the allenSDK is installed

deploy_glm_fits.py --version v_50_Medepalli_test --env-path /allen/aics/apps/hpc_shared/mod/anaconda3-5.3.0/envs/visual_behavior_glm --src-path /home/saaketh.medepalli/visual_behavior_glm --job-end-fraction 1

 
