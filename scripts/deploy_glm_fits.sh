#!/bin/bash
source activate visual_behavior
python deploy_glm_fits.py --glm-version 4_L2_fixed_lambda=1 --env visual_behavior
python deploy_glm_fits.py --glm-version 4_L2_optimize_by_session --env visual_behavior
python deploy_glm_fits.py --glm-version 4_L2_optimize_by_cell --env visual_behavior
python deploy_glm_fits.py --glm-version 3 --env visual_behavior
python deploy_glm_fits.py --glm-version 2 --env visual_behavior