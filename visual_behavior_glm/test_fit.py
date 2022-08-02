import visual_behavior_glm.GLM_params as glm_params
import GLM_analysis_tools as gat
import pandas as pd
import os
import time

version = '57_medepalli_omission_specific_analysis_pre_post'
run_params = glm_params.load_run_json(version)
oeid = 938002083

start = time.time()
pred_responses = gat.load_pred_responses_pkl(run_params, oeid)
end = time.time()
print('For version {}, it takes {:.4f} to load fit dict'.format(version, end-start)) 


version = '55_medepalli_omission_specific_analysis'
run_params = glm_params.load_run_json(version)
oeid = 1003444808

start = time.time()
fit = gat.load_fit_pkl(run_params, oeid)
end = time.time()
print('\nTime to load fit dictionary for {}: {} seconds'.format(oeid, end-start))
