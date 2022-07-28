import visual_behavior_glm.GLM_params as glm_params
import GLM_analysis_tools as gat
import pandas as pd
import os
import time

version = '56_medepalli_omission_specific_analysis_design_test'
run_params = glm_params.load_run_json(version)
oeid = 1088916616

start = time.time()
# pred_responses = gat.load_pred_responses_pkl(run_params, oeid)
end = time.time()
print('For version {}, it takes {:.4f} to load pred_responses dict'.format(version, end-start)) 

start = time.time()
design = gat.load_design_pkl(run_params, oeid)
end = time.time()
print('For version {}, it takes {:.4f} to design matrix'.format(version, end-start))
