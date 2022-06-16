import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft

run_params = glm_params.load_run_json('50_medepalli_test')
oeid = 957759566 
session, fit, design = gft.load_fit_experiment(oeid, run_params)

