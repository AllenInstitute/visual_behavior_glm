import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft

run_params = glm_params.load_run_json('51_medepalli_test')
oeid = 957759564 
session, fit, design = gft.fit_experiment(oeid, run_params)

a = 5
