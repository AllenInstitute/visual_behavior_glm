import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft

<<<<<<< HEAD
run_params = glm_params.load_run_json('51_medepalli_test')
oeid = 957759564 
session, fit, design = gft.fit_experiment(oeid, run_params)
=======
run_params = glm_params.load_run_json('50_medepalli_test3')
oeid = 991852002 
session, fit, design = gft.load_fit_experiment(oeid, run_params)
>>>>>>> d0695eea7ad16da924eeab1be216f1e9e9ad8ae5

a = 5
