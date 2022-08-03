import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft
# from visual_behavior_glm.glm import GLM

version = '57_medepalli_omission_specific_analysis_original'
run_params = glm_params.load_run_json(version)
oeid = 965928394
session, fit, design = gft.fit_experiment(oeid, run_params)

# log_results = True
# log_weights = True
# use_previous_fit = False
# glm = GLM(oeid, version, log_results=log_results, log_weights=log_weights, use_previous_fit=use_previous_fit)

