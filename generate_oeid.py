import visual_behavior_glm.GLM_params as glm_params

experiment_table = glm_params.get_experiment_table()
oeid = experiment_table.index.values[754]
print(oeid)
