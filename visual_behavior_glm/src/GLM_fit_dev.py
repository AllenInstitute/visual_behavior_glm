import visual_behavior_glm.src.GLM_fit_tools as gft

# Make run JSON
src_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
gft.make_run_json(VERSION, label='Demonstration of make run json',src_path=src_path)

# Load existing parameters
run_params = gft.load_run_json(VERSION)

# To start all experiments on hpc:
# cd visual_behavior_glm/scripts/
# python start_glm.py VERSION

# To run just one session:
oeid = run_params['ophys_experiment_ids'][-1]
session, fit, design = gft.fit_experiment(oeid, run_params)



