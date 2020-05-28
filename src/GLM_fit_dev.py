import visual_behavior_glm.src.GLM_fit_tools as gft

# Make run JSON
gft.make_run_json(VERSION, label='Demonstration of make run json')

# Load existing parameters
run_params = gft.load_run_json(VERSION)



