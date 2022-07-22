import visual_behavior_glm.GLM_params as glm_params
import GLM_analysis_tools as gat
import pandas as pd
import os

version = '55_medepalli_omission_specific_analysis'
run_params = glm_params.load_run_json(version)
oeid = 938002083 

fit = gat.load_fit_pkl(run_params, oeid)

filenameh5 = os.path.join(run_params['experiment_output_dir'], 'event_times_' + str(oeid) + '.h5')
if os.path.isfile(filenameh5):
    event_times = pd.read_hdf(filenameh5, "df")

oeid2 = 929653474 

fit2 = gat.load_fit_pkl(run_params, oeid2)

filenameh5 = os.path.join(run_params['experiment_output_dir'], 'event_times_' + str(oeid2) + '.h5')
if os.path.isfile(filenameh5):
    event_times2 = pd.read_hdf(filenameh5, "df")


