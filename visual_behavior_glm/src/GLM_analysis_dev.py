import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alex_utils import *
from scipy.stats import sem
plt.ion()
import GLM_analysis as g
import json
import h5py
import xarray as xr
import os
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
from allensdk.brain_observatory.behavior.image_api import ImageApi
from visual_behavior_analysis.visual_behavior.translator.allensdk_sessions import extended_stimulus_processing as esp
from visual_behavior_analysis.visual_behavior.translator.allensdk_sessions import attribute_formatting as af


# Get Data
full_path = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20200102_lambda_70/full_df.hdf'
df = pd.read_hdf(full_path)

full_path = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20200102_reward_filter_dev/full_df.hdf'
df_reward = pd.read_hdf(full_path)

# Plot cv var explained for cells in multiple sessions
a1,a3 = g.get_cells_in(df,'OPHYS_1_images_A','OPHYS_3_images_A')
b1,b3 = g.get_cells_in(df,'OPHYS_4_images_B','OPHYS_6_images_B')
ab3,ab4 = g.get_cells_in(df,'OPHYS_3_images_A','OPHYS_4_images_B')
ap1,ap2 = g.get_cells_in(df,'OPHYS_1_images_A','OPHYS_2_images_A_passive')
g.plot_session_comparison(a1,a3,'A1','A3')
g.plot_session_comparison(b1,b3,'B1','B3')
g.plot_session_comparison(ab3,ab4,'A3','B1')
g.plot_session_comparison(ap1,ap2,'A1','A2')

# For each cell, get its full dff and prediction
session_ids = df['ophys_session_id'].unique()
osid = 877946125
fit_data = g.compute_response(osid)
cells = list(fit_data['w'].keys())

# Useful Code Snippets
####################################
dff_max_df = pd.DataFrame.from_dict(fit_data['dff_max'],orient='index').rename(columns={0:
'max_dff'})
dff_max_df.index.name='cell_specimen_id'


var_expl_df = pd.DataFrame.from_dict(fit_data['cv_var_explained'],orient='index').rename(columns={0:
'var_expl'})
var_expl_df.index.name='cell_specimen_id'


var_total = np.var(fit_data['data_dff'][cell_id])
var_resid = np.var(fit_data['model_err'][cell_id])
cv = (var_total - var_resid)/ var_total

# Working on computing variance explained by flash/trial
####################################
manifest_path = os.path.join("/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache", "manifest.json")
cache = BehaviorProjectCache.from_lims(manifest=manifest_path)
session_ids = df['ophys_session_id'].unique()
all_cv, all_shuf, threshold = g.shuffle_across_sessions(session_ids,cache)
g.analyze_threshold(all_cv,all_shuf, threshold)
####################################

#trial_df = g.process_to_trials(fit_data,session)
#g.compute_variance_explained(trial_df)
# plot variance explained by time
# plot variance explained by change/non-change
# plot variance explained by omission/non-omission


image_index = sp['image_index']
omitted_index = sp['omitted']


def find_change(image_index, omitted_index):
    '''
    Args: 
        image_index (pd.Series): The index of the presented image for each flash
        omitted_index (int): The index value for omitted stimuli
    Returns:
        change (np.array of bool): Whether each flash was a change flash
    '''

change = np.diff(image_index) != 0
change = np.concatenate([np.array([False]), change])  # First flash not a change
omitted = image_index == omitted_index
omitted_inds = np.flatnonzero(omitted)
change[omitted_inds] = False

    if image_index.iloc[-1] == omitted_index:
        # If the last flash is omitted we can't set the +1 for that omitted idx
        change[omitted_inds[:-1] + 1] = False
    else:
        change[omitted_inds + 1] = False
                                               
    return change









