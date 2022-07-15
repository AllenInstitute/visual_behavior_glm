import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psy_output_tools as po
import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_strategy_tools as gst
from importlib import reload
from alex_utils import *
plt.ion()
'''
Development notes for connecting behavioral strategies and GLM analysis
'''
# Get GLM data (Takes a few minutes)
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(GLM_VERSION)
# get behavior data (fast)
BEHAVIOR_VERSION = 21
summary_df  = po.get_ophys_summary_table(BEHAVIOR_VERSION)
change_df = po.get_change_table(BEHAVIOR_VERSION)
licks_df = po.get_licks_table(BEHAVIOR_VERSION)
bouts_df = po.build_bout_table(licks_df)


results_beh = gst.add_behavior_metrics(results_pivoted,summary_df)
weights_beh = gst.add_behavior_metrics(weights_df,summary_df)

# TODO
# sanity check add_behavior_metrics: add validation, probably dont need to add all the columns. Don't call things "ophys_table"
# make average image seems redundant with recent advances for weights_df
# set up regression against prior_exposure_to_omissions
# Separate computation and plotting code
# analyze engaged/disengaged separatation 
# set up folder for saving figures
# set up automatic figure saving
# save fits dictionary somewhere
# on scatter plot, add binned values on regression
# on scatter plot, include regression values (r^2 and slope)
# disengaged regression has nans
# set up regression by exposure number
# maybe try regressing against hit/miss difference?
# what filtering do we need to do on cells and sessions?



