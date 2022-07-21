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

# Add behavior information to GLM dataframes
results_beh = gst.add_behavior_metrics(results_pivoted,summary_df)
weights_beh = gst.add_behavior_metrics(weights_df,summary_df)

# Omission Kernel Plots 
gst.compare_cre_kernels(weights_beh, run_params,ym='omissions')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Slc17a7-IRES2-Cre')

# Hit Kernel Plots 
gst.compare_cre_kernels(weights_beh, run_params,ym='hits')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='hits', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='hits', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='hits', cre_line='Slc17a7-IRES2-Cre')

# Miss Kernel Plots
gst.compare_cre_kernels(weights_beh, run_params,ym='misses')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='misses', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='misses', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='misses', cre_line='Slc17a7-IRES2-Cre')

# Image Kernel Plots
gst.compare_cre_kernels(weights_beh, run_params,ym='all-images')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='all-images', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='all-images', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='all-images', cre_line='Slc17a7-IRES2-Cre')

# Preferred Image Kernel Plots
gst.compare_cre_kernels(weights_beh, run_params,ym='preferred_image')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='preferred_image', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='preferred_image', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='preferred_image', cre_line='Slc17a7-IRES2-Cre')

# currently crash 
gst.scatter_by_cell(results_beh, run_params)
gst.scatter_dataset(results_beh, run_params)
gst.scatter_by_session(results_beh, run_params, 
    cre_line ='Slc17a7-IRES2-Cre',ymetric='hits') 
gst.scatter_by_session(results_beh, run_params, 
    cre_line ='Vip-IRES-Cre',ymetric='omissions')
gst.plot_strategy(results_beh, run_params)




