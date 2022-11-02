import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psy_output_tools as po
import visual_behavior_glm.PSTH as psth
import visual_behavior_glm.image_regression as ir
import visual_behavior_glm.build_dataframes as bd
import visual_behavior_glm.hierarchical_bootstrap as hb
import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_strategy_tools as gst
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.GLM_schematic_plots as gsm
from importlib import reload
from alex_utils import *
plt.ion()

'''
Development notes for connecting behavioral strategies and GLM analysis
GLM_strategy_tools.py adds behavioral splits to the kernel regression model
build_dataframes.py generates response_dataframes
PSTH.py plots average population activity using the response_df
image_regression.py analyzes the image by image activity of every cell based on peak_response_df 
'''
## Example ophys schematic
################################################################################
experiment_table = glm_params.get_experiment_table()
oeid = 956903412 
cell_id = 1086505751
run_params = glm_params.load_run_json(GLM_VERSION)
session = gft.load_data(oeid, run_params)
time=[1220.5, 1226.25]
gsm.strategy_paper_ophys_example(session, cell_id, time)
 

## Kernel Regression Analyses
################################################################################

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
results_beh = gst.add_behavior_session_metrics(results,summary_df)
results_pivoted_beh = gst.add_behavior_session_metrics(results_pivoted,summary_df)
weights_beh = gst.add_behavior_session_metrics(weights_df,summary_df)

# Basic plots
gst.plot_dropout_summary_population(results_beh, run_params)
gst.plot_fraction_summary_population(results_pivoted_beh, run_params)
gst.plot_population_averages(results_pivoted_beh, run_params) 

# Kernel Plots 
gst.compare_cre_kernels(weights_beh, run_params,ym='omissions')
gst.compare_cre_kernels(weights_beh, run_params,ym='omissions',
    compare=['strategy_labels_with_mixed'])
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Vip-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Sst-IRES-Cre')
gst.plot_kernels_by_strategy_by_session(weights_beh, run_params,
    ym='omissions', cre_line='Slc17a7-IRES2-Cre')

# Dropout Scatter plots
gst.scatter_by_cell(results_pivoted_beh, run_params)
gst.scatter_by_experience(results_pivoted_beh, run_params, 
    cre_line ='Vip-IRES-Cre',ymetric='omissions')
gst.scatter_dataset(results_pivoted_beh, run_params)

# Perturbation plots
gpt.analysis(weights_beh, run_params, 'omissions')
gpt.analysis(weights_beh, run_params, 'hits')
gpt.analysis(weights_beh, run_params, 'misses')
gpt.analysis(weights_beh, run_params, 'all-images')
gpt.analysis(weights_beh, run_params, 'preferred_image')

## Generate response dataframes
################################################################################

# Build single session
oeid = summary_df.iloc[0]['ophys_experiment_id'][0]
session = bd.load_data(oeid)
bd.build_response_df_experiment(session,'filtered_events')
bd.build_behavior_df_experiment(session)

# Aggregate from hpc results
bd.build_population_df(summary_df,'full_df','Vip-IRES-Cre','filtered_events')
bd.build_population_df(summary_df,'full_df','Vip-IRES-Cre','events')
bd.build_population_df(summary_df,'full_df','Vip-IRES-Cre','dff')

# load finished dataframes
vip_image_filtered = bd.load_population_df('filtered_events','image_df','Vip-IRES-Cre')
vip_full_filtered = bd.load_population_df('filtered_events','full_df','Vip-IRES-Cre')

## Running controls
################################################################################

# Load image_dfs 
vip_omission = psth.load_vip_omission_df(summary_df, bootstrap=False)
vip_image = psth.load_vip_image_df(summary_df) 

# Generate bootstrapped errorbars:
bootstraps_omission = psth.compute_running_bootstrap(vip_omission,'omission')
bootstraps_image = psth.compute_running_bootstrap(vip_image,'image')

# Generate figures with bootstraps
psth.running_responses(vip_omission, 'omission',bootstraps=bootstraps_omission)
psth.running_responses(vip_image, 'image',bootstraps=bootstraps_image)

# Generate figures without bootstraps
psth.running_responses(vip_omission, 'omission')
psth.running_responses(vip_image, 'image')
psth.running_responses(vip_omission, 'omission',split='engagement_v2')
psth.running_responses(vip_image, 'image',split='engagement_v2')

## VIP Omission
################################################################################

# Make summary plot of mean response
vip_omission, bootstrap_means = psth.load_vip_omission_df(summary_df,bootstrap=True)
psth.plot_vip_omission_summary(vip_omission, bootstrap_means)

## Change response across hierarchy 
################################################################################
# Loading the image_df is very slow and uses a ton of memory. care must be taken

# Load exc image_df for change images only
exc_change = psth.load_change_df(summary_df, 'Slc17a7-IRES2-Cre')

# Same thing for images
exc_image = psth.load_image_df(summary_df,'Slc17a7-IRES2-Cre') 

# Look at changes and images together
exc_both = psth.load_image_and_change_df(summary_df, 'Slc17a7-IRES2-Cre')

# Plot hierarchy for change
psth.plot_hierarchy(exc_change)
psth.plot_hierarchy(exc_change,depth='binned_depth')
psth.plot_hierarchy(exc_change.query('hit == 1'),splits=['visual_strategy_session'],
    extra='hit - ')
psth.plot_hierarchy(exc_change.query('hit == 1'),splits=['visual_strategy_session'],
    extra='hit - ',depth='binned_depth')
psth.plot_hierarchy(exc_change.query('visual_strategy_session'),splits=['hit'],
    extra='visual_strategy_')
psth.plot_hierarchy(exc_change.query('visual_strategy_session'),splits=['hit'],
    extra='visual_strategy_',depth='binned_depth')
psth.plot_hierarchy(exc_change.query('not visual_strategy_session'),splits=['hit'],
    extra='timing_strategy_')
psth.plot_hierarchy(exc_change.query('not visual_strategy_session'),splits=['hit'],
    extra='timing_strategy_',depth='binned_depth')
psth.plot_hierarchy(exc_change,splits=['engagement_v2'])
psth.plot_hierarchy(exc_change,splits=['engagement_v2'],depth='binned_depth')
psth.plot_hierarchy(exc_change.query('visual_strategy_session'),splits=['engagement_v2'],
    extra='visual_strategy_')
psth.plot_hierarchy(exc_change.query('visual_strategy_session'),splits=['engagement_v2'],
    extra='visual_strategy_',depth='binned_depth')
psth.plot_hierarchy(exc_change.query('not visual_strategy_session'),
    splits=['engagement_v2'],extra='timing_strategy_')
psth.plot_hierarchy(exc_change.query('not visual_strategy_session'),
    splits=['engagement_v2'], extra='timing_strategy_',depth='binned_depth')

# plot hierarchy for images
psth.plot_hierarchy(exc_image,response_type='image')
psth.plot_hierarchy(exc_image,response_type='image',depth='binned_depth')
psth.plot_hierarchy(exc_image,response_type='image',splits=['visual_strategy_session'])
psth.plot_hierarchy(exc_image,response_type='image',splits=['visual_strategy_session'],
    depth='binned_depth')
psth.plot_hierarchy(exc_image,response_type='image',splits=['engagement_v2'])
psth.plot_hierarchy(exc_image,response_type='image',splits=['engagement_v2'],
    depth='binned_depth')
psth.plot_hierarchy(exc_image.query('visual_strategy_session'),response_type='image',
    splits=['engagement_v2'],extra='visual_strategy_')
psth.plot_hierarchy(exc_image.query('visual_strategy_session'),response_type='image',
    splits=['engagement_v2'],extra='visual_strategy_',depth='binned_depth')
psth.plot_hierarchy(exc_image.query('not visual_strategy_session'),response_type='image',
    splits=['engagement_v2'],extra='timing_strategy_')
psth.plot_hierarchy(exc_image.query('not visual_strategy_session'),response_type='image',
    splits=['engagement_v2'],extra='timing_strategy_',depth='binned_depth')

# plot hierarchy
psth.plot_hierarchy(exc_both, response_type='both',splits=['is_change'])
psth.plot_hierarchy(exc_both, response_type='both',splits=['is_change'],
    depth='binned_depth')
psth.plot_hierarchy(exc_both.query('visual_strategy_session'), response_type='both',
    splits=['is_change'],extra='visual_strategy_')
psth.plot_hierarchy(exc_both.query('visual_strategy_session'), response_type='both',
    splits=['is_change'],depth='binned_depth',extra='visual_strategy_')
psth.plot_hierarchy(exc_both.query('not visual_strategy_session'), response_type='both',
    splits=['is_change'],extra='timing_strategy_')
psth.plot_hierarchy(exc_both.query('not visual_strategy_session'), response_type='both',
    splits=['is_change'],depth='binned_depth',extra='timing_strategy_')

psth.plot_hierarchy(exc_both.query('visual_strategy_session & engagement_v2'), 
    response_type='both',splits=['is_change'],extra='engaged_visual_strategy_')
psth.plot_hierarchy(exc_both.query('visual_strategy_session & engagement_v2'), 
    response_type='both',splits=['is_change'],extra='engaged_visual_strategy_',
    depth='binned_depth')
psth.plot_hierarchy(exc_both.query('(not visual_strategy_session) & engagement_v2'), 
    response_type='both',splits=['is_change'],extra='engaged_timing_strategy_')
psth.plot_hierarchy(exc_both.query('(not visual_strategy_session) & engagement_v2'), 
    response_type='both',splits=['is_change'],extra='engaged_timing_strategy_',
    depth='binned_depth')

psth.plot_hierarchy(exc_both.query('visual_strategy_session & (not engagement_v2)'), 
    response_type='both',splits=['is_change'],extra='disengaged_visual_strategy_')
psth.plot_hierarchy(exc_both.query('visual_strategy_session & (not engagement_v2)'), 
    response_type='both',splits=['is_change'],extra='disengaged_visual_strategy_',
    depth='binned_depth')
psth.plot_hierarchy(exc_both.query('(not visual_strategy_session) & (not engagement_v2)'),
    response_type='both',splits=['is_change'],extra='disengaged_timing_strategy_')
psth.plot_hierarchy(exc_both.query('(not visual_strategy_session) & (not engagement_v2)'),
    response_type='both',splits=['is_change'],extra='disengaged_timing_strategy_',
    depth='binned_depth')


## PSTH - Population average response
################################################################################

# Load each cell type
vip_full_filtered = bd.load_population_df('filtered_events','full_df','Vip-IRES-Cre')
sst_full_filtered = bd.load_population_df('filtered_events','full_df','Sst-IRES-Cre')
exc_full_filtered = bd.load_population_df('filtered_events','full_df','Slc17a7-IRES2-Cre')

# Add area, depth
experiment_table = glm_params.get_experiment_table()
vip_full_filtered = bd.add_area_depth(vip_full_filtered, experiment_table)
sst_full_filtered = bd.add_area_depth(sst_full_filtered, experiment_table)
exc_full_filtered = bd.add_area_depth(exc_full_filtered, experiment_table)

# merge cell types
dfs_filtered = [exc_full_filtered, sst_full_filtered, vip_full_filtered]
labels =['Excitatory','Sst Inhibitory','Vip Inhibitory']

# Make Figure 4 panels
psth.plot_figure_4_averages(dfs_filtered, data='filtered_events')
    
# Plot population response
ax = psth.plot_condition(dfs_filtered,'omission',labels,data='filtered_events')
ax = psth.plot_condition(dfs_filtered,'image',labels,data='filtered_events')
ax = psth.plot_condition(dfs_filtered,'change',labels,data='filtered_events')
ax = psth.plot_condition(dfs_filtered,'hit',labels,data='filtered_events')
ax = psth.plot_condition(dfs_filtered,'miss',labels,data='filtered_events')

# Can split by engagement, generally should plot one strategy at a time
ax = psth.plot_condition(dfs_filtered, 'omission',labels,
    split_by_engaged=True,plot_strategy='visual',data='filtered_events')
ax = psth.plot_condition(dfs_filtered, 'omission',labels,
    split_by_engaged=True,plot_strategy='timing',data='filtered_events')

# Can compare any set of conditions
ax = psth.compare_conditions(dfs_filtered, ['hit','miss'], labels, plot_strategy='visual',
    data='filtered_events')


## Population heatmaps
################################################################################
psth.plot_heatmap(vip_full_filtered,'Vip', 'omission','Familiar',\
	data='filtered_events')
psth.plot_heatmap(vip_full_filtered,'Vip', 'omission','Novel 1',\
	data='filtered_events')
psth.plot_heatmap(vip_full_filtered,'Vip', 'omission','Novel >1',\
	data='filtered_events')

psth.plot_heatmap(sst_full_filtered,'Sst', 'omission','Familiar',\
	data='filtered_events')
psth.plot_heatmap(sst_full_filtered,'Sst', 'omission','Novel 1',\
	data='filtered_events')
psth.plot_heatmap(sst_full_filtered,'Sst', 'omission','Novel >1',\
	data='filtered_events')

psth.plot_heatmap(exc_full_filtered,'Exc', 'omission','Familiar',\
	data='filtered_events')
psth.plot_heatmap(exc_full_filtered,'Exc', 'omission','Novel 1',\
	data='filtered_events')
psth.plot_heatmap(exc_full_filtered,'Exc', 'omission','Novel >1',\
	data='filtered_events')


## QQ Plots 
################################################################################
ax = psth.plot_QQ_strategy(vip_full_filtered, 'Vip','omission','Familiar',\
	data='filtered_events')
ax = psth.plot_QQ_engagement(vip_full_filtered, 'Vip','omission','Familiar',\
	data='filtered_events')




