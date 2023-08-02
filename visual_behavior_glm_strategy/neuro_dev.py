import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psy_output_tools as po
import visual_behavior_glm_strategy.PSTH as psth
import visual_behavior_glm_strategy.build_dataframes as bd
import visual_behavior_glm_strategy.hierarchical_bootstrap as hb
import visual_behavior_glm_strategy.GLM_fit_dev as gfd
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt
import visual_behavior_glm_strategy.GLM_analysis_tools as gat
import visual_behavior_glm_strategy.GLM_strategy_tools as gst
import visual_behavior_glm_strategy.GLM_params as glm_params
import visual_behavior_glm_strategy.GLM_fit_tools as gft
import visual_behavior_glm_strategy.GLM_schematic_plots as gsm
import visual_behavior_glm_strategy.GLM_perturbation_tools as gpt
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
gpt.plot_perturbation(weights_beh, run_params, 'omissions')
gpt.plot_perturbation(weights_beh, run_params, 'omissions',show_steps=True)



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
vip_omission = psth.load_omission_df(summary_df, cre='Vip-IRES-Cre',data='events')
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events')

# To make things work on the HPC, you can compute just one running bin
psth.compute_running_bootstrap_bin(vip_omission,'omission','vip',bin_num,data='events',
    nboots=nboots)

# Generate bootstrapped errorbars:
bootstraps_omission = psth.compute_running_bootstrap(vip_omission,'omission','vip',
    data='events',nboots=nboots)
bootstraps_image = psth.compute_running_bootstrap(vip_image,'image','vip',
    data='events',nboots=nboots)

# load already computed bootstrapped errors:
bootstraps_omission = psth.get_running_bootstraps('vip','omission','events',nboots)
bootstraps_image = psth.get_running_bootstraps('vip','image','events',nboots)

# Generate figures with bootstraps
psth.running_responses(vip_omission, 'omission',bootstraps=bootstraps_omission)
psth.running_responses(vip_image, 'image',bootstraps=bootstraps_image)

# Generate figures without bootstraps
psth.running_responses(vip_omission, 'omission')
psth.running_responses(vip_image, 'image')
psth.running_responses(vip_omission, 'omission',split='engagement_v2')
psth.running_responses(vip_image, 'image',split='engagement_v2')


## VIP Engagement Running
################################################################################

boot_image_visual = psth.get_engagement_running_bootstraps('vip','image',
    'events',nboots,'visual')
boot_image_timing = psth.get_engagement_running_bootstraps('vip','image',
    'events',nboots,'timing')
psth.engagement_running_responses(vip_image,'image',vis_boots=boot_image_visual,
    tim_boots=boot_image_timing)

## Hierarchy plots with bootstraps
################################################################################

psth.get_and_plot('vip','omission','events','binned_depth',
    nboots, splits=['visual_strategy_session'])

psth.get_and_plot('vip',['change','image'],'events','binned_depth',
    nboots, strategy='visual')


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

vip_full_filtered = pd.merge(vip_full_filtered, experiment_table.reset_index()[['ophys_experiment_id','binned_depth']],on='ophys_experiment_id')
sst_full_filtered = pd.merge(sst_full_filtered, experiment_table.reset_index()[['ophys_experiment_id','binned_depth']],on='ophys_experiment_id')
exc_full_filtered = pd.merge(exc_full_filtered, experiment_table.reset_index()[['ophys_experiment_id','binned_depth']],on='ophys_experiment_id')

# merge cell types
dfs_filtered = [exc_full_filtered, sst_full_filtered, vip_full_filtered]
labels =['Excitatory','Sst Inhibitory','Vip Inhibitory']
 
# Plot population response
ax = psth.plot_condition(dfs_filtered,'omission',labels,data='filtered_events')

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


## QQ Plots 
################################################################################
ax = psth.plot_QQ_strategy(vip_full_filtered, 'Vip','omission','Familiar',\
	data='filtered_events')
ax = psth.plot_QQ_engagement(vip_full_filtered, 'Vip','omission','Familiar',\
	data='filtered_events')


## Mesoscope check
################################################################################

# Get data split by equipment
sci_dfs, mes_dfs = psth.check_equipment()

# Figure out how many cells we have across strategy/area/depth/equipment
counts = psth.get_equipment_counts()

# Check PSTHs across all area/layers
psth.plot_figure_4_averages(sci_dfs,data='events')
psth.plot_figure_4_averages(mes_dfs,data='events')

# Based on the cell counts, there are 4 cell_type/area/depth combinations to check
# Check PSTHs at specific depths
psth.plot_figure_4_by_equipment(sci_dfs, mes_dfs, 'Excitatory',counts,
    data='events', areas=['VISp'],depths=[375],depth='binned_depth')
psth.plot_figure_4_by_equipment(sci_dfs, mes_dfs, 'Excitatory',counts,
    data='events', areas=['VISp'],depths=[175],depth='binned_depth')
psth.plot_figure_4_by_equipment(sci_dfs, mes_dfs, 'Sst',counts,
    data='events', areas=['VISp'],depths=[275],depth='binned_depth')
psth.plot_figure_4_by_equipment(sci_dfs, mes_dfs, 'Vip',counts,
    data='events', areas=['VISp'],depths=[175],depth='binned_depth')

# Look at response magnitudes summaries
BEHAVIOR_VERSION = 21
summary_df  = po.get_ophys_summary_table(BEHAVIOR_VERSION)
experiment_table = glm_params.get_experiment_table()

psth.plot_equipment_comparison(summary_df, experiment_table,
    'Vip','omission',depths=['175'],second=False, first=False)
psth.plot_equipment_comparison(summary_df, experiment_table, 
    'Sst','omission',depths=['275'],second=True)
psth.plot_equipment_comparison(summary_df, experiment_table,
    'Sst','change', depths=['275'],second=True)
psth.plot_equipment_comparison(summary_df, experiment_table,
    'Excitatory','change', depths=['175','375'],image=True)
psth.plot_equipment_comparison(summary_df, experiment_table,
    'Excitatory','change', depths=['175'],image=True)
psth.plot_equipment_comparison(summary_df, experiment_table,
    'Excitatory','change', depths=['375'],image=True)
psth.plot_equipment_comparison(summary_df, experiment_table,
    'Vip','pre_change',depths=['175'],second=True)

## effects of running on other cell types 
################################################################################

sst_image = psth.load_image_df(summary_df, cre='Sst-IRES-Cre',data='events')
bootstraps_image = psth.get_running_bootstraps('sst','image','events',10000)
psth.running_responses(sst_image, 'image',bootstraps=bootstraps_image,cre='sst')

sst_omission = psth.load_omission_df(summary_df, cre='Sst-IRES-Cre',data='events')
bootstraps_omission = psth.get_running_bootstraps('sst','omission','events',10000)
psth.running_responses(sst_omission, 'omission',bootstraps=bootstraps_omission,cre='sst')

exc_image = psth.load_image_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events')
bootstraps_image = psth.get_running_bootstraps('exc','image','events',10000)
psth.running_responses(exc_image, 'image',bootstraps=bootstraps_image,cre='exc')

exc_omission = psth.load_omission_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events')
bootstraps_omission = psth.get_running_bootstraps('exc','omission','events',10000)
psth.running_responses(exc_omission, 'omission',bootstraps=bootstraps_omission,cre='exc')



## Microcircuit with GLM 
################################################################################
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(GLM_VERSION)
weights_beh = gst.add_behavior_session_metrics(weights_df, summary_df)

# Plot kernels over time, compare cell types
gpt.analysis(weights_beh, run_params, 'all-images')
gpt.analysis(weights_beh, run_params, 'omissions')
gpt.analysis(weights_beh, run_params, 'hits')
gpt.analysis(weights_beh, run_params, 'misses')

# Plot kernels over time, compare strategies
gst.kernels_by_cre(weights_beh, run_params, 'all-images')
gst.kernels_by_cre(weights_beh, run_params, 'omissions')
gst.kernels_by_cre(weights_beh, run_params, 'hits')
gst.kernels_by_cre(weights_beh, run_params, 'misses')

# Plot state space plots
gpt.plot_perturbation(weights_beh, run_params, 'all-images')
gpt.plot_perturbation(weights_beh, run_params, 'omissions')
gpt.plot_perturbation(weights_beh, run_params, 'hits')
gpt.plot_perturbation(weights_beh, run_params, 'misses')



## effects of cell selection with mixed or no-strategy labels
################################################################################

# Plot what the labels look like
import psy_visualization as pv
pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index',
    categories='strategy_labels_with_none',flip1=True, flip2=True)
pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index',
    categories='strategy_labels_with_mixed',flip1=True, flip2=True)

# Plot the neural activity
dfs = psth.get_figure_4_psth(data='events')
dfs = psth.add_cell_selection_labels(dfs, summary_df)
psth.plot_figure_4_averages_cell_selection(dfs, data='events',
    strategy='strategy_labels_with_none')
psth.plot_figure_4_averages_cell_selection(dfs, data='events',
    strategy='strategy_labels_with_mixed')


