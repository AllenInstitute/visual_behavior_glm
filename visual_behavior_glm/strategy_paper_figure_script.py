import psy_output_tools as po
import visual_behavior_glm.PSTH as psth
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_schematic_plots as gsm
import visual_behavior_glm.GLM_perturbation_tools as gpt
import visual_behavior_glm.GLM_strategy_tools as gst
import matplotlib.pyplot as plt
plt.ion()
from importlib import reload
from alex_utils import *

BEHAVIOR_VERSION = 21
summary_df  = po.get_ophys_summary_table(BEHAVIOR_VERSION)

## Fig. 4C - Example ophys schematic
################################################################################
experiment_table = glm_params.get_experiment_table()
oeid = 956903412 
cell_id = 1086505751
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params = glm_params.load_run_json(GLM_VERSION)
session = gft.load_data(oeid, run_params)
time=[1221.25, 1225.75]
gsm.strategy_paper_ophys_example(session, cell_id, time)

## Fig. 4D - Population average response
################################################################################
dfs = psth.get_figure_4_psth(data='events')
psth.plot_figure_4_averages(dfs, data='events')

# Determine significant for Vip image
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    first=False, second=False)
psth.plot_summary_bootstrap_image_strategy(vip_image, 'vip',first=False, second=False)

# Determine significance for SST omission
sst_omission = psth.load_omission_df(summary_df,cre='Sst-IRES-Cre',data='events',
    first=False, second=True)
psth.plot_summary_bootstrap_omission_strategy(sst_omission,'sst',first=False,
    second=True)

# Determine significance for VIP omission
vip_omission = psth.load_omission_df(summary_df,cre='Vip-IRES-Cre',data='events',
    second=False, first=False)
psth.plot_summary_bootstrap_omission_strategy(vip_omission,'vip',first=False,
    second=False)

# Determine significance for Exc hit/miss
exc_change = psth.load_change_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events',
    first=False, second=False, image=True)
psth.plot_summary_bootstrap_strategy_hit(exc_change, 'exc', first=False, second=False,
    image=True)

# Determine significance for Sst hit/miss
sst_change = psth.load_change_df(summary_df, cre='Sst-IRES-Cre',data='events',
   first=False, second=True)
psth.plot_summary_bootstrap_strategy_hit(sst_change,'sst',first=False, second=True)

# determine pre-change for Vip
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events',
    first=False, second=True)
psth.plot_summary_bootstrap_strategy_pre_change(vip_image,'vip',first=False, second=True)

## Fig. 4F - Running VIP control Omission
################################################################################

vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events')
bootstraps_image = psth.get_running_bootstraps('vip','image','events',10000)
psth.running_responses(vip_image, 'image',bootstraps=bootstraps_image)

## Fig. 4G - Running VIP control Omission
################################################################################

vip_omission = psth.load_omission_df(summary_df, cre='Vip-IRES-Cre',data='events')
bootstraps_omission = psth.get_running_bootstraps('vip','omission','events',10000)
psth.running_responses(vip_omission, 'omission',bootstraps=bootstraps_omission)

## Running Supplement
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

## Fig 5
################################################################################

# Just need for filepaths
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params = glm_params.load_run_json(GLM_VERSION)
labels = ['Excitatory','Sst Inhibitory','Vip Inhibitory']

# Plot PSTH over time
gpt.PSTH_analysis(dfs,  'image',run_params)
gpt.PSTH_analysis(dfs,  'omission',run_params)
gpt.PSTH_analysis(dfs,  'hit',run_params)
gpt.PSTH_analysis(dfs,  'miss',run_params)

# Plot state space plots
gpt.plot_PSTH_perturbation(dfs,labels,'image',run_params)
gpt.plot_PSTH_perturbation(dfs,labels,'omission',run_params)
gpt.plot_PSTH_perturbation(dfs,labels,'hit',run_params)
gpt.plot_PSTH_perturbation(dfs,labels,'miss',run_params)

# Plot 3D state space plots
gpt.plot_PSTH_3D(dfs,labels,'image',run_params)

# Supplemental figures
gpt.plot_PSTH_perturbation(dfs,labels,'image',run_params,x='Sst')
gpt.plot_PSTH_perturbation(dfs,labels,'omission',run_params,x='Sst')
gpt.plot_PSTH_perturbation(dfs,labels,'hit',run_params,x='Sst')
gpt.plot_PSTH_perturbation(dfs,labels,'miss',run_params,x='Sst')

gpt.plot_PSTH_perturbation(dfs,labels,'image',run_params,y='Sst')
gpt.plot_PSTH_perturbation(dfs,labels,'omission',run_params,y='Sst')
gpt.plot_PSTH_perturbation(dfs,labels,'hit',run_params,y='Sst')
gpt.plot_PSTH_perturbation(dfs,labels,'miss',run_params,y='Sst')

gpt.plot_PSTH_3D(dfs,labels,'image',run_params,supp_fig=True)
gpt.plot_PSTH_3D(dfs,labels,'omission',run_params,supp_fig=True)
gpt.plot_PSTH_3D(dfs,labels,'hit',run_params,supp_fig=True)
gpt.plot_PSTH_3D(dfs,labels,'miss',run_params,supp_fig=True)

## Fig S5 - GLM Supplement
################################################################################
GLM_VERSION = '24_events_all_L2_optimize_by_session'
run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(GLM_VERSION)
weights_beh = gst.add_behavior_session_metrics(weights_df, summary_df)
gpt.analysis(weights_beh, run_params, 'omissions')
gst.kernels_by_cre(weights_beh, run_params)


## Fig. 6 Engagement PSTHs
################################################################################
dfs = psth.get_figure_4_psth(data='events')
psth.plot_engagement(dfs,data='events')

exc_change = psth.load_change_df(summary_df, cre='Slc17a7-IRES2-Cre',data='events')
psth.plot_summary_bootstrap_strategy_engaged_miss(exc_change,cell_type='exc',
    first=True, second=False,nboots=10000)

## Fig. S6 - Running VIP control image
################################################################################
vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events')
boot_image_visual = psth.compute_engagement_running_bootstrap(vip_image,'image',
    'vip','visual',nboots=10000)
boot_image_timing = psth.compute_engagement_running_bootstrap(vip_image,'image',
    'vip','timing',nboots=10000)
psth.engagement_running_responses(vip_image, 'image',vis_boots=boot_image_visual,
    tim_boots=boot_image_timing, plot_list=['visual'])
psth.engagement_running_responses(vip_image, 'image',vis_boots=boot_image_visual,
    tim_boots=boot_image_timing, plot_list=['timing'])


## Fig. S6 - Running VIP control Omission
################################################################################
vip_omission = psth.load_omission_df(summary_df, cre='Vip-IRES-Cre',data='events')
boot_omission_visual = psth.compute_engagement_running_bootstrap(vip_omission,
    'omission','vip','visual',nboots=10000)
boot_omission_timing = psth.compute_engagement_running_bootstrap(vip_omission,
    'omission','vip','timing',nboots=10000)
psth.engagement_running_responses(vip_omission, 'omission',
    vis_boots=boot_omission_visual,
    tim_boots=boot_omission_timing, plot_list=['visual'])
psth.engagement_running_responses(vip_omission, 'omission',
    vis_boots=boot_omission_visual,
    tim_boots=boot_omission_timing, plot_list=['timing'])

## Novelty Supplement 
################################################################################
dfs_novel = psth.get_figure_4_psth(data='events',experience_level='Novel 1')
psth.plot_figure_4_averages(dfs_novel, data='events',experience_level='Novel 1')






