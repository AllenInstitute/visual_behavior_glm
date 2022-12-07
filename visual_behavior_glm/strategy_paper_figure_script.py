import psy_output_tools as po
import visual_behavior_glm.PSTH as psth
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_fit_tools as gft
import visual_behavior_glm.GLM_schematic_plots as gsm

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
time=[1220.5, 1226.25]
gsm.strategy_paper_ophys_example(session, cell_id, time)

## Fig. 4D - Population average response
################################################################################
dfs_filtered = psth.get_figure_4_psth(data='filtered_events')
psth.plot_figure_4_averages(dfs_filtered, data='filtered_events')

## Fig. 4E - Running VIP control Omission
################################################################################

vip_omission = psth.load_omission_df(summary_df, cre='Vip-IRES-Cre',data='events')
bootstraps_omission = psth.get_running_bootstraps('vip','omission','events',10000)
psth.running_responses(vip_omission, 'omission',bootstraps=bootstraps_omission)

## Fig. 4F - Running VIP control Omission
################################################################################

vip_image = psth.load_image_df(summary_df, cre='Vip-IRES-Cre',data='events')
bootstraps_image = psth.get_running_bootstraps('vip','image','events',10000)
psth.running_responses(vip_image, 'image',bootstraps=bootstraps_image)

## Fig. 4G - EXC Hit/Miss
################################################################################
psth.plot_exc_change_summary()





