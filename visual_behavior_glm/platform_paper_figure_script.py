'''
    This script generates all of the figures for the visual behavior ophys
    platform paper. 

    02/22/2022 Alex Piet (alexpiet@gmail.com)

    All functions print the location of where figures are saved

'''

### TODO, Note SDK version
### TODO, Note VBA version
### TODO, Note GLM repo version


### Import packages
import visual_behavior_glm.GLM_fit_dev as gfd
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_schematic_plots as gsm
import visual_behavior_glm.GLM_cell_metrics as gcm
from visual_behavior_glm.glm import GLM
import matplotlib.pyplt as plt
plt.ion()



### Define model version
VERSION = '24_events_all_L2_optimize_by_session'



### Load data
# This function requires access to Allen Institute internal resources
# I should save out these dataframes to a file
# Takes abouts 10 minutes to load and process data
run_params, results, results_pivoted, weights_df = gfd.get_analysis_dfs(VERSION)



### Main paper figures
## Panel A - Example inputs and outputs
# Load example experiment, which takes a few minutes
oeid = 967008471
g=glm.GLM(oeid, VERSION, use_previous_fit=True, log_results=False, log_weights=False)

# Plot Inputs and model outputs for an example cell
cell_specimen_id = 1086492467
gsm.plot_glm_example(g,cell_specimen_id, run_params)

## Panel B - Omission kernel heatmap for familiar sessions
 #TODO

## Panel C - Omission kernels for each experience level
 #TODO

## Panel D - Dropout summaries
# TODO, stats
gvt.plot_dropout_summary_population(results,run_params) 

## Panel E - Dropout averages for experience and cre-line
# TODO, stats
gvt.plot_population_averages(results_pivoted, run_params) 

## Panel F - Coding fraction by experience/cre
# TODO, stats
gvt.plot_fraction_summary_population(results_pivoted, run_params)



### Supplemental figures

## S1 - all dropouts
# TODO, need stats
gvt.plot_dropout_individual_population(results,run_params)
gvt.plot_dropout_individual_population(results,run_params,use_single=True)

## S2 - all dropouts by experience
# TODO, need stats
gvt.plot_population_averages(results_pivoted, run_params,
        dropouts_to_show=['licks','running','pupil','hits','misses'])

## S3 - cell selection
# Panel A - Comparing strictly matched cells (last familiar, first novel repeat)
# TODO, need stats
gvt.plot_population_averages(results_pivoted, run_params,
        strict_experience_matching=True)

# Panel B - Cells with explained variance > 0.5%
# TODO, need stats
gvt.plot_population_averages(results_pivoted, run_params,include_zero_cells=False)

## S4 - area
# TODO, need stats
gvt.plot_population_averages_by_area(results_pivoted, run_params)

## S5 - Depth V1
# TODO, need stats
gvt.plot_population_averages_by_depth(results_pivoted,run_params, area='VISp')

## S5 - Depth LM
# TODO, need stats
gvt.plot_population_averages_by_depth(results_pivoted,run_params, area='VISl')   

## S6 - hits breakdown
gsm.change_breakdown_schematic(run_params)
 #TODO

## S7 - misses breakdown
gsm.change_breakdown_schematic(run_params)
 #TODO

## S8 - task breakdown
gsm.change_breakdown_schematic(run_params)
 #TODO

## S9 - omission breakdown
gsm.omission_breakdown_schematic(run_params)
 #TODO

## S10 - omission excitation
# Panel A - Average kernels #TODO
# Panel B - Coding Fraction #TODO
# Panel C - dropout averages #TODO

## S11 - model validation
# Panel A - kernel support
# Here "g" is the example experiment loaded above for main figure panel A 
# Doesn't save anywhere
gvt.plot_kernel_support(g,start=45144,end=45757)

# Panel B - Explained Variance by experience/cre-line
# TODO, need stats
gvt.var_explained_by_experience(results_pivoted, run_params)

# Panel C - Explained Variance vs SNR
# r2 is a dataframe of r^2 values for different subsets of the data
# the figure legend contains the high level summary statistics
r2 = gcm.compute_event_metrics(results_pivoted, run_params)

## S12 - kernel images #TODO
## S13 - kernel hits #TODO
## S14 - kernel misses #TODO
## S15 - kernel omissions #TODO
## S16 - kernel licks #TODO
## S17 - kernel pupil #TODO
## S18 - kernel running #TODO









