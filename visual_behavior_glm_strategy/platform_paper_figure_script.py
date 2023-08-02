'''
    This script generates all of the figures for the visual behavior ophys
    platform paper. 

    02/24/2022 Alex Piet (alexpiet@gmail.com)

    All functions print the location of where figures are saved. Some 
    functions return summary statistics tables. The exceptions are the kernel
    functions for which we did not perform any quantification. The entire 
    script will take about 30 minutes to run, although I recommend just 
    running it line by line since it will generate so many figures. 

    This script was tested on the following code versions
    visual_behavior_glm_strategy v1.0.0
    visual_behavior_analysis origin/master, commit d37ba1bd
    AllenSDK origin/master version rc/2.13.4

    Note that these functions will NOT save figures by default. All of the
    functions have an input argument savefig=False that you can adjust to
    save the figures.  

'''

### Import packages
import visual_behavior_glm_strategy.GLM_fit_dev as gfd
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt
import visual_behavior_glm_strategy.GLM_analysis_tools as gat
import visual_behavior_glm_strategy.GLM_schematic_plots as gsm
import visual_behavior_glm_strategy.GLM_cell_metrics as gcm
import visual_behavior_glm_strategy.GLM_across_session as gas
from visual_behavior_glm_strategy.glm import GLM
import matplotlib.pyplot as plt
plt.ion()

### Define model version
VERSION = '24_events_all_L2_optimize_by_session'

### Load data
run_params, results, results_pivoted, weights_df = gfd.load_analysis_dfs(VERSION)


### Main paper figures
## Panel A - Example inputs and outputs
# Load example experiment, which takes a few minutes
oeid = 967008471
g=GLM(oeid, VERSION, use_previous_fit=True, log_results=False, log_weights=False)

# Plot Inputs and model outputs for an example cell
cell_specimen_id = 1086492467
gsm.plot_glm_example(g,cell_specimen_id, run_params)

## Panel B - Omission kernel heatmap for familiar sessions
# This generates several figures, Panel B is `omissions_heatmap_with_dropout_Familiar.svg`
gvt.kernel_evaluation(weights_df, run_params, 'omissions', session_filter=['Familiar'])

## Panel C - Omission kernels for each experience level
# This generates 6 figures,  3 by experience, and 3 by cre-line
# only the 3 by experience are used in the main figure
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'omissions')

## Panel D - Dropout summaries
# Returns a dataframe with rows for cre/dropout, and columns describing
# the dropout score
stats_D = gvt.plot_dropout_summary_population(results,run_params) 
results_pivoted_active = results_pivoted.query('not passive').copy()
anova, tukey = gvt.test_significant_across_cell(results_pivoted_active,'all-images')
anova, tukey = gvt.test_significant_across_cell(results_pivoted_active,'omissions')
anova, tukey = gvt.test_significant_across_cell(results_pivoted_active,'behavioral')
anova, tukey = gvt.test_significant_across_cell(results_pivoted_active,'task')

## Panel E - Dropout averages for experience and cre-line
'''
    Returns a dictionary containing two entries for each dropout in
    `dropouts_to_show`. The default option is the 4 primary dropouts, but the 
    supplemental figures below use different dropouts.

    "<dropout> stats" is a dictionary, which has a 2-tuple value for each cell type
        The stats are computed in `gvt.test_significant_dropout_averages` 
        The first entry is the results of scipy.stats.f_oneway anova test comparing
            across experience levels
        The second entry is the results of multiple comparison adjusted tukey tests
            between each individual experience level comparison.

    "<dropout> data" is a dictionary of cell types and data filters.
        "<cell type> all data" is a pandas "describe()" output on all cells
        "<cell type> matched data" is just for matched cells
        "<cell type> strict matched data" is available only if requested 
'''
stats_E = gvt.plot_population_averages(results_pivoted, run_params) 

## Panel F - Coding fraction by experience/cre
# Returns a dataframe with rows for cre/experience, and columns with the fraction of
# cells coding for each regressor, and the CI value (which is the value +/- from the mean)
stats_F = gvt.plot_fraction_summary_population(results_pivoted, run_params)



### Supplemental figures

## S1 - all dropouts
# Returns a dataframe with rows for cre/dropout, and columns describing
# the dropout score
stats_S1A = gvt.plot_dropout_individual_population(results,run_params)
stats_S1B = gvt.plot_dropout_individual_population(results,run_params,use_single=True)


## S2 - all dropouts by experience
stats_S2 = gvt.plot_population_averages(results_pivoted, run_params,
        dropouts_to_show=['licks','running','pupil','hits','misses'])


## S3 - cell selection
# Panel A - Comparing strictly matched cells (last familiar, first novel repeat)
stats_S3A = gvt.plot_population_averages(results_pivoted, run_params,
        strict_experience_matching=True)

# Panel B - Cells with explained variance > 0.5%
stats_S3B = gvt.plot_population_averages(results_pivoted, run_params,
        matched_with_variance_explained=True, matched_ve_threshold=0.05)


## S4 - area
'''
    stats is a dictionary containing entries for cell_type statistics and data
    [<cell type> stats][<feature>][<experience>] is the output of a ttest
    [<cell type> data][<feature>] is a pandas "describe()" of that dropout

'''
stats_S4 = gvt.plot_population_averages_by_area(results_pivoted, run_params) 


## S5 - Depth V1
stats_S5_V1 = gvt.plot_population_averages_by_depth(results_pivoted,run_params, 
        area=['VISp']) 


## S5 - Depth LM
stats_S5_LM = gvt.plot_population_averages_by_depth(results_pivoted,run_params, 
        area=['VISl'])   

stats_S5 = gvt.plot_population_averages_by_depth(results_pivoted,run_params, 
        area=['VISl','VISp'])   


## For Supplemental figures S6-S9, you need to load the results from a different version
# This takes a few minutes
VERSION_b = '24_events_all_L2_optimize_by_session_task_and_omission_breakdown'
run_params_b, results_b, results_pivoted_b, weights_df_b = gfd.get_analysis_dfs(VERSION_b)

# For each of S6-S9, "hits" is just the (0,0.75s) window, and "post-hits" is (0.75, end)
# In the supplemental figure, I use the VERSION figure for the entire interval

## S6 - hits breakdown
gsm.change_breakdown_schematic(run_params)
stats_S6 = gvt.plot_population_averages(results_pivoted_b, run_params_b,
    dropouts_to_show=['hits','post-hits'], extra='_breakdown')


## S7 - misses breakdown
gsm.change_breakdown_schematic(run_params)
stats_S7 = gvt.plot_population_averages(results_pivoted_b, run_params_b,
    dropouts_to_show=['misses','post-misses'], extra='_breakdown')


## S8 - task breakdown
gsm.change_breakdown_schematic(run_params)
stats_S8 = gvt.plot_population_averages(results_pivoted_b, run_params_b,
    dropouts_to_show=['task','post-task'], extra='_breakdown')


## S9 - omission breakdown 
gsm.omission_breakdown_schematic(run_params)
stats_S9 = gvt.plot_population_averages(results_pivoted_b, run_params_b,
    dropouts_to_show=['omissions','post-omissions'], extra='_breakdown')


## S10 - omission excitation
# This annotates omission excited versus inhibited cells
results_pivoted = gat.append_kernel_excitation(weights_df, results_pivoted)

# Panel A - Average kernels
# This generates several figures, you want 
# `omissions_comparison_by_omissions_excited_slc_sessions_Familiar.svg`
gvt.plot_kernel_comparison_by_kernel_excitation(weights_df, run_params,'omissions')

# Panel B - Coding Fraction 
# Returns a dataframe with rows for cre/experience, and columns with the fraction of
# cells coding for each regressor, and the CI value (which is the value +/- from the mean)
stats_S10B = gvt.plot_fraction_summary_population(results_pivoted, run_params, 
    kernel_excitation=True,kernel='omissions')

# Panel C - dropout averages
stats_S10C = gvt.plot_population_averages(results_pivoted, run_params, 
    dropouts_to_show=['omissions','omissions_positive','omissions_negative'])


## S11 - model validation
# Panel A - kernel support
# Here "g" is the example experiment loaded above for main figure panel A 
# Doesn't save anywhere
gvt.plot_kernel_support(g,start=45144,end=45757)

# Panel B - Explained Variance by experience/cre-line
# Returns a pandas dataframe describing the explained variance by experience/cre-line
stats_S11B = gvt.var_explained_by_experience(results_pivoted, run_params)

# Panel C - Explained Variance vs SNR
# r2 is a dataframe of r^2 values for different subsets of the data
# the figure legend contains the high level summary statistics
r2 = gcm.compute_event_metrics(results_pivoted, run_params)

# Panel D - Explained variance for matched and non-matched cells
# Returns a pandas dataframe describing the explained variance by experience/cre-line/matched
stats_S11D = gvt.var_explained_matched(results_pivoted, run_params)

## S12 - kernel images 
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'all-images')
# For each experience level, generate:
#    'all-images_heatmap_with_dropout_<experience>' (which is all cells)
#    'all-images_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'all-images', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'all-images', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'all-images', session_filter=['Novel >1'])


## S13 - kernel hits 
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'hits')
# For each experience level, generate:
#    'hits_heatmap_with_dropout_<experience>' (which is all cells)
#    'hits_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'hits', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'hits', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'hits', session_filter=['Novel >1'])


## S14 - kernel misses
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'misses')
# For each experience level, generate:
#    'misses_heatmap_with_dropout_<experience>' (which is all cells)
#    'misses_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'misses', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'misses', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'misses', session_filter=['Novel >1'])


## S15 - kernel omissions 
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'omissions')
# For each experience level, generate:
#    'omissions_heatmap_with_dropout_<experience>' (which is all cells)
#    'omissions_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'omissions', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'omissions', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'omissions', session_filter=['Novel >1'])

## S16 - kernel licks
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'licks')
# For each experience level, generate:
#    'licks_heatmap_with_dropout_<experience>' (which is all cells)
#    'licks_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'licks', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'licks', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'licks', session_filter=['Novel >1'])

## S17 - kernel pupil
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'pupil')
# For each experience level, generate:
#    'pupil_heatmap_with_dropout_<experience>' (which is all cells)
#    'pupil_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'pupil', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'pupil', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'pupil', session_filter=['Novel >1'])

## S18 - kernel running 
# Generate the 6 average kernel panels
gvt.plot_kernel_comparison_by_experience(weights_df, run_params, 'running')
# For each experience level, generate:
#    'running_heatmap_with_dropout_<experience>' (which is all cells)
#    'running_heatmap_with_dropoutdropout_<experience>' (which is cells with non-zero dropout scores)
gvt.kernel_evaluation(weights_df, run_params, 'running', session_filter=['Familiar'])
gvt.kernel_evaluation(weights_df, run_params, 'running', session_filter=['Novel 1'])
gvt.kernel_evaluation(weights_df, run_params, 'running', session_filter=['Novel >1'])


# S19 - Across Session normalized average dropout scores
across_run_params, across_df = gas.load_cells(run_params)
across_df = gas.append_kernel_excitation_across(weights_df, across_df)

# Plot the population averages across experience/cre line
gas.plot_across_summary(across_df, across_run_params) 


