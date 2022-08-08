import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

import psy_output_tools as po
import visual_behavior_glm.GLM_strategy_tools as gst
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.ophys.platform_paper_figures as ppf

# TODO
# what is the filtering? 1249 to 402? I'm ok with removing passive, but not others
# separate hit/miss, engaged/disengaged?
# split by strategy
# move code from ppf to here
# add cre line labels
# add an image psth?
# what sanity checks can I do on the raw data?   
 
def change_mdf():
    data_type='events'
    interpolate=True
    output_sampling_rate=30
    inclusion_criteria = 'platform_experiment_table'
    event_type='all'
    conditions=['cell_specimen_id','is_change']
    change_mdf = loading.get_multi_session_df_for_conditions(data_type, 
        event_type, conditions, inclusion_criteria, interpolate=interpolate, 
        output_sampling_rate=output_sampling_rate, epoch_duration_mins=None)
    change_mdf = change_mdf[change_mdf.is_change==True]

    summary_df = po.get_ophys_summary_table(21)
    change_mdf = gst.add_behavior_metrics(change_mdf, summary_df)

    return change_mdf

def plot_change_mdf(change_mdf):
    df = change_mdf.copy()
    ppf.plot_population_averages_for_cell_types_across_experience(df,
        xlim_seconds=[-1,0.75],data_type='events',event_type='changes')

def omission_mdf():
    data_type='events'
    interpolate=True
    output_sampling_rate=30
    inclusion_criteria = 'platform_experiment_table'
    event_type='all'
    conditions=['cell_specimen_id','omitted']
    omission_mdf = loading.get_multi_session_df_for_conditions(data_type, 
        event_type, conditions, inclusion_criteria, interpolate=interpolate, 
        output_sampling_rate=output_sampling_rate, epoch_duration_mins=None)
    omission_mdf = omission_mdf[omission_mdf.omitted==True]

    summary_df = po.get_ophys_summary_table(21)
    omission_mdf = gst.add_behavior_metrics(omission_mdf, summary_df)

    return omission_mdf

def plot_omission_mdf(omission_mdf):
    df = omission_mdf.copy()
    ppf.plot_population_averages_for_cell_types_across_experience(df,
        xlim_seconds=[-1,1.5],data_type='events',event_type='omissions')


