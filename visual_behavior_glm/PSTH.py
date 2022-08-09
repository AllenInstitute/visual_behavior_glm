import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

import psy_output_tools as po
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_strategy_tools as gst
import visual_behavior.data_access.loading as loading


# TODO
# what is the filtering? 1249 to 402? I'm ok with removing passive, but not others
# separate hit/miss, engaged/disengaged?
# split by strategy
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
    plot_population_averages_for_cell_types_across_experience(df,
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
    plot_population_averages_for_cell_types_across_experience(df,
        xlim_seconds=[-1,1.5],data_type='events',event_type='omissions')

def plot_population_averages_for_cell_types_across_experience(multi_session_df, 
    xlim_seconds=[-1.25, 1.5],data_type='events', event_type='changes', interval_sec=1):

    # get important information
    cell_types = np.sort(multi_session_df.cell_type.unique())
    #palette = utilities.get_experience_level_colors()
    palette = gvt.project_colors()

    # define plot axes
    axes_column = 'experience_level'
    hue_column = 'strategy_labels'
    xlabel='time (s)' 
    ylabel='population\nresponse'

    format_fig = True
    figsize = (10, 8)
    fig, ax = plt.subplots(3, 3, figsize=figsize, sharey='row', sharex='col')
    ax = ax.ravel()

    for i, cell_type in enumerate(cell_types):
        df = multi_session_df[(multi_session_df.cell_type == cell_type)]
        ax[i * 3:(i * 3 + 3)] = plot_population_averages_for_conditions(df, 
            data_type, event_type, axes_column, hue_column,horizontal=True,
            xlim_seconds=xlim_seconds,interval_sec=interval_sec,palette=palette,
            ax=ax[i * 3:(i * 3 + 3)])

    for dex,i in enumerate([0, 3, 6]):
        ax[i].set_ylabel(cell_types[dex])
    for i in np.arange(3, 9):
        ax[i].set_title('')
    for i in np.arange(0, 6):
        ax[i].set_xlabel('')
    for i in np.arange(6, 9):
        ax[i].set_xlabel(xlabel)
    fig.tight_layout()


def plot_population_averages_for_conditions(multi_session_df, data_type, event_type, 
    axes_column, hue_column,project_code=None, timestamps=None, palette=None, 
    title=None, suptitle=None, horizontal=True, xlim_seconds=None, interval_sec=1,
    save_dir=None, folder=None, suffix='', ax=None):

    if palette is None:
        palette = utils.get_experience_level_colors()

    sdf = multi_session_df.copy()
    if 'trace_timestamps' in sdf.keys():
        timestamps = sdf.trace_timestamps.values[0]
    elif timestamps is not None:
        timestamps = timestamps
    else:
        print('provide a multi_session_df with a trace_timestamps column')

    if xlim_seconds is None:
        xlim_seconds = [timestamps[0], timestamps[-1]]
    if 'dff' in data_type:
        ylabel = 'dF/F'
    elif 'events' in data_type:
        ylabel = 'population response'
    elif 'pupil' in data_type:
        ylabel = data_type + '\n normalized'
    elif 'running' in data_type:
        ylabel = 'running speed (cm/s)'
    else:
        ylabel = 'response'
    if event_type == 'omissions':
        omitted = True
        change = False
        xlabel = 'time after omission (s)'
    elif event_type == 'changes':
        omitted = False
        change = True
        xlabel = 'time after change (s)'
    else:
        omitted = False
        change = False
        xlabel = 'time (s)'

    if hue_column == 'experience_level':
        hue_conditions = ['Familiar', 'Novel 1', 'Novel >1']
    else:
        hue_conditions = np.sort(sdf[hue_column].unique())
    if axes_column == 'experience_level':
        axes_conditions = ['Familiar', 'Novel 1', 'Novel >1']
    else:
        axes_conditions = np.sort(sdf[axes_column].unique())[::-1]
    # if there is only one axis condition, set n conditions for plotting to 2 so it can still iterate
    if len(axes_conditions) == 1:
        n_axes_conditions = 2
    else:
        n_axes_conditions = len(axes_conditions)
    if ax is None:
        format_fig = True
        if horizontal:
            figsize = (3 * n_axes_conditions, 3)
            fig, ax = plt.subplots(1, n_axes_conditions, figsize=figsize, sharey=True)
        else:
            figsize = (5, 3.5 * n_axes_conditions)
            fig, ax = plt.subplots(n_axes_conditions, 1, figsize=figsize, sharex=True)
    else:
        format_fig = False
    for i, axis in enumerate(axes_conditions):
        for c, hue in enumerate(hue_conditions):
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            traces = cdf.mean_trace.values
            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, 
                ylabel=ylabel,legend_label=hue, color=palette[hue], 
                interval_sec=interval_sec,xlim_seconds=xlim_seconds, ax=ax[i])
            ax[i] = utils.plot_flashes_on_trace(ax[i], timestamps, change=change, 
                omitted=omitted)
            if omitted:
                omission_color = sns.color_palette()[9]
                ax[i].axvline(x=0, ymin=0, ymax=1, linestyle='--', color=omission_color)
            if title == 'metadata':
                metadata_string = utils.get_container_metadata_string(
                    utils.get_metadata_for_row_of_multi_session_df(cdf))
                ax[i].set_title(metadata_string)
            else:
                if axes_column == 'experience_level':
                    ax[i].set_title(axis, color=palette[i], fontsize=20)
                else:
                    ax[i].set_title(axis)
            ax[i].set_xlim(xlim_seconds)
            ax[i].set_xlabel(xlabel, fontsize=16)
            if horizontal:
                ax[i].set_ylabel('')
            else:
                ax[i].set_ylabel(ylabel)
                ax[i].set_xlabel('')
    if format_fig:
        if horizontal:
            ax[0].set_ylabel(ylabel)
        else:
            ax[i].set_xlabel(xlabel)

    if project_code:
        if suptitle is None:
            suptitle = 'population average - ' + data_type + ' response - ' \
                + project_code[14:]
    else:
        if suptitle is None:
            suptitle = 'population average response - ' + data_type + '_' + event_type
    if format_fig:
        plt.suptitle(suptitle, x=0.52, y=1.04, fontsize=18)
        fig.tight_layout()
        if horizontal:
            fig.subplots_adjust(wspace=0.3)

    return ax


