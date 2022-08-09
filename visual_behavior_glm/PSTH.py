import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

import psy_output_tools as po
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_strategy_tools as gst
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils
 
def change_mdf():
    data_type='events'
    interpolate=True
    output_sampling_rate=30
    inclusion_criteria = 'platform_experiment_table'
    inclusion_criteria = ['active_only']
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
        xlim_seconds=[-2,2],data_type='events',event_type='changes')

def omission_mdf():
    data_type='events'
    interpolate=True
    output_sampling_rate=30
    inclusion_criteria = 'platform_experiment_table'
    inclusion_criteria = ['active_only']
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
        xlim_seconds=[-2,2],data_type='events',event_type='omissions')

def plot_population_averages_for_cell_types_across_experience(multi_session_df, 
    xlim_seconds=[-1.25, 1.5],data_type='events', event_type='changes', interval_sec=1):

    # get important information
    cell_types = np.sort(multi_session_df.cell_type.unique())
    palette = gvt.project_colors()

    # define plot axes
    axes_column = 'experience_level'
    hue_column = 'strategy_labels'
    xlabel='time from '+event_type[0:-1]+' (s)' 
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
        ax[i] = plot_flashes_on_trace(ax[i], timestamps, change=change, 
            omitted=omitted)
        for c, hue in enumerate(hue_conditions):
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            traces = cdf.mean_trace.values
            ax[i] = utils.plot_mean_trace(np.asarray(traces), timestamps, 
                ylabel=ylabel,legend_label=hue, color=palette[hue], 
                interval_sec=interval_sec,xlim_seconds=xlim_seconds, ax=ax[i])
        if title == 'metadata':
            metadata_string = utils.get_container_metadata_string(
                utils.get_metadata_for_row_of_multi_session_df(cdf))
            ax[i].set_title(metadata_string)
        else:
            if axes_column == 'experience_level':
                ax[i].set_title(axis, color=palette[axis], fontsize=20)
            else:
                ax[i].set_title(axis)
        ax[i].set_xlim(xlim_seconds)
        ax[i].set_xlabel(xlabel, fontsize=16)
        if horizontal:
            ax[i].set_ylabel('', fontsize=16)
        else:
            ax[i].set_ylabel(ylabel, fontsize=16)
            ax[i].set_xlabel('', fontsize=16)
        ax[i].xaxis.set_tick_params(labelsize=12)
        ax[i].yaxis.set_tick_params(labelsize=12)
    if format_fig:
        if horizontal:
            ax[0].set_ylabel(ylabel, fontsize=16)
        else:
            ax[i].set_xlabel(xlabel, fontsize=16)

    if format_fig:
        fig.tight_layout()
        if horizontal:
            fig.subplots_adjust(wspace=0.3)

    return ax

def plot_flashes_on_trace(ax, timestamps, change=None, omitted=False):
    """
    plot stimulus flash durations on the given axis according to the provided timestamps
    """
    stim_duration = 0.250
    blank_duration = 0.50
    change_color=[0.121,0.466,.7058]
    change_time = 0
    start_time = timestamps[0]
    end_time = timestamps[-1]
    interval = (blank_duration + stim_duration)
    # after time 0
    if omitted:
        array = np.arange((change_time + interval), end_time, interval)  
        ax.axvline(x=change_time, ymin=0, ymax=1, linestyle='--', 
            color=sns.color_palette()[9], linewidth=1.5)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        if change and (i == 0):
            change_color = sns.color_palette()[0]
            ax.axvspan(amin, amax, color=change_color, 
                alpha=.25, zorder=1)
        else:
            ax.axvspan(amin, amax, color='k',alpha=.1, zorder=1)

    # before time 0
    array = np.arange(change_time, start_time - interval, -interval)
    array = array[1:]
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, color='k',alpha=.1,zorder=1)
    return ax
