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

PSTH_DIR = '/home/alex.piet/codebase/behavior/PSTH/'

'''
    color control
    error bars (Standard error? Confidence interval over cells? or hierarchical boot?)
    figure saving
    ylimits seem to be broken
'''

def plot_condition(dfs, condition,labels=None):
    
    if type(dfs)!=list:
        dfs = [dfs]
    
    num_rows = len(dfs)
    fig, ax = plt.subplots(num_rows,3,figsize=(10,2.75*num_rows),sharey='row',squeeze=False)
   
    for index, full_df in enumerate(dfs): 
        if labels is None:
            ylabel='Population Average'
        else:
            ylabel=labels[index]
        plot_condition_experience(full_df, condition, 'Familiar','visual_strategy_session',
            ax=ax[index, 0], title=index==0,ylabel=ylabel)
        plot_condition_experience(full_df, condition, 'Novel 1','visual_strategy_session',
            ax=ax[index, 1],title=index==0,ylabel='')
        plot_condition_experience(full_df, condition, 'Novel >1','visual_strategy_session',
            ax=ax[index, 2],title=index==0,ylabel='')
    return ax

def plot_condition_experience(full_df, condition, experience_level, split, 
    ax=None,ylabel='Population Average',xlabel=True,title=False):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    df = full_df.query('(condition ==@condition)&(experience_level ==@experience_level)')
    
    split_vals = np.sort(df[split].unique())
    for val in split_vals:
        plot_split(df.query('{}==@val'.format(split)),ax)

    # Annotate figure
    omitted = 'omission' in condition
    change = 'change' in condition
    timestamps = df.iloc[0]['time']
    plot_flashes_on_trace(ax, timestamps, change=change, omitted=omitted)

    # Clean up axis 
    if xlabel:
        ax.set_xlabel('time (s)',fontsize=16)
    if ylabel != '':
        ax.set_ylabel(ylabel,fontsize=16)
    if title:
        ax.set_title(experience_level,fontsize=16)
    ax.set_ylim(bottom=0)
    ax.set_xlim(timestamps[0],timestamps[-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()

    return ax 

def plot_split(df, ax):
    # Plot mean
    time =df['time'].mean()
    response = df['response'].mean()
    ax.plot(time,response)
 
    # plot uncertainty


def plot_change_mdf(change_mdf,savefig=False,extra=''):
    df = change_mdf.copy()
    plot_population_averages_for_cell_types_across_experience(df,
        xlim_seconds=[-2,2],data_type='events',event_type='changes')
    
    if savefig:
        filename = PSTH_DIR + 'change_psth'+extra+'.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def plot_omission_mdf(omission_mdf,savefig=False,extra=''):
    df = omission_mdf.copy()
    plot_population_averages_for_cell_types_across_experience(df,
        xlim_seconds=[-2,2],data_type='events',event_type='omissions')

    if savefig:
        filename = PSTH_DIR + 'omission_psth'+extra+'.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)


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
        print('{:<15} (Visual, Timing)'.format(cell_type))
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
        num_cells = dict()
        for c, hue in enumerate(hue_conditions):
            cdf = sdf[(sdf[axes_column] == axis) & (sdf[hue_column] == hue)]
            num_cells[hue] = np.shape(cdf)[0]
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
        print('{:<9} ({:<4}, {:<5})'.format(axis,num_cells['visual'],num_cells['timing']))
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
