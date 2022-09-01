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
    error bars (Standard error? Confidence interval over cells? or hierarchical boot?)
'''

def plot_condition(dfs, condition,labels=None,savefig=False):
    
    if type(dfs)!=list:
        dfs = [dfs]
    
    num_rows = len(dfs)
    fig, ax = plt.subplots(num_rows,3,figsize=(10,2.75*num_rows),sharey='row',squeeze=False)

   
    for index, full_df in enumerate(dfs): 
        if labels is None:
            ylabel='Population Average'
        else:
            ylabel=labels[index]
        max_y = [0,0,0]
        max_y[0] = plot_condition_experience(full_df, condition, 'Familiar',
            'visual_strategy_session', ax=ax[index, 0], title=index==0,ylabel=ylabel)
        max_y[1] = plot_condition_experience(full_df, condition, 'Novel 1',
            'visual_strategy_session', ax=ax[index, 1],title=index==0,ylabel='')
        max_y[2] = plot_condition_experience(full_df, condition, 'Novel >1',
            'visual_strategy_session', ax=ax[index, 2],title=index==0,ylabel='')
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))

    if savefig:
        filename = PSTH_DIR + condition+'_psth.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)

    return ax

def plot_condition_experience(full_df, condition, experience_level, split, 
    ax=None,ylabel='Population Average',xlabel=True,title=False):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    df = full_df.query('(condition ==@condition)&(experience_level ==@experience_level)')
    colors = gvt.project_colors() 
    split_vals = np.sort(df[split].unique())
    responses = []
    for val in split_vals:
        if (split == 'visual_strategy_session') and val:
            color = colors['visual']
        elif  (split == 'visual_strategy_session') and (not val):
            color = colors['timing']
        else:
            color = 'k'
        r = plot_split(df.query('{}==@val'.format(split)),ax,color=color)
        responses.append(r)

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

    return np.max(responses)

def plot_split(df, ax,color):
    # Plot mean
    time =df['time'].mean()
    response = df['response'].mean()
    ax.plot(time,response,color=color)
 
    # plot uncertainty

    return np.max(response)

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



