import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
plt.ion()

import psy_style as pstyle
import psy_output_tools as po
import visual_behavior_glm_strategy.GLM_params as glm_params
import visual_behavior_glm_strategy.build_dataframes as bd
import visual_behavior_glm_strategy.hierarchical_bootstrap as hb
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt
import visual_behavior_glm_strategy.GLM_strategy_tools as gst
import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils

PSTH_DIR = '/home/alex.piet/codebase/behavior/PSTH/'

def plot_all_heatmaps(dfs, labels,data='events'):
    conditions = dfs[0]['condition'].unique()
    cres = ['Exc','Sst','Vip']
    experience = ['Familiar','Novel 1','Novel >1']   
 
    for cre_dex, cre in enumerate(cres):
        for e in experience:
            for c in conditions:
                try:
                    plot_heatmap(dfs[cre_dex],cre,c,e,savefig=True,data=data)
                except Exception as ex:
                    print(c)
                    print(ex)

def plot_all_conditions(dfs, labels,data='events'):
    conditions = dfs[0]['condition'].unique()
    for c in conditions:
        try:
            plot_condition(dfs, c, labels, savefig=True,data=data)
        except Exception as e:
            print(c)
            print(e)

def compare_conditions(dfs, conditions, labels, savefig=False, plot_strategy='both',
    data='filtered_events',areas=['VISp','VISl'],depths=['upper','lower'],depth='layer'):
    # If we have just one cell type, wrap in a list 
    if type(dfs)!=list:
        dfs = [dfs]
    
    # Determine the number of cell types
    num_rows = len(dfs)
    fig, ax = plt.subplots(num_rows,3,figsize=(10,2.75*num_rows),sharey='row',
        squeeze=False)

    colors = plt.get_cmap('tab10') 
    # Iterate over conditions, and cell types
    for index, full_df, in enumerate(dfs):
        if labels is None:
            ylabel='Population Average'
        else:
            ylabel=labels[index]
        max_y = []
        for cdex, condition in enumerate(conditions):
            color = colors(cdex+2)
            max_y.append(plot_condition_experience(full_df, condition, 'Familiar',
                'visual_strategy_session', ax=ax[index, 0], title=index==0,
                ylabel=ylabel, 
                plot_strategy=plot_strategy,set_color=color,depths=depths,areas=areas,
                depth=depth))
            max_y.append(plot_condition_experience(full_df, condition, 'Novel 1',
                'visual_strategy_session', ax=ax[index, 1],title=index==0,ylabel='', 
                plot_strategy=plot_strategy,set_color=color,depths=depths,areas=areas,
                depth=depth))
            max_y.append(plot_condition_experience(full_df, condition, 'Novel >1',
                'visual_strategy_session', ax=ax[index, 2],title=index==0,ylabel='', 
                plot_strategy=plot_strategy,set_color=color,depths=depths,areas=areas,
                depth=depth))
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))

    # Add Title 
    title_str = plot_strategy+': '+', '.join(conditions)
    plt.suptitle(title_str,fontsize=16)
    plt.tight_layout()

    # Save Figure
    if savefig:
        filename = PSTH_DIR + data+'/population_averages/'+\
            plot_strategy+'_'+'_'.join(conditions)+'_psth.svg'
        print('Figure saved to: '+filename)
        plt.savefig(filename)

    return ax

def plot_condition_by_depth(dfs, condition,labels,savefig=False,error_type='sem',
    plot_strategy='both',data='events',areas=['VISp','VISl'],depths=['upper','lower'],
    depth='layer'):
   
    # If we have just one cell type, wrap in a list 
    if type(dfs)!=list:
        dfs = [dfs]

    # Determine the number of cell types
    num_rows = len(dfs)
    num_cols = len(depths)
    fig, ax = plt.subplots(num_rows,num_cols,figsize=((10/3)*num_cols,2.75*num_rows),
        sharey='row', squeeze=False)

    # Iterate through cell types   
    for index, full_df in enumerate(dfs): 
        if labels is None:
            ylabel='Population Average'
        else:
            ylabel=labels[index]
        max_y = [0]*num_cols
        for dex, col in enumerate(depths):
            max_y[dex] = plot_condition_experience(full_df, condition, 'Familiar',
            'visual_strategy_session', ax=ax[index, dex], title=index==0,ylabel=ylabel,
            error_type=error_type,areas=areas, depths=[col],depth=depth)
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))
    
    # Add Title    
    title_str = condition
    plt.suptitle(title_str,fontsize=16)
    plt.tight_layout()


def plot_condition(dfs, condition,labels=None,savefig=False,error_type='sem',
    split_by_engaged=False,plot_strategy='both',data='filtered_events',
    areas=['VISl','VISp'],depths=['upper','lower'],depth='layer'):
    '''
        Plot the population average response to condition for each element of dfs

        dfs (dataframe or list) if dfs is one dataframe, then will plot just one 
            row. If dfs is a list, will plot a row for each dataframe 
        condition (str) the feature aligned to
        labels (str or list) the labels corresponding to elements of dfs
    '''
   
    # If we have just one cell type, wrap in a list 
    if type(dfs)!=list:
        dfs = [dfs]
    
    # Determine the number of cell types
    num_rows = len(dfs)
    fig, ax = plt.subplots(num_rows,3,figsize=(10,2.75*num_rows),sharey='row',
        squeeze=False)

    # Iterate through cell types   
    for index, full_df in enumerate(dfs): 
        if labels is None:
            ylabel='Population Average'
        else:
            ylabel=labels[index]
        if not split_by_engaged:
            max_y = [0,0,0]
            max_y[0] = plot_condition_experience(full_df, condition, 'Familiar',
                'visual_strategy_session', ax=ax[index, 0], title=index==0,ylabel=ylabel,
                error_type=error_type,areas=areas, depths=depths,depth=depth)
            max_y[1] = plot_condition_experience(full_df, condition, 'Novel 1',
                'visual_strategy_session', ax=ax[index, 1],title=index==0,ylabel='',
                error_type=error_type,depth=depth)
            max_y[2] = plot_condition_experience(full_df, condition, 'Novel >1',
                'visual_strategy_session', ax=ax[index, 2],title=index==0,ylabel='',
                error_type=error_type,depth=depth)
            ax[index,0].set_ylim(top = 1.05*np.max(max_y))
        else:
            max_y = [] 
            temp = plot_condition_experience(full_df,'engaged_v1_'+condition,'Familiar',
                'visual_strategy_session', ax=ax[index, 0], title=index==0,ylabel=ylabel,
                error_type=error_type,split_by_engaged=True,plot_strategy=plot_strategy,
                depth=depth)
            max_y.append(temp)
            temp = plot_condition_experience(full_df,'disengaged_v1_'+condition,
                'Familiar','visual_strategy_session', ax=ax[index, 0], title=index==0,
                ylabel=ylabel,error_type=error_type,split_by_engaged=True,
                plot_strategy=plot_strategy,depth=depth)
            max_y.append(temp)
            temp = plot_condition_experience(full_df,'engaged_v1_'+condition,'Novel 1',
                'visual_strategy_session', ax=ax[index, 1], title=index==0,ylabel=ylabel,
                error_type=error_type,split_by_engaged=True,plot_strategy=plot_strategy,
                depth=depth)
            max_y.append(temp)
            temp = plot_condition_experience(full_df,'disengaged_v1_'+condition,
                'Novel 1','visual_strategy_session', ax=ax[index, 1], title=index==0,
                ylabel=ylabel,error_type=error_type,split_by_engaged=True,
                plot_strategy=plot_strategy,depth=depth)
            max_y.append(temp)
            temp = plot_condition_experience(full_df,'engaged_v1_'+condition,'Novel >1',
                'visual_strategy_session', ax=ax[index, 2], title=index==0,ylabel=ylabel,
                error_type=error_type,split_by_engaged=True,plot_strategy=plot_strategy,
                depth=depth)
            max_y.append(temp)
            temp = plot_condition_experience(full_df,'disengaged_v1_'+condition,
                'Novel >1','visual_strategy_session', ax=ax[index, 2], title=index==0,
                ylabel=ylabel,error_type=error_type,split_by_engaged=True,
                plot_strategy=plot_strategy,depth=depth)
            max_y.append(temp)
            ax[index,0].set_ylim(top = 1.05*np.max(max_y))
    
    # Add Title    
    title_str = condition
    if split_by_engaged:
        title_str = condition + ', '+plot_strategy +' sessions split by engagement' 
    plt.suptitle(title_str,fontsize=16)
    plt.tight_layout()

    # Save Figure
    if savefig:
        if split_by_engaged:
            extra_split = '_'+plot_strategy+'_split_by_engagement'
        else:
            extra_split = ''
        filename = PSTH_DIR + data+'/population_averages/'+\
            condition+'_psth'+extra_split+'.svg'
        print('Figure saved to: '+filename)
        plt.savefig(filename)

    return ax

def get_figure_4_psth(data='events',experience_level='Familiar',mesoscope_only=False):
 
    # Load each cell type
    vip_full_filtered = bd.load_population_df(data,'full_df',\
    'Vip-IRES-Cre',experience_level=experience_level)
    sst_full_filtered = bd.load_population_df(data,'full_df',\
    'Sst-IRES-Cre',experience_level=experience_level)
    exc_full_filtered = bd.load_population_df(data,'full_df',\
        'Slc17a7-IRES2-Cre',experience_level=experience_level)

    # Add area, depth
    experiment_table = glm_params.get_experiment_table()
    vip_full_filtered = bd.add_area_depth(vip_full_filtered, experiment_table)
    sst_full_filtered = bd.add_area_depth(sst_full_filtered, experiment_table)
    exc_full_filtered = bd.add_area_depth(exc_full_filtered, experiment_table)
    vip_full_filtered = pd.merge(vip_full_filtered, experiment_table.reset_index()\
        [['ophys_experiment_id','binned_depth']],on='ophys_experiment_id')
    sst_full_filtered = pd.merge(sst_full_filtered, experiment_table.reset_index()\
        [['ophys_experiment_id','binned_depth']],on='ophys_experiment_id')
    exc_full_filtered = pd.merge(exc_full_filtered, experiment_table.reset_index()\
        [['ophys_experiment_id','binned_depth']],on='ophys_experiment_id')

    if mesoscope_only:
        experiment_table = glm_params.get_experiment_table().reset_index()
        vip_full_filtered = pd.merge(vip_full_filtered,
            experiment_table[['ophys_experiment_id','equipment_name']],
            on='ophys_experiment_id')
        vip_full_filtered = vip_full_filtered.query('equipment_name == "MESO.1"').copy()
        sst_full_filtered = pd.merge(sst_full_filtered,
            experiment_table[['ophys_experiment_id','equipment_name']],
            on='ophys_experiment_id')
        sst_full_filtered = sst_full_filtered.query('equipment_name == "MESO.1"').copy()
        exc_full_filtered = pd.merge(exc_full_filtered,
            experiment_table[['ophys_experiment_id','equipment_name']],
            on='ophys_experiment_id')
        exc_full_filtered = exc_full_filtered.query('equipment_name == "MESO.1"').copy()

    # merge cell types
    dfs_filtered = [exc_full_filtered, sst_full_filtered, vip_full_filtered]
    labels =['Excitatory','Sst Inhibitory','Vip Inhibitory']

    return dfs_filtered


def plot_figure_4_averages(dfs,data='filtered_events',savefig=False,\
    areas=['VISp','VISl'],depths=['upper','lower'],experience_level='Familiar',
    strategy = 'visual_strategy_session',depth='layer',meso=False):

    fig, ax = plt.subplots(3,3,figsize=(10,7.75),sharey='row',squeeze=False) 
    labels=['Excitatory','Sst Inhibitory','Vip Inhibitory']
    error_type='sem'
    for index, full_df in enumerate(dfs): 
        max_y = [0,0,0]
        ylabel=labels[index] +'\n(Ca$^{2+}$ events)'
        max_y[0] = plot_condition_experience(full_df, 'omission', experience_level,
            strategy, ax=ax[index, 0], ylabel=ylabel,
            error_type=error_type,areas=areas,depths=depths,depth=depth)
        max_y[1] = plot_condition_experience(full_df, 'hit', experience_level,
            strategy, ax=ax[index, 1],ylabel='',
            error_type=error_type,areas=areas,depths=depths,depth=depth)
        max_y[2] = plot_condition_experience(full_df, 'miss', experience_level,
            strategy, ax=ax[index, 2],ylabel='',
            error_type=error_type,areas=areas,depths=depths,depth=depth)
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))
    for x in [0,1,2]:
            ax[x,0].set_xlabel('time from omission (s)',fontsize=16)
            ax[x,1].set_xlabel('time from hit (s)',fontsize=16)
            ax[x,2].set_xlabel('time from miss (s)',fontsize=16)

    # Clean up
    plt.tight_layout()
    if savefig:
        if meso:
            filename = PSTH_DIR + data + '/population_averages/'+\
                'figure_4_comparisons_psth_meso_'+experience_level+'.svg'        
        else:
            filename = PSTH_DIR + data + '/population_averages/'+\
                'figure_4_comparisons_psth_'+experience_level+'.svg' 
        print('Figure saved to: '+filename)
        plt.savefig(filename)

def add_cell_selection_labels(dfs, summary_df):
    for i in [0,1,2]:
        dfs[i] = pd.merge(dfs[i], summary_df[['behavior_session_id',
            'strategy_labels','strategy_labels_with_mixed','strategy_labels_with_none']],
            on='behavior_session_id')
    return dfs

def plot_figure_4_averages_cell_selection(dfs,data='filtered_events',savefig=False,\
    areas=['VISp','VISl'],depths=['upper','lower'],experience_level='Familiar',
    strategy = 'strategy_labels_with_none'):
    
    if 'strategy_labels' not in dfs[0]:
        print('call this function first:')
        print('dfs = add_cell_selection_labels(dfs, summary_df)')
        return

    fig, ax = plt.subplots(3,3,figsize=(10,7.75),sharey='row',squeeze=False) 
    labels=['Excitatory','Sst Inhibitory','Vip Inhibitory']
    error_type='sem'
    for index, full_df in enumerate(dfs): 
        full_df = full_df.query('{} in ["visual","timing"]'.format(strategy))
        max_y = [0,0,0]
        ylabel=labels[index] +'\n(Ca$^{2+}$ events)'
        max_y[0] = plot_condition_experience(full_df, 'omission', experience_level,
            strategy, ax=ax[index, 0], ylabel=ylabel,
            error_type=error_type,areas=areas,depths=depths)
        max_y[1] = plot_condition_experience(full_df, 'hit', experience_level,
            strategy, ax=ax[index, 1],ylabel='',
            error_type=error_type,areas=areas,depths=depths)
        max_y[2] = plot_condition_experience(full_df, 'miss', experience_level,
            strategy, ax=ax[index, 2],ylabel='',
            error_type=error_type,areas=areas,depths=depths)
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))
    for x in [0,1,2]:
            ax[x,0].set_xlabel('time from omission (s)',fontsize=16)
            ax[x,1].set_xlabel('time from hit (s)',fontsize=16)
            ax[x,2].set_xlabel('time from miss (s)',fontsize=16)

    # Clean up
    plt.tight_layout()
    if savefig:
        filename = PSTH_DIR + data + '/population_averages/'+\
            'figure_4_comparisons_psth_cell_selection_'+experience_level+'.svg' 
        print('Figure saved to: '+filename)
        plt.savefig(filename)



def plot_engagement(dfs, data='filtered_events',savefig=False,\
    areas=['VISp','VISl'],depths=['upper','lower'],error_type='sem',\
    version='v2',meso=False):

    fig, ax = plt.subplots(3,3,figsize=(10,7.75),sharey='row',squeeze=False) 
    labels=['Excitatory','Sst Inhibitory','Vip Inhibitory']
    for index, full_df in enumerate(dfs): 
        max_y = [0,0,0]
        ylabel=labels[index] +'\n(Ca$^{2+}$ events)'
        max_y[0] = plot_condition_engagement(full_df, 'omission',
            ax=ax[index, 0], ylabel=ylabel, error_type=error_type,
            areas=areas,depths=depths,version=version)
        max_y[1] = plot_condition_engagement(full_df, 'miss',
            ax=ax[index, 1],ylabel='', error_type=error_type,
            areas=areas,depths=depths,version=version)
        max_y[2] = plot_condition_engagement(full_df, 'image',
            ax=ax[index, 2],ylabel='', error_type=error_type,
            areas=areas,depths=depths,version=version)
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))
    for x in [0,1,2]:
            ax[x,0].set_xlabel('time from omission (s)',fontsize=16)
            ax[x,1].set_xlabel('time from miss (s)',fontsize=16)
            ax[x,2].set_xlabel('time from image (s)',fontsize=16)

    plt.tight_layout()
    if savefig:
        if meso:
            filename = PSTH_DIR + data + '/population_averages/'+\
                'engagement_comparisons_psth.svg' 
        else:
            filename = PSTH_DIR + data + '/population_averages/'+\
                'engagement_comparisons_meso_psth.svg' 
        print('Figure saved to: '+filename)
        plt.savefig(filename)




def plot_condition_engagement(full_df, condition,experience_level='Familiar', 
    plot_strategy='both',areas=['VISp','VISl'],depths=['upper','lower'],
    depth='layer',ax=None, error_type='sem',xlabel=True, ylabel='Population Average',
    title=False,version='v2'):

    if ax is None:
        fig, ax = plt.subplots()

    colors = gvt.project_colors()    
    responses = []

    condition1 = 'engaged_'+version+'_'+condition    
    df = full_df.query('(condition ==@condition1)&(experience_level ==@experience_level)'+\
        '&(targeted_structure in @areas)&({} in @depths)'.format(depth))
    r = plot_split(df.query('not visual_strategy_session'),ax,color=colors['timing'],
        error_type=error_type)
    responses.append(r)
    r = plot_split(df.query('visual_strategy_session'),ax,color=colors['visual'],
        error_type=error_type)
    responses.append(r)

    condition2 = 'disengaged_'+version+'_'+condition    
    df = full_df.query('(condition ==@condition2)&(experience_level ==@experience_level)'+\
        '&(targeted_structure in @areas)&({} in @depths)'.format(depth))
    r = plot_split(df.query('not visual_strategy_session'),ax,color='lightblue',
        error_type=error_type)
    responses.append(r)
    r = plot_split(df.query('visual_strategy_session'),ax,color='burlywood',
        error_type=error_type)
    responses.append(r)

    # Annotate figure
    omitted = 'omission' in condition
    change = (not omitted) and (('change' in condition) or \
        ('hit' in condition) or ('miss' in condition))
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

    return np.max([ax.get_ylim()[1],np.nanmax(responses)]) 

def plot_condition_experience(full_df, condition, experience_level, split, 
    ax=None,ylabel='Population Average',xlabel=True,title=False,error_type='sem',
    split_by_engaged=False, plot_strategy ='both',set_color=None,areas=['VISp','VISl'],
    depths=['upper','lower'],depth='layer'):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    df = full_df.query('(condition ==@condition)&(experience_level ==@experience_level)&(targeted_structure in @areas)&({} in @depths)'.format(depth))
    colors = gvt.project_colors() 
    if plot_strategy != 'both':
        if plot_strategy == 'visual':
            df = df.query('visual_strategy_session').copy()
        else:
            df = df.query('not visual_strategy_session').copy()
    split_vals = np.sort(df[split].unique())
    responses = []
    for val in split_vals:
        if set_color is not None:
            color = set_color
        else:
            if (split == 'visual_strategy_session') and val:
                color = colors['visual']
                if split_by_engaged and ('disengaged' in condition):
                    color='gray'
            elif  (split == 'visual_strategy_session') and (not val):
                color = colors['timing']
                if split_by_engaged and ('disengaged' in condition):
                    color='gray'
            elif 'strategy_labels' in split:
                if val == 'visual':
                    color = colors['visual']
                elif val == 'timing':
                    color = colors['timing']
                else:
                    color = 'k'
            else:
                color = 'k'
        r = plot_split(df.query('{}==@val'.format(split)),ax,color=color,
            error_type=error_type)
        responses.append(r)

    # Annotate figure
    omitted = 'omission' in condition
    change = (not omitted) and (('change' in condition) or \
        ('hit' in condition) or ('miss' in condition))
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

    return np.max([ax.get_ylim()[1],np.nanmax(responses)])

def plot_split(df, ax,color,error_type = 'sem',lw=2):
    # Plot mean
    x = np.vstack(df['response'].values)
    time =df['time'].mean()
    response = np.nanmean(x,axis=0)
    ax.plot(time,response,color=color,linewidth=lw)
 
    # plot uncertainty
    if error_type == 'sem':
        num_cells = np.sum(~np.isnan(np.mean(x,axis=1)))
        sem = np.nanstd(x,axis=0)/np.sqrt(num_cells)
    elif error_type == 'bootstrap':
        sem = 0
    
    upper = response + sem
    lower = response - sem
    ax.fill_between(time, upper, lower, alpha=0.25, color=color,zorder=-10)

    return np.nanmax(upper)

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

def plot_heatmap(full_df,cre,condition,experience_level,savefig=False,
    data='filtered_events'):

    # Set up multi-axis figure
    height = 5
    width = 6
    pre_h_offset = 1.
    post_h_offset = .25
    below_v_offset = .75
    above_v_offset = .35
    fig = plt.figure(figsize=(width, height))
    h = [Size.Fixed(pre_h_offset), Size.Fixed(width-pre_h_offset-post_h_offset)]
    v = [Size.Fixed(below_v_offset), 
        Size.Fixed((height-above_v_offset-below_v_offset)/2)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax1 = fig.add_axes(divider.get_position(), 
        axes_locator=divider.new_locator(nx=1,ny=1))
    v = [Size.Fixed(below_v_offset+(height-above_v_offset-below_v_offset)/2),
        Size.Fixed((height-above_v_offset-below_v_offset)/2)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax2 = fig.add_axes(divider.get_position(), 
        axes_locator=divider.new_locator(nx=1,ny=1))

    # process data
    df = full_df\
        .query('(condition==@condition)&(experience_level==@experience_level)')\
        .copy()
    df['max'] = [np.nanmax(x) for x in df['response']]
    df=df[~df['response'].isnull()]
    df = df.sort_values(by=['visual_strategy_session','max'],ascending=False)
    x = np.vstack(df['response'].values)
    vmax = np.nanpercentile(x,99)
    time = df.iloc[0]['time']

    # Visual session cells
    df_visual = df.query('visual_strategy_session').\
        sort_values(by=['max'],ascending=False)
    x_visual = np.vstack(df_visual['response'].values)
    num_visual = len(df_visual)
    ax2.imshow(x_visual, aspect='auto',vmax=vmax,extent=[time[0], time[-1],0,num_visual])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    ax2.set_ylabel('Visual Cells',fontsize=16)
    ax2.set_xticks([])
    ax2.set_title(cre+' '+experience_level+' '+condition,fontsize=16)

    # Timing session cells
    df_timing = df.query('not visual_strategy_session').\
        sort_values(by=['max'],ascending=False)
    x_timing = np.vstack(df_timing['response'].values)
    num_timing = len(df_timing)
    ax1.imshow(x_timing, aspect='auto',vmax=vmax,extent=[time[0], time[-1],0,num_timing])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.set_ylabel('Timing Cells',fontsize=16)
    ax1.set_xlabel('Time from {} (s)'.format(condition),fontsize=16)

    # Save figure
    if savefig:
        filename = PSTH_DIR + data+'/heatmap/' + \
            'heatmap_{}_{}_{}.png'.format(cre,condition,experience_level)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 
    return ax1,ax2

def plot_QQ_engagement(full_df,cre,condition,experience_level,savefig=False,
    quantiles=200,ax=None,data='filtered_events'):
    
    # Prep data
    e_condition = 'engaged_v2_'+condition
    d_condition = 'disengaged_v2_'+condition
    df_e = full_df\
            .query('(condition==@e_condition)&(experience_level==@experience_level)')\
            .copy()     
    df_e['max'] = [np.nanmax(x) for x in df_e['response']] 

    df_d = full_df\
            .query('(condition==@d_condition)&(experience_level==@experience_level)')\
            .copy()     
    df_d['max'] = [np.nanmax(x) for x in df_d['response']] 
 
    y = df_e['max'].values
    x = df_d['max'].values
    quantiles = np.linspace(start=0,stop=1,num=int(quantiles))
    x_quantiles = np.nanquantile(x,quantiles, interpolation='linear')[1:-1]
    y_quantiles = np.nanquantile(y,quantiles, interpolation='linear')[1:-1]
    max_val = np.max([np.max(x_quantiles),np.max(y_quantiles)])
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_quantiles, y_quantiles, 'o-',alpha=.5)
    ax.plot([0,1.05*max_val],[0,1.05*max_val],'k--',alpha=.5)
    ax.set_ylabel('engaged quantiles',fontsize=16)
    ax.set_xlabel('disengage quantiles',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_aspect('equal')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title('{}, {}, {}'.format(cre, condition, experience_level),fontsize=16)

    # Save figure
    if savefig:
        filename = PSTH_DIR + data+'/QQ/' +\
            'QQ_{}_{}_{}.png'.format(cre,condition,experience_level)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 

    return ax




def plot_QQ_strategy(full_df,cre,condition,experience_level,savefig=False,
    quantiles=200,ax=None,data='filtered_events'):
    
    # Prep data
    df = full_df\
            .query('(condition==@condition)&(experience_level==@experience_level)')\
            .copy()     
    df['max'] = [np.nanmax(x) for x in df['response']] 
 
    y = df.query('visual_strategy_session')['max'].values
    x = df.query('not visual_strategy_session')['max'].values
    quantiles = np.linspace(start=0,stop=1,num=int(quantiles))
    x_quantiles = np.nanquantile(x,quantiles, interpolation='linear')[1:-1]
    y_quantiles = np.nanquantile(y,quantiles, interpolation='linear')[1:-1]
    max_val = np.max([np.max(x_quantiles),np.max(y_quantiles)])
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_quantiles, y_quantiles, 'o-',alpha=.5)
    ax.plot([0,1.05*max_val],[0,1.05*max_val],'k--',alpha=.5)
    ax.set_ylabel('visual session cell responses',fontsize=16)
    ax.set_xlabel('timing session cell responses',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_aspect('equal')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title('{}, {}, {}'.format(cre, condition, experience_level),fontsize=16)

    # Save figure
    if savefig:
        filename = PSTH_DIR + data+'/QQ/' +\
            'QQ_{}_{}_{}.png'.format(cre,condition,experience_level)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 

    return ax


def plot_strategy_histogram(full_df,cre,condition,experience_level,savefig=False,
    quantiles=200,ax=None,data='filtered_events',nbins=50):
    
    # Prep data
    df = full_df\
            .query('(condition==@condition)&(experience_level==@experience_level)')\
            .copy()     
    df['max'] = [np.nanmax(x) for x in df['response']] 
 
    x = df.query('visual_strategy_session')['max'].values
    y = df.query('not visual_strategy_session')['max'].values
    max_val = np.max([np.max(x),np.max(y)])
    if cre == 'Exc':
        max_y = .25
    elif cre == 'Vip':
        max_y = .5   
    bins = np.linspace(0,max_y,nbins)

    if ax is None:
        fig, ax = plt.subplots()
    vis,bins,_ = ax.hist(x,bins=bins,color='darkorange',alpha=.5,density=True,
        label='visual')
    time,bins,_= ax.hist(y,bins=bins,color='blue',alpha=.5,density=True,label='timing')
    ax.set_ylabel('density',fontsize=16)
    ax.set_xlabel('Avg. Cell response',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_yscale('log')
    ax.set_xlim(left=0)
    ax.set_title('{}, {}, {}'.format(cre, condition, experience_level),fontsize=16)

    # Save figure
    if savefig:
        filename = PSTH_DIR + data+'/hist/' +\
            'hist_{}_{}_{}.png'.format(cre,condition,experience_level)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 

    return ax

def compute_running_bootstrap_bin(df, condition, cell_type, bin_num, nboots=10000,
    data='events',meso=False,first=False, second=False):

    filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_{}'.format(bin_num),
        meso=meso,first=first,second=second)  

    if os.path.isfile(filename):
        print('Already computed {}'.format(bin_num))
        return 

    # Figure out bins
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5#2   
    df['running_bins'] = np.floor(df['running_speed']/bin_width)
    bins = np.sort(df['running_bins'].unique())  

    # Check if any data 
    if bin_num not in bins:
        print('No data for this running bin')
        return 
   
    # Compute bootstrap for this bin 
    temp = df.query('running_bins == @bin_num')[['visual_strategy_session',
        'ophys_experiment_id','cell_specimen_id','response']]
    means = hb.bootstrap(temp, levels=['visual_strategy_session',
        'ophys_experiment_id','cell_specimen_id'],nboots=nboots)
    if (True in means) & (False in means):
        diff = np.array(means[True]) - np.array(means[False])
        pboot = np.sum(diff<0)/len(diff)
        visual = np.std(means[True])
        timing = np.std(means[False])
    else:
        pboot = 0.5
        if (True in means):
            visual = np.std(means[True])
        else:
            visual = 0 
        if (False in means):
            timing = np.std(means[False])
        else:
            timing = 0 
    bootstrap = {
        'running_bin':bin_num,
        'p_boot':pboot,
        'visual_sem':visual,
        'timing_sem':timing,
        'n_boots':nboots,
        }

    # Save to file
    filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_{}'.format(bin_num),meso=meso,
        first=first,second=second)
    with open(filename,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bin saved to {}'.format(filename)) 

def compute_running_bootstrap(df,condition,cell_type,nboots=10000,data='events',
    compute=True,split='visual_strategy_session',meso=False,first=False,second=False):
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5#2   
    df['running_bins'] = np.floor(df['running_speed']/bin_width)

    bootstraps = []
    bins = np.sort(df['running_bins'].unique())
    missing=False
    for b in bins:
        # First check if this running bin has already been computed
        filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
            [split],'running_{}'.format(int(b)),meso=meso,first=first,second=second)
        if os.path.isfile(filename):
            # Load this bin
            with open(filename,'rb') as handle:
                this_boot = pickle.load(handle)
            print('loading this boot from file: {}'.format(b))
            bootstraps.append(this_boot) 
        elif compute:
            print('Need to compute this bin: {}'.format(b))
            temp = df.query('running_bins == @b')[[split,
                'ophys_experiment_id','cell_specimen_id','response']]
            means = hb.bootstrap(temp, levels=[split,
                'ophys_experiment_id','cell_specimen_id'],nboots=nboots)
            if (True in means) & (False in means):
                diff = np.array(means[True]) - np.array(means[False])
                pboot = np.sum(diff<0)/len(diff)
                visual = np.std(means[True])
                timing = np.std(means[False])
            else:
                pboot = 0.5
                if (True in means):
                    visual = np.std(means[True])
                else:
                    visual = 0 
                if (False in means):
                    timing = np.std(means[False])
                else:
                    timing = 0 
            bootstraps.append({
                'running_bin':b,
                'p_boot':pboot,
                'visual_sem':visual,
                'timing_sem':timing,
                'n_boots':nboots,
                })
        else:
            missing=True 
            print('Missing bin: {}, and compute=False'.format(b))
    
    # Convert to dataframe, do multiple comparisons corrections
    bootstraps = pd.DataFrame(bootstraps)
    bootstraps['p_boot'] = [0.5 if ((row.visual_sem ==0)or(row.timing_sem==0)) \
                                else row.p_boot for index, row in bootstraps.iterrows()]
    bootstraps['p_boot'] = [1-x if x > .5 else x for x in bootstraps['p_boot']] 
    bootstraps['ind_significant'] = bootstraps['p_boot'] <= 0.05 
    bootstraps['location'] = bootstraps['running_bin']
    bootstraps = add_hochberg_correction(bootstraps)
    bootstraps = bootstraps.drop(columns=['location'])

    # Save file
    if not missing:
        filepath = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
            ['visual_strategy_session'],'running',meso=meso,first=first,second=second)
        print('saving bootstraps to: '+filepath)
        bootstraps.to_feather(filepath)
    return bootstraps


def compute_pre_change_running_bootstrap_bin(df, condition, cell_type, strategy,
    bin_num, nboots=10000, data='events'):

    filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_pre_change_{}_{}'.format(strategy,bin_num))  
    if os.path.isfile(filename):
        print('Already computed {}'.format(bin_num))
        return 

    # Figure out bins
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5   

    # Filter to strategy
    if strategy == 'visual':
        df = df.query('visual_strategy_session')\
            .query('(pre_hit_1==1)or(pre_miss_1==1)').copy()
    elif strategy == 'timing':
        df = df.query('not visual_strategy_session')\
            .query('(pre_hit_1==1)or(pre_miss_1==1)').copy()
    else:
        raise Exception('bad strategy')

    df['pre_hit_1'] = df['pre_hit_1'].astype(bool)
    df['running_bins'] = np.floor(df['running_speed']/bin_width)
    bins = np.sort(df['running_bins'].unique())  

    # Check if any data 
    if bin_num not in bins:
        print('No data for this running bin')
        return 
   
    # Compute bootstrap for this bin 
    temp = df.query('running_bins == @bin_num')[['pre_hit_1',
        'ophys_experiment_id','cell_specimen_id','response']]
    means = hb.bootstrap(temp, levels=['pre_hit_1',
        'ophys_experiment_id','cell_specimen_id'],nboots=nboots)
    if (True in means) & (False in means):
        diff = np.array(means[True]) - np.array(means[False])
        pboot = np.sum(diff<0)/len(diff)
        pre_hit = np.std(means[True])
        pre_miss = np.std(means[False])
        pre_hit_mean = np.mean(means[True])
        pre_miss_mean = np.mean(means[False])
    else:
        pboot = 0.5
        if (True in means):
            pre_hit = np.std(means[True])
            pre_hit_mean = np.mean(means[True])
        else:
            pre_hit = 0 
            pre_hit_mean=np.nan
        if (False in means):
            pre_miss = np.std(means[False])
            pre_miss_mean = np.mean(means[False])
        else:
            pre_miss = 0 
            pre_miss_mean= np.nan
    bootstrap = {
        'running_bin':bin_num,
        'p_boot':pboot,
        'pre_hit_sem':pre_hit,
        'pre_miss_sem':pre_miss,
        'pre_hit_mean':pre_hit_mean,
        'pre_miss_mean':pre_miss_mean,
        'n_boots':nboots,
        }

    # Save to file
    filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_pre_change_{}_{}'.format(strategy,bin_num))
    with open(filename,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bin saved to {}'.format(filename)) 



def compute_pre_change_running_bootstrap(df,condition,cell_type,strategy,
    nboots=10000,data='events',compute=True,split='pre_hit_1'):
    
    # set up running bins
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5#2  

    # Set up data
    if strategy == 'visual':
        df = df.query('visual_strategy_session')\
            .query('(pre_hit_1==1)or(pre_miss_1==1)').copy()
    elif strategy == 'timing':
        df = df.query('not visual_strategy_session')\
            .query('(pre_hit_1==1)or(pre_miss_1==1)').copy()
    else:
        raise Exception('bad strategy')
   
    df['pre_hit_1'] = df['pre_hit_1'].astype(bool) 
    df['running_bins'] = np.floor(df['running_speed']/bin_width)

    bootstraps = []
    bins = np.sort(df['running_bins'].unique())
    missing=False
    for b in bins:
        # First check if this running bin has already been computed
        filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
            ['visual_strategy_session'],'running_pre_change_{}_{}'.format(strategy,int(b))) 
        if os.path.isfile(filename):
            # Load this bin
            with open(filename,'rb') as handle:
                this_boot = pickle.load(handle)
            print('loading this boot from file: {}'.format(b))
            bootstraps.append(this_boot) 
        elif compute:
            print('Need to compute this bin: {}'.format(b))
            temp = df.query('running_bins == @b')[['pre_hit_1',
                'ophys_experiment_id','cell_specimen_id','response']]
            means = hb.bootstrap(temp, levels=['pre_hit_1',
                'ophys_experiment_id','cell_specimen_id'],nboots=nboots)
            if (True in means) & (False in means): 
                diff = np.array(means[True]) - np.array(means[False])
                pboot = np.sum(diff<0)/len(diff)
                pre_hit = np.std(means[True])
                pre_miss = np.std(means[False])
            else:
                pboot = 0.5
                if (True in means):
                    pre_hit = np.std(means[True])
                else:
                    pre_hit = 0 
                if (False in means):
                    pre_miss = np.std(means[False])
                else:
                    pre_miss = 0 
            bootstraps.append({
                'running_bin':b,
                'p_boot':pboot,
                'pre_hit_sem':pre_hit,
                'pre_miss_sem':pre_miss,
                'n_boots':nboots,
                })
        else:
            missing=True 
            print('Missing bin: {}, and compute=False'.format(b))
    
    # Convert to dataframe, do multiple comparisons corrections
    bootstraps = pd.DataFrame(bootstraps)
    bootstraps['p_boot'] = [0.5 if ((row.pre_hit_sem ==0)or(row.pre_miss_sem==0)) \
                                else row.p_boot for index, row in bootstraps.iterrows()]
    bootstraps['p_boot'] = [1-x if x > .5 else x for x in bootstraps['p_boot']] 
    bootstraps['ind_significant'] = bootstraps['p_boot'] <= 0.05 
    bootstraps['location'] = bootstraps['running_bin']
    bootstraps = add_hochberg_correction(bootstraps)
    bootstraps = bootstraps.drop(columns=['location'])

    # Save file
    if not missing:
        filepath = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
            ['visual_strategy_session'],'running_pre_change_{}'.format(strategy))
        print('saving bootstraps to: '+filepath)
        bootstraps.to_feather(filepath)
    return bootstraps



def compute_engagement_running_bootstrap_bin(df, condition, cell_type, strategy,
    bin_num, nboots=10000, data='events',meso=False,first=False,second=False):

    filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_engaged_{}_{}'.format(strategy,bin_num),
        meso=meso,first=first,second=second)  
    if os.path.isfile(filename):
        print('Already computed {}'.format(bin_num))
        return 

    # Figure out bins
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5#2   

    # Filter to strategy
    if strategy == 'visual':
        df = df.query('visual_strategy_session').copy()
    elif strategy == 'timing':
        df = df.query('not visual_strategy_session').copy()
    else:
        raise Exception('bad strategy')

    df['running_bins'] = np.floor(df['running_speed']/bin_width)
    bins = np.sort(df['running_bins'].unique())  

    # Check if any data 
    if bin_num not in bins:
        print('No data for this running bin')
        return 
   
    # Compute bootstrap for this bin 
    temp = df.query('running_bins == @bin_num')[['engagement_v2',
        'ophys_experiment_id','cell_specimen_id','response']]
    means = hb.bootstrap(temp, levels=['engagement_v2',
        'ophys_experiment_id','cell_specimen_id'],nboots=nboots)
    if (True in means) & (False in means):
        diff = np.array(means[True]) - np.array(means[False])
        pboot = np.sum(diff<0)/len(diff)
        engaged = np.std(means[True])
        disengaged = np.std(means[False])
        engaged_mean = np.mean(means[True])
        disengaged_mean = np.mean(means[False])
    else:
        pboot = 0.5
        if (True in means):
            engaged = np.std(means[True])
            engaged_mean = np.mean(means[True])
        else:
            engaged = 0 
            engaged_mean=np.nan
        if (False in means):
            disengaged = np.std(means[False])
            disengaged_mean = np.mean(means[False])
        else:
            disengaged = 0 
            disengaged_mean= np.nan
    bootstrap = {
        'running_bin':bin_num,
        'p_boot':pboot,
        'engaged_sem':engaged,
        'disengaged_sem':disengaged,
        'engaged_mean':engaged_mean,
        'disengaged_mean':disengaged_mean,
        'n_boots':nboots,
        }

    # Save to file
    filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_engaged_{}_{}'.format(strategy,bin_num),
        meso=meso,first=first,second=second)
    with open(filename,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bin saved to {}'.format(filename)) 



def compute_engagement_running_bootstrap(df,condition,cell_type,strategy,
    nboots=10000,data='events',compute=True,split='engagement_v2',
    meso=False,first=False,second=False):
    
    # set up running bins
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5#2  

    # Set up data
    if strategy == 'visual':
        df = df.query('visual_strategy_session').copy()
    elif strategy == 'timing':
        df = df.query('not visual_strategy_session').copy()
    else:
        raise Exception('bad strategy')
    df['running_bins'] = np.floor(df['running_speed']/bin_width)

    bootstraps = []
    bins = np.sort(df['running_bins'].unique())
    missing=False
    for b in bins:
        # First check if this running bin has already been computed
        filename = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
            ['visual_strategy_session'],'running_engaged_{}_{}'.format(strategy,int(b)),
            meso=meso,first=first,second=second) 
        if os.path.isfile(filename):
            # Load this bin
            with open(filename,'rb') as handle:
                this_boot = pickle.load(handle)
            print('loading this boot from file: {}'.format(b))
            bootstraps.append(this_boot) 
        elif compute:
            print('Need to compute this bin: {}'.format(b))
            temp = df.query('running_bins == @b')[[split,
                'ophys_experiment_id','cell_specimen_id','response']]
            means = hb.bootstrap(temp, levels=[split,
                'ophys_experiment_id','cell_specimen_id'],nboots=nboots)
            if (True in means) & (False in means):
                diff = np.array(means[True]) - np.array(means[False])
                pboot = np.sum(diff<0)/len(diff)
                engaged = np.std(means[True])
                disengaged = np.std(means[False])
            else:
                pboot = 0.5
                if (True in means):
                    engaged = np.std(means[True])
                else:
                    engaged = 0 
                if (False in means):
                    disengaged = np.std(means[False])
                else:
                    disengaged = 0 
            bootstraps.append({
                'running_bin':b,
                'p_boot':pboot,
                'engaged_sem':engaged,
                'disengaged_sem':disengaged,
                'n_boots':nboots,
                })
        else:
            missing=True 
            print('Missing bin: {}, and compute=False'.format(b))
    
    # Convert to dataframe, do multiple comparisons corrections
    bootstraps = pd.DataFrame(bootstraps)
    bootstraps['p_boot'] = [0.5 if ((row.engaged_sem ==0)or(row.disengaged_sem==0)) \
                                else row.p_boot for index, row in bootstraps.iterrows()]
    bootstraps['p_boot'] = [1-x if x > .5 else x for x in bootstraps['p_boot']] 
    bootstraps['ind_significant'] = bootstraps['p_boot'] <= 0.05 
    bootstraps['location'] = bootstraps['running_bin']
    bootstraps = add_hochberg_correction(bootstraps)
    bootstraps = bootstraps.drop(columns=['location'])

    # Save file
    if not missing:
        filepath = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
            ['visual_strategy_session'],'running_engaged_{}'.format(strategy),
            meso=meso,first=first,second=second)
        print('saving bootstraps to: '+filepath)
        bootstraps.to_feather(filepath)
    return bootstraps


def get_running_bootstraps(cell_type, condition,data,nboots,meso=False,first=False,second=False):
    filepath = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running',meso=meso,first=first,second=second) 
    if os.path.isfile(filepath):
        bootstraps = pd.read_feather(filepath)
        return bootstraps
    else:
        print('file not found, compute the running bootstraps first')

def get_engagement_running_bootstraps(cell_type, condition,data,nboots,strategy,
    meso=False,first=False,second=False):
    filepath = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_engaged_{}'.format(strategy),
        meso=meso,first=first,second=second) 
    if os.path.isfile(filepath):
        bootstraps = pd.read_feather(filepath)
        return bootstraps
    else:
        print('file not found, compute the running bootstraps first')

def get_pre_change_running_bootstraps(cell_type, condition,data,nboots,strategy):
    filepath = get_hierarchy_filename(cell_type,condition,data,'all',nboots,
        ['visual_strategy_session'],'running_pre_change_{}'.format(strategy)) 
    if os.path.isfile(filepath):
        bootstraps = pd.read_feather(filepath)
        return bootstraps
    else:
        print('file not found, compute the running bootstraps first')

def running_responses(df, condition, cre='vip', bootstraps=None, savefig=False,
    data='events', split='visual_strategy_session',meso=False):
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5

    fig, ax = plt.subplots(figsize=(4.5,2.75)) #3.75

    df['running_bins'] = np.floor(df['running_speed']/bin_width)

    summary = df.groupby([split,'running_bins'])['response'].mean()\
        .reset_index()
    visual = summary.query(split)
    timing = summary.query('not {}'.format(split))
   
    summary_sem = df.groupby([split,'running_bins'])['response'].sem()\
        .reset_index()
    visual_sem = summary_sem.query(split)
    timing_sem = summary_sem.query('not {}'.format(split))

    # Remove running bins with less than 100 responses
    counts = df.groupby(['running_bins','visual_strategy_session'])\
        ['response'].count().unstack().reset_index()
    counts['remove'] = [(row[False]<100)or(row[True]<100) for \
        index, row in counts.iterrows()]
    counts['running_bin'] = counts['running_bins'].astype(int)
    if bootstraps is not None:
        bootstraps = pd.merge(bootstraps, counts[['running_bin','remove']],\
            on='running_bin')

    if split == "visual_strategy_session":
        vis_color = 'darkorange'
        tim_color = 'blue'
        vis_label = 'visual strategy'
        tim_label = 'timing strategy'
    else:
        vis_color = 'darkorange'
        tim_color = 'red'
        vis_label = 'engaged'
        tim_label = 'disengaged'

    if bootstraps is not None:
        vtemp = visual.set_index('running_bins')
        ttemp = timing.set_index('running_bins')
        bootstraps = bootstraps.set_index('running_bin')
        for b in visual['running_bins'].unique():
            if b in bootstraps.index.values:
                plt.errorbar(b*bin_width, vtemp.loc[b].response,
                    yerr=bootstraps.loc[b]['visual_sem'],color=vis_color,fmt='o')   
            else:
                print('missing value')
        for b in timing['running_bins'].unique():
            if b in bootstraps.index.values:
                plt.errorbar(b*bin_width, ttemp.loc[b].response,
                    yerr=bootstraps.loc[b]['timing_sem'],color=tim_color,fmt='o')    
            else:
                print('missing value')
        plt.plot(visual.running_bins*bin_width, visual.response,'o',
            color=vis_color,label=vis_label)
        plt.plot(timing.running_bins*bin_width, timing.response,'o',
            color=tim_color,label=tim_label)
    else:
        plt.errorbar(visual.running_bins*bin_width, visual.response,
            yerr=visual_sem.response,color=vis_color,fmt='o',label=vis_label)
        plt.errorbar(timing.running_bins*bin_width, timing.response,
            yerr=timing_sem.response,color=tim_color,fmt='o',label=tim_label)
    ax.set_ylabel(cre.capitalize()+' '+condition+'\n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.set_xlabel('running speed (cm/s)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12) 

    if (cre == 'vip') and (condition =='omission'):
        ax.set_ylim(0,.07) 
    elif (cre == 'vip') and (condition =='image'):
        ax.set_ylim(0,.02)
    else:
        ax.set_ylim(bottom=0)

    if (bootstraps is not None) and ('bh_significant' in bootstraps.columns):
        y =  ax.get_ylim()[1]*1.05
        bootstraps = bootstraps.reset_index()
        for index, row in bootstraps.iterrows():
            if row.bh_significant:
                if row.remove:
                    print('Not significant b/c of low count: {}'.format(row.running_bin))
                else:
                    ax.plot(row.running_bin*bin_width, y, 'k*')  
        ax.set_ylim(top=y*1.075)

    ax.set_xlim(-1,51)
    plt.legend()

    plt.tight_layout() 

    # Save fig
    if savefig:
        if meso:
            filename = PSTH_DIR + data+'/running/'+\
                'running_{}_familiar_meso_{}_{}.svg'.format(cre,condition,split)   
        else:
            filename = PSTH_DIR + data+'/running/'+\
                'running_{}_familiar_{}_{}.svg'.format(cre,condition,split)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename)

def pre_change_running_responses(df, condition, cre='vip', bootstraps=None, savefig=False,
    data='events', strategy='visual_strategy_session'):
    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5
    min_events=10

    fig, ax = plt.subplots(figsize=(3.75,2.75))

    df = df.query(strategy).query('(pre_hit_1==1)or(pre_miss_1==1)').copy()
    df['running_bins'] = np.floor(df['running_speed']/bin_width)

    split='pre_hit_1'
    df['pre_hit_1'] = df['pre_hit_1'].astype(bool)
    summary = df.groupby(['pre_hit_1','running_bins'])['response'].mean()\
        .reset_index()
    df1 = summary.query(split)
    df2=  summary.query('not {}'.format(split))
   
    summary_sem = df.groupby([split,'running_bins'])['response'].sem()\
        .reset_index()
    df1_sem = summary_sem.query(split)
    df2_sem = summary_sem.query('not {}'.format(split))

    # Remove running bins with less than 100 responses
    counts = df.groupby(['running_bins','visual_strategy_session'])\
        ['response'].count().unstack().reset_index()
    counts['remove'] = [(row[False]<min_events)or(row[True]<min_events) for \
        index, row in counts.iterrows()]
    counts['running_bin'] = counts['running_bins'].astype(int)
    if bootstraps is not None:
        bootstraps = pd.merge(bootstraps, counts[['running_bin','remove']],\
            on='running_bin')

    if strategy == 'visual_strategy_session':
        hit_color = 'darkorange'
        mis_color = 'darkorange'
    else:
        hit_color = 'blue'
        mis_color = 'blue'
    hit_label = 'pre hit'
    mis_label = 'pre miss'
    if bootstraps is not None:
        temp1 = df1.set_index('running_bins')
        temp2 = df2.set_index('running_bins')
        bootstraps = bootstraps.set_index('running_bin')
        bins = set(bootstraps.index.values).intersection(set(temp1.index.values))
        for b in bins:
            plt.errorbar(b*bin_width, temp1.loc[b].response,
                yerr=bootstraps.loc[b]['pre_hit_sem'],color=hit_color,fmt='o')   
        bins = set(bootstraps.index.values).intersection(set(temp2.index.values))
        for b in bins:
            plt.errorbar(b*bin_width, temp2.loc[b].response,
                yerr=bootstraps.loc[b]['pre_miss_sem'],color=mis_color,fmt='x')     
        plt.plot(df1.running_bins*bin_width, df1.response,'o',
            color=hit_color,label=hit_label)
        plt.plot(df2.running_bins*bin_width, df2.response,'x',
            color=mis_color,label=mis_label)
    else:
        plt.errorbar(df1.running_bins*bin_width, df1.response,
            yerr=df1_sem.response,color=hit_color,fmt='o',label=hit_label)
        plt.errorbar(df2.running_bins*bin_width, df2.response,
            yerr=df2_sem.response,color=mis_color,fmt='x',label=mis_label)
    ax.set_ylabel(cre.capitalize()+' '+condition+'\n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.set_xlabel('running speed (cm/s)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12) 

    ax.set_ylim(0,.04)

    if (bootstraps is not None) and ('bh_significant' in bootstraps.columns):
        y =  ax.get_ylim()[1]*1.05
        bootstraps = bootstraps.reset_index()
        for index, row in bootstraps.iterrows():
            if row.bh_significant:
                if row.remove:
                    print('Not significant b/c of low count: {}'.format(row.running_bin))
                else:
                    ax.plot(row.running_bin*bin_width, y, 'k*')  
        ax.set_ylim(top=y*1.075)

    ax.set_xlim(-1,61)
    plt.legend()

    plt.tight_layout() 

    # Save fig
    if savefig:
        filename = PSTH_DIR + data+'/running/'+\
            'pre_change_running_{}_familiar_{}.svg'.format(cre,strategy)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename)



def engagement_running_responses(df, condition, cre='vip', vis_boots=None, 
    tim_boots=None, savefig=False, data='events', split='visual_strategy_session',
    plot_list=['visual','timing'],first=False,second=False,meso=False):

    if condition =='omission':
        bin_width=5        
    elif condition =='image':
        bin_width=5

    fig, ax = plt.subplots(figsize=(3.75,2.75))

    df['running_bins'] = np.floor(df['running_speed']/bin_width)

    summary = df.groupby(['visual_strategy_session','engagement_v2','running_bins'])\
        ['response'].mean().reset_index()
    evisual = summary.query('(visual_strategy_session) &(engagement_v2)')
    dvisual = summary.query('(visual_strategy_session) &(not engagement_v2)')
    etiming = summary.query('(not visual_strategy_session) &(engagement_v2)')
    dtiming = summary.query('(not visual_strategy_session) &(not engagement_v2)')

    # If we don't have bootstrap, use sem of responses
    summary_sem = df.groupby(['visual_strategy_session','engagement_v2','running_bins'])\
        ['response'].sem().reset_index()
    evisual_sem = summary_sem.query('(visual_strategy_session) &(engagement_v2)')
    dvisual_sem = summary_sem.query('(visual_strategy_session) &(not engagement_v2)')
    etiming_sem = summary_sem.query('(not visual_strategy_session) &(engagement_v2)')
    dtiming_sem = summary_sem.query('(not visual_strategy_session) &(not engagement_v2)')

    # Remove running bins with less than 100 responses
    counts = df.groupby(['running_bins','visual_strategy_session','engagement_v2'])\
        ['response'].count().unstack().reset_index()
    counts['remove'] = [(row[False]<100)or(row[True]<100) for \
        index, row in counts.iterrows()]
    counts['running_bin'] = counts['running_bins'].astype(int)
    vis_counts = counts.query('visual_strategy_session')
    tim_counts = counts.query('not visual_strategy_session')
    if vis_boots is not None:
        vis_boots = pd.merge(vis_boots, vis_counts[['running_bin','remove']],\
            on='running_bin')
    if tim_boots is not None:
        tim_boots = pd.merge(tim_boots, tim_counts[['running_bin','remove']],\
            on='running_bin')

    evis_color = 'darkorange'
    etim_color = 'blue'
    dvis_color = 'burlywood'
    dtim_color = 'lightblue'
    evis_label = 'engaged visual strategy'
    etim_label = 'engaged timing strategy'
    dvis_label = 'disengaged visual strategy'
    dtim_label = 'disengaged timing strategy'

    if 'visual' in plot_list:
        if vis_boots is not None:
            evis = evisual.set_index('running_bins')
            dvis = dvisual.set_index('running_bins')
            vis_boots = vis_boots.set_index('running_bin')
            bins = set(vis_boots.index.values).intersection(set(evis.index.values))
            for b in bins:
                plt.errorbar(b*bin_width, evis.loc[b].response,
                    yerr=vis_boots.loc[b]['engaged_sem'],color=evis_color,fmt='o')   
                plt.errorbar(b*bin_width, dvis.loc[b].response,
                    yerr=vis_boots.loc[b]['disengaged_sem'],color=dvis_color,fmt='o')   
        else:
            plt.errorbar(evisual.running_bins*bin_width, evisual.response,
                yerr=evisual_sem.response,color=evis_color,fmt='o',label=evis_label)
            plt.errorbar(dvisual.running_bins*bin_width, dvisual.response,
                yerr=dvisual_sem.response,color=dvis_color,fmt='o',label=dvis_label)
    if 'timing' in plot_list:
        if tim_boots is not None:
            etim = etiming.set_index('running_bins')
            dtim = dtiming.set_index('running_bins')
            tim_boots = tim_boots.set_index('running_bin')
            bins = set(tim_boots.index.values).intersection(set(etim.index.values))
            for b in bins:
                plt.errorbar(b*bin_width, etim.loc[b].response,
                    yerr=tim_boots.loc[b]['engaged_sem'],color=etim_color,fmt='o')   
                plt.errorbar(b*bin_width, dtim.loc[b].response,
                    yerr=tim_boots.loc[b]['disengaged_sem'],color=dtim_color,fmt='o')   
        else:
            plt.errorbar(etiming.running_bins*bin_width, etiming.response,
                yerr=etiming_sem.response,color=etim_color,fmt='o',label=etim_label)
            plt.errorbar(dtiming.running_bins*bin_width, dtiming.response,
                yerr=dtiming_sem.response,color=dtim_color,fmt='o',label=dtim_label)

    ax.set_ylabel(cre.capitalize()+' '+condition+'\n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.set_xlabel('running speed (cm/s)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12) 

    if (cre == 'vip') and (condition =='omission'):
        ax.set_ylim(0,.1) 
    elif (cre == 'vip') and (condition =='image'):
        ax.set_ylim(0,.027)
    else:
        ax.set_ylim(bottom=0)

    y =  ax.get_ylim()[1]*1.05

    if (vis_boots is not None) & ('visual' in plot_list):
        vis_boots=vis_boots.reset_index()
        for index, row in vis_boots.iterrows():     
            if row.bh_significant:
                if row.remove:
                    print('not significant b/c of low count: {}'.format(row.running_bin))
                else:
                    ax.plot(row.running_bin*bin_width, y, '*',color='darkorange')  

    if (tim_boots is not None) & ('timing' in plot_list):
        tim_boots=tim_boots.reset_index()
        for index, row in tim_boots.iterrows():     
            if row.bh_significant:
                if row.remove:
                    print('not significant b/c of low count: {}'.format(row.running_bin))
                else:
                    ax.plot(row.running_bin*bin_width, y*1.05, 'b*')  
    ax.set_ylim(top=y*1.1)   

    ax.set_xlim(-1,61)
    #plt.legend()
    plt.tight_layout() 

    # Save fig
    if savefig:
        plot_str = '_'.join(plot_list)
        extra =''
        if meso:
            extra += '_meso'
        if first:
            extra += '_first'
        if second:
            extra += '_second'
        filename = PSTH_DIR + data+'/running/'+\
            'engagement_running_{}{}_familiar_{}_{}_{}.svg'.\
            format(cre,extra,condition,split,plot_str)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 

def compute_summary_bootstrap_image_strategy(df,data='events',nboots=10000,cell_type='exc',
    first=True,second=False,post=False,meso=False):

    df['group'] = df['visual_strategy_session'].astype(str)
    mapper = {
        'True':'visual',
        'False':'timing',
    }
    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_image_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if post:
        filepath += '_post'
    if meso:
        filepath += '_meso'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 

def get_summary_bootstrap_image_strategy(data='events',nboots=10000,cell_type='exc',
    first=True,second=False,post=False,meso=False):

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type\
        +'_image_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if post:
        filepath += '_post'
    if meso:
        filepath += '_meso'
    filepath = filepath+'.feather'

    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')

def plot_summary_bootstrap_image_strategy(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False,post=False,meso=False):
    
    bootstrap = get_summary_bootstrap_image_strategy(data, nboots,cell_type,
        first,second,post,meso)   
 
    fig,ax = plt.subplots(figsize=(2.5,2.75))
    visual_mean = df.query('visual_strategy_session')['response'].mean()
    timing_mean = df.query('not visual_strategy_session')['response'].mean()
    means={}
    means['visual'] = visual_mean
    means['timing'] = timing_mean
    visual_sem = np.std(bootstrap['visual'])
    timing_sem = np.std(bootstrap['timing'])
    ax.plot(0, visual_mean,'o',color='darkorange')
    ax.plot(1,timing_mean,'o',color='blue')
    ax.plot([0,0],[visual_mean-visual_sem,visual_mean+visual_sem],'-',color='darkorange')
    ax.plot([1,1],[timing_mean-timing_sem,timing_mean+timing_sem],'-',color='blue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual','timing')
    if (p < 0.05) or(p>.95):
        ylim = ax.get_ylim()[1]
        plt.plot([0,1],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([0,0],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([1,1],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0.5,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    plt.tight_layout()   

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_image_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        if post:
            filepath += '_post'
        if meso:
            filepath += '_meso'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)
    print_bootstrap_summary(means,bootstrap,p)


def compute_summary_bootstrap_omission_strategy(df,data='events',nboots=10000,cell_type='exc',
    first=True,second=False,image=False,post=False,meso=False):

    df['group'] = df['visual_strategy_session'].astype(str)
    mapper = {
        'True':'visual',
        'False':'timing',
    }
    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_omission_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if post:
        filepath += '_post'
    if meso:
        filepath += '_meso'
    if image:
        filepath += '_image'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 

def get_summary_bootstrap_omission_strategy(data='events',nboots=10000,cell_type='exc',
    first=True,second=False,post=False,meso=False,image=False):

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type\
        +'_omission_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if post:
        filepath += '_post'
    if meso:
        filepath += '_meso'
    if image:
        filepath += '_image'
    filepath = filepath+'.feather'

    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')

def print_bootstrap_summary(means,bootstrap, p,keys=['visual','timing']):
    sems={}
    for key in keys:
        sems[key] = np.std(bootstrap[key])
        print('{}: {:.6f} +/- {:.6f}'.format(key,means[key],sems[key]))
    print('p: {}'.format(p))

def plot_summary_bootstrap_omission_strategy(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False,post=False,meso=False,image=False):
    
    bootstrap = get_summary_bootstrap_omission_strategy(data, nboots,cell_type,
        first,second,post,meso,image)   
 
    fig,ax = plt.subplots(figsize=(2.5,2.75))
    visual_mean = df.query('visual_strategy_session')['response'].mean()
    timing_mean = df.query('not visual_strategy_session')['response'].mean()
    means={}
    means['visual'] = visual_mean
    means['timing'] = timing_mean
    visual_sem = np.std(bootstrap['visual'])
    timing_sem = np.std(bootstrap['timing'])
    ax.plot(0, visual_mean,'o',color='darkorange')
    ax.plot(1,timing_mean,'o',color='blue')
    ax.plot([0,0],[visual_mean-visual_sem,visual_mean+visual_sem],'-',color='darkorange')
    ax.plot([1,1],[timing_mean-timing_sem,timing_mean+timing_sem],'-',color='blue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual','timing')
    if (p < 0.05) or(p>.95):
        ylim = ax.get_ylim()[1]
        plt.plot([0,1],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([0,0],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([1,1],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0.5,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    plt.tight_layout()   

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_omission_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        if post:
            filepath += '_post'
        if meso:
            filepath += '_meso'
        if image:
            filepath += '_image'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)
    print_bootstrap_summary(means,bootstrap,p)

def compute_summary_bootstrap_strategy_omission_lick(df,data='events',nboots=10000,
    cell_type='exc',first=False,second=True):

    df = df.query('(pre_omission_no_lick)or(pre_omission_lick)').copy()
    df['group'] = df['visual_strategy_session'].astype(str)+df['pre_omission_lick'].astype(str)
    mapper = {
        'TrueTrue':'visual_lick',
        'TrueFalse':'visual_no_lick',
        'FalseTrue':'timing_lick',
        'FalseFalse':'timing_no_lick',
    }
    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_omission_lick_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 


def get_summary_bootstrap_strategy_omission_lick(data='events',nboots=10000,cell_type='exc',
    first=True,second=False):
    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_omission_lick_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    filepath = filepath+'.feather'
    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')

def plot_summary_bootstrap_strategy_omission_lick(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False):
   
    df = df.query('pre_omission_lick or pre_omission_no_lick').copy()
    bootstrap = get_summary_bootstrap_strategy_omission_lick(data, nboots,cell_type,
        first,second)   
 
    fig,ax = plt.subplots(figsize=(2.5,2.75))
    visual_lick_mean = df.query('(visual_strategy_session)&(pre_omission_lick)')['response'].mean()
    visual_no_lick_mean = df.query('(visual_strategy_session)&(pre_omission_no_lick)')['response'].mean()
    timing_lick_mean = df.query('(not visual_strategy_session)&(pre_omission_lick)')['response'].mean()
    timing_no_lick_mean = df.query('(not visual_strategy_session)&(pre_omission_no_lick)')['response'].mean()
    visual_lick_sem = np.std(bootstrap['visual_lick'])
    timing_lick_sem = np.std(bootstrap['timing_lick'])
    visual_no_lick_sem = np.std(bootstrap['visual_no_lick'])
    timing_no_lick_sem = np.std(bootstrap['timing_no_lick'])

    plt.plot(-.05,visual_lick_mean,'o',color='darkorange',label='visual lick')
    plt.plot(0.05,visual_no_lick_mean,'x',color='darkorange',label='visual no_lick')
    plt.plot(.95, timing_lick_mean,'o',color='blue',label='timing lick')
    plt.plot(1.05,timing_no_lick_mean,'x',color='blue',label='timing mis')
    plt.plot([-.05,-0.05],[visual_lick_mean-visual_lick_sem,visual_lick_mean+visual_lick_sem],
        '-',color='darkorange')
    plt.plot([.05,0.05],[visual_no_lick_mean-visual_no_lick_sem,visual_no_lick_mean+visual_no_lick_sem],
        '-',color='darkorange')
    plt.plot([.95,.95],[timing_lick_mean-timing_lick_sem,timing_lick_mean+timing_lick_sem],
        '-',color='blue')
    plt.plot([1.05,1.05],[timing_no_lick_mean-timing_no_lick_sem,timing_no_lick_mean+timing_no_lick_sem],
        '-',color='blue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual_lick','timing_lick')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([-.05,.95],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([-.05,-.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([.95,.95],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0.5,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'visual_no_lick','timing_no_lick')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([.05,1.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([.05,.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([1.05,1.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0.5,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'visual_lick','visual_no_lick')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([-.05,.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([-.05,-.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([.05,.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'timing_lick','timing_no_lick')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([.95,1.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([.95,.95],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([1.05,1.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(1,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)



    plt.tight_layout()   

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_omission_lick_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)





def compute_summary_bootstrap_strategy_pre_change(df,data='events',nboots=10000,cell_type='exc',
    first=True,second=False,meso=False):

    df = df.query('(pre_hit_1 ==1)or(pre_miss_1==1)').copy()
    df['group'] = df['visual_strategy_session'].astype(str)+df['pre_hit_1'].astype(str)
    mapper = {
        'True1.0':'visual_hit',
        'True0.0':'visual_miss',
        'False1.0':'timing_hit',
        'False0.0':'timing_miss',
    }
    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_pre_change_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if meso:
        filepath += '_meso'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 

def get_summary_bootstrap_strategy_pre_change(data='events',nboots=10000,cell_type='exc',
    first=True,second=False,meso=False):
    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_pre_change_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if meso:
        filepath += '_meso'
    filepath = filepath+'.feather'
    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')
  
def plot_summary_bootstrap_strategy_pre_change(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False,meso=False):
   
    df = df.query('(pre_hit_1 ==1)or(pre_miss_1==1)').copy()
    bootstrap = get_summary_bootstrap_strategy_pre_change(data, nboots,cell_type,
        first,second)   
 
    fig,ax = plt.subplots(figsize=(2.5,2.75))
    visual_hit_mean = df.query('(visual_strategy_session)&(pre_hit_1==1)')['response'].mean()
    visual_miss_mean = df.query('(visual_strategy_session)&(pre_hit_1==0)')['response'].mean()
    timing_hit_mean = df.query('(not visual_strategy_session)&(pre_hit_1==1)')['response'].mean()
    timing_miss_mean = df.query('(not visual_strategy_session)&(pre_hit_1==0)')['response'].mean()
    visual_hit_sem = np.std(bootstrap['visual_hit'])
    timing_hit_sem = np.std(bootstrap['timing_hit'])
    visual_miss_sem = np.std(bootstrap['visual_miss'])
    timing_miss_sem = np.std(bootstrap['timing_miss'])
    means = {}
    means['visual_hit'] = visual_hit_mean
    means['visual_miss'] = visual_miss_mean
    means['timing_hit'] = timing_hit_mean
    means['timing_miss'] = timing_miss_mean
    dx = .15
    x1 = -dx
    x2 = dx
    x3 = 1-dx
    x4 = 1+dx
    plt.plot(x1,visual_hit_mean,'o',color='darkorange',label='visual hit')
    plt.plot(x2,visual_miss_mean,'x',color='darkorange',label='visual miss')
    plt.plot(x3, timing_hit_mean,'o',color='blue',label='timing hit')
    plt.plot(x4,timing_miss_mean,'x',color='blue',label='timing mis')
    plt.plot([x1,x1],[visual_hit_mean-visual_hit_sem,visual_hit_mean+visual_hit_sem],
        '-',color='darkorange')
    plt.plot([x2,x2],[visual_miss_mean-visual_miss_sem,visual_miss_mean+visual_miss_sem],
        '-',color='darkorange')
    plt.plot([x3,x3],[timing_hit_mean-timing_hit_sem,timing_hit_mean+timing_hit_sem],
        '-',color='blue')
    plt.plot([x4,x4],[timing_miss_mean-timing_miss_sem,timing_miss_mean+timing_miss_sem],
        '-',color='blue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual_hit','timing_hit')

    ylim = ax.get_ylim()[1]
    plt.plot([x1,x3],[ylim*1.1,ylim*1.1],'k-')
    plt.plot([x1,x1],[ylim*1.05,ylim*1.1],'k-')
    plt.plot([x3,x3],[ylim*1.05,ylim*1.1],'k-')
    if (p < 0.05) or (p >.95):
        plt.plot(np.mean([x1,x3]),ylim*1.15, 'k*')

    p = bootstrap_significance(bootstrap, 'visual_miss','timing_miss')
    if (p < 0.05) or (p >.95):
        plt.plot(np.mean([x2,x4]),ylim*1.25, 'k*')
    plt.plot([x2,x4],[ylim*1.2,ylim*1.2],'k-')
    plt.plot([x2,x2],[ylim*1.15,ylim*1.2],'k-')
    plt.plot([x4,x4],[ylim*1.15,ylim*1.2],'k-')
    ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'visual_hit','visual_miss')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([x1,x2],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([x1,x1],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([x2,x2],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    plt.tight_layout()   

    phh = bootstrap_significance(bootstrap, 'visual_hit','timing_hit')
    pmm = bootstrap_significance(bootstrap, 'visual_miss','timing_miss')
    pvhm = bootstrap_significance(bootstrap, 'visual_hit','visual_miss')
    pthm = bootstrap_significance(bootstrap, 'timing_hit','timing_miss')
    print_bootstrap_summary(means, bootstrap, phh,keys=['visual_hit','timing_hit'])
    print_bootstrap_summary(means, bootstrap, pmm,keys=['visual_miss','timing_miss'])
    print_bootstrap_summary(means, bootstrap, pvhm,keys=['visual_hit','visual_miss'])
    print_bootstrap_summary(means, bootstrap, pthm,keys=['timing_hit','timing_miss'])

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_pre_change_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        if meso:
            filepath += '_meso'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)


 

def compute_summary_bootstrap_strategy_hit(df,data='events',nboots=10000,cell_type='exc',
    first=True,second=False,image=False,meso=False):

    df['group'] = df['visual_strategy_session'].astype(str)+df['hit'].astype(str)
    mapper = {
        'True1.0':'visual_hit',
        'True0.0':'visual_miss',
        'False1.0':'timing_hit',
        'False0.0':'timing_miss',
    }
    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_hit_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if image:
        filepath += '_image_period'
    if meso:
        filepath += '_meso'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 

def get_summary_bootstrap_strategy_hit(data='events',nboots=10000,cell_type='exc',
    first=True,second=False,image=False,meso=False):
    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_hit_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if image:
        filepath += '_image_period'
    if meso:
        filepath += '_meso'

    filepath = filepath+'.feather'
    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')
   
def plot_summary_bootstrap_strategy_hit(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False,image=False,meso=False):
    
    bootstrap = get_summary_bootstrap_strategy_hit(data, nboots,cell_type,
        first,second,image,meso)   
 
    fig,ax = plt.subplots(figsize=(2.5,2.75))
    visual_hit_mean = df.query('(visual_strategy_session)&(hit==1)')['response'].mean()
    visual_miss_mean = df.query('(visual_strategy_session)&(hit==0)')['response'].mean()
    timing_hit_mean = df.query('(not visual_strategy_session)&(hit==1)')['response'].mean()
    timing_miss_mean = df.query('(not visual_strategy_session)&(hit==0)')['response'].mean()
    visual_hit_sem = np.std(bootstrap['visual_hit'])
    timing_hit_sem = np.std(bootstrap['timing_hit'])
    visual_miss_sem = np.std(bootstrap['visual_miss'])
    timing_miss_sem = np.std(bootstrap['timing_miss'])
    means = {}
    means['visual_hit'] = visual_hit_mean
    means['visual_miss'] = visual_miss_mean
    means['timing_hit'] = timing_hit_mean
    means['timing_miss'] = timing_miss_mean 

    dx = .15
    x1 = -dx
    x2 = dx
    x3 = 1-dx
    x4 = 1+dx
    plt.plot(x1,visual_hit_mean,'o',color='darkorange',label='visual hit')
    plt.plot(x2,visual_miss_mean,'x',color='darkorange',label='visual miss')
    plt.plot(x3, timing_hit_mean,'o',color='blue',label='timing hit')
    plt.plot(x4,timing_miss_mean,'x',color='blue',label='timing mis')
    plt.plot([x1,x1],[visual_hit_mean-visual_hit_sem,visual_hit_mean+visual_hit_sem],
        '-',color='darkorange')
    plt.plot([x2,x2],[visual_miss_mean-visual_miss_sem,visual_miss_mean+visual_miss_sem],
        '-',color='darkorange')
    plt.plot([x3,x3],[timing_hit_mean-timing_hit_sem,timing_hit_mean+timing_hit_sem],
        '-',color='blue')
    plt.plot([x4,x4],[timing_miss_mean-timing_miss_sem,timing_miss_mean+timing_miss_sem],
        '-',color='blue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual_hit','timing_hit')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([x1,x3],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([x1,x1],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([x3,x3],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(np.mean([x1,x3]),ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.15)

    p = bootstrap_significance(bootstrap, 'visual_hit','visual_miss')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([x1,x2],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([x1,x1],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([x2,x2],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    plt.tight_layout()   
    phh = bootstrap_significance(bootstrap, 'visual_hit','timing_hit')
    pmm = bootstrap_significance(bootstrap, 'visual_miss','timing_miss')
    pvhm = bootstrap_significance(bootstrap, 'visual_hit','visual_miss')
    pthm = bootstrap_significance(bootstrap, 'timing_hit','timing_miss')
    print_bootstrap_summary(means, bootstrap, phh,keys=['visual_hit','timing_hit'])
    print_bootstrap_summary(means, bootstrap, pmm,keys=['visual_miss','timing_miss'])
    print_bootstrap_summary(means, bootstrap, pvhm,keys=['visual_hit','visual_miss'])
    print_bootstrap_summary(means, bootstrap, pthm,keys=['timing_hit','timing_miss'])

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_hit_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        if image:
            filepath += '_image_period'
        if meso:
            filepath += '_meso'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)


def compute_summary_bootstrap_strategy_engaged_miss(df,data='events',nboots=10000,
    cell_type='exc',first=True,second=False,meso=False,image=False):

    df = df.query('miss==1').copy()
    df['group'] = df['visual_strategy_session'].astype(str)\
        +df['engagement_v2'].astype(str)
    mapper = {
        'TrueTrue':'visual_engaged',
        'TrueFalse':'visual_disengaged',
        'FalseTrue':'timing_engaged',
        'FalseFalse':'timing_disengaged',
    }
    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, 
        levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type\
        +'_engaged_miss_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if meso:
        filepath += '_meso'
    if image:
        filepath += '_image'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 


def get_summary_bootstrap_strategy_engaged_miss(data='events',nboots=10000,
    cell_type='exc',first=True,second=False,image=False,meso=False):

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+'_engaged_miss_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if meso:
        filepath += '_meso'
    if image:
        filepath += '_image'
    filepath = filepath+'.feather'
    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')
   
def plot_summary_bootstrap_strategy_engaged_miss(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False,image=False,meso=False):
    
    bootstrap = get_summary_bootstrap_strategy_engaged_miss(data, nboots,cell_type,
        first,second,image=image,meso=meso)   
 
    fig,ax = plt.subplots(figsize=(3,2.75))
    df = df.query('miss==1').copy()
    visual_engaged_mean = df.query('(visual_strategy_session)&(engagement_v2)')['response'].mean()
    visual_disengaged_mean = df.query('(visual_strategy_session)&(not engagement_v2)')['response'].mean()
    timing_engaged_mean = df.query('(not visual_strategy_session)&(engagement_v2)')['response'].mean()
    timing_disengaged_mean = df.query('(not visual_strategy_session)&(not engagement_v2)')['response'].mean()
    visual_engaged_sem = np.std(bootstrap['visual_engaged'])
    timing_engaged_sem = np.std(bootstrap['timing_engaged'])
    visual_disengaged_sem = np.std(bootstrap['visual_disengaged'])
    timing_disengaged_sem = np.std(bootstrap['timing_disengaged'])
    means = {}
    means['visual_engaged'] = visual_engaged_mean
    means['visual_disengaged'] = visual_disengaged_mean
    means['timing_engaged'] = timing_engaged_mean
    means['timing_disengaged'] = timing_disengaged_mean 

    plt.plot(-.05,visual_engaged_mean,'x',color='darkorange',label='visual engaged')
    plt.plot(0.05,visual_disengaged_mean,'x',color='burlywood',label='visual disengaged')
    plt.plot(.95, timing_engaged_mean,'x',color='blue',label='timing engaged')
    plt.plot(1.05,timing_disengaged_mean,'x',color='lightblue',label='timing disengaged')
    plt.plot([-.05,-0.05],[visual_engaged_mean-visual_engaged_sem,
        visual_engaged_mean+visual_engaged_sem],
        '-',color='darkorange')
    plt.plot([.05,0.05],[visual_disengaged_mean-visual_disengaged_sem,
        visual_disengaged_mean+visual_disengaged_sem],
        '-',color='burlywood')
    plt.plot([.95,.95],[timing_engaged_mean-timing_engaged_sem,
        timing_engaged_mean+timing_engaged_sem],
        '-',color='blue')
    plt.plot([1.05,1.05],[timing_disengaged_mean-timing_disengaged_sem,
        timing_disengaged_mean+timing_disengaged_sem],
        '-',color='lightblue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual_engaged','timing_engaged')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([-.05,.95],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([-.05,-.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([.95,.95],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0.5,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'visual_engaged','visual_disengaged')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([-.05,.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([-.05,-.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([.05,.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'timing_engaged','timing_disengaged')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([.95,1.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([.95,.95],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([1.05,1.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(1,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    plt.tight_layout()   

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_engaged_miss_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        if image:
            filepath += '_image'
        if meso:
            filepath += '_meso'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)
    pee = bootstrap_significance(bootstrap, 'visual_engaged','timing_engaged')
    pdd = bootstrap_significance(bootstrap, 'visual_disengaged','timing_disengaged')
    pved = bootstrap_significance(bootstrap, 'visual_engaged','visual_disengaged')
    pted = bootstrap_significance(bootstrap, 'timing_engaged','timing_disengaged')
    print_bootstrap_summary(means, bootstrap, pee,keys=['visual_engaged','timing_engaged'])
    print_bootstrap_summary(means, bootstrap, pdd,keys=['visual_disengaged','timing_disengaged'])
    print_bootstrap_summary(means, bootstrap, pved,keys=['visual_engaged','visual_disengaged'])
    print_bootstrap_summary(means, bootstrap, pted,keys=['timing_engaged','timing_disengaged'])



def compute_summary_bootstrap_strategy_engaged_omission(df,data='events',nboots=10000,
    cell_type='exc',first=True,second=False,image=False,post=False,meso=False):

    df['group'] = df['visual_strategy_session'].astype(str)\
        +df['engagement_v2'].astype(str)
    mapper = {
        'TrueTrue':'visual_engaged',
        'TrueFalse':'visual_disengaged',
        'FalseTrue':'timing_engaged',
        'FalseFalse':'timing_disengaged',
    }

    df['group'] = [mapper[x] for x in df['group']]
    bootstrap = hb.bootstrap(df, levels=['group','ophys_experiment_id','cell_specimen_id'],
        nboots=nboots)

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type+\
        '_engaged_omission_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if post:
        filepath += '_post'
    if meso:
        filepath += '_meso'
    if image:
        filepath += '_image'
    filepath = filepath+'.feather'

    with open(filepath,'wb') as handle:
        pickle.dump(bootstrap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('bootstrap saved to {}'.format(filepath)) 

def get_summary_bootstrap_strategy_engaged_omission(data='events',nboots=10000,cell_type='exc',
    first=True,second=False,post=False,meso=False,image=False):

    filepath = PSTH_DIR + data +'/bootstraps/'+cell_type\
        +'_engaged_omission_strategy_summary_'+str(nboots)
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if post:
        filepath += '_post'
    if meso:
        filepath += '_meso'
    if image:
        filepath += '_image'
    filepath = filepath+'.feather'

    if os.path.isfile(filepath):
        # Load this bin
        with open(filepath,'rb') as handle:
            this_boot = pickle.load(handle)
        print('loading from file')
        return this_boot
    else:
        print('file not found')

def plot_summary_bootstrap_strategy_engaged_omission(df,cell_type,savefig=False,data='events',
    nboots=10000,first=True, second=False,post=False,meso=False,image=False):
   
    bootstrap = get_summary_bootstrap_strategy_engaged_omission(data, nboots,cell_type,
        first,second,post,meso,image)   

    fig,ax = plt.subplots(figsize=(3,2.75))
    visual_engaged_mean = df.query('(visual_strategy_session)&(engagement_v2)')['response'].mean()
    visual_disengaged_mean = df.query('(visual_strategy_session)&(not engagement_v2)')['response'].mean()
    timing_engaged_mean = df.query('(not visual_strategy_session)&(engagement_v2)')['response'].mean()
    timing_disengaged_mean = df.query('(not visual_strategy_session)&(not engagement_v2)')['response'].mean()
    visual_engaged_sem = np.std(bootstrap['visual_engaged'])
    timing_engaged_sem = np.std(bootstrap['timing_engaged'])
    visual_disengaged_sem = np.std(bootstrap['visual_disengaged'])
    timing_disengaged_sem = np.std(bootstrap['timing_disengaged'])
    means = {}
    means['visual_engaged'] = visual_engaged_mean
    means['visual_disengaged'] = visual_disengaged_mean
    means['timing_engaged'] = timing_engaged_mean
    means['timing_disengaged'] = timing_disengaged_mean 

    plt.plot(-.05,visual_engaged_mean,'x',color='darkorange',label='visual engaged')
    plt.plot(0.05,visual_disengaged_mean,'x',color='burlywood',label='visual disengaged')
    plt.plot(.95, timing_engaged_mean,'x',color='blue',label='timing engaged')
    plt.plot(1.05,timing_disengaged_mean,'x',color='lightblue',label='timing disengaged')
    plt.plot([-.05,-0.05],[visual_engaged_mean-visual_engaged_sem,
        visual_engaged_mean+visual_engaged_sem],
        '-',color='darkorange')
    plt.plot([.05,0.05],[visual_disengaged_mean-visual_disengaged_sem,
        visual_disengaged_mean+visual_disengaged_sem],
        '-',color='burlywood')
    plt.plot([.95,.95],[timing_engaged_mean-timing_engaged_sem,
        timing_engaged_mean+timing_engaged_sem],
        '-',color='blue')
    plt.plot([1.05,1.05],[timing_disengaged_mean-timing_disengaged_sem,
        timing_disengaged_mean+timing_disengaged_sem],
        '-',color='lightblue')

    mapper={
        'exc':'Excitatory',
        'sst':'Sst Inhibitory',
        'vip':'Vip Inhibitory'
        }
    nice_cell =mapper[cell_type] 

    ax.set_ylabel(nice_cell+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(bottom=0)
    
    p = bootstrap_significance(bootstrap, 'visual_engaged','timing_engaged')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([-.05,.95],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([-.05,-.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([.95,.95],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0.5,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'visual_engaged','visual_disengaged')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([-.05,.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([-.05,-.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([.05,.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(0,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    p = bootstrap_significance(bootstrap, 'timing_engaged','timing_disengaged')
    if (p < 0.05) or (p >.95):
        ylim = ax.get_ylim()[1]
        plt.plot([.95,1.05],[ylim*1.1,ylim*1.1],'k-')
        plt.plot([.95,.95],[ylim*1.05,ylim*1.1],'k-')
        plt.plot([1.05,1.05],[ylim*1.05,ylim*1.1],'k-')
        plt.plot(1,ylim*1.15, 'k*')
        ax.set_ylim(top=ylim*1.2)

    plt.tight_layout()   

    if savefig:
        filepath = PSTH_DIR + data +'/summary/'+cell_type+'_engaged_omission_strategy_summary_'+str(nboots)
        if first:
            filepath += '_first'
        if second:
            filepath += '_second'
        if post:
            filepath += '_post'
        if meso:
            filepath += '_meso'
        if image:
            filepath += '_image'
        filepath = filepath+'.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath)

    pee = bootstrap_significance(bootstrap, 'visual_engaged','timing_engaged')
    pdd = bootstrap_significance(bootstrap, 'visual_disengaged','timing_disengaged')
    pved = bootstrap_significance(bootstrap, 'visual_engaged','visual_disengaged')
    pted = bootstrap_significance(bootstrap, 'timing_engaged','timing_disengaged')
    print_bootstrap_summary(means, bootstrap, pee,keys=['visual_engaged','timing_engaged'])
    print_bootstrap_summary(means, bootstrap, pdd,keys=['visual_disengaged','timing_disengaged'])
    print_bootstrap_summary(means, bootstrap, pved,keys=['visual_engaged','visual_disengaged'])
    print_bootstrap_summary(means, bootstrap, pted,keys=['timing_engaged','timing_disengaged'])


 
def bootstrap_significance(bootstrap, k1, k2):
    diff = np.array(bootstrap[k1]) - np.array(bootstrap[k2])
    p = np.sum(diff >= 0)/len(diff)
    if p > 0.5:
        p = 1-p
    return p


def load_df_and_compute_running(summary_df, cell_type, response, data, nboots, 
    bin_num,meso=False,first=False, second=False):

    mapper = {
        'exc':'Slc17a7-IRES2-Cre',
        'sst':'Sst-IRES-Cre',
        'vip':'Vip-IRES-Cre'
        }
    if response == 'image':
        df = load_image_df(summary_df, mapper[cell_type], data,
            meso=meso,first=first, second=second)
    elif response == 'omission':
        df = load_omission_df(summary_df, mapper[cell_type], data,
            meso=meso,first=first,second=second)
    compute_running_bootstrap_bin(df,response, cell_type, bin_num, nboots=nboots,
            meso=meso,first=first,second=second)

def load_df_and_compute_engagement_running(summary_df, cell_type, response, data, 
    nboots, bin_num,meso=False,first=False,second=False):

    mapper = {
        'exc':'Slc17a7-IRES2-Cre',
        'sst':'Sst-IRES-Cre',
        'vip':'Vip-IRES-Cre'
        }
    if response == 'image':
        df = load_image_df(summary_df, mapper[cell_type], data,
            meso=meso,first=first,second=second)
    elif response == 'omission':
        df = load_omission_df(summary_df, mapper[cell_type], data,
            meso=meso,first=first,second=second)
    compute_engagement_running_bootstrap_bin(df,response, cell_type, 'visual',bin_num, 
        nboots=nboots,meso=meso,first=first,second=second)
    compute_engagement_running_bootstrap_bin(df,response, cell_type, 'timing',bin_num, 
        nboots=nboots,meso=meso,first=first,second=second)



def load_df_and_compute_hierarchy(summary_df, cell_type, response, data, depth, nboots, 
    splits, query='', extra='',first=False,second=False):
    mapper = {
        'exc':'Slc17a7-IRES2-Cre',
        'sst':'Sst-IRES-Cre',
        'vip':'Vip-IRES-Cre'
        }
    if response == 'image':
        print('loading image')
        df = load_image_df(summary_df, mapper[cell_type], data,first,second)
    elif response == 'omission':
        df = load_omission_df(summary_df, mapper[cell_type], data,first,second)
    elif response == 'change':
        df = load_change_df(summary_df, mapper[cell_type], data,first,second)

    # Append preferred image
    df = find_preferred(df)
    
    if query is not '':
        print('querying!') 
        df = df.query(query)

    hierarchy = compute_hierarchy(df, cell_type, response, data, depth, splits=splits, 
        nboots=nboots, extra=extra,first=first,second=second)


def compute_hierarchy(df, cell_type, response, data, depth, splits=[],bootstrap=True,
    extra='',nboots=10000,alpha=0.05,first=False,second=False):
    '''
    Generates a dataframe with the mean + bootstraps values for each split of the data
    saves dataframe to file for fast access
    '''
    # Get mean response in each area/depth/split
    mean_df = df.groupby(splits+['targeted_structure',depth])['response']\
        .mean().reset_index()
    sem_df = df.groupby(splits+['targeted_structure',depth])['response']\
        .sem().reset_index()
    mean_df['sem'] = sem_df['response']*1.96
    mean_df['location'] = mean_df['targeted_structure'] + '_' +\
         mean_df[depth].astype(str)
    if depth == 'layer':
        mapper = {
            'VISp_upper':1,
            'VISp_lower':2,
            'VISl_upper':3,
            'VISl_lower':4
            }
    elif depth == 'binned_depth':
        mapper = {
            'VISp_75':1,
            'VISp_175':2,
            'VISp_275':3,
            'VISp_375':4,
            'VISl_75':5,
            'VISl_175':6,
            'VISl_275':7,
            'VISl_375':8,
            }
    mean_df['xloc'] = [mapper[x] for x in mean_df['location']] 

    # Get number of cells and experiments in each area/depth/split
    counts = df.groupby(splits+['targeted_structure',depth])[['ophys_experiment_id',\
        'cell_specimen_id']].nunique().reset_index()
    mean_df = pd.merge(mean_df,counts, on=splits+['targeted_structure',depth],
        validate='1:1')
    mean_df = mean_df.rename(columns={
        'ophys_experiment_id':'num_oeid',
        'cell_specimen_id':'num_cells'
        })
    mean_df['bootstraps'] = [ [] for x in range(0,len(mean_df))] 

    # Add the bootstrap, can be slow
    if bootstrap:
        groups = mean_df[['targeted_structure',depth]].drop_duplicates()
        for index, row in groups.iterrows():
            row_depth = row[depth]
            temp = df.query(\
                '(targeted_structure == @row.targeted_structure)&({} == @row_depth)'\
                .format(depth))

            if len(splits) == 0:
                bootstrap = hb.bootstrap(temp,levels=splits+\
                    ['ophys_experiment_id','cell_specimen_id'],nboots=nboots,
                    no_top_level=True)
                sem = np.std(bootstrap)
                dex = (mean_df['targeted_structure'] == row.targeted_structure)&\
                    (mean_df[depth] == row[depth])
                mean_df.loc[dex,'bootstrap_sem'] = sem
                mean_df.loc[dex, 'n_boots'] = nboots
                index_dex = dex[dex].index.values[0]
                mean_df.at[index_dex,'bootstraps'] = bootstrap
            else:
                bootstrap = hb.bootstrap(temp,levels=splits+\
                    ['ophys_experiment_id','cell_specimen_id'],nboots=nboots)
                keys = list(bootstrap.keys()) 
                if len(keys) == 2:
                    diff = np.array(bootstrap[keys[0]]) > np.array(bootstrap[keys[1]])
                    p_boot = np.sum(diff)/len(diff)
                else:
                    p_boot = 0.5
                for key in keys:
                    sem = np.std(bootstrap[key])
                    dex = (mean_df['targeted_structure'] == row.targeted_structure)&\
                        (mean_df[depth] == row[depth])&\
                        (mean_df[splits[0]] == key)
                    mean_df.loc[dex,'bootstrap_sem'] = sem
                    mean_df.loc[dex,'p_boot'] = p_boot 
                    mean_df.loc[dex, 'n_boots'] = nboots
                    index_dex = dex[dex].index.values[0]
                    mean_df.at[index_dex,'bootstraps'] = bootstrap[key]
       
        # Determine significance
        if len(splits) > 0:
            mean_df['p_boot'] = [1-x if x > .5 else x for x in mean_df['p_boot']] 
            mean_df['ind_significant'] = mean_df['p_boot'] <= alpha
    
            # Benjamini Hochberg Correction 
            mean_df = add_hochberg_correction(mean_df)

    # Save file
    filepath = get_hierarchy_filename(cell_type,response,data,depth,nboots,splits,
        extra,first,second)
    print('saving bootstraps to: '+filepath)
    mean_df.to_feather(filepath)
    
    return mean_df

def add_hochberg_correction(table):
    '''
        Performs the Benjamini Hochberg correction
    '''    
    # Sort table by pvalues
    table = table.sort_values(by=['p_boot','location']).reset_index()
    table['test_number'] = (~table['location'].duplicated()).cumsum()   
 
    # compute the corrected pvalue based on the rank of each test
    table['imq'] = (table['test_number'])/table['test_number'].max()*0.05

    # Find the largest pvalue less than its corrected pvalue
    # all tests above that are significant
    table['bh_significant'] = False
    passing_tests = table[table['p_boot'] < table['imq']]
    if len(passing_tests) >0:
        last_index = table[table['p_boot'] < table['imq']].tail(1).index.values[0]
        table.at[last_index,'bh_significant'] = True
        table.at[0:last_index,'bh_significant'] = True
    
    # reset order of table and return
    table = table.sort_values(by='index').reset_index(drop=True)
    table = table.drop(columns='index')
    return table

def get_hierarchy_filename(cell_type, response, data, depth, nboots, splits, extra,
    first=False,second=False,meso=False):
    filepath = PSTH_DIR + data +'/bootstraps/' +\
        '_'.join([cell_type,response,depth,str(nboots)]+splits)
    if extra != '':
        filepath = filepath +'_'+extra
    if first:
        filepath += '_first'
    if second:
        filepath += '_second'
    if meso:
        filepath += '_meso'
    filepath = filepath+'.feather'
    return filepath

def get_hierarchy(cell_type, response, data, depth, nboots,splits=[],extra='',
    first=False,second=False):
    '''
        loads the dataframe from file
    '''
    filepath = get_hierarchy_filename(cell_type,response,data,depth,nboots,
        splits,extra,first,second)
    if os.path.isfile(filepath):
        hierarchy = pd.read_feather(filepath)
        return hierarchy
    else:
        print('file not found, compute the hierarchy first')

def compare_hierarchy(response,data,depth,splits):

    ax = get_and_plot('vip',response,data,depth,splits=splits)
    ax = get_and_plot('sst',response,data,depth,splits=splits,ax=ax)
    
    #ax2 = ax.twinx()
    ax = get_and_plot('exc',response,data,depth,splits=splits,ax=ax)


def get_and_plot(cell_type, response, data, depth, nboots=10000,splits=[], extra='', 
    savefig=False,ax=None,strategy =None,first=False,second=False):
    style = pstyle.get_style()
    colors={
        'omission':style['schematic_omission'],
        'image':'gray',
        'change':style['schematic_change']
        }
    if type(response) is list:
        hierarchies = []
    
        # plot each response type
        for r in response:
            if strategy is not None:
                hierarchy = get_hierarchy(cell_type, r, data, depth,nboots,
                    ['visual_strategy_session'], extra,first,second)
                if strategy == 'visual':
                    hierarchy=hierarchy.query('visual_strategy_session').copy()
                    label = 'visual ' + cell_type + ' ' + r
                else:
                    hierarchy=hierarchy.query('not visual_strategy_session').copy()
                    label = 'timing ' + cell_type + ' ' + r
            else:
                hierarchy = get_hierarchy(cell_type, r, data, depth,nboots, splits, 
                    extra,first,second)
                label = cell_type +' '+r
            ax = plot_hierarchy(hierarchy, cell_type, response, data, depth, [], 
                savefig=False,extra=extra,ax=ax,in_color=colors[r],in_label=label,
                polish=False,first=first,second=second)
            hierarchy['response_type'] = r
            hierarchies.append(hierarchy)
        mean_df = pd.concat(hierarchies)

        # do stats
        if len(response) == 2:
            location = mean_df['location'].unique()
            for l in location:
                temp =mean_df.query('location == @l') 
                num_groups = len(temp)
                if num_groups == 2:
                    diff = temp.iloc[0]['bootstraps'] > temp.iloc[1]['bootstraps']
                    p_boot = np.sum(diff)/len(diff)
                else:
                    p_boot = 0.5
                for r in response:
                    dex = (mean_df['location'] == l)&(mean_df['response_type']==r)
                    mean_df.loc[dex,'p_boot'] = p_boot 

            # determine significance with hochberg correction
            mean_df['p_boot'] = [1-x if x > .5 else x for x in mean_df['p_boot']] 
            mean_df['ind_significant'] = mean_df['p_boot'] <= 0.05 
            mean_df = add_hochberg_correction(mean_df)

            # Add significance stars to plot
            y =  ax.get_ylim()[1]*1.05
            for index, row in mean_df.iterrows():
                if row.bh_significant:
                    ax.plot(row.xloc, y, 'k*')  
            ax.set_ylim(top=y*1.075)

        plt.tight_layout()

    else:
        hierarchy = get_hierarchy(cell_type, response, data, depth,nboots, splits, 
            extra,first,second)
        ax = plot_hierarchy(hierarchy, cell_type, response, data, depth, splits, 
            savefig=savefig,extra=extra,ax=ax,first=first,second=second)
    return ax

def plot_hierarchy(hierarchy, cell_type, response, data, depth, splits, savefig=False,
    ylim=None,extra='',ax=None,in_color=None,in_label=None,polish=True,first=False,second=False):

    if ax is None:
        if depth == 'layer':
            fig,ax = plt.subplots(figsize=(5,4))
        else:
            fig,ax = plt.subplots(figsize=(8,4))

    if len(splits) >0:
        for index, value in enumerate(splits):
            unique = hierarchy[value].unique()
            for sdex, split_value in enumerate(unique):
                if value == 'hit':
                    if bool(split_value):
                        color = 'black'
                        label = 'hit'
                    else:
                        color = 'lightgray'
                        label = 'miss'
                elif value == 'visual_strategy_session':
                    if bool(split_value):
                        color = 'darkorange'
                        label = 'visual strategy'                   
                    else:
                        color = 'blue'
                        label = 'timing strategy'
                elif value == 'is_change':
                    if bool(split_value):
                        color=(0.1215,0.466,.7058)
                        label='change'
                    else:
                        color='lightgray'
                        label='repeat'
                elif value == 'engagement_v2':
                    if bool(split_value):
                        color = 'darkorange'
                        label='engaged'
                    else:
                        color = 'red'
                        label='disengaged'
                else:
                    color= plt.cm.get_cmap('tab10').colors[sdex]
                    label = str(value)+' '+str(split_value) 
                if isinstance(split_value, str):
                    temp = hierarchy.query('{} == @split_value'.format(value))         
                else:
                    temp = hierarchy.query('{} == {}'.format(value, split_value))        
                ax.errorbar(temp['xloc'],temp['response'],yerr=temp['bootstrap_sem'],
                    fmt='o',color=color,alpha=.5)
                ax.plot(temp['xloc'],temp['response'],'o',color=color,label= label)
    else:
        colors={
            'sst':(158/255,218/255,229/255),
            'exc':(255/255,152/255,150/255),
            'vip':(197/255,176/255,213/255),
        }
        if in_color is not None:
            colors[cell_type] = in_color
        if in_label is not None:
            label=in_label
        else:
            label=cell_type
        ax.plot(hierarchy['xloc'],hierarchy['response'], 'o',label=label,
            color=colors[cell_type])
        ax.errorbar(hierarchy['xloc'],hierarchy['response'],
            yerr=hierarchy['bootstrap_sem'],fmt='o',alpha=.5,color=colors[cell_type])

    # Determine xlabels
    xlabels = hierarchy.sort_values(by='xloc')\
        .drop_duplicates(subset=['targeted_structure',depth])
    mapper = {
        'VISp':'V1',
        'VISl':'LM'
        }
    xlabels['labels'] = [mapper[row.targeted_structure]\
        +' '+str(row[depth]) for index, row in xlabels.iterrows()]
    ax.set_xlabel('area & depth (um)',fontsize=16)
    ax.set_xticks(xlabels['xloc'])
    ax.set_xticklabels(xlabels['labels'])

    # Determine y labels and limits
    ylabel = cell_type +' '+response
    ax.set_ylabel(ylabel,fontsize=16)   
    if polish:
        plt.ylim(bottom=0)
    if ylim is not None:
        plt.ylim(top=ylim)

    # Annotate significance
    if ('bh_significant' in hierarchy.columns) and (polish):
        y =  ax.get_ylim()[1]*1.05
        for index, row in hierarchy.iterrows():
            if row.bh_significant:
                ax.plot(row.xloc, y, 'k*')  
        ax.set_ylim(top=y*1.075)

    # Clean up
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.legend()
    plt.tight_layout()

    # Save figure 
    if savefig:
        extra = extra + '_'.join(splits)
        if first:
            extra+='_first'
        if second:
            extra+='_second'
        filename = PSTH_DIR + data+'/hierarchy/'+\
            '{}_hierarchy_{}_{}_{}.svg'.format(cell_type,response,depth,extra) 
        print('Figure saved to: '+filename)
        plt.savefig(filename)
    
    return ax

def load_change_df(summary_df,cre,data='events',first=False,second=False,image=False,meso=False):

    # Load everything
    df = bd.load_population_df(data,'image_df',cre,first=first,second=second,image=image)

    # filter to changes
    df.drop(df[~df['is_change']].index,inplace=True)

    # add additional details
    experiment_table = glm_params.get_experiment_table()
    df = bd.add_area_depth(df,experiment_table)
    df = pd.merge(df, experiment_table.reset_index()[['ophys_experiment_id',
    'binned_depth','equipment_name']],on='ophys_experiment_id')
    df = pd.merge(df, summary_df[['behavior_session_id',
        'visual_strategy_session','experience_level']])

    # limit to familiar
    df = df.query('experience_level == "Familiar"')
    
    # limit to meso
    if meso:
        df = df.query('equipment_name == "MESO.1"').copy()

    return df

def load_image_df(summary_df, cre,data='events',first=False,second=False,meso=False,image=False):
    '''
        This function is optimized for memory conservation
    '''

    # Load everything
    df = bd.load_population_df(data,'image_df',cre,first=first,second=second,image=image)

    # Drop changes and omissions
    df.drop(df[df['is_change'] | df['omitted']].index,inplace=True)

    # drop familiar sessions
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)

    # Add additional details
    df = pd.merge(df,summary_df[['behavior_session_id','visual_strategy_session']])
    experiment_table = glm_params.get_experiment_table()
    df = bd.add_area_depth(df, experiment_table)
    df = pd.merge(df, experiment_table.reset_index()[['ophys_experiment_id',
        'binned_depth','equipment_name']],on='ophys_experiment_id')

    # Limit to Mesoscope
    if meso:
        df = df.query('equipment_name == "MESO.1"').copy()

    return df

def load_image_and_change_df(summary_df, cre,data='events',first=False,second=False):
    # load everything
    df = bd.load_population_df(data,'image_df',cre,first=first,second=second)
    
    # drop omissions
    df.drop(df[df['omitted']].index,inplace=True)

    # limit to familiar
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)

    # Add additional details
    df = pd.merge(df,summary_df[['behavior_session_id','visual_strategy_session']])
    experiment_table = glm_params.get_experiment_table()
    df = bd.add_area_depth(df, experiment_table)
    df = pd.merge(df, experiment_table.reset_index()[['ophys_experiment_id',
        'binned_depth']],on='ophys_experiment_id')
    return df

def load_omission_df(summary_df, cre, data='events',first=False,second=False,meso=False):
    # load everything
    df = bd.load_population_df(data,'image_df',cre,first=first,second=second)
    
    # drop omissions
    df.drop(df[~df['omitted']].index,inplace=True)

    # limit to familiar
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)

    # Add additional details
    df = pd.merge(df,summary_df[['behavior_session_id','visual_strategy_session']])
    experiment_table = glm_params.get_experiment_table()
    df = bd.add_area_depth(df, experiment_table)
    df = pd.merge(df, experiment_table.reset_index()[['ophys_experiment_id',
        'binned_depth','equipment_name']],on='ophys_experiment_id')

    # Limit to Mesoscope
    if meso:
        df = df.query('equipment_name == "MESO.1"').copy()
    return df

def load_vip_omission_df(summary_df,bootstrap=False,data='events'):
    '''
        Load the Vip omission responses, compute the bootstrap intervals
    '''
    print('loading vip image_df')
    vip_image_filtered= bd.load_population_df(data,'image_df','Vip-IRES-Cre')
    vip_omission = vip_image_filtered.query('omitted').copy()
    vip_omission = pd.merge(vip_omission, 
        summary_df[['behavior_session_id','visual_strategy_session','experience_level']],
        on='behavior_session_id')
    vip_omission = vip_omission.query('experience_level=="Familiar"').copy()
    
    if bootstrap:
        # processing
        print('bootstrapping')
        vip_omission = vip_omission[['visual_strategy_session','ophys_experiment_id',
            'cell_specimen_id','stimulus_presentations_id','response']]
        bootstrap_means = hb.bootstrap(vip_omission, levels=['visual_strategy_session',
            'ophys_experiment_id','cell_specimen_id'],nboots=10000)
        return vip_omission, bootstrap_means
    else:
        return vip_omission

def plot_vip_omission_summary(vip_omission, bootstrap_means):
    '''
        Plot a summary figure of the average omission response split by
        strategy
    '''

    diff = np.array(bootstrap_means[True])-np.array(bootstrap_means[False])
    p_boot = np.sum(diff<=0)/len(diff)
    visual_sem = np.std(bootstrap_means[True])
    timing_sem = np.std(bootstrap_means[False])

    x = vip_omission.groupby(['visual_strategy_session'])['response'].mean()
    y = vip_omission.groupby(['visual_strategy_session'])['response'].sem()

    plt.figure(figsize=(1.75,2.75))
    plt.plot(1,x.loc[True],'o',color='darkorange')
    plt.plot(2,x.loc[False],'o',color='blue')
    plt.errorbar(1,x.loc[True],y.loc[True],color='darkorange')
    plt.errorbar(2,x.loc[False],y.loc[False],color='blue')
    plt.errorbar(1,x.loc[True],visual_sem,color='darkorange')
    plt.errorbar(2,x.loc[False],timing_sem,color='blue')
    plt.xlim(.5,2.5)
    plt.ylim(0,.0425)
    plt.plot([1,2],[.038,.038],'k')
    plt.plot([1,1],[.037,.038],'k')
    plt.plot([2,2],[.037,.038],'k')
    if p_boot < .05:
        plt.plot(1.5,.0395,'k*' )
    plt.ylabel('Vip omission',fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([1,2])
    ax.set_xticklabels(['V','T'],fontsize=16)
    plt.tight_layout()

    data = 'events'
    filename = PSTH_DIR + data+'/summary/'+\
        'vip_omission_summary.svg'
        
    print('Figure saved to: '+filename)
    plt.savefig(filename)

def plot_exc_change_summary():
    cell_type = 'exc'
    response='change'
    data='events'
    depth='binned_depth'
    nboots=10000
    splits=['hit']
    first=True
    second=False
    area = 'VISp'
    layer='275'
    visual = get_hierarchy(cell_type, response, data, depth, nboots,
        splits,'visual',first,second)
    timing = get_hierarchy(cell_type, response, data, depth, nboots,
        splits,'timing',first,second)
    visual = visual.query('targeted_structure == @area')\
        .query('{}=={}'.format(depth, layer))\
        .set_index('hit')
    timing = timing.query('targeted_structure == @area')\
        .query('{}=={}'.format(depth, layer))\
        .set_index('hit')

    #plt.figure(figsize=(3,2.75))
    plt.figure(figsize=(2.5,2.75))
    plt.plot(-.05,visual.loc[1]['response'],'o',color='darkorange',label='visual hit')
    plt.plot(0.05,visual.loc[0]['response'],'x',color='darkorange',label='visual miss')
    plt.plot(.95,timing.loc[1]['response'],'o',color='blue',label='timing hit')
    plt.plot(1.05,timing.loc[0]['response'],'x',color='blue',label='timing mis')
    plt.errorbar(0.05,visual.loc[0]['response'],
        visual.loc[0]['bootstrap_sem'],color='darkorange')
    plt.errorbar(-0.05,visual.loc[1]['response'],
        visual.loc[1]['bootstrap_sem'],color='darkorange')
    plt.errorbar(1.05,timing.loc[0]['response'],
        timing.loc[0]['bootstrap_sem'],color='blue')
    plt.errorbar(.95,timing.loc[1]['response'],
        timing.loc[1]['bootstrap_sem'],color='blue')

    plt.plot(0,.012,'k*' )
    plt.ylim(0,.0125)
    plt.xlim(-.5,1.5)
    
    plt.ylabel('Excitatory \n(avg. Ca$^{2+}$ events)',fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Vis.','Tim.'],fontsize=16)
    plt.legend()
    plt.tight_layout()

    data = 'events'
    filename = PSTH_DIR + data+'/summary/'+\
        'exc_change_summary.svg'
        
    print('Figure saved to: '+filename)
    plt.savefig(filename)
 

def find_preferred(df):
    g = df.groupby(['cell_specimen_id','image_index'])['response'].mean()
    g = g.unstack()
    g['max'] = g.idxmax(axis=1)
    df['preferred_image'] = df['cell_specimen_id'].map(g['max'])
    df['is_preferred'] = df['preferred_image'] == df['image_index']
    return df


def check_equipment():
    '''
        Returns PSTH datasets split by imaging equipment type
        Familiar sessions only    
    '''
    dfs = get_figure_4_psth(data='events')
    experiment_table = glm_params.get_experiment_table().reset_index()
    for index, df in enumerate(dfs):
        dfs[index] = pd.merge(dfs[index], 
            experiment_table[['ophys_experiment_id','equipment_name']],
            on='ophys_experiment_id')

    sci_dfs = []
    mes_dfs = []
    sci_equip = ['CAM2P.3','CAM2P.4','CAM2P.5']
    for index, df in enumerate(dfs):
        sci_dfs.append(df.query('equipment_name in @sci_equip').copy())
        mes_dfs.append(df.query('equipment_name == "MESO.1"').copy())

    return sci_dfs, mes_dfs

def plot_figure_4_by_equipment(sci_df,mes_df,cell_type,cell_counts,data='filtered_events',
    savefig=False,areas=['VISp','VISl'],depths=['upper','lower'],experience_level='Familiar',
    strategy ='visual_strategy_session',depth='layer'):


    fig, ax = plt.subplots(3,3,figsize=(10,7.5),sharey='row',squeeze=False) 
    labels = ['Sci. '+cell_type, 'Meso. '+cell_type, 'Meso. '+cell_type+'\nexcluding the area']
    error_type='sem'
    mapper = {
        'Excitatory':0,
        'excitatory':0,
        'exc':0,
        'Sst':1,
        'sst':1,
        'Vip':2,
        'vip':2
        }
    cell_index=mapper[cell_type]
    dfs = [sci_df[cell_index], mes_df[cell_index]]
    for index, full_df in enumerate(dfs): 
        max_y = [0,0,0]
        ylabel=labels[index] +'\n(Ca$^{2+}$ events)'
        max_y[0] = plot_condition_experience(full_df, 'omission', experience_level,
            strategy, ax=ax[index, 0], ylabel=ylabel,
            error_type=error_type,areas=areas,depths=depths,depth=depth)
        max_y[1] = plot_condition_experience(full_df, 'hit', experience_level,
            strategy, ax=ax[index, 1],ylabel='',
            error_type=error_type,areas=areas,depths=depths,depth=depth)
        max_y[2] = plot_condition_experience(full_df, 'miss', experience_level,
            strategy, ax=ax[index, 2],ylabel='',
            error_type=error_type,areas=areas,depths=depths,depth=depth)
        ax[index,0].set_ylim(top = 1.05*np.max(max_y))

    exclusion_df = mes_df[cell_index]
    exclusion_df = exclusion_df.\
        query('~((targeted_structure == @areas[0])&(binned_depth ==@depths[0]))')
    max_y = [0,0,0]
    index=2
    ylabel=labels[index] +'\n(Ca$^{2+}$ events)'
    max_y[0] = plot_condition_experience(exclusion_df, 'omission', experience_level,
        strategy, ax=ax[index, 0], ylabel=ylabel,
        error_type=error_type)
    max_y[1] = plot_condition_experience(exclusion_df, 'hit', experience_level,
        strategy, ax=ax[index, 1],ylabel='',
        error_type=error_type)
    max_y[2] = plot_condition_experience(exclusion_df, 'miss', experience_level,
        strategy, ax=ax[index, 2],ylabel='',
        error_type=error_type)
    ax[index,0].set_ylim(top = 1.05*np.max(max_y))

    for x in [0,1,2]:
            ax[x,0].set_xlabel('time from omission (s)',fontsize=16)
            ax[x,1].set_xlabel('time from hit (s)',fontsize=16)
            ax[x,2].set_xlabel('time from miss (s)',fontsize=16)

    # Clean up
    area_depth = areas[0]+'_'+str(depths[0])
    mapper={
        'Excitatory':'Slc17a7-IRES2-Cre',
        'Vip':'Vip-IRES-Cre',
        'Sst':'Sst-IRES-Cre'
        }
    cre = mapper[cell_type]
    sci_vis = cell_counts.loc[cre, 'SCI',area_depth]['visual']
    sci_tim = cell_counts.loc[cre, 'SCI',area_depth]['timing']
    mes_vis = cell_counts.loc[cre, 'MESO',area_depth]['visual']
    mes_tim = cell_counts.loc[cre, 'MESO',area_depth]['timing']
    title = ('{}, {}, {} cortical depth\n'+\
        'Scientifica cells: {: >8.0f} visual, {: >8.0f} timing\n'+\
        'Mesoscope cells: {: >8.0f} visual, {: >8.0f} timing').\
        format(cell_type,areas[0],depths[0],sci_vis, sci_tim, mes_vis, mes_tim)
    plt.suptitle(title)
    plt.tight_layout()
    if savefig:
        filename = PSTH_DIR + data + '/population_averages/'+\
            'equipment_psth_'+cell_type+'_{}_{}.png'.format(depths[0],areas[0])
        print('Figure saved to: '+filename)
        plt.savefig(filename)

def get_equipment_counts(BEHAVIOR_VERSION=21):
    cell_table = loading.get_cell_table(platform_paper_only=True,
        add_extra_columns=False,
        include_4x2_data=False)
    cell_table = cell_table.query('experience_level == "Familiar"').copy()
    experiment_table = glm_params.get_experiment_table()
    cell_table = pd.merge(cell_table, 
        experiment_table.reset_index()[['ophys_experiment_id','area_binned_depth']],
        on='ophys_experiment_id')
    cell_table['equipment'] = ['MESO' if x == "MESO.1" else 'SCI' \
        for x in cell_table['equipment_name']]
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    cell_table = pd.merge(cell_table, summary_df[['behavior_session_id','strategy_labels']],
        on='behavior_session_id')
    counts = cell_table.groupby(['cre_line','equipment','area_binned_depth',\
        'strategy_labels'])['cell_specimen_id'].nunique().unstack().fillna(value=0)
    return counts

def plot_equipment_comparison(summary_df, experiment_table, cell_type, comparison_type,
    depths = [],first=False, second=False, image=False, data='events',savefig=False):

    mapper={
        'Excitatory':'Slc17a7-IRES2-Cre',
        'Vip':'Vip-IRES-Cre',
        'Sst':'Sst-IRES-Cre'
        }
    cre = mapper[cell_type]
    
    if comparison_type == 'omission':
        all_df= load_omission_df(summary_df,cre=cre,data=data,
            first=first, second=second)
    elif comparison_type == 'change':
        all_df= load_change_df(summary_df,cre=cre,data=data,
            first=first, second=second, image=image)
    elif comparison_type == 'pre_change':
        all_df = load_image_df(summary_df, cre=cre,data=data,
            first=first, second=second)
        all_df = all_df.query('(pre_hit_1==1)or(pre_miss_1==1)').copy()
    
    all_df = pd.merge(all_df, 
        experiment_table.reset_index()[['ophys_experiment_id','equipment_name']],
        on='ophys_experiment_id')

    all_df['equipment'] = ['MESO' if x == "MESO.1" else 'SCI' \
        for x in all_df['equipment_name']]
   
    if comparison_type =='omission': 
        df = all_df.query('targeted_structure == "VISp"').query('binned_depth in @depths').\
            groupby(['equipment','visual_strategy_session'])['response'].describe()
    elif comparison_type == 'change':
        df = all_df.query('targeted_structure == "VISp"').query('binned_depth in @depths').\
            groupby(['equipment','visual_strategy_session','hit'])['response'].describe()
    elif comparison_type == 'pre_change':
        df = all_df.query('targeted_structure == "VISp"').query('binned_depth in @depths').\
            groupby(['equipment','visual_strategy_session','pre_hit_1'])['response'].describe()
    df = df[['count','mean','std']]
    df['sem'] = df['std']/np.sqrt(df['count'])
    print(df)

    plot_equipment_comparison_inner(df, cell_type, comparison_type, depths,data,savefig=savefig)

def plot_equipment_comparison_inner(df,cell_type,comparison_type,depths,data,savefig=False):

    # Plot mean+CI for each group
    fig, ax = plt.subplots(1,2,figsize=(5,2.75))
    colors = ['darkorange','blue']
    if comparison_type == 'omission':
        for index, val in enumerate(['MESO','SCI']):
            for jndex, jval in enumerate([True, False]):
                mean = df.loc[val, jval]['mean'] 
                sem = df.loc[val,jval]['sem']*1.96   
                ax[index].plot(jndex+1,mean,'o',color=colors[jndex])
                ax[index].plot([jndex+1,jndex+1],[mean-sem,mean+sem],'-',color=colors[jndex])
    elif comparison_type in ['change','pre_change']:
        markers=['x','o']
        for index, val in enumerate(['MESO','SCI']):
            for jndex, jval in enumerate([True, False]):
                for kndex, kval, in enumerate([0,1]):
                    mean = df.loc[val, jval,kval]['mean'] 
                    sem = df.loc[val,jval,kval]['sem']*1.96   
                    if kndex == 0:
                        dx = .15
                    else:
                        dx = -.15
                    ax[index].plot(jndex+1+dx,mean,markers[kndex],color=colors[jndex])
                    ax[index].plot([jndex+1+dx,jndex+1+dx],[mean-sem,mean+sem],'-',
                        color=colors[jndex])
        

    # Clean up plot
    equipment = ['Mesoscope','Scientifica']
    for index in [0,1]:
        ax[index].set_ylabel(equipment[index]+' \n(avg. Ca$^{2+}$ events)',fontsize=16)
        ax[index].spines['right'].set_visible(False)
        ax[index].spines['top'].set_visible(False)
        ax[index].xaxis.set_tick_params(labelsize=12)
        ax[index].yaxis.set_tick_params(labelsize=12)
        ax[index].set_xticks([1,2])
        ax[index].set_xticklabels(['Vis.','Tim.'],fontsize=16)
        ax[index].set_xlim(.5,2.5)
        ax[index].set_ylim(bottom=0)

    title = '{}, {}, {}, {}'.format(cell_type, 'VISp',', '.join(depths),comparison_type)
    plt.suptitle(title,fontsize=16)
    plt.tight_layout()
    
    # Save
    if savefig:
        filename = PSTH_DIR + data + '/summary/'+\
            'equipment_psth_'+cell_type+'_{}_{}.png'.format(comparison_type,'_'.join(depths))
        print('Figure saved to: '+filename)
        plt.savefig(filename)



def bootstrap_summary_multiple_comparisons():
    tests = {}

    # Vip, image repeats, comparing strategies
    vip_image = get_summary_bootstrap_image_strategy(cell_type='vip',second=True,first=False,
        meso=True)
    p = bootstrap_significance(vip_image,'visual','timing')
    tests['vip_image']=p

    # Sst, image repeats, comparing strategies
    sst_image = get_summary_bootstrap_image_strategy(cell_type='sst',second=False,first=True,
        meso=True)
    p = bootstrap_significance(sst_image,'visual','timing')
    tests['sst_image']=p

    # Sst, omissions, comparing strategies    
    sst_omission = get_summary_bootstrap_omission_strategy(cell_type='sst',first=False,
        second=True,meso=True)
    p = bootstrap_significance(sst_omission,'visual','timing')
    tests['sst_omission']=p

    # Vip, omissions, comparing strategies
    vip_omission = get_summary_bootstrap_omission_strategy(cell_type='vip',first=False,second=False,
        meso=True)
    p = bootstrap_significance(vip_omission,'visual','timing')
    tests['vip_omission']=p   

    # Exc, changes, comparing strategies
    exc_hit = get_summary_bootstrap_strategy_hit(cell_type='exc', first=False, second=False,
        image=True,meso=True)
    phh = bootstrap_significance(exc_hit, 'visual_hit','timing_hit')
    pvhm = bootstrap_significance(exc_hit, 'visual_hit','visual_miss')
    tests['exc_hit_hit']=phh
    tests['exc_visual_hit_miss']=pvhm

    # Sst, changes, comparing strategies
    sst_hit = get_summary_bootstrap_strategy_hit(cell_type='sst', first=False, second=True,
        image=False,meso=True)
    phh = bootstrap_significance(sst_hit, 'visual_hit','timing_hit')
    pvhm = bootstrap_significance(sst_hit, 'visual_hit','visual_miss')
    tests['sst_hit_hit']=phh
    tests['sst_visual_hit_miss']=pvhm

    # Vip, changes, comparing strategies
    vip_hit = get_summary_bootstrap_strategy_pre_change(cell_type='vip', first=False, second=True,
        meso=True)
    phh = bootstrap_significance(vip_hit, 'visual_hit','timing_hit')
    pmm = bootstrap_significance(vip_hit, 'visual_miss','timing_miss')
    pvhm = bootstrap_significance(vip_hit, 'visual_hit','visual_miss')
    tests['vip_hit_hit']=phh
    tests['vip_miss_miss']=pmm
    tests['vip_visual_hit_miss']=pvhm

    ## Engagement
    # Exc, misses, comparing engagement 
    exc_miss = get_summary_bootstrap_strategy_engaged_miss(data='events',nboots=10000,
        cell_type='exc',first=False,second=False,image=True,meso=True)
    pved = bootstrap_significance(exc_miss, 'visual_engaged','visual_disengaged')
    tests['exc_misses_visual_engaged_disengaged']=pved

    # Exc, post omission, comparing engagement
    exc_post_engagement = get_summary_bootstrap_strategy_engaged_omission(data='events',
        nboots=10000,cell_type = 'exc', first=False, second=False, image=True,meso=True,post=True)
    pved = bootstrap_significance(exc_post_engagement, 'visual_engaged','visual_disengaged')
    tests['exc_post_omission_visual_engaged_disengaged']=pved  

    #return tests
    tests = pd.DataFrame.from_dict(tests,orient='index')
    tests = tests.rename(columns = {0:'p'})
    tests['p'] = [1-p if p > .5 else p for p in tests['p']]
    
    tests = tests.reset_index()
    tests = tests.rename(columns={
        'index':'location',
        'p':'p_boot'
        })
    tests['significant'] = tests['p_boot'] < 0.05
    tests = add_hochberg_correction(tests)

    cols=['location','p_boot','imq','bh_significant']
    print('\nThe following tests are significant with multiple comparisons corrections: ')
    print(tests.query('bh_significant').sort_values(by='imq')[cols])   

    print('\nThe following significant tests do not survive multiple comparisons corrections:')
    print(tests.query('significant & (not bh_significant)').sort_values(by='imq')[cols])

    print('\nThe following tests were not significant:')
    print(tests.query('not significant').sort_values(by='imq')[cols])

    return tests
