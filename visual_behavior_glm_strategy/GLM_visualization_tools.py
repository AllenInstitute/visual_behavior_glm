import visual_behavior.plotting as vbp
import visual_behavior.utilities as vbu
import visual_behavior.data_access.utilities as utilities
import visual_behavior.data_access.loading as loading
import visual_behavior_glm_strategy.GLM_analysis_tools as gat
import visual_behavior_glm_strategy.GLM_params as glm_params
from mpl_toolkits.axes_grid1 import Divider, Size

import copy
import visual_behavior.database as db
import matplotlib as mpl
import seaborn as sns
import scipy
import numpy as np
import pandas as pd
import pickle
import os
import time
from tqdm import tqdm
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc
from scipy import ndimage
from scipy import stats
import statsmodels.stats.multicomp as mc
import scipy.cluster.hierarchy as sch
import visual_behavior.visualization.utils as utils
from sklearn.decomposition import PCA

def project_colors():
    '''
        Defines a color scheme for various conditions
    '''
    tab20= plt.get_cmap("tab20c")
    set1 = plt.get_cmap('Set1')
    colors = {
        0:set1(0),
        1:set1(1),
        2:set1(2),
        3:set1(3),
        4:set1(4),
        5:set1(5),
        6:set1(6),
        'Sst-IRES-Cre visual':(50/255,218/255,229/255),
        'Slc17a7-IRES2-Cre visual':(255/255,100/255,150/255),
        'Vip-IRES-Cre visual':(197/255,126/255,213/255),
        'Sst-IRES-Cre timing':(158/255,218/255,229/255),
        'Slc17a7-IRES2-Cre timing':(255/255,152/255,150/255),
        'Vip-IRES-Cre timing':(197/255,176/255,213/255),
        'Sst-IRES-Cre':(158/255,218/255,229/255),
        'sst':(158/255,218/255,229/255),
        'Sst Inhibitory':(158/255,218/255,229/255),
        'Slc17a7-IRES2-Cre':(255/255,152/255,150/255),
        'slc':(255/255,152/255,150/255),
        'Excitatory':(255/255,152/255,150/255),
        'exc':(255/255,152/255,150/255),
        'Vip-IRES-Cre':(197/255,176/255,213/255),
        'vip':(197/255,176/255,213/255),
        'Vip Inhibitory':(197/255,176/255,213/255),
        '1':(148/255,29/255,39/255),
        '2':(222/255,73/255,70/255),
        '3':(239/255,169/255,150/255),
        '4':(43/255,80/255,144/255),
        '5':(100/255,152/255,193/255),
        '6':(195/255,216/255,232/255),
        '1.0':(148/255,29/255,39/255),
        '2.0':(222/255,73/255,70/255),
        '3.0':(239/255,169/255,150/255),
        '4.0':(43/255,80/255,144/255),
        '5.0':(100/255,152/255,193/255),
        '6.0':(195/255,216/255,232/255),
        'active':(.8,.8,.8),
        'passive':(.4,.4,.4),
        'familiar':(222/255,73/255,70/255),
        'novel':(100/255,152/255,193/255),
        'Familiar':(0.66,0.06,0.086),
        'Novel 1':(0.044,0.33,0.62),
        'Novel >1':(0.34,.17,0.57),
        'deep':'r',
        'shallow':'b',
        'VISp':'C0',
        'V1':'C0',
        'VISl':'C1',
        'LM':'C1',
        'VISal':'C2',
        'AL':'C2',
        'VISam':'C3',
        'AM':'C3',
        'Full': (.7,.7,.7),
        'visual':tab20(0), 
        'all-images':tab20(1),
        'expectation':tab20(2),
        'behavioral':tab20(8), 
        'licking':tab20(9),
        'pupil_and_running':tab20(10),
        'face_motion_energy':tab20(11),
        'cognitive':tab20(5), 
        'task':tab20(6),
        'beh_model':tab20(7),
        'behavioral_model':tab20(7),
        'licks':color_interpolate(tab20(9),tab20(11),6,1),
        'pupil':color_interpolate(tab20(10),tab20(11),5,0),
        'running':color_interpolate(tab20(10),tab20(11),5,2),
        'face_motion_PC_0':color_interpolate(tab20(10),tab20(11),5,5),
        'face_motion_PC_1':color_interpolate(tab20(10),tab20(11),5,6),
        'face_motion_PC_2':color_interpolate(tab20(10),tab20(11),5,7),
        'face_motion_PC_3':color_interpolate(tab20(10),tab20(11),5,8),
        'face_motion_PC_4':color_interpolate(tab20(10),tab20(11),5,9),
        'hits':color_interpolate(tab20(6),tab20(7),5,0),
        'misses':color_interpolate(tab20(6),tab20(7),5,1),
        'passive_change':color_interpolate(tab20(6),tab20(7),5,2), 
        'correct_rejects':color_interpolate(tab20(6),tab20(7),5,3),
        'false_alarms':color_interpolate(tab20(6),tab20(7),5,4),
        'model_bias':color_interpolate(tab20(6),tab20(7),5,5),
        'model_omissions1':color_interpolate(tab20(6),tab20(7),5,6),
        'model_task0':color_interpolate(tab20(6),tab20(7),5,7),
        'model_timing1D':color_interpolate(tab20(6),tab20(7),5,8),
        'bias strategy':color_interpolate(tab20(6),tab20(7),5,5),
        'post omission strategy':color_interpolate(tab20(6),tab20(7),5,6),
        'task strategy':color_interpolate(tab20(6),tab20(7),5,7),
        'timing strategy':color_interpolate(tab20(6),tab20(7),5,8),
        'visual':'darkorange',
        'timing':'blue',
        'image0':color_interpolate(tab20(1), tab20(3),8,0),
        'image1':color_interpolate(tab20(1), tab20(3),8,1),
        'image2':color_interpolate(tab20(1), tab20(3),8,2),
        'image3':color_interpolate(tab20(1), tab20(3),8,3),
        'image4':color_interpolate(tab20(1), tab20(3),8,4),
        'image5':color_interpolate(tab20(1), tab20(3),8,5),
        'image6':color_interpolate(tab20(1), tab20(3),8,6),
        'image7':color_interpolate(tab20(1), tab20(3),8,7),
        'omissions':color_interpolate(tab20(1), tab20(3),8,8),
        'Mesoscope':'c',
        'Scientifica':'y',
        'schematic_change': sns.color_palette()[0],
        'schematic_omission':sns.color_palette()[-1]
        } 
    return colors

def color_interpolate(start, end, num,position):
    diff = (np.array(start) - np.array(end))/num
    return tuple(start-diff*position)

def get_problem_sessions():
    '''
        Returns a list of ophys_session_ids that break various plotting codes. Specifically the problem is that these sessions were collected with a different ophys_sampling rate on mesoscope, and they need to be interpolated onto scientifica timestamps separately. For now, I am just excluding them in analyses that require interpolation onto common timestamps. 
    
        This list may not be exhaustive. Its just a manual list I generated when code broke. 
    '''
    problem_sessions = [873720614, 962045676, 1048363441,1049240847, 1050231786,1050597678, 1051107431,1051319542,1052096166,1052330675, 1052512524,1056065360, 1056238781, 1052752249,1049240847,1050929040,1052330675]
    return problem_sessions

def plot_kernel_support(glm,include_cont = True,plot_bands=True,plot_ticks=True,start=10000,end=11000):
    '''
        Plots the time points where each kernel has support 
        INPUTS:
        glm, glm object for the session to plot
        include_cont, if True, includes the continuous kernels which have support everywhere
        plot_bands, if True, plots diagonal bands to asses how kernels overlap
        plot_ticks, if True, plots a tick mark at the triggering event for each kernel
 
    '''  
    discrete = [x for x in glm.run_params['kernels'] if (glm.run_params['kernels'][x]['type']=='discrete') or (x == 'lick_model') or (x == 'groom_model')]
    continuous = [x for x in glm.run_params['kernels'] if glm.run_params['kernels'][x]['type']=='continuous']
    if include_cont:
        kernels = continuous + discrete
    else:
        kernels = discrete

    # Basic figure set up
    if plot_bands:
        plt.figure(figsize=(12,10))
    else:
        plt.figure(figsize=(12,6))
    time_vec = glm.fit['fit_trace_timestamps'][start:end]
    start_t = time_vec[0]
    end_t = time_vec[-1]
    ones = np.ones(np.shape(time_vec))
    colors = sns.color_palette('hls', len(discrete)+len(continuous)) 

    # Set up visualization parameters
    dk = 5
    dt = .4
    ms = 2
    if not plot_bands:
        dt = 0
        dk = 1
        ms = 10
    count = 0
    starts = []
    ends = []
    stim_points = {} # Create a dictionary of vertical position of each kernel

    # Plot the kernels
    for index, d in enumerate(kernels):
        starts.append(count)
        X = glm.design.get_X(kernels = [d])
        for dex in range(0,np.shape(X)[1]): 
            support = X.values[start:end,dex] != 0 
            plt.plot(time_vec[support],count*ones[support], 'o',color=colors[index],markersize=ms)
            count +=dt
        ends.append(count)
        count+=dk
        stim_points[d] = (starts[-1],ends[-1])
    ticks = [np.mean([x,y]) for (x,y) in zip(starts,ends)]
    all_k = kernels
    frame_rate = glm.fit['ophys_frame_rate']

    # Plot Rewards
    if 'rewards' in glm.run_params['kernels']:
        reward_dex = stim_points['rewards'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['rewards']['offset'])*frame_rate)
    elif 'hits' in glm.run_params['kernels']:
        reward_dex = stim_points['hits'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['hits']['offset'])*frame_rate)
    else:
        reward_dex = 0
    if plot_ticks:
        rewards =glm.session.rewards.query('timestamps < @end_t & timestamps > @start_t')['timestamps']
        #plt.plot(rewards, reward_dex*np.ones(np.shape(rewards)),'ro')
    
    # Stimulus Presentations
    stim = glm.session.stimulus_presentations.query('start_time > @start_t & start_time < @end_t & not omitted & not is_change')
    for index, time in enumerate(stim['start_time'].values):
        plt.axvspan(time, time+0.25, color='k',alpha=.1)
    if plot_ticks:
        for index in range(0,8):
            image = glm.session.stimulus_presentations.query('start_time >@start_t & start_time < @end_t & image_index == @index')['start_time']
            image_dex = stim_points['image'+str(index)][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['image'+str(index)]['offset'])*frame_rate)
            plt.plot(image, image_dex*np.ones(np.shape(image)),'k|')

    # Stimulus Changes
    change = glm.session.stimulus_presentations.query('start_time > @start_t & start_time < @end_t & is_change')
    if plot_ticks:
        if 'change' in glm.run_params['kernels']:
            change_dex = stim_points['change'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['change']['offset'])*frame_rate)
            plt.plot(change['start_time'], change_dex*np.ones(np.shape(change['start_time'])),'k|')
    for index, time in enumerate(change['start_time'].values):
        plt.axvspan(time, time+0.25, color=project_colors()['schematic_change'],alpha=.5,edgecolor=None)

    # Stimulus Omissions
    if plot_ticks:
        if 'omissions' in glm.run_params['kernels']:
            omitted = glm.session.stimulus_presentations.query('start_time >@start_t & start_time < @end_t & omitted')['start_time']
            omitted_dex = stim_points['omissions'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['omissions']['offset'])*frame_rate)
            plt.plot(omitted, omitted_dex*np.ones(np.shape(omitted)),'k|')
            for index, time in enumerate(omitted):
                plt.axvline(time, color=project_colors()['schematic_omission'], linestyle='--',zorder=-np.inf, linewidth=1.5)

    # Image Expectation
    if plot_ticks & ('image_expectation' in glm.run_params['kernels']):
        expectation = glm.session.stimulus_presentations.query('start_time >@start_t & start_time < @end_t & not omitted')['start_time']
        expectation_dex = stim_points['image_expectation'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['image_expectation']['offset'])*frame_rate)
        plt.plot(expectation, expectation_dex*np.ones(np.shape(expectation)),'k|')

    # Licks
    if plot_ticks:
        licks = glm.session.licks.query('timestamps < @end_t & timestamps > @start_t')['timestamps']
        
        if 'pre_lick_bouts' in glm.run_params['kernels']:
            bouts = glm.session.licks.query('timestamps < @end_t & timestamps > @start_t & bout_start')['timestamps']
            pre_dex = stim_points['pre_lick_bouts'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['pre_lick_bouts']['offset'])*frame_rate)
            post_dex = stim_points['post_lick_bouts'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['post_lick_bouts']['offset'])*frame_rate)
            plt.plot(bouts, pre_dex*np.ones(np.shape(bouts)),'k|')
            plt.plot(bouts, post_dex*np.ones(np.shape(bouts)),'k|')
        if 'lick_bouts' in glm.run_params['kernels']:
            bouts = glm.session.licks.query('timestamps < @end_t & timestamps > @start_t & bout_start')['timestamps']
            dex = stim_points['lick_bouts'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['lick_bouts']['offset'])*frame_rate)
            plt.plot(bouts, dex*np.ones(np.shape(bouts)),'k|')

        if 'pre_licks' in glm.run_params['kernels']:
            pre_dex = stim_points['pre_licks'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['pre_licks']['offset'])*frame_rate)
            post_dex = stim_points['post_licks'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['post_licks']['offset'])*frame_rate)
            plt.plot(licks, pre_dex*np.ones(np.shape(licks)),'k|')
            plt.plot(licks, post_dex*np.ones(np.shape(licks)),'k|')
        if 'licks' in glm.run_params['kernels']:
            dex = stim_points['licks'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['licks']['offset'])*frame_rate)
            plt.plot(licks, dex*np.ones(np.shape(licks)),'k|')


    # Trials
    if plot_ticks:
        types = ['hit','miss','false_alarm','correct_reject']
        ks = ['hits','misses','false_alarms','correct_rejects']
        trials = glm.session.trials.query('change_time < @end_t & change_time > @start_t')
        for index, t in enumerate(types):
            if ks[index] in glm.run_params['kernels']:
                try:
                    change_time = trials[trials[t]]['change_time'] 
                    trial_dex = stim_points[ks[index]][0] + dt*np.ceil(np.abs(glm.run_params['kernels'][ks[index]]['offset'])*frame_rate)
                    plt.plot(change_time, trial_dex*np.ones(np.shape(change_time)),'k|')
                except:
                    print('error plotting - '+t)

    # Clean up the plot
    plt.xlabel('Time (s)',fontsize=18)
    plt.yticks(ticks,all_k)
    plt.tick_params(axis='both',labelsize=16)
    plt.xlim(stim.iloc[0].start_time, stim.iloc[-1].start_time+.75)
    #plt.title(str(glm.session.metadata['ophys_experiment_id']) +' '+glm.session.metadata['equipment_name'])
    plt.tight_layout()
    return

def plot_glm_version_comparison_histogram(comparison_table=None, results=None, versions_to_compare=None, savefig=True):
    assert not (comparison_table is None and versions_to_compare is None), 'must pass either a comparison table or a list of two versions to compare'
    assert not (comparison_table is not None and results is not None), 'must pass either a comparison table or a results dataframe, not both'

    if results is not None:
        if versions_to_compare is None:
            versions_to_compare = results['glm_version'].unique()
        assert len(versions_to_compare) == 2, 'can only compare two glm_versions. Either pass a list of two versions, or pass a results table with two versions'

    if comparison_table is None:
        comparison_table = gat.get_glm_version_comparison_table(versions_to_compare=versions_to_compare, results=results)

    cre_lines = np.sort(comparison_table['cre_line'].dropna().unique())

    comparison_table['Diff'] = comparison_table[versions_to_compare[0]] -comparison_table[versions_to_compare[1]]
    plt.figure()
    jointplot = sns.histplot(
        comparison_table,
        x= 'Diff',
        hue='cre_line',
        hue_order = cre_lines, 
        palette = [project_colors()[cre_line] for cre_line in cre_lines],
        element='step',
        stat='density',
        common_norm=False,
    )
    plt.xlim(-.1,.1)
    plt.axvline(0, color='k',alpha=.25, linestyle='--')
    plt.xlabel(versions_to_compare[0] +'\n minus \n'+ versions_to_compare[1] +'\n Variance Explained')
    plt.tight_layout()    

    # Save a figure for each version 
    if savefig and (versions_to_compare is not None):
        version_strings = '_'.join([x.split('_')[0] for x in versions_to_compare])
        for version in versions_to_compare:
            run_params = glm_params.load_run_json(version)
            filepath = os.path.join(run_params['figure_dir'], 'version_comparison_histogram_'+version_strings+'.png')
            plt.savefig(filepath)

    return jointplot

def plot_glm_version_comparison(comparison_table=None, results=None, versions_to_compare=None, savefig=True):
    '''
    makes a scatterplot comparing cellwise performance on two GLM versions

    if a comparison table is not passed, the versions to compare must be passed (as a list of strings)
    comparison table will be built using GLM_analysis_tools.get_glm_version_comparison_table, which takes about 2 minutes
    
    savefig (bool) if True, saves a figure for each version in versions_to_compare
    '''
    assert not (comparison_table is None and versions_to_compare is None), 'must pass either a comparison table or a list of two versions to compare'
    assert not (comparison_table is not None and results is not None), 'must pass either a comparison table or a results dataframe, not both'

    if results is not None:
        if versions_to_compare is None:
            versions_to_compare = results['glm_version'].unique()
        assert len(versions_to_compare) == 2, 'can only compare two glm_versions. Either pass a list of two versions, or pass a results table with two versions'

    if comparison_table is None:
        comparison_table = gat.get_glm_version_comparison_table(versions_to_compare=versions_to_compare, results=results)

    cre_lines = np.sort(comparison_table['cre_line'].dropna().unique())
    jointplot = sns.jointplot(
        data = comparison_table,
        x = versions_to_compare[0],
        y = versions_to_compare[1],
        hue = 'cre_line',
        hue_order = cre_lines,
        alpha = 0.15,
        marginal_kws = {'common_norm':False},
        palette = [project_colors()[cre_line] for cre_line in cre_lines],
        height = 10,
    )

    # add a diagonal black line
    jointplot.ax_joint.plot(
        [0,1],
        [0,1], 
        color='k',
        linewidth=2,
        alpha=0.5,
        zorder=np.inf
    )
   
    # Save a figure for each version 
    if savefig and (versions_to_compare is not None):
        version_strings = '_'.join([x.split('_')[0] for x in versions_to_compare])
        for version in versions_to_compare:
            run_params = glm_params.load_run_json(version)
            filepath = os.path.join(run_params['figure_dir'], 'version_comparison_'+version_strings+'.png')
            plt.savefig(filepath)

    return jointplot

def plot_significant_cells(results_pivoted,dropout, dropout_threshold=0,save_fig=False,filename=None):
    sessions = np.array([1,2,3,4,5,6])
    cre = ["Sst-IRES-Cre", "Vip-IRES-Cre","Slc17a7-IRES2-Cre"]
    colors=['C0','C1','C2']
    plt.figure(figsize=(6,4))
    
    # Iterate over cre lines 
    for i,c in enumerate(cre):
        cells = results_pivoted.query('cre_line == @c')       
        num_cells = len(cells)
        cell_count = np.array([np.sum(cells.query('session_number == @x')[dropout] < dropout_threshold) for x in sessions])
        cell_p = cell_count/num_cells
        cell_err = 1.98*np.sqrt((cell_p*(1-cell_p))/cell_count)
        plt.errorbar(sessions-0.05, cell_p, yerr=cell_err, color=colors[i],label=c)

    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel('Session #')
    plt.ylabel('Fraction Cells Significant')
    plt.title(dropout + ', threshold: '+str(dropout_threshold))
    plt.tight_layout()
    if save_fig:
        filename = os.path.join(filename, dropout+'.png')
        plt.savefig(filename)

def plot_all_significant_cells(results_pivoted,run_params):
    dropouts = set(run_params['dropouts'].keys())
    dropouts.remove('Full')
    filename = run_params['output_dir']+'/'+'significant_cells/'
    for d in dropouts:
        plot_significant_cells(results_pivoted, d, save_fig=True, filename=filename)
        plt.close(plt.gcf().number)

def plot_regressor_correlation(glm, add_lines=True,save_plot=False):
    '''
        Plots the correlation of the design matrix for this glm object
        
        glm, the session to look at
        add_lines (bool), if True, plots faint lines to devide the correlation matrix
    '''   

    # Look at the discrete event kernels 
    discrete = [x for x in glm.run_params['kernels'] if glm.run_params['kernels'][x]['type']=='discrete']
    if 'intercept' in discrete:
        discrete.remove('intercept')
    discrete = np.sort(discrete)
    X = glm.design.get_X(kernels=discrete).values
    corr = np.corrcoef(X.T) # remove intercept
    plt.figure(figsize=(10,10))
    p = plt.gca().imshow(corr,cmap='Blues')
    plt.gcf().colorbar(p, ax=plt.gca())
    plt.title('Discrete Regressors')
    plt.xlabel('Regressor')
    plt.ylabel('Regressor')

    # Add ticks to mark each kernel
    start = 0
    end = -1
    ticks =[]
    locs = []
    for x in discrete:
        end += glm.design.kernel_dict[x]['kernel_length_samples'] 
        ticks.append(x)
        locs.append(np.mean([start,end]))
        start += glm.design.kernel_dict[x]['kernel_length_samples'] 
        if add_lines:
            plt.gca().axvline(end+0.5,color='k',alpha=0.05)
            plt.gca().axhline(end+0.5,color='k',alpha=0.05)
    plt.xticks(ticks=locs, labels=ticks,rotation=90)
    plt.yticks(ticks=locs, labels=ticks)
    plt.tight_layout()
    if save_plot:
        filename = os.path.join(glm.run_params['figure_dir'], 'discrete_regressor_correlation.png')
        plt.savefig(filename)

    # Look at the continuous kernels
    cont = [x for x in glm.run_params['kernels'] if glm.run_params['kernels'][x]['type']=='continuous']
    if 'intercept' in cont:
        cont.remove('intercept')
    cont = np.sort(cont)
    X = glm.design.get_X(kernels=cont).values
    corr = np.corrcoef(X.T) # remove intercept
    plt.figure(figsize=(10,10))
    p = plt.gca().imshow(corr,cmap='Blues')
    plt.gcf().colorbar(p, ax=plt.gca())
    plt.title('Continuous Regressors')
    plt.xlabel('Regressor')
    plt.ylabel('Regressor')
    
    # Add ticks to mark each kernel
    start = 0
    end = -1
    ticks =[]
    locs = []
    for x in cont:
        end += glm.design.kernel_dict[x]['kernel_length_samples'] 
        ticks.append(x)
        locs.append(np.mean([start,end]))
        start += glm.design.kernel_dict[x]['kernel_length_samples'] 
        if add_lines:
            plt.gca().axvline(end+0.5,color='k',alpha=0.05)
            plt.gca().axhline(end+0.5,color='k',alpha=0.05)
    plt.xticks(ticks=locs, labels=ticks,rotation=90)
    plt.yticks(ticks=locs, labels=ticks)
    plt.tight_layout() 
    if save_plot:
        filename = os.path.join(glm.run_params['figure_dir'], 'continuous_regressor_correlation.png')
        plt.savefig(filename)

    # Plot the correlations between the timeseries with no delay for the continuous kernels
    cont_events = np.vstack([glm.design.events[x] for x in cont])
    plt.figure(figsize=(10,10))
    corr = np.corrcoef(cont_events) # remove intercept
    p = plt.gca().imshow(corr,cmap='Blues')
    plt.gcf().colorbar(p, ax=plt.gca())

    # Add faint lines
    for dex,x in enumerate(cont):
        if add_lines:
            plt.gca().axvline(dex+0.5,color='k',alpha=0.05)
            plt.gca().axhline(dex+0.5,color='k',alpha=0.05)

    # Clean up plot and save   
    plt.title('Continuous Timeseries')
    plt.xlabel('Regressors')
    plt.ylabel('Regressors')
    plt.xticks(ticks=range(0,len(cont)), labels=cont,rotation=90)
    plt.yticks(ticks=range(0,len(cont)), labels=cont)
    plt.tight_layout()  
    if save_plot:
        filename = os.path.join(glm.run_params['figure_dir'], 'continuous_events_correlation.png')
        plt.savefig(filename)

def plot_PCA_var_explained(pca, figsize=(10,8)):
    fig,ax=plt.subplots(2,1,figsize=figsize, sharex=True)
    ax[0].plot(
        np.arange(40),
        pca.explained_variance_ratio_,
        'o-k'
    )
    ax[1].plot(
        np.arange(40),
        np.cumsum(pca.explained_variance_ratio_),
        'o-k'
    )

    ax[0].axhline(0, color='gray')
    ax[1].axhline(1, color='gray')
    ax[1].set_xlabel('PC number')
    ax[0].set_ylabel('variance explained')
    ax[1].set_ylabel('cumulative variance explained')
    ax[0].set_title('variance explained by PC')
    ax[1].set_title('cumulative variance explained by PC')
    fig.tight_layout()
    return fig, ax

def pc_component_heatmap(pca, figsize=(18,4)):
    components = pd.DataFrame(pca.components_, columns=pca.component_names)
    sorted_cols = np.array(pca.component_names)[np.argsort(pca.components_[0,:])]
    fig,ax=plt.subplots(figsize=figsize)
    sns.heatmap(
        components[sorted_cols[::-1]].iloc[:10],
        cmap='seismic',
        ax=ax,
        vmin=-1,
        vmax=1
    )
    ax.set_title('Principal Component Vectors')
    ax.set_xticks(np.arange(0.5,len(pca.component_names)+0.5))
    ax.set_xticklabels(sorted_cols[::-1],rotation=45,ha='right')
    ax.set_ylabel('PC number')
    fig.tight_layout()
    return fig, ax

def var_explained_matched(results_pivoted, run_params):
    # Remove passive sessions
    results_pivoted = results_pivoted.query('not passive').copy()
    colors = project_colors()
    colors['Matched'] = 'k'
    colors['Non-matched'] = 'gray'

    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory'
        }
    results_pivoted['cell_type'] = [mapper[x] for x in results_pivoted['cre_line']]
    results_pivoted['variance_explained_percent'] = results_pivoted['variance_explained_full']*100

    # load cells table to get matched cells
    cells_table = loading.get_cell_table(platform_paper_only=True,include_4x2_data=run_params['include_4x2_data']) 
    cells_table = cells_table.query('not passive').copy()
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    matched_cells = cells_table.cell_specimen_id.unique()
    results_pivoted['matched'] = ['Matched' if x in np.array(matched_cells) else 'Non-matched' for x in results_pivoted['cell_specimen_id']]

    fig,ax = plt.subplots(1,3,figsize=(10,3.5))
    cres = ['Excitatory','Sst Inhibitory','Vip Inhibitory']
    for index,cre in enumerate(cres):
        all_data = results_pivoted.query('cell_type==@cre')
        ax[index] = sns.boxplot(
            x='experience_level',
            y='variance_explained_percent',
            hue='matched',
            data=all_data,
            hue_order=['Matched', 'Non-matched'],
            order=['Familiar','Novel 1','Novel >1'],
            palette=[colors[cre],'gray'],
            linewidth=1,
            fliersize=0,
            ax=ax[index],
        )
        if index == 0:
            ax[index].set_ylabel('Variance Explained (%)',fontsize=18)
        else:
            ax[index].set_ylabel('',fontsize=18)
        ax[index].set_xlabel(cre,fontsize=18)
        ax[index].spines['top'].set_visible(False)
        ax[index].spines['right'].set_visible(False)
        ax[index].set_ylim(0,30)
        ax[index].tick_params(axis='both',labelsize=14)
        handles, labels = ax[index].get_legend_handles_labels()
        ax[index].legend(handles=handles, labels=labels,loc='upper right')
    plt.tight_layout() 
    filename = run_params['figure_dir']+'/variance_explained_matched.svg'
    plt.savefig(run_params['figure_dir']+'/variance_explained_matched.png')
    print('Figure saved to: ' + filename)
    plt.savefig(filename)
    return results_pivoted.groupby(['cell_type','experience_level','matched'])['variance_explained_percent'].describe()

 
 

def var_explained_by_experience(results_pivoted, run_params,threshold = 0,savefig=False):
    
    if threshold != 0:
        results_pivoted = results_pivoted.query('(not passive) & (variance_explained_full > @threshold)').copy()
    else:
         results_pivoted = results_pivoted.query('not passive').copy()   

    colors = project_colors()
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory'
        }
    results_pivoted['cell_type'] = [mapper[x] for x in results_pivoted['cre_line']]
    results_pivoted['variance_explained_percent'] = results_pivoted['variance_explained_full']*100
    plt.figure()
    ax = sns.boxplot(
        x='cell_type',
        y='variance_explained_percent',
        hue='experience_level',
        data=results_pivoted,
        hue_order=['Familiar','Novel 1','Novel >1'],
        order=['Vip Inhibitory','Sst Inhibitory','Excitatory'],
        palette=colors,
        fliersize=0,
        linewidth=1,
    )
    ax.set_ylabel('Variance Explained (%)',fontsize=18)
    ax.set_xlabel('Cell Type',fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0,40)
    plt.tick_params(axis='both',labelsize=14)
    plt.tight_layout() 
    if savefig:
        if threshold !=0:
            filename = run_params['figure_dir']+'/variance_explained_by_experience_filtered.svg'
            plt.savefig(run_params['figure_dir']+'/variance_explained_by_experience_filtered.png')
            print('Figure saved to: ' + filename)
            plt.savefig(filename)
        else:
            filename = run_params['figure_dir']+'/variance_explained_by_experience.svg'
            plt.savefig(run_params['figure_dir']+'/variance_explained_by_experience.png')
            print('Figure saved to: ' + filename)
            plt.savefig(filename)
    return results_pivoted.groupby(['cell_type','experience_level'])['variance_explained_percent'].describe()

def compare_var_explained_by_version(results=None, fig=None, ax=None, test_data=True, figsize=(9,5), use_violin=True,cre=None,metric='Full',show_equipment=True,zoom_xlim=True,sort_by_signal=True):
    '''
    make a boxplot comparing variance explained for each version in the database
    inputs:
        results: a dataframe of results (if None, will be retreived from database)
        fig, ax: figure and axis handles. If None, will be created
        figsize: size of figure
        outlier_threshold: Proportion of the IQR past the low and high quartiles to extend the plot whiskers. Points outside this range will be identified as outliers. (from seaborn docs)

    returns:
        figure and axis handles (tuple)
    '''

    # set up figure axis
    if results is None:
        results_dict = gat.retrieve_results()
        results = results_dict['full']
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize, sharey=True, sharex='col')

    # determine what data to plot
    hue = 'cre_line'
    split=False
    inner= None
    hue_order = np.sort(results['cre_line'].unique())
    colors = project_colors() 
    num_versions = len(results['glm_version'].unique())   

    if cre is not None:
        results = results.query('cre_line == @cre').copy()
        hue_order = np.sort(results['cre_line'].unique())
        inner = 'quartile'
        if show_equipment:
            hue = 'meso'
            results['meso'] = ['Mesoscope' if "MESO" in x else 'Scientifica' for x in results['equipment_name']]
            hue_order = np.sort(results['meso'].unique())
            split=True
    else:
        results = results.copy()

    if num_versions < 3:
        inner = 'quartile'    

    if sort_by_signal:
        results['dff'] = ['dff' in x for x in results['glm_version']]

        if num_versions > 2:
            glm_version_order = np.concatenate([np.sort(results.query('dff')['glm_version'].unique()),[''],np.sort(results.query('not dff')['glm_version'].unique())])
        else:
            glm_version_order = np.concatenate([np.sort(results.query('dff')['glm_version'].unique()),np.sort(results.query('not dff')['glm_version'].unique())])
    else:
        glm_version_order = np.sort(results['glm_version'].unique())
    
    if test_data:
        dataset = 'test'
    else:
        dataset = 'train'

    # plot main data
    if use_violin:
        plot1 = sns.violinplot(
            data=results,
            y='glm_version',
            x=metric+'__avg_cv_var_{}'.format(dataset),
            order = glm_version_order,
            hue=hue,
            hue_order=hue_order,
            inner=inner,
            linewidth=1,
            ax=ax,
            palette=colors,
            cut=0,
            split=split
        )
        lines = plot1.get_lines()
        for index, line in enumerate(lines):
            if np.mod(index,3) == 0:
                line.set_linewidth(0)
            elif np.mod(index,3) == 1:
                line.set_linewidth(1)
                line.set_color('r')
                line.set_linestyle('-')
            elif np.mod(index,3) == 2:
                line.set_linewidth(0)
    else:
        plot1 = sns.boxplot(
            data=results,
            x='glm_version',
            y=metric+'__avg_cv_var_{}'.format(dataset),
            order = glm_version_order,
            hue='cre_line',
            hue_order=cre_line_order,
            fliersize=0,
            whis=1.5,
            ax=ax,
        )      
    
    # Label axes and title 
    ax.set_xlabel(metric+' Model Variance Explained on {} set'.format(dataset),fontsize=16)
    ax.set_ylabel('GLM version',fontsize=16)
    if cre is not None:
        ax.set_title(cre,fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=12)
    ax.legend()

    # Add gray boxes to separate model versions
    ax.axvline(0, color='black',alpha=.25)
    mids = ax.get_yticks()
    edges = np.diff(mids)*.5 + mids[0:-1]
    orig_ylim = ax.get_ylim()
    ylim = sorted(list(orig_ylim))
    edges = [ylim[0]]+list(edges)+[ylim[1]]
    for dex, edge in enumerate(edges[:-1]):
        if np.mod(dex,2) == 0:
            plt.axhspan(edges[dex], edges[dex+1], color='k',alpha=.1)
    ax.set_ylim(orig_ylim)
    
    if zoom_xlim & show_equipment:
        ax.set_xlim(-0.05,.2)
 
    # Clean up and save
    fig.tight_layout()
    extra = '_'+metric+'_'+dataset
    if cre is not None:
        extra = extra+"_"+cre
        if show_equipment:
            extra = extra+"_equipment"
        if sort_by_signal:
            extra = extra+"_by_dff"
    filepath= '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/version_comparisons/variance_explained'+extra+'.png'
    print(filepath)
    plt.savefig(filepath)

    return fig, ax


def plot_licks(session, ax, y_loc=0, t_span=None):
    if t_span:
        df = session.licks.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        df = session.licks
    ax.plot(
        df['timestamps'],
        y_loc*np.ones_like(df['timestamps']),
        marker='o',
        color='white',
        linestyle='none',
        alpha=0.9
    )

def plot_rewards(session, ax, y_loc=0, t_span=None):
    if t_span:
        df = session.rewards.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        df = session.licks
    ax.plot(
        df['timestamps'],
        y_loc*np.ones_like(df['timestamps']),
        marker='o',
        color='skyblue',
        linestyle='none',
        alpha=0.9,
        markersize=12,
    )


def plot_running(session, ax, t_span=None):
    if t_span:
        running_df = session.running_speed.reset_index().query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        running_df = session.running_speed.reset_index()
    ax.plot(
        running_df['timestamps'],
        running_df['speed'],
        color='skyblue',
        linewidth=3
    )
    ax.set_ylim(
        session.running_speed['speed'].min(),
        session.running_speed['speed'].max(),
    )

def plot_pupil(session, ax, t_span=None):
    '''shares axis with running'''
    vbp.initialize_legend(ax=ax, colors=['skyblue','LemonChiffon'],linewidth=3)
    if t_span:
        pupil_df = session.eye_tracking.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        pupil_df = session.eye_tracking
    ax.plot(
        pupil_df['timestamps'],
        pupil_df['pupil_area'],
        color='LemonChiffon',
        linewidth=3
    )

    ax.legend(
        ['running','pupil'],
        loc='upper left',
        ncol=10, 
    )



def plot_omissions(session, ax, y_loc=0, t_span=None):
    omissions = session.stimulus_presentations.query('omitted == True')
    ax.plot(
        omissions['start_time'],
        y_loc*np.ones_like(omissions['start_time']),
        marker='*',
        color='red',
        linestyle='none'
    )


def plot_stimuli(stimulus_presentations, ax, t_span=None, alpha=.35):

    buffer = 0.25
    images = stimulus_presentations['image_name'].unique()

    colors = {image: color for image, color in zip(
        np.sort(images), sns.color_palette("Set2", 8))}

    if t_span:
        query_string = 'start_time >= {0} - {2} and stop_time <= {1} + {2}'.format(
            t_span[0], t_span[1], buffer)
        visual_stimuli = stimulus_presentations.query(
            'omitted == False').query(query_string).copy()
    else:
        visual_stimuli = stimulus_presentations.query(
            'omitted == False').copy()

    visual_stimuli['color'] = visual_stimuli['image_name'].map(
        lambda i: colors[i])
    visual_stimuli['change'] = visual_stimuli['image_name'] != visual_stimuli['image_name'].shift()

    for idx, stimulus in visual_stimuli.iterrows():
        ax.axvspan(
            stimulus['start_time'],
            stimulus['stop_time'],
            color=stimulus['color'],
            alpha=alpha,
            edgecolor=None,
        )
    
def get_movie_filepath(session_id, session_type='OphysSession', movie_type='RawBehaviorTrackingVideo'):
    well_known_files = db.get_well_known_files(session_id, session_type)
    behavior_video_path = ''.join(well_known_files.loc[movie_type][[
                                  'storage_directory', 'filename']].tolist())
    return behavior_video_path

def get_sync_data(session_id, session_type='OphysSession'):
    sync_key_map = {
        'OphysSession': 'OphysRigSync',
        'EcephysSession': 'EcephysRigSync',
    }
    well_known_files = db.get_well_known_files(session_id, session_type)
    sync_path = ''.join(well_known_files.loc[sync_key_map[session_type]][[
                        'storage_directory', 'filename']].tolist())
    sync_data = vbu.get_sync_data(sync_path)
    return sync_data

def build_simulated_FOV(session, F_dataframe, column):

    assert len(session.cell_specimen_table) == len(F_dataframe)

    arr = np.zeros_like(session.max_projection)
    for ii, cell_specimen_id in enumerate(session.cell_specimen_ids):

        F_cell = F_dataframe.loc[cell_specimen_id][column]
        # arr += session.cell_specimen_table.loc[cell_specimen_id]['image_mask']*F_cell
        arr += session.cell_specimen_table.loc[cell_specimen_id]['roi_mask']*F_cell

    return arr


def plot_kernels(kernel_df, ax, palette_df, t_span=None, legend=False, annotate=True, t0=0, t1=np.inf):
    # kernels_to_exclude_from_plot = []#['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    # kernels_to_exclude_from_plot = ['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    kernels_to_exclude_from_plot = ['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    kernels_to_include_in_plot = [k for k in kernel_df['kernel_name'].unique() if k not in kernels_to_exclude_from_plot]
    palette = palette_df.query('kernel_name in @kernels_to_include_in_plot')['kernel_color'].to_list()

    if t_span:
        t0,t1 = t_span
        data_to_plot = kernel_df.query('timestamps >= @t0 and timestamps <= @t1 and kernel_name not in @kernels_to_exclude_from_plot')
    else:
        data_to_plot = kernel_df.query('kernel_name not in @kernels_to_exclude_from_plot')

    sns.lineplot(
        data = data_to_plot,
        x='timestamps',
        y='kernel_outputs',
        hue='kernel_name',
        n_boot=0,
        ci=None,
        ax=ax,
        palette = palette,
        alpha=0.75,
        legend=legend,
        linewidth=3,
    )
    if legend:
        ax.legend(
            data_to_plot['kernel_name'].unique(),
            loc='upper left',
            ncol=10, 
            mode="expand", 
            framealpha = 0.5,
        )
    if annotate:
        max_locs = get_max_locs_df(data_to_plot)
        percentile_threshold = 95
        for idx,row in max_locs.iterrows():
            kernel_name = row['kernel_name']
            if row['percentile'] > percentile_threshold:
                va = 'top' if row['abs_max_sign'] < 0 else 'bottom'
                ax.text(
                    row['time'], 
                    row['abs_max_sign']*row['abs_max_value'],
                    row['kernel_name'],
                    ha='center',
                    va=va,
                    fontweight='bold',
                    color=palette_df.query('kernel_name == @kernel_name')['kernel_color'].iloc[0],
                    fontsize=15
                )
    qs = 'timestamps >= {} and timestamps <= {}'.format(
        t0,
        t1
    )
    ax.set_ylim(
        kernel_df.query(qs)['kernel_outputs'].min(),
        kernel_df.query(qs)['kernel_outputs'].max(),
    )

def plot_session_summary(glm):
    plt.figure()
    plt.plot(glm.dropout_summary.query('dropout=="Full"')['variance_explained'].sort_values().values)
    plt.axhline(0.00, color='k',alpha=.25)
    plt.axhline(0.01, color='k',alpha=.25)
    plt.gca().axhspan(-.1,0.01, color='k',alpha=0.25)
    plt.ylim(bottom=-.1)
    plt.ylabel('Full Model CV Variance Explained')
    plt.xlabel('Cells')

def plot_dropout_summary(results_summary, cell_specimen_id, ax, 
        dropouts_to_show=None, dropouts_to_plot='both', dropouts_to_exclude=[],
        ylabel_fontsize=22, ticklabel_fontsize=21, title_fontsize=22, legend_fontsize=18):

    '''
    makes bar plots of results summary
    inputs:
        glm -- glm object
        cell_specimen_id -- cell to plot
        ax -- axis on which to plot
        dropouts_to_plot -- 'single', 'standard' or 'both'. 'both' will show both in two hues.
    '''
    data_to_plot = (
        results_summary
        .query('cell_specimen_id == @cell_specimen_id')
        .sort_values(by='adj_fraction_change_from_full', ascending=False)
    ).copy().reset_index(drop=True)

    dropouts = data_to_plot.dropout.unique()

    all_identified_dropouts_to_exclude = []
    for dropout in dropouts_to_exclude:
        identified_dropouts_to_exclude = [d for d in dropouts if dropout in d]
        all_identified_dropouts_to_exclude += identified_dropouts_to_exclude
    
    data_to_plot = data_to_plot.query('dropout not in @all_identified_dropouts_to_exclude')

    
    single_dropouts = [d for d in dropouts if d.startswith('single-')]
    standard_dropouts = [d.split('single-')[1] for d in single_dropouts]

    for idx,row in data_to_plot.iterrows():
        if row['dropout'] in single_dropouts:
            data_to_plot.at[idx,'dropout_type']='single'
            data_to_plot.at[idx,'dropout_simple']=row['dropout'].split('single-')[1]
        elif row['dropout'] in standard_dropouts:
            data_to_plot.at[idx,'dropout_type']='standard'
            data_to_plot.at[idx,'dropout_simple']=row['dropout']

    if dropouts_to_plot == 'both':
        sort_by = 'single'
    else:
        sort_by = dropouts_to_plot
        data_to_plot = data_to_plot.query('dropout_type == @dropouts_to_plot')

    yorder = (
        data_to_plot
        .query('dropout_type == @sort_by')
        .sort_values(by='adj_fraction_change_from_full',ascending=False)['dropout_simple']
        .values
    )
    
    bp = sns.barplot(
        data = data_to_plot.sort_values(by='adj_fraction_change_from_full', ascending=False),
        x = 'adj_fraction_change_from_full',
        y = 'dropout_simple',
        ax=ax,
        hue='dropout_type' if dropouts_to_plot == 'both' else None,
        hue_order=['standard','single'] if dropouts_to_plot == 'both' else None,
        order=yorder,
        palette=['magenta','cyan'] if dropouts_to_plot == 'both' else ['cyan']
    )
    bar_colors = ['black','gray']
    for row in ax.get_yticks():
        ax.axhspan(row - 0.5, row + 0.5, color = bar_colors[row%2], alpha=0.25)
    ax.set_ylim(ax.get_yticks().max()+0.5, ax.get_yticks().min()-0.5)
    plt.legend(ncol=1, loc='upper left')
    plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=legend_fontsize) # for legend title
    
    
    ax.set_ylabel('', fontsize=ylabel_fontsize)
    # ax.set_xlabel('Fraction Change in Var Explained', fontsize=22)
    bp.tick_params(labelsize=ticklabel_fontsize)
    ax.set_title('Fraction Change\nin Variance Explained', fontsize=title_fontsize)


def plot_filters(glm, cell_specimen_id, n_cols=5):
    '''plots all filters for a given cell'''
    kernel_list = list(glm.design.kernel_dict.keys())
    all_weight_names = glm.X.weights.values
    n_rows = int(np.ceil(len(kernel_list)/5))

    fig,ax=plt.subplots(int(n_rows),int(n_cols), figsize=(2.5*n_cols,2.5*n_rows),sharey=True)

    ii = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if ii <= len(kernel_list) - 1:
                kernel_name = kernel_list[ii]
                
                t_kernel, w_kernel = get_kernel_weights(glm, kernel_name, cell_specimen_id)

                ax[row,col].plot(t,w_kernel,marker='.')
                ax[row,col].set_title(kernel_name)
                ax[row,col].axvline(0, color='k',linestyle=':')
                ax[row,col].axhline(0, color='k',linestyle=':')
                
            else:
                ax[row,col].axis('off')
            
            if ii >= len(kernel_list) - n_rows:
                ax[row,col].set_xlabel('time from event (s)')
            if col == 0:
                ax[row,col].set_ylabel('$\Delta$F/F')
            ii += 1

    fig.tight_layout()

    return fig, ax


def get_title(ophys_experiment_id, cell_specimen_id, glm_version):
    '''
    generate a standardized figure title containing identifying information
    '''
    experiments_table = loading.get_filtered_ophys_experiment_table().reset_index()

    row = experiments_table.query('ophys_experiment_id == @ophys_experiment_id').iloc[0].to_dict()
    title = '{}_exp_id={}_{}_{}_depth={}_cell_id={}_glm_version={}'.format(
        row['cre_line'],
        row['ophys_experiment_id'],
        row['session_type'],
        row['targeted_structure'],
        row['imaging_depth'],
        cell_specimen_id,
        glm_version.split('_')[0],
    )
    return title

def get_max_locs_df(df_in):
    '''
    find max location of each kernel in the kernel_df
    '''
    df_in = df_in.copy()
    df_in['kernel_outputs_abs'] = df_in['kernel_outputs'].abs()
    max_df = df_in.groupby('kernel_name')[['kernel_outputs','kernel_outputs_abs']].max().sort_values(by='kernel_outputs', ascending=False)
    max_locs = []
    for kernel_name,row in max_df.iterrows():
        kernel_subset = df_in.query('kernel_name == @kernel_name')
        m = kernel_subset['kernel_outputs_abs'].abs().max()
        max_locs.append({
            'kernel_name': kernel_name,
            'abs_max_value': kernel_subset['kernel_outputs_abs'].abs().max(),
            'abs_max_sign': np.sign(kernel_subset.loc[kernel_subset['kernel_outputs_abs'].idxmax()]['kernel_outputs']),
            'idx': kernel_subset['kernel_outputs_abs'].idxmax(),
            'time': kernel_subset.loc[kernel_subset['kernel_outputs_abs'].idxmax()]['timestamps'],
            'percentile':stats.percentileofscore(df_in['kernel_outputs_abs'], kernel_subset['kernel_outputs_abs'].abs().max(), kind='strict')
        })
    return pd.DataFrame(max_locs)

class GLM_Movie(object):

    def __init__(self, glm, cell_specimen_id, start_frame, end_frame, frame_interval=1, fps=10, destination_folder=None, verbose=False):

        self.verbose = verbose
        if self.verbose:
            print('initializing')
        # note that ffmpeg must be installed on your system
        # this is tested on linux (not sure if it works on windows)
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

        # plt.style.use('seaborn-white')
        plt.style.use('dark_background')
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        mpl.rcParams['legend.fontsize'] = 16

        self.glm = glm
        self.cell_specimen_id = cell_specimen_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_interval = frame_interval

        try:
            self.parameter_to_fit = 'events' if self.glm.run_params['use_events'] else 'dff'
        except KeyError:
            # GLM versions 10 and earlier lack the 'use_events' key and always fit dff
            self.parameter_to_fit = 'dff'

        self.model_timestamps = glm.fit['fit_trace_arr']['fit_trace_timestamps'].values
        self.initial_time = self.model_timestamps[self.start_frame]
        self.final_time = self.model_timestamps[self.end_frame]

        self.title = get_title(self.glm.oeid, self.cell_specimen_id, self.glm.version)

        self.kernel_df = gat.build_kernel_df(self.glm, self.cell_specimen_id)

        self.real_2p_movie = loading.load_motion_corrected_movie(self.glm.oeid)

        self.frames = np.arange(self.start_frame, self.end_frame, self.frame_interval)
        self.fps = fps

        if destination_folder is None:
            # if destination_folder is not specified, set it to {run_params['output_dir']}/output_files
            base_path = self.glm.run_params['output_dir'].split('/v_')[0]
            save_folder = os.path.join(base_path, 'output_files')
            self.destination_folder = os.path.join(base_path, 'output_files')
        else:
            self.destination_folder = destination_folder

        self.palette_df = pd.DataFrame({
            'kernel_name':self.kernel_df['kernel_name'].unique(),
            'kernel_color':vbp.generate_random_colors(
                len(self.kernel_df['kernel_name'].unique()), 
                lightness_range=(0.6,1), 
                saturation_range=(0.75,1), 
                random_seed=3, 
                order_colors=False
            )
        })

        self.sync_data = get_sync_data(glm.ophys_session_id)
        self.tracking_movies = {}
        for ii,movie_name in enumerate(['Behavior','Eye']):
            try:
                sync_timestamps = self.sync_data['cam{}_exposure_rising'.format(ii+1)]
            except KeyError:
                if movie_name == 'Eye':
                    sync_timestamps = self.sync_data['eye_tracking_rising']
                elif movie_name == 'Behavior':
                    sync_timestamps = self.sync_data['behavior_monitoring_rising']
            self.tracking_movies[movie_name.lower()] = vbu.Movie(
                get_movie_filepath(glm.ophys_session_id, 
                movie_type='Raw{}TrackingVideo'.format(movie_name)), 
                sync_timestamps=sync_timestamps,
            )

        # try to make destination folder if it doesn't already exist
        if os.path.exists(self.destination_folder) == False:
                os.mkdir(self.destination_folder)


        self.results_summary = glm.dropout_summary #gat.generate_results_summary(self.glm).reset_index()
        self.dropout_summary_plotted = False
        self.cell_roi_plotted = False

        self.this_cell = self.glm.cell_results_df.query('cell_specimen_id == @cell_specimen_id')
        self.stimulus_presentations = self.glm.session.stimulus_presentations

        self.fig, self.ax = self.set_up_axes()
        self.writer = self.set_up_writer()

        self.dff_color = 'lightgreen'
        self.events_color = 'salmon'
        self.fit_color = 'white'

        if self.verbose:
            print('done initializing')

    def make_cell_movie_frame(self, ax, glm, F_index, cell_specimen_id, t_before=10, t_after=10):
        
        if self.verbose:
            ti = time.time()
            print('starting frame plotting process')
        

        if self.verbose:
            print('done getting cell info at {:0.2f} seconds'.format(time.time() - ti))

        cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

        if self.verbose:
            print('done getting cell index at {:0.2f} seconds'.format(time.time() - ti))

        t_now = self.model_timestamps[F_index]
        t_span = [t_now - t_before, t_now + t_after]

        if self.verbose:
            print('done setting up tspan at {:0.2f} seconds'.format(time.time() - ti))

        dropouts_to_exclude=[
            'visual',
            'cognitive',
            'behavioral',
            'licking',
            'running_and_omissions',
            'pupil_and_omissions',
            'pupil_and_running',
            'expectation',
            'face_motion_PC_1',
            'face_motion_PC_2',
            'face_motion_PC_3',
            'face_motion_PC_4',
            'face_motion_PC_0',
        ]

        if not self.dropout_summary_plotted and len(self.glm.dropout_summary['dropout'].unique()) > 1:
            try:
                plot_dropout_summary(
                    self.results_summary, 
                    self.cell_specimen_id, 
                    ax['dropout_summary'], 
                    dropouts_to_plot='standard',
                    dropouts_to_exclude = dropouts_to_exclude
                )
            except Exception as e:
                # this fails if all dropouts are not defined
                warnings.warn('Failed to plot dropout summary')
            self.dropout_summary_plotted = True

        if self.verbose:
            print('done plotting dropout summary at {:0.2f} seconds'.format(time.time() - ti))

        for axis_name in ax.keys():
            if axis_name != 'dropout_summary' and axis_name != 'cell_roi':
                ax[axis_name].cla()

        if self.verbose:
            print('done clearing axes at {:0.2f} seconds'.format(time.time() - ti))
        
        # 2P ROI images:
        if not self.cell_roi_plotted:
            cell_roi_id = gat.retrieve_results(
                {
                    'ophys_experiment_id':self.glm.ophys_experiment_id, 
                    'cell_specimen_id':self.cell_specimen_id, 
                    'glm_version':self.glm.version
                }, 
                results_type='full'
            )['cell_roi_id'][0]
            if cell_specimen_id in glm.session.cell_specimen_table.index.tolist():
                print('FOUND CELL SPECIMEN ID')
                self.com = ndimage.measurements.center_of_mass(glm.session.cell_specimen_table.loc[cell_specimen_id]['roi_mask'])
            elif cell_roi_id in glm.session.cell_specimen_table['cell_roi_id'].tolist():
                print('FOUND CELL ROI ID')
                correct_csid= self.glm.session.cell_specimen_table.query('cell_roi_id == @cell_roi_id').index[0]
                # self.com = ndimage.measurements.center_of_mass(glm.session.cell_specimen_table.query('cell_roi_id == @cell_roi_id')['roi_mask'].values[0])
                self.com = ndimage.measurements.center_of_mass(glm.session.cell_specimen_table.loc[cell_specimen_id]['roi_mask'])
                print(self.com)
            else:
                print('COULD NOT FIND CELL')
                self.com = None
            self.cell_roi_plotted = True

        if self.verbose:
            print('done getting roi info at {:0.2f} seconds'.format(time.time() - ti))

        for movie_name in ['behavior','eye']:
            frame = self.tracking_movies[movie_name].get_frame(time=t_now)[:,:,0]
            ax['{}_movie'.format(movie_name)].imshow(
                frame ,
                cmap='gray'
            )
            ax['{}_movie'.format(movie_name)].axis('off')
            ax['{}_movie'.format(movie_name)].set_title('{} tracking movie'.format(movie_name), fontsize=22)

        if self.verbose:
            print('done plotting behavior videos at {:0.2f} seconds'.format(time.time() - ti))

        # make a crude approximation of pixelwise df/f 
        frame_2p = self.this_cell.query('fit_trace_timestamps >= @t_now')['frame_index'].iloc[0]
        f0 = self.real_2p_movie[frame_2p - 100:frame_2p + 100, :, :].mean(axis=0)
        f = self.real_2p_movie[frame_2p - 3:frame_2p + 3, :, :].mean(axis=0)
        dff = (f-f0)/f0
        dff[pd.isnull(dff)]=0
        cmax = np.percentile(dff, 95) #set cmax to 95th percentile of this image
        ax['real_fov'].imshow(dff[10:-10,10:-10], cmap='gray', clim=[0, cmax])

        ax['real_fov'].set_title('2P Field of View', fontsize=22)

        for axis_name in ['real_fov']: #,'reconstructed_fov','simulated_fov']:
            ax[axis_name].set_xticks([])
            ax[axis_name].set_yticks([])
            # if self.com:
            #     ax[axis_name].axvline(self.com[1],color='MediumAquamarine',alpha=0.5)
            #     ax[axis_name].axhline(self.com[0],color='MediumAquamarine',alpha=0.5)

        if self.verbose:
            print('done plotting 2P FOV at {:0.2f} seconds'.format(time.time() - ti))

        # time series plots:
        query_string = 'fit_trace_timestamps >= {} and fit_trace_timestamps <= {}'.format(
            t_span[0],
            t_span[1]
        )
        local_df = self.this_cell.query(query_string)

        vbp.initialize_legend(
            ax=ax['dff'], 
            colors=[self.dff_color, self.events_color, self.fit_color],
            linewidth=3
        )

        ax['dff'].plot(
            local_df['fit_trace_timestamps'],
            local_df['dff'],
            alpha=0.9,
            color=self.dff_color,
            linewidth=3,
        )

        ax['events'].plot(
            local_df['fit_trace_timestamps'],
            local_df['events'],
            alpha=0.9,
            color=self.events_color,
            linewidth=3,
        )

        ax[self.parameter_to_fit].plot(
            local_df['fit_trace_timestamps'],
            local_df['model_prediction'],
            alpha=1,
            color=self.fit_color,
            linewidth=3,
        )

        qs = 'fit_trace_timestamps >= {} and fit_trace_timestamps <= {}'.format(
            self.initial_time - t_before,
            self.final_time + t_after
        )
        ax['dff'].set_ylim(
            self.this_cell.query(qs)['dff'].min() - 0.01,
            self.this_cell.query(qs)['dff'].max() + 0.01,
        )

        ax['dff'].legend(
            ['measured $\Delta$F/F','events', 'model fit'],
            loc='upper left',
            ncol=1, 
            framealpha = 0.2,
        )

        if self.verbose:
            print('done plotting Fluorscence timeseries at {:0.2f} seconds'.format(time.time() - ti))

        plot_rewards(glm.session, ax['licks'], t_span=t_span)
        plot_licks(glm.session, ax['licks'], t_span=t_span)

        if self.verbose:
            print('done plotting rewards and licks at {:0.2f} seconds'.format(time.time() - ti))
        
        query_string = 'timestamps >= {} and timestamps <= {}'.format(
            self.initial_time - t_before,
            self.final_time + t_after
        )
        plot_running(glm.session, ax['running'], t_span=t_span)
        # set running y lims
        ax['running'].set_ylim(
            self.glm.session.running_speed.query(query_string)['speed'].min() - 5,
            self.glm.session.running_speed.query(query_string)['speed'].max() + 5
        )
        plot_pupil(glm.session, ax['pupil'], t_span=t_span)
        # set pupil ylims
        pupil_query = self.glm.session.eye_tracking.query(query_string)
        ax['pupil'].set_ylim(
            pupil_query['pupil_area'].min() - 100,
            pupil_query['pupil_area'].max() + 100,
        )

        if self.verbose:
            print('done plotting pupil and running at {:0.2f} seconds'.format(time.time() - ti))

        plot_kernels(self.kernel_df, ax['kernel_contributions'], self.palette_df, t_span)

        # set limits on kernel plot

        ax['kernel_contributions'].set_ylim(
            self.kernel_df.query(query_string)['kernel_outputs'].min() - 0.05,
            self.kernel_df.query(query_string)['kernel_outputs'].max() + 0.05
        )

        if self.verbose:
            print('done plotting kernels at {:0.2f} seconds'.format(time.time() - ti))

        # some axis formatting: 
        for axis_name in ['licks', 'dff', 'running','kernel_contributions']:
            ax[axis_name].axvline(t_now, color='white', linewidth=3, alpha=0.5)
            plot_stimuli(self.stimulus_presentations, ax[axis_name], t_span=t_span)
            if axis_name != 'kernel_contributions':
                ax[axis_name].set_xticklabels([])

        if self.verbose:
            print('done plotting stimulus spans at {:0.2f} seconds'.format(time.time() - ti))

        ax['dff'].set_title('Time series plots for cell {}'.format(cell_specimen_id), fontsize=22)
        ax['licks'].set_xlim(t_span[0], t_span[1])
        ax['licks'].set_yticks([])

        ax['dff'].set_xticklabels('')

        ax['licks'].set_xlabel('time')

        ax['licks'].set_ylabel('licks/rewards       ', rotation=0,ha='right', va='center')
        ax['dff'].set_ylabel('$\Delta$F/F', rotation=0, ha='right', va='center')
        ax['events'].set_ylabel('event\nmagnitude', rotation=0, ha='left', va='center')
        ax['running'].set_ylabel('Running\nSpeed\n(cm/s)', rotation=0, ha='right', va='center')
        ax['pupil'].set_ylabel('Pupil\nArea\n(pix^2)', rotation=0, ha='left', va='center')
        ax['kernel_contributions'].set_ylabel('kernel\ncontributions\nto predicted\nsignal\n($\Delta$F/F)', rotation=0, ha='right', va='center')
        ax['kernel_contributions'].set_xlabel('time (s)', fontsize=20)

        if self.verbose:
            print('done axis formatting at {:0.2f} seconds'.format(time.time() - ti))

    def update(self, frame_number):
        '''
        method to update figure
        animation class will call this

        the print statement is there to help track progress
        '''
        self.make_cell_movie_frame(
            self.ax, self.glm, F_index=frame_number, cell_specimen_id=self.cell_specimen_id)

        self.pbar.update(1)
        gc.collect()

    def set_up_axes(self):
        fig = plt.figure(figsize=(24, 14))
        ax = {
            'real_fov': vbp.placeAxesOnGrid(fig, xspan=(0.3, 0.49), yspan=(0, 0.25)),
            'behavior_movie': vbp.placeAxesOnGrid(fig, xspan=(0.5, 0.75), yspan=(0, 0.25)),
            'eye_movie': vbp.placeAxesOnGrid(fig, xspan=(0.75, 1), yspan=(0, 0.25)),
            'dropout_summary':vbp.placeAxesOnGrid(fig, xspan=[0.05,0.18], yspan=[0,1]),
            'dff': vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.30, 0.5]),
            'licks': vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.5, 0.525]),
            'running': vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.525, 0.625]),
            'kernel_contributions':vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.625, 1]),
            
        }
        ax['pupil'] = ax['running'].twinx()
        ax['events'] = ax['dff'].twinx()

        ax['licks'].get_shared_x_axes().join(ax['licks'], ax['dff'])
        ax['running'].get_shared_x_axes().join(ax['running'], ax['dff'])
        ax['events'].get_shared_x_axes().join(ax['events'], ax['dff'])
        ax['kernel_contributions'].get_shared_x_axes().join(ax['kernel_contributions'], ax['dff'])

        variance_explained_string = 'Variance explained (full model) = {:0.1f}%'.format(100*self.glm.results.loc[self.cell_specimen_id]['Full__avg_cv_var_test'])
        fig.suptitle(self.title+'\n'+variance_explained_string, fontsize=18)

        return fig, ax

    def set_up_writer(self):

        writer = animation.FFMpegWriter(
            fps=self.fps,
            codec='mpeg4',
            bitrate=-1,
            extra_args=['-pix_fmt', 'yuv420p', '-q:v', '5']
        )
        return writer

    def make_movie(self):
        self.dropout_summary_plotted = False
        self.cell_roi_plotted = False

        a = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=1/self.fps*1000,
            repeat=False,
            blit=False
        )

        filename = self.title+'_frame_{}_to_{}.mp4'.format(self.start_frame, self.end_frame)

        with tqdm(total=len(self.frames)) as self.pbar:
            a.save(
                os.path.join(self.destination_folder, filename),
                writer=self.writer
            )





def plot_all_kernel_comparison(weights_df, run_params, drop_threshold=0,session_filter=[1,2,3,4,5,6],equipment_filter="all",depth_filter=[0,1000],cell_filter="all",area_filter=['VISp','VISl'],compare=['cre_line'],plot_errors=False):  ##TODO update
    '''
        Generated kernel comparison plots for all dropouts
        weights_df, dataframe of kernels
        run_params, run json of model version
    '''
    
    # Keep track of what is failing 
    fail = []

    # Set up which sessions to plot
    active_only  = ['licks','hits','misses']
    passive_only = ['passive_change']
 
    # Iterate over list of dropouts
    for kernel in run_params['kernels']:
        if kernel in ['intercept','time']:
            continue

        # Determine which sessions to plot
        if kernel in active_only:
            session_filter = [1,3,4,6]
        elif kernel in passive_only:
            session_filter = [2,5]
        else:
            session_filter = [1,2,3,4,5,6]

        try:
            # plot the coding fraction
            filepath = run_params['fig_kernels_dir']
            plot_kernel_comparison(weights_df, run_params, kernel, drop_threshold=drop_threshold, session_filter=session_filter, equipment_filter=equipment_filter, depth_filter=depth_filter, cell_filter=cell_filter, area_filter=area_filter, compare=compare, plot_errors=plot_errors)
        except Exception as e:
            print(e)
            # Track failures
            fail.append(kernel)
    
        # Close figure
        plt.close(plt.gcf().number)
    
    # Report failures
    if len(fail) > 0:
        print('The following kernels failed')
        print(fail) 

def plot_compare_across_kernels(weights_df, run_params, kernels,session_filter=[1,2,3,4,5,6], equipment_filter="all",cell_filter="all", compare=[],area_filter=['VISp','VISl'],title=None): #TODO
    '''
        compare multiple different kernels to each other 
    ''' 

    if 'dropout_threshold' in run_params:
        threshold = run_params['dropout_threshold']
    else:
        threshold = 0.005

    version = run_params['version']
    filter_string = ''
    problem_sessions = get_problem_sessions() 

    # Filter by Equipment
    equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"]
    if equipment_filter == "scientifica": 
        equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5"]
        filter_string += '_scientifica'
    elif equipment_filter == "mesoscope":
        equipment_list = ["MESO.1"]
        filter_string += '_mesoscope'
    
    # Filter by Cell Type    
    cell_list = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre']     
    if cell_filter == "sst":
        cell_list = ['Sst-IRES-Cre']
        filter_string += '_sst'
    elif cell_filter == "vip":
        cell_list = ['Vip-IRES-Cre']
        filter_string += '_vip'
    elif cell_filter == "slc":
        cell_list = ['Slc17a7-IRES2-Cre']
        filter_string += '_slc'

    # Determine filename
    if session_filter != [1,2,3,4,5,6]:
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if area_filter != ['VISp','VISl']:
        filter_string+='_area_'+'_'.join(area_filter)
    if title is None:
        title=filter_string   
 
    weights = weights_df.query('(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&(session_number in @session_filter) & (ophys_session_id not in @problem_sessions) & (variance_explained_full > @threshold)').copy()

    # Set up time vectors.
    time_vec = np.arange(run_params['kernels'][kernels[0]]['offset'], run_params['kernels'][kernels[0]]['offset'] + run_params['kernels'][kernels[0]]['length'],1/31)
    time_vec = np.round(time_vec,2)

    # Plotting settings
    fig,ax=plt.subplots(figsize=(8,4))
    
    # Define color scheme for project
    colors = project_colors()

    # Define linestyles
    lines = {
        0:'-',
        1:'--',
        2:':',
        3:'-.',
        4:(0,(1,10)),
        5:(0,(5,10))
        }

    # Filter for this group, and plot
    kernel_weights = [x+'_weights' for x in kernels]
 
    # Determine unique groups of cells by the categorical attributes in compare
    if len(compare) == 0:
        weights_dfiltered = weights[kernel_weights]
        plot_compare_across_kernels_inner(ax,weights_dfiltered,kernel_weights,'', 'k',lines, time_vec) 
    else:
        groups = list(weights.groupby(compare).groups.keys())
  
        # Iterate over groups of cells
        for dex,group in enumerate(groups):
            # Build color, linestyle, and query string for this group
            query_str = '({0} == @group)'.format(compare[0])
            color = colors.setdefault(group,(100/255,100/255,100/255)) 
            weights_dfiltered = weights.query(query_str)[kernel_weights].copy()
            plot_compare_across_kernels_inner(ax,weights_dfiltered,kernel_weights,group, color,lines, time_vec) 

    # Clean Plot, and add details
    ax.axhline(0, color='k',linestyle='--',alpha=0.25)
    ax.axvline(0, color='k',linestyle='--',alpha=0.25)
    ax.set_ylabel('Kernel Weights ($\Delta$f/f)',fontsize=18)   
    ax.set_xlabel('Time (s)',fontsize=18)
    ax.set_xlim(time_vec[0],time_vec[-1])   
    add_stimulus_bars(ax,kernels[0],alpha=.1)
    plt.tick_params(axis='both',labelsize=16)
    plt.legend(loc='upper right',title=' & '.join(compare),handlelength=4)

    plt.title(title)
    plt.tight_layout()

def plot_compare_across_kernels_inner(ax, df,kernels,group,color,linestyles,time_vec,linewidth=4):
    '''
        Plots the average kernel for the cells in df
        
        ax, the axis to plot on
        df, series of cells with column that is the kernel to plot
        label, what to label this group of cells
        color, the line color for this group of cells
        linestyle, the line style for this group of cells
        time_vec, the time basis to plot on
        linewidth, the width of the mean line
    '''

    # Normalize kernels, and interpolate to time_vec
    df = df.dropna(axis=0,subset=kernels)

    mean_kernels = {}
    for k in kernels:
        mean_kernels[k] = np.mean(np.vstack(df[k]),0)
   
    # Plot mean and error bar
    for dex,k in enumerate(kernels):
        ax.plot(time_vec, mean_kernels[k],linestyle=linestyles[dex],color=color,label=group+' '+k,linewidth=linewidth)

## TODO update
def plot_perturbation(weights_df, run_params, kernel, drop_threshold=0,session_filter=[1,2,3,4,5,6],equipment_filter="all",depth_filter=[0,1000],cell_filter="all",area_filter=['VISp','VISl'],normalize=True,CMAP='Blues',in_ax=None):

    if 'dropout_threshold' in run_params:
        threshold = run_params['dropout_threshold']
    else:
        threshold = 0.005

    filter_string = ''
    problem_sessions = get_problem_sessions()
 
    # Filter by Equipment
    equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"]
    if equipment_filter == "scientifica": 
        equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5"]
        filter_string += '_scientifica'
    elif equipment_filter == "mesoscope":
        equipment_list = ["MESO.1"]
        filter_string += '_mesoscope'
    
    # Filter by Cell Type    
    cell_list = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre']     
    if cell_filter == "sst":
        cell_list = ['Sst-IRES-Cre']
        filter_string += '_sst'
    elif cell_filter == "vip":
        cell_list = ['Vip-IRES-Cre']
        filter_string += '_vip'
    elif cell_filter == "slc":
        cell_list = ['Slc17a7-IRES2-Cre']
        filter_string += '_slc'

    # Determine filename
    if session_filter != [1,2,3,4,5,6]:
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if depth_filter !=[0,1000]:
        filter_string+='_depth_'+str(depth_filter[0])+'_'+str(depth_filter[1])
    if area_filter != ['VISp','VISl']:
        filter_string+='_area_'+'_'.join(area_filter)
    filename2 = os.path.join(run_params['fig_kernels_dir'],kernel+filter_string+'_perturbation_validation.png')
    filename1 = os.path.join(run_params['fig_kernels_dir'],kernel+filter_string+'_perturbation.png')

    # Applying hard thresholds to dataset
    weights = weights_df.query('(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&(session_number in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))

    # Set up time vectors.
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)
    meso_time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/11)#1/10.725)
    groups = list(weights.groupby(['cre_line']).groups.keys())
    
    kernel_means = {}
    for dex,group in enumerate(groups):
        query_str = 'cre_line == @group'
        df = weights.query(query_str)[kernel+'_weights']
        if normalize:
            df_norm = [x/np.max(np.abs(x)) for x in df[~df.isnull()].values]
        else:
            df_norm = [x for x in df[~df.isnull()].values]
        df_norm = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in df_norm]
        df_norm = np.vstack(df_norm)
        kernel_means[group]=df_norm.mean(axis=0)     
    
    mids_range = range(int(np.floor(time_vec[0]/0.75)),int(np.floor(time_vec[-1]/0.75))+1)
    mids = [x*0.75+0.75/2 for x in mids_range] 
    print(mids)
    times = {}
    for flash_num in mids_range:
        times[flash_num] = (time_vec >= 0.75*flash_num) & (time_vec < 0.75*(flash_num+1))
    
    avg_vals = pd.DataFrame({'Slc17a7-IRES2-Cre':[], 'Sst-IRES-Cre':[],'Vip-IRES-Cre':[]})
    for dex, group in enumerate(groups):
        avg_vals[group] = [np.mean(kernel_means[group][times[x]]) for x in mids_range] 

    avg_vals['vip-sst'] = avg_vals['Vip-IRES-Cre']-avg_vals['Sst-IRES-Cre']
    colors = project_colors()   


    plt.figure()
    for k in kernel_means.keys():
        plt.plot(time_vec, kernel_means[k],  color=colors[k])
        plt.plot(mids,avg_vals[k],'o',color=colors[k])
    plt.ylabel(kernel,fontsize=16)
    plt.xlabel('Time',fontsize=16)
    plt.tick_params(axis='both',labelsize=12)
    plt.tight_layout()
    print('Figure Saved to: '+filename2)
    plt.savefig(filename2) 

    if in_ax is None:
        plt.figure()
        ax = plt.gca()
    else:
        ax = in_ax 
    
    #cmap = plt.get_cmap('tab20c')
    cmap = plt.get_cmap(CMAP)(np.linspace(0.3,1,len(mids_range)))
    
    for dex,flash_num in enumerate(mids_range):
        if flash_num == 0:
            label = kernel
        elif flash_num < 0:
            label = 'Pre '+kernel+' '+str(flash_num)          
        else:
            label = 'Post '+kernel+' '+str(flash_num)
        ax.plot(avg_vals.loc[dex]['vip-sst'],avg_vals.loc[dex]['Slc17a7-IRES2-Cre'],'o',color=cmap[dex],markersize=10,label=label)

    for dex, flash_num in enumerate(mids_range):
        if dex == 0:
            startxy = [0,0]
        else:
            startxy=[avg_vals.loc[dex-1]['vip-sst'],avg_vals.loc[dex-1]['Slc17a7-IRES2-Cre']]

        endxy  =[avg_vals.loc[dex]['vip-sst'],avg_vals.loc[dex]['Slc17a7-IRES2-Cre']]
        dx = (startxy[0]-endxy[0])*0.1
        dy = (startxy[1]-endxy[1])*0.1
        sxy =(startxy[0]-dx,startxy[1]-dy)
        exy =(endxy[0]+dx,endxy[1]+dy)
        ax.annotate("",xy=exy,xytext=sxy,arrowprops=dict(arrowstyle="->",color=cmap[dex]))
   
    plt.legend(loc='upper left')
    ylim = plt.ylim()
    xlim = plt.xlim()
    yrange = ylim[1]-ylim[0]
    xrange = xlim[1]-xlim[0]
    ax.set_ylim(ylim[0]-.25*yrange, ylim[1]+.25*yrange)
    ax.set_xlim(xlim[0]-.25*xrange, xlim[1]+.25*xrange)
    ax.set_ylabel('Excitatory',fontsize=16)
    ax.set_xlabel('VIP - SST',fontsize=16)
    plt.tick_params(axis='both',labelsize=12)
    plt.tight_layout()
    print('Figure Saved to: '+filename1)
    plt.savefig(filename1)

    plt.figure()
    plt.plot(kernel_means['Vip-IRES-Cre']-kernel_means['Sst-IRES-Cre'],kernel_means['Slc17a7-IRES2-Cre'])
    plt.ylabel('Excitatory',fontsize=16)
    plt.xlabel('VIP - SST',fontsize=16)
    plt.tick_params(axis='both',labelsize=12)
    plt.tight_layout()
   
    return ax,kernel_means

def plot_kernel_comparison_by_experience(weights_df, run_params, kernel,threshold=0,drop_threshold=0,savefig=False):
    weights_df = weights_df.query('not passive').copy()
    
    extra=''
    if threshold !=0:
        extra=extra+'_threshold_'+str(threshold)
    if drop_threshold !=0:
        extra=extra+'_drop_threshold_'+str(drop_threshold)
    k, fig_f , ax_f = plot_kernel_comparison(weights_df, run_params, kernel, save_results=False, session_filter=['Familiar'],threshold=threshold, drop_threshold=drop_threshold) 
    k, fig_n , ax_n = plot_kernel_comparison(weights_df, run_params, kernel, save_results=False, session_filter=['Novel 1'],threshold=threshold, drop_threshold=drop_threshold) 
    k, fig_np, ax_np =plot_kernel_comparison(weights_df, run_params, kernel, save_results=False, session_filter=['Novel >1'],threshold=threshold, drop_threshold=drop_threshold) 

 
    ylims = list(ax_f.get_ylim()) +  list(ax_n.get_ylim()) +  list(ax_np.get_ylim()) 
    new_y = [np.min(ylims), np.max(ylims)]
    ax_f.set_ylim(new_y)
    ax_n.set_ylim(new_y)
    ax_np.set_ylim(new_y)
    if savefig:
        fig_f.savefig(run_params['fig_kernels_dir']+'/'+kernel+'_familiar_kernel'+extra+'.svg')
        fig_n.savefig(run_params['fig_kernels_dir']+'/'+kernel+'_novel1_kernel'+extra+'.svg')
        fig_np.savefig(run_params['fig_kernels_dir']+'/'+kernel+'_novelp1_kernel'+extra+'.svg')
        print('Figure saved to: '+run_params['fig_kernels_dir']+'/'+kernel+'_familiar_kernel'+extra+'.svg')
        print('Figure saved to: '+run_params['fig_kernels_dir']+'/'+kernel+'_novel1_kernel'+extra+'.svg')
        print('Figure saved to: '+run_params['fig_kernels_dir']+'/'+kernel+'_novelp1_kernel'+extra+'.svg')

    k, fig_v , ax_v =plot_kernel_comparison(weights_df,run_params,kernel,save_results=False,session_filter=['Familiar','Novel 1','Novel >1'],cell_filter='Vip-IRES-Cre',compare=['experience_level'],threshold=threshold,drop_threshold=drop_threshold)   
    k, fig_s , ax_s =plot_kernel_comparison(weights_df,run_params,kernel,save_results=False,session_filter=['Familiar','Novel 1','Novel >1'],cell_filter='Sst-IRES-Cre',compare=['experience_level'],threshold=threshold,drop_threshold=drop_threshold)   
    k, fig_e , ax_e =plot_kernel_comparison(weights_df,run_params,kernel,save_results=False,session_filter=['Familiar','Novel 1','Novel >1'],cell_filter='Slc17a7-IRES2-Cre',compare=['experience_level'],threshold=threshold,drop_threshold=drop_threshold)   
    if savefig:
        fig_v.savefig(run_params['fig_kernels_dir']+'/'+kernel+'_vip_kernel'+extra+'.svg')
        fig_s.savefig(run_params['fig_kernels_dir']+'/'+kernel+'_sst_kernel'+extra+'.svg')
        fig_e.savefig(run_params['fig_kernels_dir']+'/'+kernel+'_exc_kernel'+extra+'.svg')
        print('Figure saved to: '+run_params['fig_kernels_dir']+'/'+kernel+'_vip_kernel'+extra+'.svg')
        print('Figure saved to: '+run_params['fig_kernels_dir']+'/'+kernel+'_sst_kernel'+extra+'.svg')
        print('Figure saved to: '+run_params['fig_kernels_dir']+'/'+kernel+'_exc_kernel'+extra+'.svg')

def plot_kernel_comparison_by_kernel_excitation(weights_df, run_params,kernel,savefig=False):
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory'
        }
    nk = kernel.replace('all-','')

    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Familiar'],cell_filter='Slc17a7-IRES2-Cre',compare=[kernel+'_excited'],set_title='Excitatory, Familiar, '+nk,save_results=savefig)
    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Novel 1'],cell_filter='Slc17a7-IRES2-Cre',compare=[kernel+'_excited'], set_title='Excitatory, Novel 1, '+nk,save_results=savefig)
    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Novel >1'],cell_filter='Slc17a7-IRES2-Cre',compare=[kernel+'_excited'],set_title='Excitatory, Novel >1, '+nk,save_results=savefig)

    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Familiar'],cell_filter='Sst-IRES-Cre',compare=[kernel+'_excited'],set_title='Sst Inhibitory, Familiar, '+nk,save_results=savefig)
    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Novel 1'],cell_filter='Sst-IRES-Cre',compare=[kernel+'_excited'], set_title='Sst Inhibitory, Novel 1, '+nk,save_results=savefig)
    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Novel >1'],cell_filter='Sst-IRES-Cre',compare=[kernel+'_excited'],set_title='Sst Inhibitory, Novel >1, '+nk,save_results=savefig)

    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Familiar'],cell_filter='Vip-IRES-Cre',compare=[kernel+'_excited'],set_title='Vip Inhibitory, Familiar, '+nk,save_results=savefig)
    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Novel 1'],cell_filter='Vip-IRES-Cre',compare=[kernel+'_excited'], set_title='Vip Inhibitory, Novel 1, '+nk,save_results=savefig)
    plot_kernel_comparison(weights_df,run_params,kernel,session_filter=['Novel >1'],cell_filter='Vip-IRES-Cre',compare=[kernel+'_excited'],set_title='Vip Inhibitory, Novel >1, '+nk,save_results=savefig)

def plot_kernel_comparison(weights_df, run_params, kernel, save_results=True, drop_threshold=0,session_filter=['Familiar','Novel 1','Novel >1'],equipment_filter="all",depth_filter=[0,1000],cell_filter="all",area_filter=['VISp','VISl'],compare=['cre_line'],plot_errors=False,save_kernels=False,fig=None, ax=None,fs1=20,fs2=16,show_legend=True,filter_sessions_on='experience_level',image_set=['familiar','novel'],threshold=0,set_title=None): 
    '''
        Plots the average kernel across different comparisons groups of cells
        First applies hard filters, then compares across remaining cells

        INPUTS:
        run_params              = glm_params.load_run_params(<version>) 
        results_pivoted         = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
        weights_df              = gat.build_weights_df(run_params, results_pivoted)
        kernel                  The name of the kernel to be plotted
        save_results            if True, saves a figure to the directory in run_params['output_dir']
        drop_threshold,         the minimum adj_fraction_change_from_full for the dropout model of just dropping this kernel
        session_filter,         The list of session numbers to include
        equipment_filter,       "scientifica" or "mesoscope" filter, anything else plots both 
        cell_filter,            "sst","vip","slc", anything else plots all types
        area_filter,            the list of targeted_structures to include
        compare (list of str)   list of categorical labels in weights_df to split on and compare
                                First entry of compare determines color of the line, second entry determines linestyle
        plot_errors (bool)      if True, plots a shaded error bar for each group of cells
    
    '''
    version = run_params['version']
    filter_string = ''
    problem_sessions = get_problem_sessions()   
    weights_df = weights_df.copy()
    
    #if 'dropout_threshold' in run_params:
    #    threshold = run_params['dropout_threshold']
    #else:
    #    threshold = 0.005
    #threshold = 0
 
    # Filter by Equipment
    equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"]
    if equipment_filter == "scientifica": 
        equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5"]
        filter_string += '_scientifica'
    elif equipment_filter == "mesoscope":
        equipment_list = ["MESO.1"]
        filter_string += '_mesoscope'
    
    # Filter by Cell Type    
    cell_list = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre']     
    if (cell_filter == "sst") or (cell_filter == "Sst-IRES-Cre"):
        cell_list = ['Sst-IRES-Cre']
        filter_string += '_sst'
    elif (cell_filter == "vip") or (cell_filter == "Vip-IRES-Cre"):
        cell_list = ['Vip-IRES-Cre']
        filter_string += '_vip'
    elif (cell_filter == "slc") or (cell_filter == "Slc17a7-IRES2-Cre"):
        cell_list = ['Slc17a7-IRES2-Cre']
        filter_string += '_slc'

    # Determine filename
    if session_filter != [1,2,3,4,5,6]:
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if depth_filter !=[0,1000]:
        filter_string+='_depth_'+str(depth_filter[0])+'_'+str(depth_filter[1])
    if area_filter != ['VISp','VISl']:
        filter_string+='_area_'+'_'.join(area_filter)
    filename = os.path.join(run_params['fig_kernels_dir'],kernel+'_comparison_by_'+'_and_'.join(compare)+filter_string+'.svg')

    # Set up time vectors.
    if kernel in ['preferred_image', 'all-images']:
        run_params['kernels'][kernel] = run_params['kernels']['image0'].copy()
    if kernel == 'all-omissions':
        run_params['kernels'][kernel] = run_params['kernels']['omissions'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['omissions']['length'] + run_params['kernels']['post-omissions']['length']
    if kernel == 'all-hits':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['hits']['length'] + run_params['kernels']['post-hits']['length']   
    if kernel == 'all-misses':
        run_params['kernels'][kernel] = run_params['kernels']['misses'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['misses']['length'] + run_params['kernels']['post-misses']['length']   
    if kernel == 'all-passive_change':
        run_params['kernels'][kernel] = run_params['kernels']['passive_change'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['passive_change']['length'] + run_params['kernels']['post-passive_change']['length']   
    if kernel == 'task':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()   
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2) 
    if 'image' in kernel:
        time_vec = time_vec[:-1]
    if ('omissions' == kernel) & ('post-omissions' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('hits' == kernel) & ('post-hits' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('misses' == kernel) & ('post-misses' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('passive_change' == kernel) & ('post-passive_change' in run_params['kernels']):
        time_vec = time_vec[:-1]
 
    if '-' in kernel:
        weights_df= weights_df.rename(columns={
            'all-omissions':'all_omissions',
            'all-omissions_weights':'all_omissions_weights',
            'post-omissions':'post_omissions',
            'post-omissions_weights':'post_omissions_weights',
            'all-hits':'all_hits',
            'all-hits_weights':'all_hits_weights',
            'post-hits':'post_hits',
            'post-hits_weights':'post_hits_weights', 
            'all-misses':'all_misses',
            'all-misses_weights':'all_misses_weights',
            'post-misses':'post_misses',
            'post-misses_weights':'post_misses_weights', 
            'all-passive_change':'all_passive_change',
            'all-passive_change_weights':'all_passive_change_weights',
            'post-passive_change':'post_passive_change',
            'post-passive_change_weights':'post_passive_change_weights',  
            'all-images':'all_images',
            'all-images_weights':'all_images_weights',
            'all-images_excited':'all_images_excited'
            })
        kernel = kernel.replace('-','_')
    compare = [x.replace('-','_') for x in compare]
        

    # Applying hard thresholds to dataset
    if kernel in weights_df:
        weights = weights_df.query('(not passive)&(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&({0} in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > @threshold) & ({1} <= @drop_threshold)'.format(filter_sessions_on, kernel))
        use_dropouts=True
    else:
        weights = weights_df.query('(not passive)&(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&({0} in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > @threshold)'.format(filter_sessions_on))
        print('Dropouts not included, cannot use drop filter')
        use_dropouts=False

    # Plotting settings
    if ax is None:
        #fig,ax=plt.subplots(figsize=(8,4))
        height = 4
        width=8
        pre_horz_offset = 1.5
        post_horz_offset = 2.5
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(pre_horz_offset),Size.Fixed(width-pre_horz_offset-post_horz_offset)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  
    
    # Define color scheme for project
    colors = project_colors()

    # Define linestyles
    lines = {
        0:'-',
        1:'--',
        2:':',
        3:'-.',
        4:(0,(1,10)),
        5:(0,(5,10))
        }

    # Determine unique groups of cells by the categorical attributes in compare
    groups = list(weights.groupby(compare).groups.keys())
    if len(compare) >1:
        # Determine number of 2nd level attributes for linestyle definitions 
        num_2nd = len(list(weights[compare[1]].unique()))
   
    outputs={}
    # Iterate over groups of cells
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory',
        'Familiar':'Familiar',
        'Novel 1':'Novel',
        'Novel >1':'Novel +'
        }
    for dex,group in enumerate(groups):

        # Build color, linestyle, and query string for this group
        if len(compare) ==1:
            query_str = '({0} == @group)'.format(compare[0])
            linestyle = '-'
            color = colors.setdefault(group,(100/255,100/255,100/255)) 
        else:
            query_str = '&'.join(['('+x[0]+'==\"'+x[1]+'\")' for x in zip(compare,group)])
            linestyle = lines.setdefault(np.mod(dex,num_2nd),'-')
            color = colors.setdefault(group[0],(100/255,100/255,100/255)) 
    
        # Filter for this group, and plot
        weights_dfiltered = weights.query(query_str)[kernel+'_weights']
        k=plot_kernel_comparison_inner(ax,weights_dfiltered,mapper[group], color,linestyle, time_vec, plot_errors=plot_errors) 
        outputs[group]=k

    # Clean Plot, and add details
    if set_title is not None:
        plt.title(set_title, fontsize=fs1)
    else:
        mapper = {
            'Slc17a7-IRES2-Cre':'Excitatory',
            'Sst-IRES-Cre':'Sst Inhibitory',
            'Vip-IRES-Cre':'Vip Inhibitory',
            'Familiar':'Familiar',
            'Novel 1':'Novel',
            'Novel >1':'Novel +'
            }

        if len(session_filter) > 1:
            session_title=cell_filter
            session_title=mapper[session_title]
        else:
            session_title = mapper[session_filter[0]]
 
        #plt.title(run_params['version']+'\n'+kernel+' '+cell_filter+' '+session_title)
        plt.title(kernel+' kernels, '+session_title,fontsize=fs1)
    ax.axhline(0, color='k',linestyle='--',alpha=0.25)
    #ax.axvline(0, color='k',linestyle='--',alpha=0.25)
    ax.set_ylabel('Kernel Weights',fontsize=fs1)      
    if kernel == 'omissions':
        ax.set_xlabel('Time from omission (s)',fontsize=fs1)
    elif kernel in ['hits','misses']:
        ax.set_xlabel('Time from image change (s)',fontsize=fs1)
    else:
        ax.set_xlabel('Time (s)',fontsize=fs1)

    ax.set_xlim(time_vec[0]-0.05,time_vec[-1])  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_stimulus_bars(ax,kernel,alpha=.1)
    plt.tick_params(axis='both',labelsize=fs2)
    if show_legend:
        ax.legend(loc='upper left',bbox_to_anchor=(1.05,1),title=' & '.join(compare).replace('_',' '),handlelength=4)


 
    ## Final Clean up and Save
    #plt.tight_layout()
    if save_results:
        print('Figure Saved to: '+filename)
        plt.savefig(filename) 
    if save_kernels:
        outputs['time'] = time_vec
        filename2 = os.path.join(run_params['fig_kernels_dir'],kernel+'_comparison_by_'+'_and_'.join(compare)+filter_string+'.pkl')
        print('Kernels Saved to: '+filename2)
        file_temp = open(filename2,'wb')
        pickle.dump(outputs, file_temp)
        file_temp.close()
    return outputs, fig,ax

def plot_kernel_comparison_inner(ax, df,label,color,linestyle,time_vec, plot_errors=True,linewidth=4,alpha=.25):
    '''
        Plots the average kernel for the cells in df
        
        ax, the axis to plot on
        df, series of cells with column that is the kernel to plot
        label, what to label this group of cells
        color, the line color for this group of cells
        linestyle, the line style for this group of cells
        time_vec, the time basis to plot on
        meso_time_vec, the time basis for mesoscope kernels (will be interpolated to time_vec)
        plot_errors (bool), if True, plots a shaded error bar
        linewidth, the width of the mean line
        alpha, the alpha for the shaded error bar
    '''

    # Normalize kernels, and interpolate to time_vec
    df_norm = [x for x in df[~df.isnull()].values]
    
    # Needed for stability
    if len(df_norm)>0:
        df_norm = np.vstack(df_norm)
    else:
        df_norm = np.empty((2,len(time_vec)))
        df_norm[:] = np.nan
    
    # Plot mean and error bar
    if plot_errors:
        ax.fill_between(time_vec, df_norm.mean(axis=0)-df_norm.std(axis=0)/np.sqrt(df_norm.shape[0]), df_norm.mean(axis=0)+df_norm.std(axis=0)/np.sqrt(df_norm.shape[0]),facecolor=color, alpha=alpha)   
    ax.plot(time_vec, df_norm.mean(axis=0),linestyle=linestyle,label=label,color=color,linewidth=linewidth)
    return df_norm.mean(axis=0)

def kernel_evaluation(weights_df, run_params, kernel, save_results=False, drop_threshold=0,session_filter=['Familiar','Novel 1','Novel >1'],equipment_filter="all",cell_filter='all',area_filter=['VISp','VISl'],depth_filter=[0,1000],filter_sessions_on='experience_level',plot_dropout_sorted=True):  
    '''
        Plots the average kernel for each cell line. 
        Plots the heatmap of the kernels sorted by time. 
        Plots the distribution of dropout scores for this kernel.   
        Does that analysis for all cells, just cells with a significant variance_explained, and just cells with a significant dropout score. 

        INPUTS:
        run_params              = glm_params.load_run_params(<version>) 
        results_pivoted         = gat.build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
        weights_df              = gat.build_weights_df(run_params, results_pivoted)
        kernel                  The name of the kernel to be plotted
        save_results            if True, saves a figure to the directory in run_params['output_dir']
        drop_threshold,         the minimum adj_fraction_change_from_full for the dropout model of just dropping this kernel
        session_filter,         The list of session numbers to include
        equipment_filter,       "scientifica" or "mesoscope" filter, anything else plots both 
    '''
    
    # Check for confusing sign
    if drop_threshold > 0:
        print('Are you sure you dont want to use -'+str(drop_threshold)+' ?')
    if drop_threshold <= -1:
        print('Are you sure you mean to use a drop threshold beyond -1?')

    # Filter by Equipment
    filter_string=''
    equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"]
    if equipment_filter == "scientifica": 
        equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5"]
        filter_string += '_scientifica'
    elif equipment_filter == "mesoscope":
        equipment_list = ["MESO.1"]
        filter_string += '_mesoscope'
    
    # Filter by Cell Type    
    cell_list = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre']     
    if (cell_filter == "sst") or (cell_filter == "Sst-IRES-Cre"):
        cell_list = ['Sst-IRES-Cre']
        filter_string += '_sst'
    elif (cell_filter == "vip") or (cell_filter == "Vip-IRES-Cre"):
        cell_list = ['Vip-IRES-Cre']
        filter_string += '_vip'
    elif (cell_filter == "slc") or (cell_filter == "Slc17a7-IRES2-Cre"):
        cell_list = ['Slc17a7-IRES2-Cre']
        filter_string += '_slc'

    # Determine filename
    if session_filter != ['Familiar','Novel 1','Novel >1']:
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if depth_filter !=[0,1000]:
        filter_string+='_depth_'+str(depth_filter[0])+'_'+str(depth_filter[1])
    if area_filter != ['VISp','VISl']:
        filter_string+='_area_'+'_'.join(area_filter)
    filename = os.path.join(run_params['fig_kernels_dir'],kernel+'_evaluation_'+filter_string+'.png')
    filename_svg = os.path.join(run_params['fig_kernels_dir'],kernel+'_evaluation_'+filter_string+'.svg')
    problem_sessions = get_problem_sessions()

    # Filter by overall VE
    if 'dropout_threshold' in run_params:
        threshold = run_params['dropout_threshold']
    else:
        threshold = 0.005

    # Set up time vectors.
    if kernel in ['preferred_image', 'all-images']:
        run_params['kernels'][kernel] = run_params['kernels']['image0'].copy()
    if kernel == 'all-omissions':
        run_params['kernels'][kernel] = run_params['kernels']['omissions'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['omissions']['length'] + run_params['kernels']['post-omissions']['length']
    if kernel == 'all-hits':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['hits']['length'] + run_params['kernels']['post-hits']['length']   
    if kernel == 'all-misses':
        run_params['kernels'][kernel] = run_params['kernels']['misses'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['misses']['length'] + run_params['kernels']['post-misses']['length']   
    if kernel == 'all-passive_change':
        run_params['kernels'][kernel] = run_params['kernels']['passive_change'].copy()
        run_params['kernels'][kernel]['length'] = run_params['kernels']['passive_change']['length'] + run_params['kernels']['post-passive_change']['length']   
    if kernel == 'task':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()   
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)
    if 'image' in kernel:
        time_vec = time_vec[:-1]
    if ('omissions' == kernel) & ('post-omissions' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('hits' == kernel) & ('post-hits' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('misses' == kernel) & ('post-misses' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('passive_change' == kernel) & ('post-passive_change' in run_params['kernels']):
        time_vec = time_vec[:-1]

    # Make dropout list
    drop_list = [d for d in run_params['dropouts'].keys() if (
                    (run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['kernels'])) 
                    or ((not run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['dropped_kernels']))]
    if (len(drop_list) == 0) & (kernel == 'all-images'):
        drop_list = ['all_images']
    if (len(drop_list) == 0) & (kernel == 'all-omissions'):
        drop_list = ['all_omissions']
    if (len(drop_list) == 0) & (kernel == 'task'):
        drop_list = ['task']

    if '-' in kernel:
        weights_df= weights_df.rename(columns={
            'all-omissions':'all_omissions',
            'all-omissions_weights':'all_omissions_weights',
            'post-omissions':'post_omissions',
            'post-omissions_weights':'post_omissions_weights', 
            'all-hits':'all_hits',
            'all-hits_weights':'all_hits_weights',
            'post-hits':'post_hits',
            'post-hits_weights':'post_hits_weights', 
            'all-misses':'all_misses',
            'all-misses_weights':'all_misses_weights',
            'post-misses':'post_misses',
            'post-misses_weights':'post_misses_weights', 
            'all-passive_change':'all_passive_change',
            'all-passive_change_weights':'all_passive_change_weights',
            'post-passive_change':'post_passive_change',
            'post-passive_change_weights':'post_passive_change_weights',  
            'all-images':'all_images',
            'all-images_weights':'all_images_weights',
            'single-post-omissions':'single_post_omissions',
            'single-all-images':'single_all_omissions',
            })

        kernel = kernel.replace('-','_')
        drop_list = [x.replace('-','_') for x in drop_list]

    # Applying hard thresholds to dataset
    # don't apply overall VE, or dropout threshold limits here, since we look at the effects of those criteria below. 
    # we do remove NaN dropouts here
    if kernel in weights_df:
        weights = weights_df.query('(not passive)&(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&({0} in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > 0) & ({1} <= 0)'.format(filter_sessions_on, kernel))
        use_dropouts=True
    else:
        weights = weights_df.query('(not passive)&(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&({0} in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > 0)'.format(filter_sessions_on)) 
        print('Dropouts not included, cannot use drop filter')
        use_dropouts=False

    # Have to do a manual filtering step here because weird things happen when combining
    # two kernels
    if kernel == 'task':
        weights = weights[~weights['task_weights'].isnull()]

    # Plotting settings
    colors = project_colors()
    line_alpha = 0.25
    width=0.25

    # Get all cells data and plot Average Trajectories
    fig,ax=plt.subplots(3,3,figsize=(12,9))
    sst = weights.query('cre_line == "Sst-IRES-Cre"')[kernel+'_weights']
    vip = weights.query('cre_line == "Vip-IRES-Cre"')[kernel+'_weights']
    slc = weights.query('cre_line == "Slc17a7-IRES2-Cre"')[kernel+'_weights']
    if plot_dropout_sorted:
        sst_drop = weights.query('cre_line == "Sst-IRES-Cre"')[kernel]
        vip_drop = weights.query('cre_line == "Vip-IRES-Cre"')[kernel]
        slc_drop = weights.query('cre_line == "Slc17a7-IRES2-Cre"')[kernel]
        sst_table = weights.query('cre_line == "Sst-IRES-Cre"')[[kernel+'_weights',kernel]]
        vip_table = weights.query('cre_line == "Vip-IRES-Cre"')[[kernel+'_weights',kernel]]
        slc_table = weights.query('cre_line == "Slc17a7-IRES2-Cre"')[[kernel+'_weights',kernel]]
    
    n_sst = len(sst)
    n_vip = len(vip)
    n_slc = len(slc)     
 
    # Make into 2D array, but only if we have results.
    # Else make a 2D array of NaNs 
    if len(sst)>0:
        sst = np.vstack(sst)
    else:
        sst = np.empty((2,len(time_vec)))
        sst[:] = np.nan
    if len(vip)>0:
        vip = np.vstack(vip)
    else:
        vip = np.empty((2,len(time_vec)))
        vip[:] = np.nan
    if len(slc)>0:
        slc = np.vstack(slc)
    else:
        slc = np.empty((2,len(time_vec)))
        slc[:] = np.nan

    # Plot
    ax[0,0].plot(time_vec, sst.mean(axis=0),label='SST (n='+str(n_sst)+')',color=colors['sst'],linewidth=2)
    ax[0,0].plot(time_vec, vip.mean(axis=0),label='VIP (n='+str(n_vip)+')',color=colors['vip'],linewidth=2)
    ax[0,0].plot(time_vec, slc.mean(axis=0),label='SLC (n='+str(n_slc)+')',color=colors['slc'],linewidth=2)
    ax[0,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    #ax[0,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[0,0].set_ylabel('Weights (df/f)')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].legend()
    ax[0,0].set_title('Average kernel')
    ax[0,0].set_xlim(time_vec[0]-0.05,time_vec[-1])   
    add_stimulus_bars(ax[0,0],kernel)
    sst = sst.T
    vip = vip.T
    slc = slc.T

    # Get Full model filtered data, and plot average kernels
    sst_f = weights.query('cre_line == "Sst-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    vip_f = weights.query('cre_line == "Vip-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    slc_f = weights.query('cre_line == "Slc17a7-IRES2-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    if plot_dropout_sorted:
        sst_drop_f = weights.query('cre_line == "Sst-IRES-Cre" & variance_explained_full > @threshold')[kernel]
        vip_drop_f = weights.query('cre_line == "Vip-IRES-Cre" & variance_explained_full > @threshold')[kernel]
        slc_drop_f = weights.query('cre_line == "Slc17a7-IRES2-Cre" & variance_explained_full > @threshold')[kernel]
    n_sst = len(sst_f)
    n_vip = len(vip_f)
    n_slc = len(slc_f)     
    if len(sst_f)>0:
        sst_f = np.vstack(sst_f)
    else:
        sst_f = np.empty((2,len(time_vec)))
        sst_f[:] = np.nan
    if len(vip_f)>0:
        vip_f = np.vstack(vip_f)
    else:
        vip_f = np.empty((2,len(time_vec)))
        vip_f[:] = np.nan
    if len(slc_f)>0:
        slc_f = np.vstack(slc_f)
    else:
        slc_f = np.empty((2,len(time_vec)))
        slc_f[:] = np.nan
    ax[1,0].plot(time_vec, sst_f.mean(axis=0),label='SST (n='+str(n_sst)+')',color=colors['sst'],linewidth=2)
    ax[1,0].plot(time_vec, vip_f.mean(axis=0),label='VIP (n='+str(n_vip)+')',color=colors['vip'],linewidth=2)
    ax[1,0].plot(time_vec, slc_f.mean(axis=0),label='SLC (n='+str(n_slc)+')',color=colors['slc'],linewidth=2)
    ax[1,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    #ax[1,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[1,0].set_ylabel('Weights (df/f)')
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].legend()
    ax[1,0].set_title('Filtered on Full Model VE > '+str(threshold))
    ax[1,0].set_xlim(time_vec[0]-0.05,time_vec[-1])   
    add_stimulus_bars(ax[1,0],kernel)
    sst_f = sst_f.T
    vip_f = vip_f.T
    slc_f = slc_f.T

    # Get Dropout filtered data, and plot average kernels
    if use_dropouts:
        sst_df = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel+'_weights']
        vip_df = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel+'_weights']
        slc_df = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel+'_weights']
        if plot_dropout_sorted:
            sst_drop_df = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel]
            vip_drop_df = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel]
            slc_drop_df = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel]

            sst_table_df = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[[kernel+'_weights',kernel]]
            vip_table_df = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[[kernel+'_weights',kernel]]
            slc_table_df = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[[kernel+'_weights',kernel]]

        n_sst = len(sst_df)
        n_vip = len(vip_df)
        n_slc = len(slc_df)     
     
        if len(sst_df)>0:
            sst_df = np.vstack(sst_df)
        else:
            sst_df = np.empty((2,len(time_vec)))
            sst_df[:] = np.nan
        if len(vip_df)>0:
            vip_df = np.vstack(vip_df)
        else:
            vip_df = np.empty((2,len(time_vec)))
            vip_df[:] = np.nan
        if len(slc_df)>0:
            slc_df = np.vstack(slc_df)
        else:
            slc_df = np.empty((2,len(time_vec)))
            slc_df[:] = np.nan
    
        ax[2,0].plot(time_vec, sst_df.mean(axis=0),label='SST (n='+str(n_sst)+')',color=colors['sst'],linewidth=2)
        ax[2,0].plot(time_vec, vip_df.mean(axis=0),label='VIP (n='+str(n_vip)+')',color=colors['vip'],linewidth=2)
        ax[2,0].plot(time_vec, slc_df.mean(axis=0),label='SLC (n='+str(n_slc)+')',color=colors['slc'],linewidth=2)
        ax[2,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
        #ax[2,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
        ax[2,0].set_ylabel('Weights (df/f)')
        ax[2,0].set_xlabel('Time (s)')
        ax[2,0].legend()
        ax[2,0].set_title('Filtered on Dropout Score < '+str(drop_threshold))
        ax[2,0].set_xlim(time_vec[0]-.05,time_vec[-1])   
        add_stimulus_bars(ax[2,0],kernel)
        sst_df = sst_df.T
        vip_df = vip_df.T
        slc_df = slc_df.T
        
    # Plot Heat maps
    sst_sorted = sst[:,np.argsort(np.argmax(sst,axis=0))]
    vip_sorted = vip[:,np.argsort(np.argmax(vip,axis=0))]
    slc_sorted = slc[:,np.argsort(np.argmax(slc,axis=0))]
    if plot_dropout_sorted:
        sst_drop_sorted = sst[:,sst_drop.values.argsort()]
        vip_drop_sorted = vip[:,vip_drop.values.argsort()] 
        slc_drop_sorted = slc[:,slc_drop.values.argsort()]
    weights_sorted = np.hstack([slc_sorted,sst_sorted, vip_sorted])
    cbar = ax[0,1].imshow(weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted)[1]],cmap='bwr')
    cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted),95),np.nanpercentile(np.abs(weights_sorted),95))
    color_bar=fig.colorbar(cbar, ax=ax[0,1])
    color_bar.ax.set_ylabel('Weights')   
    ax[0,1].set_ylabel('{0} Cells\n sorted by peak'.format(np.shape(weights_sorted)[1]))
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].axhline(np.shape(vip)[1],color='k',linewidth='1')
    ax[0,1].axhline(np.shape(vip)[1] + np.shape(sst)[1],color='k',linewidth='1')
    ax[0,1].set_yticks([np.shape(vip)[1]/2,np.shape(vip)[1]+np.shape(sst)[1]/2, np.shape(vip)[1]+np.shape(sst)[1]+np.shape(slc)[1]/2])
    ax[0,1].set_yticklabels(['Vip','Sst','Exc'])
    ax[0,1].set_title(kernel)

    # Plot Heatmap of filtered cells
    sst_sorted_f = sst_f[:,np.argsort(np.argmax(sst_f,axis=0))]
    vip_sorted_f = vip_f[:,np.argsort(np.argmax(vip_f,axis=0))]
    slc_sorted_f = slc_f[:,np.argsort(np.argmax(slc_f,axis=0))]
    if plot_dropout_sorted:
        sst_drop_sorted_f = sst_f[:,sst_drop_f.values.argsort()]
        vip_drop_sorted_f = vip_f[:,vip_drop_f.values.argsort()] 
        slc_drop_sorted_f = slc_f[:,slc_drop_f.values.argsort()]
    weights_sorted_f = np.hstack([slc_sorted_f,sst_sorted_f, vip_sorted_f])
    cbar = ax[1,1].imshow(weights_sorted_f.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_f)[1]],cmap='bwr')
    cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted_f),95),np.nanpercentile(np.abs(weights_sorted_f),95))
    color_bar = fig.colorbar(cbar, ax=ax[1,1])
    color_bar.ax.set_ylabel('Weights')   
    ax[1,1].set_ylabel('{0} Cells\n sorted by peak'.format(np.shape(weights_sorted_f)[1]))
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].axhline(np.shape(vip_f)[1],color='k',linewidth='1')
    ax[1,1].axhline(np.shape(vip_f)[1] + np.shape(sst_f)[1],color='k',linewidth='1')
    ax[1,1].set_yticks([np.shape(vip_f)[1]/2,np.shape(vip_f)[1]+np.shape(sst_f)[1]/2, np.shape(vip_f)[1]+np.shape(sst_f)[1]+np.shape(slc_f)[1]/2])
    ax[1,1].set_yticklabels(['Vip','Sst','Exc'])
    ax[1,1].set_title('Filtered on Full Model')

    # Plot Heatmap of filtered cells
    if use_dropouts:
        sst_sorted_df = sst_df[:,np.argsort(np.argmax(sst_df,axis=0))]
        vip_sorted_df = vip_df[:,np.argsort(np.argmax(vip_df,axis=0))]
        slc_sorted_df = slc_df[:,np.argsort(np.argmax(slc_df,axis=0))]
        if plot_dropout_sorted:
            sst_drop_sorted_df = sst_df[:,sst_drop_df.values.argsort()]
            vip_drop_sorted_df = vip_df[:,vip_drop_df.values.argsort()]
            slc_drop_sorted_df = slc_df[:,slc_drop_df.values.argsort()]
        weights_sorted_df = np.hstack([slc_sorted_df,sst_sorted_df, vip_sorted_df])
        cbar = ax[2,1].imshow(weights_sorted_df.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_df)[1]],cmap='bwr')
        cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted_df),95),np.nanpercentile(np.abs(weights_sorted_df),95))
        color_bar = fig.colorbar(cbar, ax=ax[2,1])
        color_bar.ax.set_ylabel('Weights')   
        ax[2,1].set_ylabel('{0} Cells\n sorted by peak'.format(np.shape(weights_sorted_df)[1]))
        ax[2,1].set_xlabel('Time (s)')
        ax[2,1].axhline(np.shape(vip_df)[1],color='k',linewidth='1')
        ax[2,1].axhline(np.shape(vip_df)[1] + np.shape(sst_df)[1],color='k',linewidth='1')
        ax[2,1].set_yticks([np.shape(vip_df)[1]/2,np.shape(vip_df)[1]+np.shape(sst_df)[1]/2, np.shape(vip_df)[1]+np.shape(sst_df)[1]+np.shape(slc_df)[1]/2])
        ax[2,1].set_yticklabels(['Vip','Sst','Exc'])
        ax[2,1].set_title('Filtered on Dropout')

    ## Right Column, Dropout Scores 
    # Make list of dropouts that contain this kernel
    medianprops = dict(color='k')

    if plot_dropout_sorted:
        weights_sorted = np.hstack([slc_drop_sorted,sst_drop_sorted, vip_drop_sorted])
        cbar = ax[0,2].imshow(weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted)[1]],cmap='bwr')
        cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted),95),np.nanpercentile(np.abs(weights_sorted),95))
        color_bar=fig.colorbar(cbar, ax=ax[0,2])
        color_bar.ax.set_ylabel('Weights')   
        ax[0,2].set_ylabel('{0} Cells\n sorted by dropout'.format(np.shape(weights_sorted)[1]))
        if kernel == 'omissions':
            ax[0,2].set_xlabel('Time from omission (s)')  
        elif kernel in ['hits','misses']:
            ax[0,2].set_xlabel('Time from image change (s)')          
        else:
            ax[0,2].set_xlabel('Time (s)')
        ax[0,2].axhline(np.shape(vip)[1],color='k',linewidth='1')
        ax[0,2].axhline(np.shape(vip)[1] + np.shape(sst)[1],color='k',linewidth='1')
        ax[0,2].set_yticks([np.shape(vip)[1]/2,np.shape(vip)[1]+np.shape(sst)[1]/2, np.shape(vip)[1]+np.shape(sst)[1]+np.shape(slc)[1]/2])
        ax[0,2].set_yticklabels(['Vip','Sst','Exc'])
        ax[0,2].set_title(kernel)
        ncells={
            'vip':np.shape(vip)[1],
            'sst':np.shape(sst)[1],
            'exc':np.shape(slc)[1],
            }

        zlims = plot_kernel_heatmap(weights_sorted,time_vec, kernel, run_params,ncells,session_filter=session_filter,savefig=save_results)
        #zlims_test = plot_kernel_heatmap_with_dropout(vip_table, sst_table, slc_table,time_vec, kernel, run_params,ncells,session_filter=session_filter)
    else:
    
        # All Cells
        # For each dropout, plot the score distribution by cre line
        for index, dropout in enumerate(drop_list):
            drop_sst = weights.query('cre_line=="Sst-IRES-Cre"')[dropout]
            drop_vip = weights.query('cre_line=="Vip-IRES-Cre"')[dropout]
            drop_slc = weights.query('cre_line=="Slc17a7-IRES2-Cre"')[dropout]
            drop_sst = drop_sst[~drop_sst.isnull()].values
            drop_vip = drop_vip[~drop_vip.isnull()].values
            drop_slc = drop_slc[~drop_slc.isnull()].values
            drops = ax[0,2].boxplot([drop_sst,drop_vip,drop_slc],
                                    positions=[index-width,index,index+width],
                                    labels=['SST','VIP','SLC'],
                                    showfliers=False,
                                    patch_artist=True,
                                    medianprops=medianprops,
                                    widths=.2)
            for patch, color in zip(drops['boxes'],[colors['sst'],colors['vip'],colors['slc']]):
                patch.set_facecolor(color)
    
        # Clean up plot
        ax[0,2].set_ylabel('Adj. Fraction from Full')
        ax[0,2].set_xticks(np.arange(0,len(drop_list)))
        ax[0,2].set_xticklabels(drop_list,rotation=60,fontsize=8,ha='right')
        ax[0,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
        ax[0,2].set_ylim(-1.05,.05)
        ax[0,2].set_title('Dropout Scores')
    
    # Filtered by Full model
    if plot_dropout_sorted:
        weights_sorted_f = np.hstack([slc_drop_sorted_f,sst_drop_sorted_f, vip_drop_sorted_f])
        cbar = ax[1,2].imshow(weights_sorted_f.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_f)[1]],cmap='bwr')
        cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted_f),95),np.nanpercentile(np.abs(weights_sorted_f),95))
        color_bar = fig.colorbar(cbar, ax=ax[1,2])
        color_bar.ax.set_ylabel('Weights')   
        ax[1,2].set_ylabel('{0} Cells\n sorted by dropout'.format(np.shape(weights_sorted_f)[1]))
        if kernel == 'omissions':
            ax[1,2].set_xlabel('Time from omission (s)')  
        elif kernel in ['hits','misses']:
            ax[1,2].set_xlabel('Time from image change (s)')          
        else:
            ax[1,2].set_xlabel('Time (s)')
        ax[1,2].axhline(np.shape(vip_f)[1],color='k',linewidth='1')
        ax[1,2].axhline(np.shape(vip_f)[1] + np.shape(sst_f)[1],color='k',linewidth='1')
        ax[1,2].set_yticks([np.shape(vip_f)[1]/2,np.shape(vip_f)[1]+np.shape(sst_f)[1]/2, np.shape(vip_f)[1]+np.shape(sst_f)[1]+np.shape(slc_f)[1]/2])
        ax[1,2].set_yticklabels(['Vip','Sst','Exc'])
        ax[1,2].set_title('Filtered on Full Model')
        ncells_f={
            'vip':np.shape(vip_f)[1],
            'sst':np.shape(sst_f)[1],
            'exc':np.shape(slc_f)[1],
            }
        zlims = plot_kernel_heatmap(weights_sorted_f,time_vec, kernel, run_params,ncells_f,extra='full_model',zlims=zlims,session_filter=session_filter,savefig=save_results) 
    else:
        # For each dropout, plot score
        for index, dropout in enumerate(drop_list):
            drop_sst = weights.query('cre_line=="Sst-IRES-Cre" & variance_explained_full > @threshold')[dropout]
            drop_vip = weights.query('cre_line=="Vip-IRES-Cre" & variance_explained_full > @threshold')[dropout]
            drop_slc = weights.query('cre_line=="Slc17a7-IRES2-Cre" & variance_explained_full > @threshold')[dropout]
            drop_sst = drop_sst[~drop_sst.isnull()].values
            drop_vip = drop_vip[~drop_vip.isnull()].values
            drop_slc = drop_slc[~drop_slc.isnull()].values
            drops = ax[1,2].boxplot([drop_sst,drop_vip,drop_slc],
                                    positions=[index-width,index,index+width],
                                    labels=['SST','VIP','SLC'],
                                    showfliers=False,
                                    patch_artist=True,
                                    medianprops=medianprops,
                                    widths=.2)
            for patch, color in zip(drops['boxes'],[colors['sst'],colors['vip'],colors['slc']]):
                patch.set_facecolor(color)
    
        # Clean up plot
        ax[1,2].set_ylabel('Adj. Fraction from Full')
        ax[1,2].set_xticks(np.arange(0,len(drop_list)))
        ax[1,2].set_xticklabels(drop_list,rotation=60,fontsize=8,ha='right')
        ax[1,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
        ax[1,2].set_ylim(-1.05,.05)
        ax[1,2].set_title('Filter on Full Model')

    # Filtered by Dropout Score
    if plot_dropout_sorted:
        weights_sorted_df = np.hstack([slc_drop_sorted_df,sst_drop_sorted_df, vip_drop_sorted_df])
        cbar = ax[2,2].imshow(weights_sorted_df.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_df)[1]],cmap='bwr')
        cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted_df),95),np.nanpercentile(np.abs(weights_sorted_df),95))
        color_bar = fig.colorbar(cbar, ax=ax[2,2])
        color_bar.ax.set_ylabel('Weights')   
        ax[2,2].set_ylabel('{0} Cells\n sorted by dropout'.format(np.shape(weights_sorted_df)[1]))
        if kernel == 'omissions':
            ax[2,2].set_xlabel('Time from omission (s)')  
        elif kernel in ['hits','misses']:
            ax[2,2].set_xlabel('Time from image change (s)')          
        else:
            ax[2,2].set_xlabel('Time (s)')
        ax[2,2].axhline(np.shape(vip_df)[1],color='k',linewidth='1')
        ax[2,2].axhline(np.shape(vip_df)[1] + np.shape(sst_df)[1],color='k',linewidth='1')
        ax[2,2].set_yticks([np.shape(vip_df)[1]/2,np.shape(vip_df)[1]+np.shape(sst_df)[1]/2, np.shape(vip_df)[1]+np.shape(sst_df)[1]+np.shape(slc_df)[1]/2])
        ax[2,2].set_yticklabels(['Vip','Sst','Exc'])
        ax[2,2].set_title('Filtered on Dropout')
        ncells_df={
            'vip':np.shape(vip_df)[1],
            'sst':np.shape(sst_df)[1],
            'exc':np.shape(slc_df)[1],
            }
        zlims = plot_kernel_heatmap(weights_sorted_df,time_vec, kernel, run_params,ncells_df,extra='dropout',zlims=None,session_filter=session_filter,savefig=save_results)
        zlims_test = plot_kernel_heatmap_with_dropout(vip_table_df, sst_table_df, slc_table_df,time_vec, kernel, run_params,ncells_df,session_filter=session_filter,zlims=zlims,extra='dropout',savefig=save_results)

        zlims = plot_kernel_heatmap(weights_sorted,time_vec, kernel, run_params,ncells,zlims=zlims,session_filter=session_filter,savefig=save_results)
        zlims_test = plot_kernel_heatmap_with_dropout(vip_table, sst_table, slc_table,time_vec, kernel, run_params,ncells,session_filter=session_filter,zlims=zlims,savefig=save_results)
    else:
        # For each dropout, plot score
        for index, dropout in enumerate(drop_list):
            drop_sst = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[dropout].values
            drop_vip = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[dropout].values
            drop_slc = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[dropout].values
            drops = ax[2,2].boxplot([drop_sst,drop_vip,drop_slc],
                                    positions=[index-width,index,index+width],
                                    labels=['SST','VIP','SLC'],
                                    showfliers=False,
                                    patch_artist=True,
                                    medianprops=medianprops,
                                    widths=.2)
            for patch, color in zip(drops['boxes'],[colors['sst'],colors['vip'],colors['slc']]):
                patch.set_facecolor(color)
    
        # Clean Up Plot
        ax[2,2].set_ylabel('Adj. Fraction from Full')
        ax[2,2].set_xticks(np.arange(0,len(drop_list)))
        ax[2,2].set_xticklabels(drop_list,rotation=60,fontsize=8,ha='right')
        ax[2,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
        ax[2,2].set_ylim(-1.05,.05)
        ax[2,2].set_title('Filter on Dropout Score')
        ax[2,2].axhline(drop_threshold, color='r',linestyle='--', alpha=line_alpha)

    
    ## Final Clean up and Save
    plt.figure(fig.number)
    plt.tight_layout()
    if save_results:
        print('Figure Saved to: '+filename)
        plt.savefig(filename) 
        plt.savefig(filename_svg)

def all_kernels_evaluation(weights_df, run_params, drop_threshold=0,session_filter=['Familiar','Novel 1','Novel >1'],equipment_filter="all",cell_filter='all',area_filter=['VISp','VISl'],depth_filter=[0,1000]): 
    '''
        Makes the analysis plots for all kernels in this model version. Excludes intercept and time kernels
                
        INPUTS:
        Same as kernel_evaluation
        
        SAVES:
        a figure for each kernel    

    '''
    kernels = set(run_params['kernels'].keys())
    kernels.remove('intercept')
    kernels.add('task')
    kernels.add('all-omissions')
    kernels.add('all-images')
    kernels.add('preferred_image')
    crashed = set()
    for k in kernels:
        print(k)
        try:
            kernel_evaluation(
                weights_df, 
                run_params, 
                k, 
                save_results=True, 
                drop_threshold=drop_threshold,
                session_filter=session_filter, 
                equipment_filter=equipment_filter,
                depth_filter=depth_filter,
                cell_filter=cell_filter,
                area_filter=area_filter
                )
            plt.close(plt.gcf().number)
        except Exception as e:
            crashed.add(k)
            plt.close(plt.gcf().number)
            print(e)

    for k in crashed:
        print('Crashed - '+k) 

def plot_kernel_heatmap(weights_sorted, time_vec,kernel, run_params, ncells = {},ax=None,extra='',zlims=None,session_filter=['Familiar','Novel 1','Novel >1'],savefig=False):
    if ax==None:
        #fig,ax = plt.subplots(figsize=(8,4))
        height = 4
        width=8
        pre_horz_offset = 1.5
        post_horz_offset = 2.5
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(pre_horz_offset),Size.Fixed(width-pre_horz_offset-post_horz_offset)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1)) 
        h = [Size.Fixed(width-post_horz_offset+.25),Size.Fixed(.25)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        cax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  

    cbar = ax.imshow(weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted)[1]],cmap='bwr')
    if zlims is None:
        zlims =[-np.nanpercentile(np.abs(weights_sorted),95),np.nanpercentile(np.abs(weights_sorted),95)]
    cbar.set_clim(zlims[0], zlims[1])
    color_bar=fig.colorbar(cbar, cax=cax)
    color_bar.ax.set_ylabel('Weight',fontsize=16)  
    color_bar.ax.tick_params(axis='both',labelsize=16)
    ax.set_ylabel('{0} cells'.format(np.shape(weights_sorted)[1]),fontsize=16) 
    if kernel == 'omissions':
        ax.set_xlabel('Time from omission (s)',fontsize=20)  
    elif kernel in ['hits','misses']:
        ax.set_xlabel('Time from image change (s)',fontsize=20)          
    else:
        ax.set_xlabel('Time (s)',fontsize=20)

    ax.axhline(ncells['vip'],color='k',linewidth='1')
    ax.axhline(ncells['vip']+ncells['sst'],color='k',linewidth='1')
    ax.set_yticks([ncells['vip']/2,ncells['vip']+ncells['sst']/2,ncells['vip']+ncells['sst']+ncells['exc']/2])
    ax.set_yticklabels(['Vip','Sst','Exc'])
    ax.tick_params(axis='both',labelsize=16)
    if extra == '':
        title = kernel +' kernels'
    elif extra == 'dropout':
        title = kernel +' kernels, coding cells'
    else:
        title = kernel +' kernels, VE cells'
    if len(session_filter) ==1:
        mapper={
            'Familiar':'Familiar',
            'Novel 1':'Novel',
            'Novel >1':'Novel +'
            }
        extra=extra+'_'+session_filter[0].replace(' ','_').replace('>','p')
        title = title + ', '+mapper[session_filter[0]]
    ax.set_title(title,fontsize=20)
    filename = os.path.join(run_params['fig_kernels_dir'],kernel+'_heatmap_'+extra+'.svg')
    if savefig:
        plt.savefig(filename) 
    return zlims
    #plt.tight_layout()

def plot_kernel_heatmap_with_dropout(vip_table, sst_table, slc_table, time_vec,kernel, run_params, ncells = {},ax=None,extra='',zlims=None,session_filter=['Familiar','Novel 1','Novel >1'],savefig=False):
    if ax==None:
        #fig,ax = plt.subplots(figsize=(8,4))
        height = 4
        width=8
        pre_horz_offset = 1.5
        post_horz_offset = 2.5
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(pre_horz_offset),Size.Fixed(width-pre_horz_offset-post_horz_offset-.25)]
        v = [Size.Fixed(vertical_offset),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax3 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))
        v = [Size.Fixed(vertical_offset+(height-vertical_offset-.5)/3),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax2 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))
        v = [Size.Fixed(vertical_offset+2*(height-vertical_offset-.5)/3),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax1 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))
        
        h = [Size.Fixed(width-post_horz_offset-.25),Size.Fixed(.25)]
        v = [Size.Fixed(vertical_offset+2*(height-vertical_offset-.5)/3),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        dax1 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  
        v = [Size.Fixed(vertical_offset+(height-vertical_offset-.5)/3),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        dax2 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  
        v = [Size.Fixed(vertical_offset),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        dax3 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  

        h = [Size.Fixed(width-post_horz_offset+.25),Size.Fixed(.25)]
        v = [Size.Fixed(vertical_offset+(height-vertical_offset-.5)/2)+.125,Size.Fixed((height-vertical_offset-.5)/2-.125)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        cax1 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  

        h = [Size.Fixed(width-post_horz_offset+.25),Size.Fixed(.25)]
        v = [Size.Fixed(vertical_offset/4),Size.Fixed((height-vertical_offset-.5)/2-.125)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        cax2 = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  


    # Sort cells
    # convert kernels to columns
    ncols = len(vip_table[kernel+'_weights'].values[0])
    vip_df = pd.DataFrame(vip_table[kernel+'_weights'].to_list(),columns = ['w'+str(x) for x in range(0,ncols)])
    vip_df['dropout'] = vip_table.reset_index()[kernel]*-1
    vip_df = vip_df.sort_values(by=['dropout'],ascending=False)    
    sst_df = pd.DataFrame(sst_table[kernel+'_weights'].to_list(),columns = ['w'+str(x) for x in range(0,ncols)])
    sst_df['dropout'] = sst_table.reset_index()[kernel]*-1
    sst_df = sst_df.sort_values(by=['dropout'],ascending=False) 
    slc_df = pd.DataFrame(slc_table[kernel+'_weights'].to_list(),columns = ['w'+str(x) for x in range(0,ncols)])
    slc_df['dropout'] = slc_table.reset_index()[kernel]*-1
    slc_df = slc_df.sort_values(by=['dropout'],ascending=False) 

    weights_sorted = np.concatenate([slc_df.to_numpy(),sst_df.to_numpy(), vip_df.to_numpy()])[:,0:-1].T
    drop_sorted = np.concatenate([slc_df.to_numpy(),sst_df.to_numpy(), vip_df.to_numpy()])[:,-1].T
    slc_weights_sorted =slc_df.to_numpy()[:,0:-1].T
    slc_drop_sorted =   slc_df.to_numpy()[:,-1].T
    sst_weights_sorted =sst_df.to_numpy()[:,0:-1].T
    sst_drop_sorted =   sst_df.to_numpy()[:,-1].T
    vip_weights_sorted =vip_df.to_numpy()[:,0:-1].T
    vip_drop_sorted =   vip_df.to_numpy()[:,-1].T

    cbar1 = ax1.imshow(vip_weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(slc_weights_sorted)[1]],cmap='bwr')
    if zlims is None:
        zlims =[-np.nanpercentile(np.abs(weights_sorted),95),np.nanpercentile(np.abs(weights_sorted),95)]
    cbar1.set_clim(zlims[0], zlims[1])
    cbar2 = ax2.imshow(sst_weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(sst_weights_sorted)[1]],cmap='bwr')
    cbar2.set_clim(zlims[0], zlims[1])
    cbar3 = ax3.imshow(slc_weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(vip_weights_sorted)[1]],cmap='bwr')
    cbar3.set_clim(zlims[0], zlims[1])

    color_bar=fig.colorbar(cbar1, cax=cax1)
    color_bar.ax.set_title('Weight',fontsize=16,loc='left')  
    color_bar.ax.tick_params(axis='both',labelsize=16)
    if kernel == 'omissions':
        ax3.set_xlabel('Time from omission (s)',fontsize=20)  
    elif kernel in ['hits','misses']:
        ax3.set_xlabel('Time from image change (s)',fontsize=20)          
    else:
        ax3.set_xlabel('Time (s)',fontsize=20)

    ax1.set_yticks([ax1.get_ylim()[1]/2])
    ax1.set_yticklabels(['Vip'])
    ax2.set_yticks([ax2.get_ylim()[1]/2])
    ax2.set_yticklabels(['Sst'])
    ax3.set_yticks([ax3.get_ylim()[1]/2])
    ax3.set_yticklabels(['Exc'])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.tick_params(axis='both',labelsize=16)
    ax2.tick_params(axis='both',labelsize=16)
    ax3.tick_params(axis='both',labelsize=16)
    title = kernel +' kernels'
    all_cells=False
    dropout=False
    VE = False
    if extra == '':
        all_cells=True
    elif extra == 'dropout':
        dropout = True
    else:
        VE = True

    if len(session_filter) ==1:
        mapper={
            'Familiar':'Familiar',
            'Novel 1':'Novel',
            'Novel >1':'Novel +'
            }
        extra=extra+'_'+session_filter[0].replace(' ','_').replace('>','p')
        title = title + ', '+mapper[session_filter[0]]
    if dropout:
        #title =title +  '\n coding cells'
        ax2.set_ylabel('Coding Cells',fontsize=16)
    if VE:
        ax2.set_ylabel('VE > 0.005 Cells',fontsize=16)

    ax1.set_title(title,fontsize=20)
    cmap = copy.copy(plt.cm.get_cmap('plasma'))
    cmap.set_under('black')
    cbar2=dax1.imshow(np.sqrt(vip_drop_sorted[:,np.newaxis]),aspect='auto',cmap=cmap,vmin=1e-10,vmax=1) 
    dax1.set_yticks([])
    dax1.set_xticks([])
    color_bar=fig.colorbar(cbar2, cax=cax2,extend='min')
    color_bar.ax.set_title('Coding \nScore',fontsize=16,loc='left') 
    color_bar.set_ticks([0,.5,1]) 
    color_bar.ax.tick_params(axis='both',labelsize=16)

    dax2.imshow(np.sqrt(sst_drop_sorted[:,np.newaxis]),aspect='auto',cmap=cmap,vmin=1e-10,vmax=1) 
    dax2.set_yticks([])
    dax2.set_xticks([])
    dax3.imshow(np.sqrt(slc_drop_sorted[:,np.newaxis]),aspect='auto',cmap=cmap,vmin=1e-10,vmax=1) 
    dax3.set_yticks([])
    dax3.set_xticks([])   
    #dax3.set_xticks([dax3.get_xlim()[1]/2])
    #dax3.set_xticklabels(['Coding\n Score'],rotation=-70,fontsize=16)
    if savefig:
        filename = os.path.join(run_params['fig_kernels_dir'],kernel+'_heatmap_with_dropout'+extra+'.svg')
        plt.savefig(filename) 
        print('Figure saved to: '+filename)
        filename = os.path.join(run_params['fig_kernels_dir'],kernel+'_heatmap_with_dropout'+extra+'.png')
        plt.savefig(filename) 
    return zlims

def add_stimulus_bars(ax, kernel,alpha=0.1):
    '''
        Adds stimulus bars to the given axis, but only for certain kernels 
    '''
    # Check if this is an image aligned kernel
    if kernel in ['change','hits','misses','false_alarms','omissions','image_expectation','image','image0','image1','image2','image3','image4','image5','image6','image7','avg_image','all_images','all-images']:
        # Define timepoints of stimuli
        lims = ax.get_xlim()
        times = set(np.concatenate([np.arange(0,lims[1],0.75),np.arange(-0.75,lims[0]-0.001,-0.75)]))
        if kernel == 'omissions':
            # For omissions, remove omitted stimuli
            times.remove(0.0)
            ax.axvline(0, color=project_colors()['schematic_omission'], linewidth=1.5,zorder=-np.inf,linestyle='--')
        if kernel in ['change','hits','misses','false_alarms']:
            # For change aligned kernels, plot the two stimuli different colors
            for flash_start in times:
                if flash_start == 0:
                    ax.axvspan(flash_start,flash_start+0.25,color=project_colors()['schematic_change'],alpha=0.2,zorder=-np.inf)                   
                else:
                    ax.axvspan(flash_start,flash_start+0.25,color='k',alpha=alpha,zorder=-np.inf)                   
        else:
            # Normal case, just plot all the same color
            for flash_start in times:
                ax.axvspan(flash_start,flash_start+0.25,color='k',alpha=alpha,zorder=-np.inf)
         
def plot_over_fitting(full_results, dropout,save_file=""):
    ''' 
        Plots an evaluation of how this dropout model contributed to overfitting. 

        INPUTS:
        full_results, with overfitting values
            full_results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='full')
            gat.compute_over_fitting_proportion(full_results,run_params)
        dropout, (str) name of dropout to plot
        save_file (str), if not empty will save figure to that location
    
        SAVES:
        a figure to the location specified by save_file, if not the empty string
     
    '''
    # Set Up Figure. Only two panels for the full model
    if dropout == "Full":
        fig, ax = plt.subplots(1,2,figsize=(10,4))   
    else:
        fig, ax = plt.subplots(1,3,figsize=(12,4))
    
    # First panel, relationship between variance explained and overfitting proportion
    ax[0].plot(full_results[dropout+'__avg_cv_var_test']*100, full_results[dropout+'__over_fit'],'ko',alpha=.1)
    ax[0].set_xlim(0,100)
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Overfitting Proportion: '+dropout)
    ax[0].set_xlabel('Test Variance Explained')

    if dropout == "Full":
        full_results = full_results.query('(Full__over_fit < 1) and (Full__over_fit >=0)').copy()    

    # Second panel, histogram of overfitting proportion, with mean/median marked
    hist_output = ax[1].hist(full_results[dropout+'__over_fit'],50,density=True)
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1.25*np.max(hist_output[0][:-1]))
    ax[1].plot(np.mean(full_results[dropout+'__over_fit']), 1.1*np.max(hist_output[0][:-1]),'rv',label='Mean')
    #ax[1].plot(np.mean(full_results[dropout+'__over_fit'][full_results[dropout+'__over_fit']<1]), 1.1*np.max(hist_output[0][:-1]),'rv',label='Mean Exclude overfit=1 cells')
    #ax[1].plot(np.median(full_results[dropout+'__over_fit']), 1.1*np.max(hist_output[0][:-1]),'bv',markerfacecolor='none',label='Median All Cells')
    #ax[1].plot(np.median(full_results[dropout+'__over_fit'][full_results[dropout+'__over_fit']<1]), 1.1*np.max(hist_output[0][:-1]),'bv',label='Median Exclude overfit=1 cells')
    ax[1].set_ylabel('Density')
    ax[1].set_xlabel('Overfitting Proportion: '+dropout)
    ax[1].legend(loc='lower right')
    
    # Third panel, distribution of dropout_overfitting_proportion compared to full model
    if dropout != "Full":
        ax[2].hist(full_results[dropout+'__dropout_overfit_proportion'].where(lambda x: (x<1)&(x>-1)),100)
        ax[2].axvline(full_results[dropout+'__dropout_overfit_proportion'].where(lambda x: (x<1)&(x>-1)).median(),color='r',linestyle='--')
        ax[2].set_xlim(-1,1)
    else:
        ax[0].tick_params(axis='both',labelsize=16)
        ax[0].set_ylabel('Overfitting Proportion',fontsize=18)
        ax[0].set_xlabel('Variance Explained (%)',fontsize=18)
        ax[1].tick_params(axis='both',labelsize=16)
        ax[1].set_xlabel('Overfitting Proportion',fontsize=18)
        ax[1].set_ylabel('Density',fontsize=18)

    # Clean up and save
    plt.tight_layout()
    if save_file !="":
        save_file = os.path.join(save_file, dropout+'.png')
        print(save_file)
        plt.savefig(save_file)

def plot_over_fitting_full_model(full_results, run_params):
    savefile = run_params['fig_overfitting_dir']
    plot_over_fitting(full_results, 'Full',save_file=savefile)

def plot_over_fitting_summary(full_results, run_params, plot_dropouts=True):
    '''
        Plots a summary figure that shows which kernels were the most responsible for overfitting.
        
        INPUTS:
        full_results, with overfitting values
            full_results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='full')
            gat.compute_over_fitting_proportion(full_results,run_params)
        run_params, the parameter dictionary for this model version
        
        SAVES:
        a summary figure
    '''
    # Set up
    plt.figure(figsize=(6,6))
    p = []
    labels = [] 

    # Iterate over model dropouts, and get mean overfitting proportion
    if plot_dropouts:
        plot_list = run_params['dropouts'].keys()
    else:
        plot_list = run_params['kernels'].keys()
    for index,d in enumerate(plot_list):
        if (d != "Full")&(not d.startswith('single-')):
            if d+'__dropout_overfit_proportion' in full_results:
                p.append(np.mean(full_results[d+'__dropout_overfit_proportion'].where(lambda x: (x<1)&(x>-1))))        
                labels.append(d)
    
    # Sort by proportion, and save order for yticks
    sort_labels=[]
    for index,x in enumerate(sorted(zip(p,labels))):
        plt.plot(x[0],index,'ko')
        sort_labels.append(x[1])

    # Clean up plot and save
    plt.yticks(range(0,len(sort_labels)),labels=sort_labels)
    plt.xlabel('Avg. Overfitting fraction from kernel')
    plt.axvline(0,color='k',alpha=.25)
    plt.tight_layout()
    filename = os.path.join(run_params['fig_overfitting_dir'],'over_fitting_summary.png')
    plt.savefig(filename)

def plot_all_over_fitting(full_results, run_params):
    '''
        Iterates over all the dropouts and plots the over_fitting_proportion
    
        INPUTS:
        full_results, with overfitting values
            full_results = gat.retrieve_results(search_dict={'glm_version':version}, results_type='full')
            gat.compute_over_fitting_proportion(full_results,run_params)
        run_params, the parameter dictionary for this run, used for where to save and which dropouts to plot

        SAVES:
        a collection of figures
    '''
    # Iterate over model dropouts
    for d in run_params['dropouts']:
        try:
            # Plot each dropout
            plot_over_fitting(full_results, d,save_file=run_params['fig_overfitting_dir'])
            plt.close(plt.gcf().number)
        except:
            # Plot crashed for some reason, print error and move on
            print('crashed - '+d)
            plt.close(plt.gcf().number)

def plot_top_level_dropouts(results_pivoted, filter_cre=False, cre='Slc17a7-IRES2-Cre',bins=150, cmax=10):
    '''
         IN DEVELOPMENT
    '''
    if filter_cre:
        rsp = results_pivoted.query('(variance_explained_full > 0.01) & (cre_line == @cre)').copy()
    else:
        rsp = results_pivoted.query('variance_explained_full > 0.01').copy()
        cre='All'
    rsp.fillna(value=0,inplace=True)

    #fig, ax = plt.subplots(1,3,figsize=(12,4))
    #ax[0].plot(rsp['visual'],rsp['behavioral'],'ko',alpha=.1)
    #ax[0].set_ylabel('behavioral')
    #ax[0].set_xlabel('visual')
    #ax[1].plot(rsp['visual'],rsp['cognitive'],'ko',alpha=.1)
    #ax[1].set_ylabel('cognitive')
    #ax[1].set_xlabel('visual')
    #ax[2].plot(rsp['cognitive'],rsp['behavioral'],'ko',alpha=.1)
    #ax[2].set_ylabel('behavioral')
    #ax[2].set_xlabel('cognitive')
    #ax[0].plot([-1,0],[-1,0],'r--')
    #ax[1].plot([-1,0],[-1,0],'r--')
    #ax[2].plot([-1,0],[-1,0],'r--')

    fig, ax = plt.subplots(1,3,figsize=(12,4))
    ax[0].hist2d(rsp['visual'],rsp['behavioral'],bins=bins,density=True, cmax=cmax,cmap='inferno')
    ax[1].hist2d(rsp['visual'],rsp['cognitive'],bins=bins,density=True, cmax=cmax,cmap='inferno')
    ax[2].hist2d(rsp['cognitive'],rsp['behavioral'],bins=bins,density=True, cmax=cmax,cmap='inferno')
    ax[0].set_ylabel('behavioral')
    ax[0].set_xlabel('visual')
    ax[1].set_ylabel('cognitive')
    ax[1].set_xlabel('visual')
    ax[2].set_ylabel('behavioral')
    ax[2].set_xlabel('cognitive')
    ax[2].set_title(cre)
    plt.tight_layout()

def plot_nested_dropouts(results_pivoted,run_params, num_levels=2,size=0.3,force_nesting=True,filter_cre=False, cre='Slc17a7-IRES2-Cre',invert=False,mixing=True,thresh=-.2,savefig=True,force_subsets=True):
    '''
        Plots the circle plots of clusterd neurons, not the rainbow plots
    '''

    if filter_cre:
        rsp = results_pivoted.query('(variance_explained_full > 0.01) & (cre_line == @cre)').copy()
    else:
        rsp = results_pivoted.query('variance_explained_full > 0.01').copy()

    fig, ax = plt.subplots(1,num_levels+1,figsize=((num_levels+1)*3+1,4))
    cmap= plt.get_cmap("tab20c")
    outer_colors = cmap(np.array([0,4,8,12]))
    inner_colors = cmap(np.array([1,2,3,5,6,7,9,10,11]))
    
    if num_levels==1:
        size=size*2

    if invert:
        r = [1-size,1]
    else:
        r = [1,1-size]   
 
    # Compute Level 1 clusters
    if mixing:
        rsp['level1'] = [np.argmin(x) for x in zip(rsp['visual'],rsp['behavioral'],rsp['cognitive'])]
        rsp['level1'] = [3 if (x[0]<thresh)&(x[1]<thresh) else x[2] for x in zip(rsp['visual'],rsp['behavioral'],rsp['level1'])]
    else:
        rsp['level1'] = [np.argmin(x) for x in zip(rsp['visual'],rsp['behavioral'],rsp['cognitive'])]
    level_1_props = rsp.groupby('level1')['level1'].count()
    if 0 not in level_1_props.index:
        level_1_props.loc[0] = 0
    if 1 not in level_1_props.index:
        level_1_props.loc[1] = 0
    if 2 not in level_1_props.index:
        level_1_props.loc[2] = 0
    level_1_props = level_1_props.sort_index(inplace=False)
    level_1_props = level_1_props/np.sum(level_1_props)

    # Compute Level 2 clusters
    if force_nesting:
        rsp['level2_0'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'])]
        rsp['level2_1'] = [np.argmin(x) for x in zip(rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'])]
        rsp['level2_2'] = [np.argmin(x) for x in zip(rsp['beh_model'],rsp['task'])]
        if mixing:
            rsp['level2_3'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'])]
            rsp['level2'] = [x[x[0]+1]+3*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'],rsp['level2_3'])]
        else:
            rsp['level2'] = [x[x[0]+1]+3*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'])]
        level_2_props = rsp.groupby('level2')['level2'].count()
        for i in range(0,9):    
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        if mixing:
            for i in range(9,15):    
                if i not in level_2_props.index:
                    level_2_props.loc[i] = 0
        level_2_props = level_2_props.sort_index(inplace=False)       
        level_2_props = level_2_props/np.sum(level_2_props)

    elif force_subsets:
        rsp['level2_0'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        rsp['level2_1'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        rsp['level2_2'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        if mixing:
            rsp['level2_3'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
            rsp['level2'] = [x[x[0]+1]+100*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'],rsp['level2_3'])] 
        else:
            rsp['level2'] = [x[x[0]+1]+100*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'])] 
        level_2_props = rsp.groupby('level2')['level2'].count()
        for i in range(0,9):
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        for i in range(100,109):
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        for i in range(200,209):
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        if mixing:
            for i in range(300,309):
                if i not in level_2_props.index:
                    level_2_props.loc[i] = 0       
        level_2_props = level_2_props.sort_index(inplace=False)       
        level_2_props = level_2_props/np.sum(level_2_props)

    else:
        rsp['level2'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        level_2_props = rsp.groupby('level2')['level2'].count()
        level_2_props.loc[8] = 0 # Add third category for cognitive
        level_2_props = level_2_props.sort_index(inplace=False)       
        level_2_props = level_2_props/np.sum(level_2_props)

    # Plot Layer 1 for legend
    wedges, texts= ax[0].pie(level_1_props,radius=0,colors=outer_colors,wedgeprops=dict(width=size,edgecolor='w'))
    ax[0].legend(wedges, ['Visual','Behavioral','Cognitive','Mixed'],loc='center')#,bbox_to_anchor=(0,-.25,1,2))
    ax[0].set_title('Level 1')

    # Plot Layer 2 for legend
    if num_levels ==2:
        wedges, texts = ax[1].pie(level_2_props,radius=0,colors=inner_colors,wedgeprops=dict(width=size,edgecolor='w'))
        ax[1].legend(wedges,['all-images','expectation','omissions','face_motion_energy','licking','pupil_and_running','beh_model','task'],loc='center')#,bbox_to_anchor=(0,-.4,1,2))
        if force_nesting:
            ax[1].set_title('Level 2\nForced Hierarchy')   
        else:
            ax[1].set_title('Level 2')
        final_ax = 2
    else:
        final_ax = 1

    # Plot Full chart
    wedges, texts = ax[final_ax].pie(level_1_props,radius=r[0],colors=outer_colors,wedgeprops=dict(width=size,edgecolor='w'))
    if num_levels ==2:
        wedges, texts = ax[final_ax].pie(level_2_props,radius=r[1],colors=inner_colors,wedgeprops=dict(width=size,edgecolor='w'))
    if filter_cre:
        ax[final_ax].set_title(cre)
    else:
        ax[final_ax].set_title('All cells')

    plt.tight_layout()
    if savefig:
        filename = os.path.join(run_params['fig_clustering_dir'], 'pie_'+str(num_levels))
        if filter_cre:
            filename+='_'+cre[0:3]
        if num_levels ==2:
            if force_nesting:
                filename+='_forced'
        if mixing:
            filename+='_mixing'
        plt.savefig(filename+'.png')
    return level_1_props, level_2_props, rsp

def plot_all_nested_dropouts(results_pivoted, run_params):
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=False, force_nesting=False, num_levels=1)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=1,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=1,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=1,cre='Sst-IRES-Cre')

    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=True, force_nesting=False, num_levels=1)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=1,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=1,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=1,cre='Sst-IRES-Cre')

    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=False, force_nesting=False, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=2,cre='Sst-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=True, force_nesting=False, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=2,cre='Sst-IRES-Cre')

    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=False, force_nesting=True, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=True, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=True, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=True, num_levels=2,cre='Sst-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=True, force_nesting=True, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=True, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=True, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=True, num_levels=2,cre='Sst-IRES-Cre')

def get_lick_triggered_motion_response(ophys_experiment_id, cell_specimen_id):
    '''
    gets lick triggered responses for:
        x-motion correcion
        y-motion correcion
        dff
    returns tidy dataframe
    '''
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    motion_correction = dataset.motion_correction
    motion_correction['timestamps'] = dataset.ophys_timestamps

    licks = dataset.licks

    cell_df = pd.DataFrame({
        'timestamps':dataset.ophys_timestamps,
        'dff':dataset.dff_traces.loc[cell_specimen_id]['dff']
    })

    etrs = {}
    for val in ['x','y']:
        etrs[val] = vbu.event_triggered_response(
            motion_correction, 
            val, 
            licks['timestamps'], 
            time_key='timestamps'
        )
    etrs['dff'] = vbu.event_triggered_response(
        cell_df, 
        'dff', 
        licks['timestamps'], 
        time_key='timestamps'
    )

    etr = etrs['x'].merge(
        etrs['y'],
        left_on=['time','event_number','event_time'],
        right_on=['time','event_number','event_time'],
    )
    etr = etr.merge(
        etrs['dff'],
        left_on=['time','event_number','event_time'],
        right_on=['time','event_number','event_time'],
    )
    return etr

def plot_lick_triggered_motion(ophys_experiment_id, cell_specimen_id, title=''):
    '''
    makes a 3x1 figure showing:
        mean +/95% CI x-motion correction
        mean +/95% CI y-motion correction
        mean +/95% CI dF/F
    surrounding every lick in the session, for a given cell ID
    '''
    event_triggered_response = get_lick_triggered_motion_response(ophys_experiment_id, cell_specimen_id)
    fig,ax=plt.subplots(3,1,figsize=(12,5),sharex=True)
    for row,key in enumerate(['x','y','dff']):
        sns.lineplot(
            data=event_triggered_response,
            x='time',
            y=key,
            n_boot=100,
            ax=ax[row],
        )
    ax[0].set_ylabel('x-correction')
    ax[1].set_ylabel('y-correction')
    ax[2].set_xlabel('time from lick (s)')
    fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


def cosyne_make_dropout_summary_plot(dropout_summary, ax=None, palette=None):
    '''
        Top level function for cosyne summary plot 
        Plots the distribution of dropout scores by cre-line for the visual, behavioral, and cognitive nested models
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    else:
        fig = ax.get_figure()

    dropouts_to_show = ['visual', 'behavioral', 'cognitive']
    plot_dropout_summary_cosyne(dropout_summary, ax=ax, dropouts_to_show=dropouts_to_show, palette=palette)
    ax.tick_params(axis='both',labelsize=20)
    ax.set_ylabel('% decrease in variance explained \n when removing sets of kernels',fontsize=24)
    ax.set_xlabel('Sets of Kernels',fontsize=24)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Visual','Behavioral','Cognitive'])
    ax.axhline(0, color='k',linestyle='--',alpha=.25)
    y = ax.get_yticks()
    ax.set_yticklabels(np.round(y*100).astype(int))
    plt.tight_layout()

    return fig, ax

def plot_population_perturbation(results_pivoted, run_params, dropouts_to_show = ['all-images','omissions','behavioral','task'],sharey=False, include_zero_cells=True):
    # Filter for cells with low variance explained
    if include_zero_cells:
        results_pivoted = results_pivoted.query('not passive').copy()       
    else:
        results_pivoted = results_pivoted.query('(variance_explained_full > 0.005)&(not passive)').copy()    

    # Convert dropouts to positive values
    for dropout in dropouts_to_show:
        results_pivoted[dropout] = results_pivoted[dropout].abs()

    # Add additional columns about experience levels
    experiments_table = loading.get_platform_paper_experiment_table(include_4x2_data=run_params['include_4x2_data'])
    experiment_table_columns = experiments_table.reset_index()[['ophys_experiment_id','last_familiar_active','second_novel_active','cell_type','binned_depth']]
    results_pivoted = results_pivoted.merge(experiment_table_columns, on='ophys_experiment_id')
   
    # Cells Matched across all three experience levels 
    cells_table = loading.get_cell_table(platform_paper_only=True, include_4x2_data=run_params['include_4x2_data'])
    cells_table = cells_table.query('not passive').copy()
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    matched_cells = cells_table.cell_specimen_id.unique()

    # plotting variables
    cell_types = results_pivoted.cell_type.unique()
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    colors = project_colors()

    # Make dataframe
    all_df = results_pivoted.groupby(['experience_level','cre_line'])[dropouts_to_show].mean()
    matched_df =  results_pivoted.query('cell_specimen_id in @matched_cells').groupby(['experience_level','cre_line'])[dropouts_to_show].mean()
    fig, ax = plt.subplots(6,len(dropouts_to_show),figsize=(10,15), sharey=sharey,sharex=sharey)
    plot_population_perturbation_inner('Exc','Sst',all_df, dropouts_to_show, sharey=sharey,ax=ax[0,:],add_title=True)
    plot_population_perturbation_inner('Exc','Vip',all_df, dropouts_to_show, sharey=sharey,ax=ax[1,:])
    plot_population_perturbation_inner('Sst','Vip',all_df, dropouts_to_show, sharey=sharey,ax=ax[2,:])
    plot_population_perturbation_inner('Exc','Vip-Sst',all_df, dropouts_to_show, sharey=sharey,ax=ax[3,:])
    plot_population_perturbation_inner('Exc-Vip','Sst',all_df, dropouts_to_show, sharey=sharey,ax=ax[4,:])
    plot_population_perturbation_inner('Exc-Sst','Vip',all_df, dropouts_to_show, sharey=sharey,ax=ax[5,:])
    plt.tight_layout()
    plt.savefig(run_params['figure_dir']+'/dropout_perturbation.svg')
    plt.savefig(run_params['figure_dir']+'/dropout_perturbation.png')  

    plot_population_perturbation_inner('Vip-Sst','Exc',all_df, dropouts_to_show, sharey=sharey,add_title=True,plot_big=True)
    plt.tight_layout()
    plt.savefig(run_params['figure_dir']+'/dropout_perturbation_VS_E.svg')
    plt.savefig(run_params['figure_dir']+'/dropout_perturbation_VS_E.png')  

def add_perturbation_arrow(startxy,endxy, colors,experience_level,ax):
        dx = (startxy[0]-endxy[0])*0.1
        dy = (startxy[1]-endxy[1])*0.1
        sxy =(startxy[0]-dx,startxy[1]-dy)
        exy =(endxy[0]+dx,endxy[1]+dy)
        ax.annotate("",xy=exy,xytext=sxy,arrowprops=dict(arrowstyle="->",color=colors[experience_level]))

def plot_population_perturbation_inner(x,y,df, dropouts_to_show,sharey=True,all_cells=True,ax=None,add_title=False,plot_big=False):
    # plot
    df= df.rename(index={'Sst-IRES-Cre':'Sst','Vip-IRES-Cre':'Vip','Slc17a7-IRES2-Cre':'Exc'})
    familiar = df.loc['Familiar']
    familiar.loc['Vip-Sst'] = familiar.loc['Vip'] - familiar.loc['Sst']
    familiar.loc['Exc-Sst'] = familiar.loc['Exc'] - familiar.loc['Sst']
    familiar.loc['Exc-Vip'] = familiar.loc['Exc'] - familiar.loc['Vip']
    novel1 = df.loc['Novel 1']
    novel1.loc['Vip-Sst'] = novel1.loc['Vip'] - novel1.loc['Sst']
    novel1.loc['Exc-Sst'] = novel1.loc['Exc'] - novel1.loc['Sst']
    novel1.loc['Exc-Vip'] = novel1.loc['Exc'] - novel1.loc['Vip']
    novelp1 = df.loc['Novel >1']
    novelp1.loc['Vip-Sst'] = novelp1.loc['Vip'] - novelp1.loc['Sst']
    novelp1.loc['Exc-Sst'] = novelp1.loc['Exc'] - novelp1.loc['Sst']
    novelp1.loc['Exc-Vip'] = novelp1.loc['Exc'] - novelp1.loc['Vip']
    colors = project_colors()
    if ax is None:
        fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(10,2.5), sharey=sharey,sharex=sharey)
    for index, feature in enumerate(dropouts_to_show):
        familiarxy =[familiar.loc[x][feature],familiar.loc[y][feature]]
        novel1xy = [novel1.loc[x][feature],novel1.loc[y][feature]]
        novelp1xy = [novelp1.loc[x][feature],novelp1.loc[y][feature]]

        if all_cells:
            ax[index].plot(familiarxy[0],familiarxy[1],'o',color=colors['Familiar'])
            ax[index].plot(novel1xy[0],novel1xy[1],'o',color=colors['Novel 1'])
            ax[index].plot(novelp1xy[0],novelp1xy[1],'o',color=colors['Novel >1'])
            add_perturbation_arrow(familiarxy, novel1xy, colors,'Familiar',ax[index])
            add_perturbation_arrow(novel1xy, novelp1xy, colors,'Novel 1',ax[index])
        else:
            ax[index].plot(familiarxy[0],familiarxy[1],'o',color='lightgray')
            ax[index].plot(novel1xy[0],novel1xy[1],'o',color='lightgray')
            ax[index].plot(novelp1xy[0],novelp1xy[1],'o',color='lightgray')
        ax[index].axis('equal')
        if plot_big:
            ax[index].set_xlabel(x,fontsize=18)
            ax[index].set_ylabel(y,fontsize=18)
            ax[index].tick_params(axis='x',labelsize=16)
            ax[index].tick_params(axis='y',labelsize=16)
        else:
            ax[index].set_xlabel(x,fontsize=12)
            ax[index].set_ylabel(y,fontsize=12)
            ax[index].tick_params(axis='x',labelsize=10)
            ax[index].tick_params(axis='y',labelsize=10)
        ax[index].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        ax[index].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        if add_title:
            if plot_big:
                ax[index].set_title(feature,fontsize=18)           
            else:
                ax[index].set_title(feature,fontsize=12)
    return ax

def plot_population_averages_by_depth(results_pivoted, run_params, dropouts_to_show = ['all-images','omissions','behavioral','task'],sharey=False,include_zero_cells=True,add_stats=True,extra='',equipment="mesoscope",area=['VISp'],savefig=False):
    '''
        Plots the average dropout scores for each cre line, on each experience level. 
        Includes all cells, and matched only cells. 
        sharey (bool) if True, shares the same y axis across dropouts of the same cre line
        include_zero_cells (bool) if False, requires cells have a minimum of 0.005 variance explained
        boxplot (bool), if True, uses boxplot instead of pointplot. In general, very hard to read
    '''  
 
    if not sharey:
        extra = extra+'_untied'
    if include_zero_cells:
        extra = extra+'_with_zero_cells'

    if equipment == "mesoscope":
        extra = extra+'_mesoscope'
        results_pivoted = results_pivoted.query('equipment_name == "MESO.1"').copy()
    else:
        extra = extra+'_scientifica'
        results_pivoted = results_pivoted.query('equipment_name != "MESO.1"').copy()

    extra = extra+'_'.join(area)
    results_pivoted=results_pivoted.query('targeted_structure in @area').copy()
 
    # Filter for cells with low variance explained
    if include_zero_cells:
        results_pivoted = results_pivoted.query('not passive').copy()       
    else:
        extra = extra + '_no_zero_cells'
        results_pivoted = results_pivoted.query('(variance_explained_full > 0.005)&(not passive)').copy()
    
    # Add binned depth
    results_pivoted['coarse_binned_depth'] = [coarse_bin_depth(x) for x in results_pivoted['imaging_depth']]   

    # Convert dropouts to positive values
    for dropout in dropouts_to_show:
        results_pivoted[dropout] = results_pivoted[dropout].abs()
    
    # Add additional columns about experience levels
    experiments_table = loading.get_platform_paper_experiment_table(include_4x2_data=run_params['include_4x2_data'])
    experiment_table_columns = experiments_table.reset_index()[['ophys_experiment_id','last_familiar_active','second_novel_active','cell_type','binned_depth']]
    results_pivoted = results_pivoted.merge(experiment_table_columns, on='ophys_experiment_id')
    
    # plotting variables
    cell_types = results_pivoted.cell_type.unique()
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    experience_level_labels = ['Familiar','Novel','Novel +']
    colors = project_colors()

    summary = {}
    # Iterate cell types and make a plot for each
    for cell_type in cell_types:
        if len(dropouts_to_show) ==3:
            fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(8.1,4), sharey=sharey)       
        else:
            fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(10.8,4), sharey=sharey)
        all_data = results_pivoted.query('cell_type ==@cell_type')
        stats = {}
        summary[cell_type + ' data'] = {}
        # Iterate dropouts and plot each by experience
        for index, feature in enumerate(dropouts_to_show):
            stats[feature] = test_significant_dropout_averages_by_depth(all_data,feature)
            summary[cell_type+' data'][feature] = all_data.groupby(['experience_level','coarse_binned_depth'])[feature].describe()
            # Plot all cells in active sessions 
            ax[index] = sns.pointplot(
                data = all_data,
                x = 'experience_level',
                y= feature,
                hue='coarse_binned_depth', 
                order=['Familiar','Novel 1','Novel >1', 'dummy'], #Fix for seaborn bug
                hue_order=['upper','lower'],
                palette={'upper':'black','lower':'gray'},
                linestyles=['-','--'],
                markers=['o','x'],
                join=True,
                dodge=True,
                ax=ax[index]
            )
            
            all_cell_points = list(ax[index].get_children())
            for x in all_cell_points:
                x.set_zorder(1000)
                        
            if index !=len(dropouts_to_show)-1: 
                ax[index].get_legend().remove()
            else:
                ax[index].get_legend().set_title('Depth') 
            title_feature = feature.replace('all-images','images')
            title_feature = title_feature.replace('omissions_positive','excited')
            title_feature = title_feature.replace('omissions_negative','inhibited')
            title_feature = title_feature.replace('_',' ')
            ax[index].set_title(title_feature,fontsize=20)

            ax[index].set_ylabel('')
            ax[index].set_xlabel('')
            ax[index].set_xticks([0,1,2])
            ax[index].set_xticklabels(experience_level_labels, rotation=90)
            ax[index].set_xlim(-.5,2.5)
            ax[index].tick_params(axis='x',labelsize=16)
            ax[index].tick_params(axis='y',labelsize=16)
            ax[index].set_ylim(bottom=0)
            ax[index].spines['top'].set_visible(False)
            ax[index].spines['right'].set_visible(False)
            ax[index].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        if add_stats:
            for index, feature in enumerate(dropouts_to_show):
                y1h = ax[index].get_ylim()[1]*1.05
                if stats[feature]['Familiar'].pvalue<0.05:
                    ax[index].text(0,y1h,'*')
                if stats[feature]['Novel 1'].pvalue<0.05:
                    ax[index].text(1,y1h,'*')
                if stats[feature]['Novel >1'].pvalue<0.05:
                    ax[index].text(2,y1h,'*')
                ax[index].set_ylim(0,y1h*1.05)
        ax[0].set_ylabel('Coding Score',fontsize=20)
        plt.suptitle(cell_type+', '+' & '.join(area),fontsize=20)
        fig.tight_layout() 
        filename = run_params['figure_dir']+'/dropout_average_by_depth_'+cell_type[0:3]+extra+'.svg'
        if savefig:
            plt.savefig(run_params['figure_dir']+'/dropout_average_by_depth_'+cell_type[0:3]+extra+'.png')
            print('Figure saved to: '+filename)
            plt.savefig(filename)
        summary[cell_type + ' stats'] = stats
    return summary

def plot_population_averages_by_area(results_pivoted, run_params, dropouts_to_show = ['all-images','omissions','behavioral','task'],sharey=False,include_zero_cells=True,add_stats=True,extra='',equipment="mesoscope",savefig=False):
    '''
        Plots the average dropout scores for each cre line, on each experience level. 
        Includes all cells, and matched only cells. 
        sharey (bool) if True, shares the same y axis across dropouts of the same cre line
        include_zero_cells (bool) if False, requires cells have a minimum of 0.005 variance explained
        boxplot (bool), if True, uses boxplot instead of pointplot. In general, very hard to read
    '''  
 
    if not sharey:
        extra = extra+'_untied'
    if include_zero_cells:
        extra = extra+'_with_zero_cells'

    if equipment == "mesoscope":
        extra = extra+'_mesoscope'
        results_pivoted = results_pivoted.query('equipment_name == "MESO.1"').copy()
    else:
        extra = extra+'_scientifica'
        results_pivoted = results_pivoted.query('equipment_name != "MESO.1"').copy()
 
    # Filter for cells with low variance explained
    if include_zero_cells:
        results_pivoted = results_pivoted.query('not passive').copy()       
    else:
        extra = extra + '_no_zero_cells'
        results_pivoted = results_pivoted.query('(variance_explained_full > 0.005)&(not passive)').copy()
     
    # Convert dropouts to positive values
    for dropout in dropouts_to_show:
        results_pivoted[dropout] = results_pivoted[dropout].abs()

    ## Rename experience level
    #results_pivoted['experience_level'][results_pivoted['experience_level'] == "Novel >1"] = "Novel +"   
    #results_pivoted['experience_level'][results_pivoted['experience_level'] == "Novel 1"] = "Novel"   
 
    # Add additional columns about experience levels
    experiments_table = loading.get_platform_paper_experiment_table(include_4x2_data=run_params['include_4x2_data'])
    experiment_table_columns = experiments_table.reset_index()[['ophys_experiment_id','last_familiar_active','second_novel_active','cell_type','binned_depth']]
    results_pivoted = results_pivoted.merge(experiment_table_columns, on='ophys_experiment_id')
   
    # plotting variables
    cell_types = results_pivoted.cell_type.unique()
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    experience_level_labels = ['Familiar','Novel','Novel +']
    colors = project_colors()
    if run_params['include_4x2_data']:
        areas = ['VISp','VISl','VISam','VISal']
        area_colors = {"VISp":'black',"VISl":'gray','VISam':'blue','VISal':'red'}       
        linestyles=['-','-','-','-']
        markers=['o','o','o','o']
        dodge=.25
    else:
        areas = ['VISp','VISl']
        area_colors = {"VISp":'black',"VISl":'gray'}
        linestyles=['-','-']
        markers=['o','o']
        dodge=False
    summary = {}
    # Iterate cell types and make a plot for each
    for cell_type in cell_types:
        if len(dropouts_to_show) == 3:
            fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(8.1,4), sharey=sharey)       
        else:
            fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(10.8,4), sharey=sharey)
        all_data = results_pivoted.query('cell_type ==@cell_type')
        stats = {}
        summary[cell_type + ' data'] = {}
        # Iterate dropouts and plot each by experience
        for index, feature in enumerate(dropouts_to_show):
            stats[feature] = test_significant_dropout_averages_by_area(all_data,feature)
            # Plot all cells in active sessions 
            summary[cell_type+' data'][feature] = all_data.groupby(['experience_level','targeted_structure'])[feature].describe()
            ax[index] = sns.pointplot(
                data = all_data,
                x = 'experience_level',
                y= feature,
                hue='targeted_structure', 
                order=['Familiar','Novel 1','Novel >1', 'dummy'], #Fix for seaborn bug
                hue_order=areas,
                palette=area_colors,
                linestyles=linestyles,
                markers=markers,
                join=False,
                dodge=dodge,
                ax=ax[index]
            )
            
            all_cell_points = list(ax[index].get_children())
            for x in all_cell_points:
                x.set_zorder(1000)
            
            if index != len(dropouts_to_show)-1:
                ax[index].get_legend().remove()
            else:
                ax[index].get_legend().set_title('Area')
            title_feature = feature.replace('all-images','images')
            title_feature = title_feature.replace('omissions_positive','excited')
            title_feature = title_feature.replace('omissions_negative','inhibited')
            title_feature = title_feature.replace('_',' ')
            ax[index].set_title(title_feature,fontsize=20)
            ax[index].set_ylabel('')
            ax[index].set_xlabel('')
            ax[index].set_xticks([0,1,2])
            ax[index].set_xticklabels(experience_level_labels, rotation=90)
            ax[index].set_xlim(-.5,2.5)
            ax[index].tick_params(axis='x',labelsize=16)
            ax[index].tick_params(axis='y',labelsize=16)
            ax[index].set_ylim(bottom=0)
            ax[index].spines['top'].set_visible(False)
            ax[index].spines['right'].set_visible(False)
            ax[index].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        if add_stats:
            for index, feature in enumerate(dropouts_to_show):
                y1h = ax[index].get_ylim()[1]*1.05
                if stats[feature]['Familiar'].pvalue<0.05:
                    ax[index].text(0,y1h,'*')
                if stats[feature]['Novel 1'].pvalue<0.05:
                    ax[index].text(1,y1h,'*')
                if stats[feature]['Novel >1'].pvalue<0.05:
                    ax[index].text(2,y1h,'*')
                ax[index].set_ylim(0,y1h*1.05)
        ax[0].set_ylabel('Coding Score',fontsize=20)
        plt.suptitle(cell_type,fontsize=20)
        fig.tight_layout() 
        filename = run_params['figure_dir']+'/dropout_average_by_area_'+cell_type[0:3]+extra+'.svg'
        if savefig:
            plt.savefig(run_params['figure_dir']+'/dropout_average_by_area_'+cell_type[0:3]+extra+'.png')
            plt.savefig(filename)
            print('Figure saved to: '+filename)
        summary[cell_type + ' stats'] = stats
    return summary

def get_matched_cells_with_ve(cells_table, results_pivoted,threshold):
    # determine max VE for each cell
    cells_table = pd.merge(cells_table.reset_index(),results_pivoted[['cell_roi_id','variance_explained_full']], on='cell_roi_id',validate='1:1')

    # get cells with maxVE about threshold
    cells_with_ve = cells_table.groupby('cell_specimen_id')['variance_explained_full'].max().to_frame().query('variance_explained_full >= @threshold')

    return cells_with_ve.index.values


def plot_population_averages(results_pivoted, run_params, dropouts_to_show = ['all-images','omissions','behavioral','task'],sharey=True,include_zero_cells=True,boxplot=False,add_stats=True,extra='',strict_experience_matching=False,plot_by_cell_type=False,across_session=False,stats_on_across=True, matched_with_variance_explained=False,matched_ve_threshold=0,savefig=False):
    '''
        Plots the average dropout scores for each cre line, on each experience level. 
        Includes all cells, and matched only cells. 

        dropouts_to_show (list of str), list of the dropout scores to show
        sharey (bool) if True, shares the same y axis across dropouts of the 
            same cre line
        include_zero_cells (bool) if False, requires cells have a minimum of 
            0.005 variance explained
        boxplot (bool), if True, uses boxplot instead of pointplot. In general, 
            very hard to read
        add_stats (bool) adds anova followed by tukeyHD stats
        extra (str) add an extra string to the filename
        strict_experience_matching (bool) if True, require matched cells are 
            strictly last familiar and first novel repeat, instead of just one 
            session of each experience level
        plot_by_cell_type (bool). Whether to make each row by cell type (True) 
            or by dropout (False)
        across_session (bool) Whether to compare within and across session dropouts. 
            if True, then dropouts_to_show should be each dropout labeled as "_within"
            and the corresponding "_across" must also be in results_pivoted
        stats_on_across (bool) Only used if across_session = True. Whether to
            perform and plot stats on the across session (True) or within session
            (False) dropout scores. 
        matched_with_variance_explained (bool) Compare with cells with 
            variance_explained_full on at least one session about matched_ve_threshold
        matched_ve_threshold (float) The threshold used by matched_with_variance_explained
        savefig (bool) whether to save the figure or not
        
    ''' 
    
    # Check to make sure across/within dropouts are being called correctly 
    if across_session:
        assert np.all(['_within' in x for x in dropouts_to_show]), 'if across_session, then dropouts_to_show must be within session dropouts'
        assert np.all([x.replace('_within','_across') in results_pivoted for x in dropouts_to_show]), 'across_session dropout not available'

    if not sharey:
        extra = extra+'_untied'
    if include_zero_cells:
        extra = extra+'_with_zero_cells'
 
    # Filter for cells with low variance explained
    if include_zero_cells:
        results_pivoted = results_pivoted.query('not passive').copy()       
    else:
        extra = extra + '_no_zero_cells'
        results_pivoted = results_pivoted.query('(variance_explained_full > 0.005)&(not passive)').copy()    

    # Convert dropouts to positive values
    for dropout in dropouts_to_show:
        if '_signed' in dropout:
            results_pivoted[dropout] = -results_pivoted[dropout]
        else:
            results_pivoted[dropout] = results_pivoted[dropout].abs()
        if across_session:
            # In addition to _within dropout, need to convert the corresponding across session dropout
            results_pivoted[dropout.replace('_within','_across')] = results_pivoted[dropout.replace('_within','_across')].abs()
    
    # Add additional columns about experience levels
    experiments_table = loading.get_platform_paper_experiment_table(include_4x2_data=run_params['include_4x2_data'])
    experiment_table_columns = experiments_table.reset_index()[['ophys_experiment_id','last_familiar_active','second_novel_active','cell_type','binned_depth']]
    if across_session:
        results_pivoted = results_pivoted.merge(experiment_table_columns, on='ophys_experiment_id',suffixes=('','_y'))
    else:
        results_pivoted = results_pivoted.merge(experiment_table_columns, on='ophys_experiment_id')

    # Cells Matched across all three experience levels 
    cells_table = loading.get_cell_table(platform_paper_only=True,include_4x2_data=run_params['include_4x2_data'])
    cells_table = cells_table.query('not passive').copy()
    cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
    matched_cells = cells_table.cell_specimen_id.unique()
    
    # Strictly matched cells in the last familiar, and second novel session
    if strict_experience_matching:
        cells_table = loading.get_cell_table(platform_paper_only=True,include_4x2_data=run_params['include_4x2_data'])
        cells_table = cells_table.query('not passive').copy()
        cells_table = utilities.limit_to_last_familiar_second_novel_active(cells_table)
        cells_table = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table)
        strict_matched_cells = cells_table.cell_specimen_id.unique()
        extra = extra + "_strict_matched"

    if matched_with_variance_explained:
        matched_cells_with_ve = get_matched_cells_with_ve(cells_table, results_pivoted,matched_ve_threshold)
        extra = extra + "_matched_with_ve_"+str(matched_ve_threshold)

    # plotting variables
    #cell_types = results_pivoted.cell_type.unique()
    cell_types = ['Vip Inhibitory','Sst Inhibitory','Excitatory']
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    experience_level_labels=['Familiar','Novel','Novel +']
    colors = project_colors()

    if plot_by_cell_type:
        # make combined across cre line plot
        fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(10,4), sharey=sharey)
        for index, feature in enumerate(dropouts_to_show):
            # plots three cre-lines in standard colors
            ax[index] = sns.pointplot(
                data = results_pivoted,
                x = 'experience_level',
                y= feature,
                hue='cre_line',
                hue_order = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre'],
                order=experience_levels,
                palette = colors,
                join=True,
                ax=ax[index],
                legend=False,
            )
            ax[index].get_legend().remove()
            ax[index].axhline(0,color='k',linestyle='--',alpha=.25)
            ax[index].set_title(feature,fontsize=20)
            ax[index].set_ylabel('')
            ax[index].set_xlabel('')
            ax[index].set_xticklabels(experience_level_labels, rotation=90)
            ax[index].tick_params(axis='x',labelsize=16)
            ax[index].tick_params(axis='y',labelsize=16)
        ax[0].set_ylabel('Coding Score',fontsize=20)
        plt.tight_layout()
        filename = run_params['figure_dir']+'/dropout_average_combined'+extra+'.svg'
        if savefig:
            print('Figure saved to: '+filename)
            plt.savefig(filename)   
    
        # Iterate cell types and make a plot for each
        for cell_type in cell_types:
            fig, ax = plt.subplots(1,len(dropouts_to_show),figsize=(10,4), sharey=sharey)
            all_data = results_pivoted.query('cell_type ==@cell_type')
            matched_data = all_data.query('cell_specimen_id in @matched_cells')
            if strict_experience_matching:
                strict_matched_data = all_data.query('cell_specimen_id in @strict_matched_cells') 
            if matched_with_variance_explained:
                matched_cells_with_ve_data = all_data.query('cell_specimen_id in @matched_cells_with_ve')

            stats = {}
            # Iterate dropouts and plot each by experience
            for index, feature in enumerate(dropouts_to_show):
                anova, tukey = test_significant_dropout_averages(all_data,feature)
                stats[feature]=(anova, tukey)
                # Plot all cells in active sessions
                if boxplot:
                    ax[index] = sns.boxplot(
                        data = all_data,
                        x = 'experience_level',
                        y= feature,
                        hue='experience_level',
                        order=['Familiar','Novel 1','Novel >1', 'dummy'], #Fix for seaborn bug
                        hue_order=experience_levels,
                        palette = colors,
                        showfliers=False,
                        ax=ax[index]
                    )
                else:
                    ax[index] = sns.pointplot(
                        data = all_data,
                        x = 'experience_level',
                        y= feature,
                        hue='experience_level', 
                        order=['Familiar','Novel 1','Novel >1', 'dummy'], #Fix for seaborn bug
                        hue_order=experience_levels,
                        palette = colors,
                        join=False,
                        ax=ax[index]
                    )
    
                all_cell_points = list(ax[index].get_children())
                for x in all_cell_points:
                    x.set_zorder(1000)
                
                # Plot cells in matched active sessions
                ax[index] = sns.pointplot(
                    data = matched_data,
                    x = 'experience_level',
                    y=feature,
                    order=experience_levels,
                    color='lightgray',
                    join=True,
                    ax=ax[index],
                )
    
                if strict_experience_matching:
                    # Plot cells in matched active sessions
                    ax[index] = sns.pointplot(
                        data = strict_matched_data,
                        x = 'experience_level',
                        y=feature,
                        order=experience_levels,
                        color='navajowhite',
                        join=True,
                        ax=ax[index],
                    )
                if matched_with_variance_explained:
                    # Plot cells in matched active sessions
                    ax[index] = sns.pointplot(
                        data = matched_cells_with_ve_data,
                        x = 'experience_level',
                        y=feature,
                        order=experience_levels,
                        color='navajowhite',
                        join=True,
                        ax=ax[index],
                    )
                 
                if index !=3:
                    ax[index].get_legend().remove() 
                #ax[index].axhline(0,color='k',linestyle='--',alpha=.25)
                ax[index].set_title(feature,fontsize=20)
                ax[index].set_ylabel('')
                ax[index].set_xlabel('')
                ax[index].set_xticks([0,1,2])
                ax[index].set_xticklabels(experience_level_labels, rotation=90)
                ax[index].set_xlim(-.5,2.5)
                ax[index].tick_params(axis='x',labelsize=16)
                ax[index].tick_params(axis='y',labelsize=16)
                ax[index].set_ylim(bottom=0)
                ax[index].spines['top'].set_visible(False)
                ax[index].spines['right'].set_visible(False)
                ax[index].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    
            if add_stats:
                ytop = ax[0].get_ylim()[1]
                y1 = ytop
                y1h = ytop*1.05
                y2 = ytop*1.1
                y2h = ytop*1.15
    
                for index, feature in enumerate(dropouts_to_show):
                    (anova, tukey) = stats[feature]
                    if anova.pvalue<0.05:
                        for tindex, row in tukey.iterrows():
                            if row.x2-row.x1 > 1:
                                y = y2
                                yh = y2h
                            else:
                                y = y1
                                yh = y1h 
                            if row.reject:
                                ax[index].plot([row.x1,row.x1,row.x2,row.x2],[y,yh,yh,y],'k-')
                                ax[index].text(np.mean([row.x1,row.x2]),yh, '*')
                    ax[index].set_ylim(0,ytop*1.2)
            ax[0].set_ylabel('Coding Score',fontsize=18)
            plt.suptitle(cell_type,fontsize=18)
            fig.tight_layout()
            filename = run_params['figure_dir']+'/dropout_average_'+cell_type[0:3]+extra+'.svg' 
            if savefig:
                plt.savefig(filename) 
                print('Figure saved to: '+filename)

    # Repeat the plots but transposed
    # Iterate cell types and make a plot for each
    summary_data = {}
    for index, feature in enumerate(dropouts_to_show):   
        fig, ax = plt.subplots(1,4,figsize=(10.8,4), sharey=sharey) 

        ax[3] = sns.pointplot(
            data = results_pivoted,
            x = 'experience_level',
            y= feature,
            hue='cre_line',
            hue_order = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre'],
            order=experience_levels,
            palette = colors,
            join=True,
            ax=ax[3],
            legend=False,
        )
        ax[3].get_legend().remove()
        ax[3].axhline(0,color='k',linestyle='--',alpha=.25)
        ax[3].set_title('Combined',fontsize=20)
        ax[3].set_ylabel('')
        ax[3].set_xlabel('')
        ax[3].set_xticklabels(experience_level_labels, rotation=90)
        ax[3].tick_params(axis='x',labelsize=16)
        ax[3].tick_params(axis='y',labelsize=16)
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)

        summary_data[feature + ' data'] = {}
        stats = {}
        # Iterate dropouts and plot each by experience
        for cindex, cell_type in enumerate(cell_types):
            all_data = results_pivoted.query('cell_type ==@cell_type')
            matched_data = all_data.query('cell_specimen_id in @matched_cells')
            if strict_experience_matching:
                strict_matched_data = all_data.query('cell_specimen_id in @strict_matched_cells')
            if matched_with_variance_explained:
                matched_cells_with_ve_data = all_data.query('cell_specimen_id in @matched_cells_with_ve')
            if across_session & stats_on_across:
                stats_feature = feature.replace('_within','_across')
            else:
                stats_feature = feature
            if matched_with_variance_explained:
                anova, tukey = test_significant_dropout_averages(matched_cells_with_ve_data,stats_feature)
            else:
                anova, tukey = test_significant_dropout_averages(all_data,stats_feature)
            stats[cell_type]=(anova, tukey)
            summary_data[feature+' data'][cell_type+' all data'] = all_data.groupby(['experience_level'])[feature].describe()
            summary_data[feature+' data'][cell_type+' matched data'] = matched_data.groupby(['experience_level'])[feature].describe()
            # Plot all cells in active sessions
            if boxplot:
                ax[cindex] = sns.boxplot(
                    data = all_data,
                    x = 'experience_level',
                    y= feature,
                    hue='experience_level',
                    order=['Familiar','Novel 1','Novel >1', 'dummy'], #Fix for seaborn bug
                    hue_order=experience_levels,
                    palette = colors,
                    showfliers=False,
                    ax=ax[cindex]
                )
            else:
                if not across_session:
                    ax[cindex] = sns.pointplot(
                        data = all_data,
                        x = 'experience_level',
                        y= feature,
                        hue='experience_level', 
                        order=['Familiar','Novel 1','Novel >1', 'dummy'], #Fix for seaborn bug
                        hue_order=experience_levels,
                        palette = colors,
                        join=False,
                        ax=ax[cindex]
                    )

            all_cell_points = list(ax[cindex].get_children())
            for x in all_cell_points:
                x.set_zorder(1000)
            
            # Plot cells in matched active sessions
            ax[cindex] = sns.pointplot(
                data = matched_data,
                x = 'experience_level',
                y=feature,
                order=experience_levels,
                color='lightgray',
                join=True,
                ax=ax[cindex],
            )

            if across_session:
                ax[cindex] = sns.pointplot(
                    data = all_data,
                    x = 'experience_level',
                    y=feature.replace('_within','_across'),
                    order=experience_levels,
                    color='yellowgreen',
                    join=True,
                    ax=ax[cindex],
                )               

            if strict_experience_matching:
                # Plot cells in matched active sessions
                summary_data[feature+' data'][cell_type+' strict matched data'] = strict_matched_data.groupby(['experience_level'])[feature].describe()
                ax[cindex] = sns.pointplot(
                    data = strict_matched_data,
                    x = 'experience_level',
                    y=feature,
                    order=experience_levels,
                    color='navajowhite',
                    join=True,
                    ax=ax[cindex],
                )
            if matched_with_variance_explained:
                # Plot cells in matched active sessions
                ax[cindex] = sns.pointplot(
                    data = matched_cells_with_ve_data,
                    x = 'experience_level',
                    y=feature,
                    order=experience_levels,
                    color='navajowhite',
                    join=True,
                    ax=ax[cindex],
                )
             
           
 
            if (cindex !=3 )&( not across_session):
                ax[cindex].get_legend().remove() 
            if '_signed' in feature:
                ax[cindex].axhline(0,color='k',linestyle='--',alpha=.25)
            ax[cindex].set_title(cell_type,fontsize=20)
            ax[cindex].set_ylabel('')
            ax[cindex].set_xlabel('')
            ax[cindex].set_xticks([0,1,2])
            ax[cindex].set_xticklabels(experience_level_labels, rotation=90)
            ax[cindex].set_xlim(-.5,2.5)
            ax[cindex].tick_params(axis='x',labelsize=16)
            ax[cindex].tick_params(axis='y',labelsize=16)
            #ax[cindex].set_ylim(bottom=0)
            ax[cindex].spines['top'].set_visible(False)
            ax[cindex].spines['right'].set_visible(False)
            ax[cindex].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        if add_stats:
            ytop = ax[0].get_ylim()[1]
            y1 = ytop
            y1h = ytop*1.05
            y2 = ytop*1.1
            y2h = ytop*1.15
            if across_session & stats_on_across:
                stats_color = 'olivedrab'
            elif matched_with_variance_explained:
                stats_color='navajowhite'
            else:
                stats_color='k'

            for cindex, cell_type in enumerate(cell_types):
                (anova, tukey) = stats[cell_type]
                if anova.pvalue<0.05:
                    for tindex, row in tukey.iterrows():
                        if row.x2-row.x1 > 1:
                            y = y2
                            yh = y2h
                        else:
                            y = y1
                            yh = y1h 
                        if row.reject:
                            ax[cindex].plot([row.x1,row.x1,row.x2,row.x2],[y,yh,yh,y],'-',color=stats_color)
                            ax[cindex].text(np.mean([row.x1,row.x2]),yh, '*')
                #ax[index].set_ylim(0,ytop*1.2)
        clean_feature = feature.replace('all-images','images')
        clean_feature = clean_feature.replace('_positive',' excited')
        clean_feature = clean_feature.replace('_negative',' inhibited')
        clean_feature = clean_feature.replace('_within','')
        clean_feature = clean_feature.replace('_', ' ')
        ax[0].set_ylabel(clean_feature+'\nCoding Score',fontsize=20)
        plt.suptitle(clean_feature,fontsize=18)
        if '_signed' not in feature:
            ax[0].set_ylim(bottom=0)
            ax[1].set_ylim(bottom=0)
            ax[2].set_ylim(bottom=0)
            ax[3].set_ylim(bottom=0)
        fig.tight_layout() 
        if across_session & stats_on_across:
            extra = extra + '_stats_on_across'
        elif across_session:
            extra = extra + '_stats_on_within'
        filename = run_params['figure_dir']+'/dropout_average_'+clean_feature.replace(' ','_')+extra+'.svg'
        if savefig:
            plt.savefig(filename)
            print('Figure saved to: '+filename)
        summary_data[feature+' stats'] = stats

    return summary_data

def test_significant_across_cell(data, feature):

    data = data[~data[feature].isnull()].copy()
    anova = stats.f_oneway(
        data.query('cre_line == "Slc17a7-IRES2-Cre"')[feature],  
        data.query('cre_line == "Sst-IRES-Cre"')[feature],  
        data.query('cre_line == "Vip-IRES-Cre"')[feature]
        )
    comp = mc.MultiComparison(data[feature], data['cre_line'])
    post_hoc_res = comp.tukeyhsd()
    tukey_table = pd.read_html(post_hoc_res.summary().as_html(),header=0, index_col=0)[0]
    tukey_table = tukey_table.reset_index()
    mapper = {
        'Slc17a7-IRES2-Cre':0,
        'Sst-IRES-Cre':1,
        'Vip-IRES-Cre':2,
        }
    tukey_table['x1'] = [mapper[str(x)] for x in tukey_table['group1']]
    tukey_table['x2'] = [mapper[str(x)] for x in tukey_table['group2']]
    return anova, tukey_table



def test_significant_dropout_averages(data,feature):
    data = data[~data[feature].isnull()].copy()
    anova = stats.f_oneway(
        data.query('experience_level == "Familiar"')[feature],  
        data.query('experience_level == "Novel >1"')[feature],  
        data.query('experience_level == "Novel 1"')[feature]
        )
    comp = mc.MultiComparison(data[feature], data['experience_level'])
    post_hoc_res = comp.tukeyhsd()
    tukey_table = pd.read_html(post_hoc_res.summary().as_html(),header=0, index_col=0)[0]
    tukey_table = tukey_table.reset_index()
    mapper = {
        'Familiar':0,
        'Novel 1':1,
        'Novel >1':2,
        }
    tukey_table['x1'] = [mapper[str(x)] for x in tukey_table['group1']]
    tukey_table['x2'] = [mapper[str(x)] for x in tukey_table['group2']]
    return anova, tukey_table


def test_significant_dropout_averages_by_area(data,feature):
    data = data[~data[feature].isnull()].copy()
    ttests = {}
    for experience in data['experience_level'].unique():
       ttests[experience] = stats.ttest_ind(
            data.query('experience_level == @experience & targeted_structure == "VISp"')[feature],  
            data.query('experience_level == @experience & targeted_structure == "VISl"')[feature],
            )
    return ttests


def test_significant_dropout_averages_by_depth(data,feature):
    data = data[~data[feature].isnull()].copy()
    ttests = {}
    for experience in data['experience_level'].unique():
       ttests[experience] = stats.ttest_ind(
            data.query('experience_level == @experience & coarse_binned_depth == "upper"')[feature],  
            data.query('experience_level == @experience & coarse_binned_depth == "lower"')[feature],
            )
    return ttests


def plot_dropout_individual_population(results, run_params,ax=None,palette=None,use_violin=False,add_median=True,include_zero_cells=True,add_title=False,use_single=False,savefig=False): 
    '''
        Makes a bar plot that shows the population dropout summary by cre line for different regressors 
        palette , color palette to use. If None, uses gvt.project_colors()
        use_violion (bool) if true, uses violin, otherwise uses boxplots
        add_median (bool) if true, adds a line at the median of each population
        include_zero_cells (bool) if true, uses all cells, otherwise uses a threshold for minimum variance explained
    '''
    dropouts_to_show = ['all-images','image0','image1','image2','image3','image4','image5','image6','image7','','all-omissions','omissions','post-omissions','','behavioral','licks','pupil','running','','task','hits','misses','all-hits','all-misses','post-hits','post-misses']

    dropouts_to_show = [x for x in dropouts_to_show if (len(x) == 0) or (x in run_params['dropouts']) ]
    if ax is None:
        height = 8
        width=18
        horz_offset = 2
        vertical_offset = 2.5
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(horz_offset),Size.Fixed(width-horz_offset-.5)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  
 
    if palette is None:
        palette = project_colors()

    if include_zero_cells:
        threshold = 0
    else:
        if 'dropout_threshold' in run_params:
            threshold = run_params['dropout_threshold']
        else:
            threshold = 0.005

    cre_lines = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']

    if 'post-omissions' not in results.dropout.unique():
        dropouts_to_show = [x for x in dropouts_to_show if x not in ['all-omissions','post-omissions']]
    if 'post-hits' not in results.dropout.unique():
        dropouts_to_show = [x for x in dropouts_to_show if x not in ['all-hits','post-hits']]
    if 'post-misses' not in results.dropout.unique():
        dropouts_to_show = [x for x in dropouts_to_show if x not in ['all-misses','post-misses']]
    if 'post-passive_change' not in results.dropout.unique():
        dropouts_to_show = [x for x in dropouts_to_show if x not in ['all-passive_change','post-passive_change']]
 
    if use_single:
        dropouts_to_show = [x if x=='' else 'single-'+x for x in dropouts_to_show]
 
    data_to_plot = results.query('not passive').query('dropout in @dropouts_to_show and variance_explained_full > {}'.format(threshold)).copy()
    data_to_plot['explained_variance'] = -1*data_to_plot['adj_fraction_change_from_full']
    if use_violin:
        plot1= sns.violinplot(
            data = data_to_plot,
            x='dropout',
            y='explained_variance',
            hue='cre_line',
            order=dropouts_to_show,
            hue_order=cre_lines,
            fliersize=0,
            ax=ax,
            inner='quartile',
            linewidth=0,
            palette=palette,
            cut = 0
        )
        if add_median:
            lines = plot1.get_lines()
            for index, line in enumerate(lines):
                if np.mod(index,3) == 0:
                    line.set_linewidth(0)
                elif np.mod(index,3) == 1:
                    line.set_linewidth(1)
                    line.set_color('r')
                    line.set_linestyle('-')
                elif np.mod(index,3) == 2:
                    line.set_linewidth(0)
        plt.axhline(0,color='k',alpha=.25)

    else:
        sns.boxplot(
            data = data_to_plot,
            x='dropout',
            y='explained_variance',
            hue='cre_line',
            order=dropouts_to_show,
            hue_order=cre_lines,
            fliersize=0,
            ax=ax,
            palette=palette
        )
    ax.set_ylim(0,1)
    h,labels =ax.get_legend_handles_labels()
    clean_labels={
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory'
        }
    mylabels = [clean_labels[x] for x in labels]
    ax.legend(h,mylabels,loc='upper right',fontsize=18)
    #ax.set_ylabel('Fraction reduction \nin explained variance',fontsize=20)
    ax.set_ylabel('Coding Score',fontsize=20)
    if use_single:
        ax.set_xlabel('Only component included',fontsize=20)
        xticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        labels = [x.get_text().replace('single-','') for x in labels]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
               
    else:
        ax.set_xlabel('Withheld component',fontsize=20)
    ax.tick_params(axis='x',labelsize=18,rotation=90)
    ax.tick_params(axis='y',labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if add_title:
        plt.title(run_params['version'])
    if use_violin:
        filename = run_params['figure_dir']+'/dropout_individual.svg'
    elif use_single:
        filename = run_params['figure_dir']+'/dropout_individual_boxplot_single.svg'
    else:
        filename = run_params['figure_dir']+'/dropout_individual_boxplot.svg'
        plt.savefig(run_params['figure_dir']+'/dropout_individual_boxplot.png')

    if savefig:
        plt.savefig(filename)
        print('Figure saved to: '+filename)

    return data_to_plot.groupby(['cre_line','dropout'])['explained_variance'].describe()

def plot_dropout_summary_population(results, run_params,dropouts_to_show =  ['all-images','omissions','behavioral','task'],ax=None,palette=None,use_violin=False,add_median=True,include_zero_cells=True,add_title=False,dropout_cleaning_threshold=None, exclusion_threshold=None,savefig=False): 
    '''
        Makes a bar plot that shows the population dropout summary by cre line for different regressors 
        palette , color palette to use. If None, uses gvt.project_colors()
        use_violion (bool) if true, uses violin, otherwise uses boxplots
        add_median (bool) if true, adds a line at the median of each population
        include_zero_cells (bool) if true, uses all cells, otherwise uses a threshold for minimum variance explained
    '''
    if ax is None:
        height = 4
        width=12
        horz_offset = 2
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(horz_offset),Size.Fixed(width-horz_offset-.5)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))  
 
    if palette is None:
        palette = project_colors()

    if include_zero_cells:
        threshold = 0
    else:
        threshold=exclusion_threshold
        # if 'dropout_threshold' in run_params:
        #     threshold = run_params['dropout_threshold']
        # else:
        #     threshold = 0.005

    cre_lines = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']

    if ('post-omissions' in results.dropout.unique())&('omissions' in dropouts_to_show):
       dropouts_to_show = ['all-omissions' if x == 'omissions' else x for x in dropouts_to_show]
    if ('post-hits' in results.dropout.unique())&('hits' in dropouts_to_show):
       dropouts_to_show = ['all-hits' if x == 'hits' else x for x in dropouts_to_show]
    if ('post-misses' in results.dropout.unique())&('misses' in dropouts_to_show):
       dropouts_to_show = ['all-misses' if x == 'misses' else x for x in dropouts_to_show]
    if ('post-passive_change' in results.dropout.unique())&('passive_change' in dropouts_to_show):
       dropouts_to_show = ['all-passive_change' if x == 'passive_change' else x for x in dropouts_to_show]
 
    data_to_plot = results.query('not passive').query('dropout in @dropouts_to_show and variance_explained_full > {}'.format(threshold)).copy()
    data_to_plot['explained_variance'] = -1*data_to_plot['adj_fraction_change_from_full']
    if dropout_cleaning_threshold is not None:
        print('Clipping dropout scores for cells with full model VE < '+str(dropout_cleaning_threshold))
        data_to_plot.loc[data_to_plot['adj_variance_explained_full']<dropout_cleaning_threshold,'explained_variance'] = 0 

    if use_violin:
        plot1= sns.violinplot(
            data = data_to_plot,
            x='dropout',
            y='explained_variance',
            hue='cre_line',
            order=dropouts_to_show,
            hue_order=cre_lines,
            fliersize=0,
            ax=ax,
            inner='quartile',
            linewidth=0,
            palette=palette,
            cut = 0
        )
        if add_median:
            lines = plot1.get_lines()
            for index, line in enumerate(lines):
                if np.mod(index,3) == 0:
                    line.set_linewidth(0)
                elif np.mod(index,3) == 1:
                    line.set_linewidth(1)
                    line.set_color('r')
                    line.set_linestyle('-')
                elif np.mod(index,3) == 2:
                    line.set_linewidth(0)
        plt.axhline(0,color='k',alpha=.25)

    else:
        sns.boxplot(
            data = data_to_plot,
            x='dropout',
            y='explained_variance',
            hue='cre_line',
            order=dropouts_to_show,
            hue_order=cre_lines,
            fliersize=0,
            ax=ax,
            palette=palette,
            width=.7,
        )
    ax.set_ylim(0,1)
    h,labels =ax.get_legend_handles_labels()
    clean_labels={
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory'
        }
    mylabels = [clean_labels[x] for x in labels]
    ax.legend(h,mylabels,loc='upper right',fontsize=16)
    #ax.set_ylabel('Fraction reduction \nin explained variance',fontsize=20)
    ax.set_ylabel('Coding Score',fontsize=20)
    ax.set_xlabel('Withheld component',fontsize=20)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['images','omissions','behavioral','task'])
    ax.tick_params(axis='x',labelsize=16)
    ax.tick_params(axis='y',labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if add_title:
        plt.title(run_params['version'])
    if dropout_cleaning_threshold is not None:
        extra ='_'+str(dropout_cleaning_threshold) 
    elif not include_zero_cells:
        extra ='_remove_'+str(exclusion_threshold)
    else:
        extra=''
    if savefig:
        if use_violin:
            filename = run_params['figure_dir']+'/dropout_summary'+extra+'.svg'
            plt.savefig(filename)
        else:
            filename = run_params['figure_dir']+'/dropout_summary_boxplot'+extra+'.svg'
            print('Figure saved to: '+filename)
            plt.savefig(filename)
            plt.savefig(run_params['figure_dir']+'/dropout_summary_boxplot'+extra+'.png')
    return data_to_plot.groupby(['cre_line','dropout'])['explained_variance'].describe() 

def plot_fraction_summary_population(results_pivoted, run_params,sharey=True,kernel_excitation=False,kernel=None,savefig=False):
    if kernel_excitation:
        assert kernel is not None, "Need to name the excited kernel"
    else:
        assert kernel is None, "Kernel Excitation is False, you should not provide a named kernel"

    # compute coding fractions
    results_pivoted = results_pivoted.query('not passive').copy()
    results_pivoted['code_anything'] = results_pivoted['variance_explained_full'] > run_params['dropout_threshold'] 
    results_pivoted['code_images'] = results_pivoted['code_anything'] & (results_pivoted['all-images'] < 0)
    results_pivoted['code_omissions'] = results_pivoted['code_anything'] & (results_pivoted['omissions'] < 0)
    results_pivoted['code_behavioral'] = results_pivoted['code_anything'] & (results_pivoted['behavioral'] < 0)
    results_pivoted['code_task'] = results_pivoted['code_anything'] & (results_pivoted['task'] < 0)
    summary_df = results_pivoted.groupby(['cre_line','experience_level'])[['code_anything','code_images','code_omissions','code_behavioral','code_task']].mean()

    # Compute Confidence intervals
    summary_df['n'] = results_pivoted.groupby(['cre_line','experience_level'])[['code_anything','code_images','code_omissions','code_behavioral','code_task']].count()['code_anything']
    summary_df['code_images_ci'] = 1.96*np.sqrt((summary_df['code_images']*(1-summary_df['code_images']))/summary_df['n'])
    summary_df['code_omissions_ci'] = 1.96*np.sqrt((summary_df['code_omissions']*(1-summary_df['code_omissions']))/summary_df['n'])
    summary_df['code_behavioral_ci'] = 1.96*np.sqrt((summary_df['code_behavioral']*(1-summary_df['code_behavioral']))/summary_df['n'])
    summary_df['code_task_ci'] = 1.96*np.sqrt((summary_df['code_task']*(1-summary_df['code_task']))/summary_df['n'])

    if kernel_excitation:
        results_pivoted['code_'+kernel] = results_pivoted['code_anything'] & (results_pivoted[kernel] < 0)
        results_pivoted['code_'+kernel+'_excited'] = results_pivoted['code_anything'] & (results_pivoted[kernel] < 0) & (results_pivoted[kernel+'_excited'])    
        results_pivoted['code_'+kernel+'_inhibited'] = results_pivoted['code_anything'] & (results_pivoted[kernel] < 0) & (results_pivoted[kernel+'_excited']==False) 
        summary_df = results_pivoted.groupby(['cre_line','experience_level'])[['code_anything','code_'+kernel,'code_'+kernel+'_excited','code_'+kernel+'_inhibited']].mean()
        summary_df['n'] = results_pivoted.groupby(['cre_line','experience_level'])[['code_anything','code_'+kernel,'code_'+kernel+'_excited','code_'+kernel+'_inhibited']].count()['code_anything']
        summary_df['code_'+kernel+'_ci'] = 1.96*np.sqrt((summary_df['code_'+kernel]*(1-summary_df['code_'+kernel]))/summary_df['n'])
        summary_df['code_'+kernel+'_excited_ci'] = 1.96*np.sqrt((summary_df['code_'+kernel+'_excited']*(1-summary_df['code_'+kernel+'_excited']))/summary_df['n'])
        summary_df['code_'+kernel+'_inhibited_ci'] = 1.96*np.sqrt((summary_df['code_'+kernel+'_inhibited']*(1-summary_df['code_'+kernel+'_inhibited']))/summary_df['n'])

    # plotting variables
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    experience_level_labels = ['Familiar','Novel','Novel +']
    colors = project_colors()

    if kernel_excitation:
        coding_groups = ['code_'+kernel,'code_'+kernel+'_excited','code_'+kernel+'_inhibited']   
        titles = [kernel.replace('all-images','images'), 'excited','inhibited']
    else:
        coding_groups = ['code_images','code_omissions','code_behavioral','code_task']
        titles = ['images','omissions','behavioral','task']

    # make combined across cre line plot
    if kernel_excitation:
        fig, ax = plt.subplots(1,len(coding_groups),figsize=(8.1,4), sharey=sharey)
    else:
        fig, ax = plt.subplots(1,len(coding_groups),figsize=(10.8,4), sharey=sharey)
    for index, feature in enumerate(coding_groups):
        # plots three cre-lines in standard colors
        ax[index].plot([0,1,2], summary_df.loc['Vip-IRES-Cre'][feature],'-',color=colors['Vip-IRES-Cre'],label='Vip Inhibitory',linewidth=3)
        ax[index].plot([0,1,2], summary_df.loc['Sst-IRES-Cre'][feature],'-',color=colors['Sst-IRES-Cre'],label='Sst Inhibitory',linewidth=3)
        ax[index].plot([0,1,2], summary_df.loc['Slc17a7-IRES2-Cre'][feature],'-',color=colors['Slc17a7-IRES2-Cre'],label='Excitatory',linewidth=3)
        
        ax[index].errorbar([0,1,2], summary_df.loc['Vip-IRES-Cre'][feature],yerr=summary_df.loc['Vip-IRES-Cre'][feature+'_ci'],color=colors['Vip-IRES-Cre'],linewidth=3)
        ax[index].errorbar([0,1,2], summary_df.loc['Sst-IRES-Cre'][feature],yerr=summary_df.loc['Sst-IRES-Cre'][feature+'_ci'],color=colors['Sst-IRES-Cre'],linewidth=3)
        ax[index].errorbar([0,1,2], summary_df.loc['Slc17a7-IRES2-Cre'][feature],yerr=summary_df.loc['Slc17a7-IRES2-Cre'][feature+'_ci'],color=colors['Slc17a7-IRES2-Cre'],linewidth=3)

        ax[index].set_title(titles[index],fontsize=20)
        ax[index].set_ylabel('')
        ax[index].set_xlabel('')
        ax[index].set_xticks([0,1,2])
        ax[index].set_xticklabels(experience_level_labels, rotation=90)
        ax[index].tick_params(axis='x',labelsize=16)
        ax[index].tick_params(axis='y',labelsize=16)
        ax[index].spines['top'].set_visible(False)
        ax[index].spines['right'].set_visible(False)
        ax[index].set_xlim(-.5,2.5)
        ax[index].set_ylim(bottom=0)
        if index ==3:
            ax[index].legend()
    ax[0].set_ylabel('Fraction of cells \n coding for ',fontsize=20)
    plt.tight_layout()
    if savefig:
        if kernel_excitation:
            filename = run_params['figure_dir']+'/coding_fraction_'+kernel+'_summary.svg'  
            plt.savefig(filename)  
            plt.savefig(run_params['figure_dir']+'/coding_fraction_'+kernel+'_summary.png') 
            print('Figure saved to: '+filename) 
        else:
            filename = run_params['figure_dir']+'/coding_fraction_summary.svg'
            plt.savefig(filename)  
            plt.savefig(run_params['figure_dir']+'/coding_fraction_summary.png')  
            print('Figure saved to: '+filename) 
    return summary_df 

def make_cosyne_schematic(glm,cell=1028768972,t_range=5,time_to_plot=3291,alpha=.25):
    '''
        Plots the summary figure for the cosyne abstract with visual, behavioral, and cognitive kernels separated.
        Additionally plots the cell response and model prediction. Hard-wired here for a specific cell, on oeid:830700781
    '''
    t_span = (time_to_plot-t_range, time_to_plot+t_range)
    fig, ax = make_cosyne_summary_figure(glm, cell, t_span,alpha=alpha)
    ax['visual_kernels'].set_ylabel('Kernel Output',fontsize=14)
    ax['visual_kernels'].set_xlabel('Time (s)',fontsize=14)
    ax['behavioral_kernels'].set_ylabel('Kernel Output',fontsize=14)
    ax['behavioral_kernels'].set_xlabel('Time (s)',fontsize=14)
    ax['cognitive_kernels'].set_ylabel('Kernel Output',fontsize=14)
    ax['cognitive_kernels'].set_xlabel('Time (s)',fontsize=14)
    ax['visual_kernels'].set_xlim(t_span) 
    ax['behavioral_kernels'].set_xlim(t_span) 
    ax['cognitive_kernels'].set_xlim(t_span)
    ax['cell_response'].set_xlim(t_span)
    ax['cell_response'].set_ylabel('$\Delta$ F/F',fontsize=14)
    ax['cell_response'].set_xlabel('Time (s)',fontsize=14)
    ax['cell_response'].tick_params(axis='both',labelsize=12)
    ax['visual_kernels'].tick_params(axis='both',labelsize=12) 
    ax['behavioral_kernels'].tick_params(axis='both',labelsize=12) 
    ax['cognitive_kernels'].tick_params(axis='both',labelsize=12)
    ax['visual_kernels'].axhline(0,color='k',alpha=.25) 
    ax['behavioral_kernels'].axhline(0,color='k',alpha=.25) 
    ax['cognitive_kernels'].axhline(0,color='k',alpha=.25)
    ax['cell_response'].axhline(0,color='k',alpha=.25)
    ax['cell_response'].set_ylim(list(np.array(ax['cell_response'].get_ylim())*1.1))
    return fig, ax


def make_cosyne_summary_figure(glm, cell_specimen_id, t_span,dropout_df,alpha =0.35):
    ### PROBABLY BROKEN BY RECENT REFACTORING
    '''
    makes a summary figure for cosyne abstract
    inputs:
        glm: glm object
        cell_specimen_id
        time_to_plot: time to show in center of plot for time-varying axes
        t_span: time range to show around time_to_plot, in seconds
        dropout_df = gsp.plot_dropouts()
    '''
    fig = plt.figure(figsize=(18,10))

    vbuffer = 0.05

    ax = {
        'visual_kernels': vbp.placeAxesOnGrid(fig, xspan=[0, 0.4], yspan=[0, 0.33 - vbuffer]),
        'behavioral_kernels': vbp.placeAxesOnGrid(fig, xspan=[0, 0.4], yspan=[0.33 + vbuffer, 0.67 - vbuffer]),
        'cognitive_kernels': vbp.placeAxesOnGrid(fig, xspan=[0, 0.4], yspan=[0.67 + vbuffer, 1]),
        'cell_response': vbp.placeAxesOnGrid(fig, xspan=[0.6, 1], yspan=[0, 0.25]),
        'dropout_quant': vbp.placeAxesOnGrid(fig, xspan=[0.6, 1], yspan=[0.4, 1]),
    }

    # add dropout summary
    results_summary = gat.generate_results_summary(glm)
    plot_dropout_summary(results_summary, cell_specimen_id, ax['dropout_quant'])

    regressors = {
        'visual': ['image0','image1'],
        'behavioral': ['pupil','running'],
        'cognitive': ['hit','miss'],
    }

    kernel_df = gat.build_kernel_df(glm, cell_specimen_id)


    run_params = glm_params.load_run_json(glm.version)
    #dropout_df = plot_dropouts(run_params)
    palette_df = dropout_df[['color']].reset_index().rename(columns={'color':'kernel_color','index':'kernel_name'})

    t0, t1 = t_span
    for regressor_category in regressors.keys():
        dropouts = np.sort(dropout_df[dropout_df['level-5'] == regressor_category].index.values).tolist()
        plot_kernels(
            kernel_df.query('kernel_name in @dropouts'), 
            ax['{}_kernels'.format(regressor_category)], 
            palette_df, 
            t_span
        )   
        plot_stimuli(glm.session.stimulus_presentations, ax['{}_kernels'.format(regressor_category)], t_span=t_span)
        ax['{}_kernels'.format(regressor_category)].set_title('{}'.format(regressor_category))
        ax['{}_kernels'.format(regressor_category)].set_ylim(
            kernel_df.query('timestamps >= @t0 and timestamps <= @t1')['kernel_outputs'].min()-0.2,
            kernel_df.query('timestamps >= @t0 and timestamps <= @t1')['kernel_outputs'].max()+0.2
        )


    # cell df/f plots:

    this_cell = glm.cell_results_df.query('cell_specimen_id == @cell_specimen_id')
    cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

    query_string = 'fit_trace_timestamps >= {} and fit_trace_timestamps <= {}'.format(
        t_span[0],
        t_span[1]
    )
    local_df = this_cell.query(query_string)

    ax['cell_response'].plot(
        local_df['fit_trace_timestamps'],
        local_df['dff'],
        alpha=0.9,
        color='darkgreen',
        linewidth=3,
    )

    ax['cell_response'].plot(
        local_df['fit_trace_timestamps'],
        local_df['dff_predicted'],
        alpha=1,
        color='black',
        linewidth=3,
    )
    qs = 'fit_trace_timestamps >= {} and fit_trace_timestamps <= {}'.format(
        t_span[0],
        t_span[1]
    )
    ax['cell_response'].set_ylim(
        this_cell.query(qs)['dff'].min(),
        this_cell.query(qs)['dff'].max(),
    )

    ax['cell_response'].legend(
        ['Actual $\Delta$F/F','Model Predicted $\Delta$F/F'],
        loc='upper left',
        ncol=2, 
        framealpha = 0.2,
    )

    plot_stimuli(glm.session.stimulus_presentations, ax['cell_response'], t_span=t_span,alpha=alpha)

    return fig, ax


def plot_all_coding_fraction(results_pivoted, run_params,drop_threshold=0,metric='fraction',compare=['cre_line']):
    '''
        Generated coding fraction plots for all dropouts
        results_pivoted, dataframe of dropout scores
        run_params, run json of model version
    '''
    
    # Keep track of what is failing 
    fail = []
    
    # Set up which sessions to plot
    active_only  = ['licks','hits','misses','false_alarms','correct_rejects', 'model_bias','model_task0','model_omissions1','model_timing1D','beh_model','licking']
    passive_only = ['passive_change']
    active_only  = active_only+['single-'+x for x in active_only]
    passive_only = passive_only+['single-'+x for x in passive_only]

    # Iterate over list of dropouts
    for dropout in run_params['dropouts']:
        try:
            # Dont plot full model
            if dropout == 'Full':
                continue

            # Determine which sessions to plot
            session ='all'
            if dropout in active_only:
                session = 'active'
            elif dropout in passive_only:
                session = 'passive'

            # plot the coding fraction
            filepath = run_params['fig_coding_dir']
            plot_coding_fraction(results_pivoted, run_params,dropout,drop_threshold=drop_threshold,savefile=filepath,session_filter=session,metric=metric,compare=compare)
        except Exception as e:
            print(e)
            # Track failures
            fail.append(dropout)
    
        # Close figure
        plt.close(plt.gcf().number)
    
    # Report failures
    if len(fail) > 0:
        print('The following kernels failed')
        print(fail)

def plot_coding_fraction(results_pivoted_in, run_params, dropout,drop_threshold=0,savefig=True,savefile='',metric='fraction',compare=['cre_line'],area_filter=['VISp','VISl'], cell_filter='all',equipment_filter='all',depth_filter=[0,1000],session_filter=[1,2,3,4,5,6]): 
    '''
        Plots coding fraction across session for each cre-line
        Applies hard filters, then uses "compare" to split data categorically and plot each group
        
        results_pivoted,            dataframe of dropout scores
        dropout (str)               name of nested model to plot
        threshold,                  level of significance for coding fraction
        savefig (bool),             if True, saves figures
        savefile (str),             pathroot to save
        metric (str),               'fraction', 'magnitude', or 'filtered_magnitude'   
        compare ([str]),            categorical condition to split the data by
        area_filter([str]),         list of targeted structures to include
        cell_filter(str)            "sst","slc", or "vip" anything else plots all cell types
        equipment_filter (str)      "mesoscope", or "scientifica" anything else plots all equipment 
        depth_filter ([min, max])   min and max depth to include
        session_filter(list)        list of sessions to include, or 'all','active','passive','familiar','novel'
        threshold(float),           minimum full model variance explain
 
        returns summary dataframe about coding fraction for this dropout
    '''   
    if 'dropout_threshold' in run_params:
        threshold = run_params['dropout_threshold']
    else:
        threshold = 0.005
 
    # Dumb stability thing because pandas doesnt like '-' in column names
    if '-' in dropout:
        # Make cleaned up dropout name
        old_dropout = dropout
        dropout = dropout.replace('-','_')
        
        # Rename dropout in results table
        if old_dropout in results_pivoted_in:
            results_pivoted_in = results_pivoted_in.rename({old_dropout:dropout},axis=1)
   
    ## Apply Hard Filters
    filter_string = ''

    # Filter by equipment 
    equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"]
    if equipment_filter == "scientifica": 
        equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5"]
        filter_string += '_scientifica'
    elif equipment_filter == "mesoscope":
        equipment_list = ["MESO.1"]
        filter_string += '_mesoscope'   

    # Filter by Cell Type    
    cell_list = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre']     
    if cell_filter == "sst":
        cell_list = ['Sst-IRES-Cre']
        filter_string += '_sst'
    elif cell_filter == "vip":
        cell_list = ['Vip-IRES-Cre']
        filter_string += '_vip'
    elif cell_filter == "slc":
        cell_list = ['Slc17a7-IRES2-Cre']
        filter_string += '_slc'

    # Unpack session filter codenames into list of sessions
    if session_filter == 'all':
        session_filter = [1,2,3,4,5,6]
    elif session_filter == 'active':
        session_filter = [1,3,4,6]
    elif session_filter == 'passive':
        session_filter = [2,5]
    elif session_filter == 'familiar':
        session_filter = [1,2,3]
    elif session_filter == 'novel':
        session_filter = [4,5,6]

    # compile filter info for filename
    if (session_filter != [1,2,3,4,5,6]):
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if depth_filter !=[0,1000]:
        filter_string+='_depth_'+str(depth_filter[0])+'_'+str(depth_filter[1])
    if area_filter != ['VISp','VISl']:
        filter_string+='_area_'+'_'.join(area_filter)

    # Apply hard filters
    results_pivoted = results_pivoted_in.query('(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&(session_number in @session_filter) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > @threshold)').copy()

    ## Set up comparisons
    # Set up indexing
    conditions  = compare+['session_number']

    # Get Total number of cells
    num_cells   = results_pivoted.groupby(conditions)['Full'].count()

    # Get number of cells with significant coding
    filter_str  = dropout +' < @drop_threshold'
    sig_cells   = results_pivoted.query(filter_str).groupby(conditions)['Full'].count()

    # Get Magnitude
    magnitude = results_pivoted.groupby(conditions)[dropout].mean()
    filtered_magnitude = results_pivoted.query(filter_str).groupby(conditions)[dropout].mean()

    # Get fraction significant
    fraction    = sig_cells/num_cells
    
    # Build dataframe
    fraction    = fraction.rename('fraction')
    sig_cells   = sig_cells.rename('num_sig')
    num_cells   = num_cells.rename('num_cells')
    magnitude   = magnitude.rename('magnitude')
    filtered_magnitude = filtered_magnitude.rename('filtered_magnitude')
    df = pd.concat([num_cells, sig_cells, fraction,magnitude, filtered_magnitude],axis=1)

    # Make Figure and set up axis labels
    plt.figure(figsize=(8,4))
    if metric=='fraction':
        plt.ylabel('% of cells with \n '+dropout+' coding',fontsize=18)
    elif metric=='magnitude':
        plt.ylabel('Avg '+dropout,fontsize=18)
    elif metric=='filtered_magnitude':
        plt.ylabel('Avg '+dropout+'\n for significant cells',fontsize=18)
    plt.xlabel('Session',fontsize=18)
    plt.tick_params(axis='both',labelsize=16)
    
    # Determine xtick labels based on what sessions were filtered out 
    names = ['F1','F2','F3','N1','N2','N3']
    xticklabels = [names[x-1] for x in session_filter]
    plt.xticks(range(0,len(xticklabels)),xticklabels, fontsize=18)

    # Set up color scheme for each cre line
    colors = project_colors()
    lines = ['-','--',':','-.']
    markers=['o','x','^','v','s']
  
    # Make a list of comparison groups to plot, but we group all the session numbers together
    groups = [(x[0:-1]) for x in df.index.values if x[-1] == session_filter[0]]
    
    labels_SAC = ['deep','superficial']

    # Iterate over groups, and plot
    for dex, group in enumerate(groups):
        # Determine color, line, and markerstyle
        color = colors.setdefault(group[0], (100/255,100/255,100/255))
        if df.index.nlevels > 2:
            linedex = np.where(group[1] == np.array(df.index.get_level_values(1).unique()))[0][0]
            linestyle = lines[np.mod(linedex,len(lines))]
        else:
            linestyle = '-'
        if df.index.nlevels > 3:
            markerdex = np.where(group[2] == np.array(df.index.get_level_values(2).unique()))[0][0]
            markerstyle = markers[np.mod(markerdex,len(markers))]
        else:
            markerstyle='o'

        # Plot
        plot_coding_fraction_inner(plt.gca(), df.loc[group], color, labels_SAC[dex], metric=metric, linestyle=linestyle,markerstyle=markerstyle)
        #plot_coding_fraction_inner(plt.gca(), df.loc[group], color, group, metric=metric, linestyle=linestyle,markerstyle=markerstyle)

    # Clean up plot
    plt.legend(loc='upper left',bbox_to_anchor=(1.05,1),handlelength=4,fontsize=16)
    plt.tight_layout()
    
    # Save figure
    if savefig:
        if len(compare) > 0:
            savefile = os.path.join(savefile,'coding_'+metric+'_by_'+'_'.join(compare)+'_'+dropout+filter_string+'.png')
        else:
            savefile = os.path.join(savefile,'coding_'+metric+'_'+dropout+filter_string+'.png')
        plt.savefig(savefile)
        print('Figure Saved to: '+ savefile)
    
    # return coding dataframe
    return df 

def plot_coding_fraction_inner(ax,df,color,label,metric='fraction',linestyle='-',markerstyle='o'):
    '''
        plots the fraction of significant cells with 95% binomial error bars    
        ax, axis to plot on
        df, dataframe with group to plot
        label, what to label this group
        metric, what information to pull from dataframe (fraction, magnitude, or filtered_magnitude)
    '''   
    # unpack fraction and get confidence interval
    if metric=='fraction':
        frac = df[metric].values 
        num  = df['num_cells'].values
        se   = 1.98*np.sqrt(frac*(1-frac)/num)
        # convert to percentages
        frac = frac*100
        se   = se*100
    else:
        frac = -df[metric].values 
        num  = df['num_cells'].values
        se   = 1.98*np.sqrt(frac*(1-frac)/num)   
        frac = -frac
        se   = -se
   
    # Plot the mean values 
    plt.plot(np.array(range(0,len(frac))), frac,marker=markerstyle,linestyle=linestyle,color=color,linewidth=4,label=label)
    
    # Iterate over lines and plot confidence intervals
    for dex, val in enumerate(zip(frac,se)):
        plt.plot([dex,dex],[val[0]+val[1],val[0]-val[1]], 'k',linewidth=1)

def plot_dendrogram(results_pivoted,regressors='all', method = 'ward', metric = 'euclidean', ax = 'none'):
    '''
    Clusters and plots dendrogram of glm regressors using the dropout scores from glm output. 
    More info: https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    Note: filling NaNs with 0 might affect the result
    
    INPUTS:
    results_pivoted - pandas dataframe of dropout scores from GLM_analysis_tools.build_pivoted_results_summary
    regressors - list of regressors to cluster; default is all
    method - string, linckage method ('centroid', 'single', etc); default = 'ward', which minimizes within cluster variance
    metric - string, metric of space in which the data is clustered; default = 'euclidean'
    ax - where to plot
    
    '''
    if regressors == 'all':
         regressors = results_pivoted.columns.to_numpy()

    if ax =='none':
        fig, ax = plt.subplots(1,1,figsize=(10,10))
            
    X = results_pivoted[regressors].fillna(0).to_numpy()
    Z = sch.linkage(X.T, method = method, metric = metric)
    dend = sch.dendrogram(Z, orientation = 'right',labels = regressors,\
                          color_threshold=None, leaf_font_size = 15, leaf_rotation=0, ax = ax)
    plt.tight_layout()
    
    return

def view_cell_across_sessions(cell_specimen_id, glm_version):
    '''
    For a given cell_specimen_id, visualize the cell and mask across sessions for which there is a GLM fit
    Shows GLM fit var explained in the title
    inputs:
        cell_specimen_id
        glm_version
    returns tuple:
        fig - figure handle
        ax - dictionary of axis handles
    '''
    search_dict = {'glm_version':glm_version, 'cell_specimen_id':cell_specimen_id, 'dropout':'Full'}


    fig = plt.figure(figsize = (1.5*11,1.5*8.5))
    axes = {
        'exp0_full': vbp.placeAxesOnGrid(fig, xspan=(0,0.33), yspan=(0,0.25)),
        'exp0_zoom': vbp.placeAxesOnGrid(fig, xspan=(0,0.33/2), yspan=(0.275,0.45)),
        'exp0_roi': vbp.placeAxesOnGrid(fig, xspan=(0.33/2,0.33), yspan=(0.275,0.45)),

        'exp1_full': vbp.placeAxesOnGrid(fig, xspan=(0.33,0.67), yspan=(0,0.25)),
        'exp1_zoom': vbp.placeAxesOnGrid(fig, xspan=(0.33,0.33+0.33/2), yspan=(0.275,0.45)),
        'exp1_roi': vbp.placeAxesOnGrid(fig, xspan=(0.33+0.33/2,0.67), yspan=(0.275,0.45)),

        'exp2_full': vbp.placeAxesOnGrid(fig, xspan=(0.67,1), yspan=(0,0.25)),
        'exp2_zoom': vbp.placeAxesOnGrid(fig, xspan=(0.67,0.67+0.33/2), yspan=(0.275,0.45)),
        'exp2_roi': vbp.placeAxesOnGrid(fig, xspan=(0.67+0.33/2,1), yspan=(0.275,0.45)),

        'exp3_full': vbp.placeAxesOnGrid(fig, xspan=(0,0.33), yspan=(0.55,0.8)),
        'exp3_zoom': vbp.placeAxesOnGrid(fig, xspan=(0,0.33/2), yspan=(0.825,1)),
        'exp3_roi': vbp.placeAxesOnGrid(fig, xspan=(0.33/2,0.33), yspan=(0.825,1)),

        'exp4_full': vbp.placeAxesOnGrid(fig, xspan=(0.33,0.67), yspan=(0.55,0.8)),
        'exp4_zoom': vbp.placeAxesOnGrid(fig, xspan=(0.33,0.33+0.33/2), yspan=(0.825,1)),
        'exp4_roi': vbp.placeAxesOnGrid(fig, xspan=(0.33+0.33/2,0.67), yspan=(0.825,1)),

        'exp5_full': vbp.placeAxesOnGrid(fig, xspan=(0.67,1), yspan=(0.55,0.8)),
        'exp5_zoom': vbp.placeAxesOnGrid(fig, xspan=(0.67,0.67+0.33/2), yspan=(0.825,1)),
        'exp5_roi': vbp.placeAxesOnGrid(fig, xspan=(0.67+0.33/2,1), yspan=(0.825,1)),
    }


    dropouts_for_cell = gat.retrieve_results(search_dict = search_dict, results_type = 'summary')
    for idx, row in dropouts_for_cell.sort_values(by='date_of_acquisition').reset_index().iterrows():
        # get the dataset
        dataset = loading.get_ophys_dataset(row['ophys_experiment_id'])

        # get the cell mask info from the cell specimen table
        cell_roi_id = row['cell_roi_id']
        mask = dataset.cell_specimen_table.query('cell_roi_id == @cell_roi_id')['roi_mask'].iloc[0].astype(float)
        mask[mask==0]=np.nan
        xmin, xmax, ymin, ymax = np.where(mask == 1)[1].min(), np.where(mask == 1)[1].max(), np.where(mask == 1)[0].min(), np.where(mask == 1)[0].max()

        # plot the max projection
        axes['exp{}_full'.format(idx)].imshow(dataset.max_projection, cmap='gray')
        axes['exp{}_full'.format(idx)].axis('off')

        # add a rectangle around the cell
        rect = patches.Rectangle(
            (xmin-20,ymin-20),
            xmax-xmin + 2*20,
            ymax-ymin + 2*20,
            linewidth=1,
            edgecolor='red',
            facecolor='none'
        )
        axes['exp{}_full'.format(idx)].add_patch(rect)

        # zoom in on the cell
        axes['exp{}_zoom'.format(idx)].imshow(dataset.max_projection, cmap='gray')
        axes['exp{}_zoom'.format(idx)].set_xlim(xmin-20, xmax+20)
        axes['exp{}_zoom'.format(idx)].set_ylim(ymax+20, ymin-20)
        axes['exp{}_zoom'.format(idx)].axis('off')
        axes['exp{}_zoom'.format(idx)].set_title('zoom in on cell', fontsize=10)

        # zoom in on the cell and overlay the mask
        axes['exp{}_roi'.format(idx)].imshow(dataset.max_projection, cmap='gray')
        axes['exp{}_roi'.format(idx)].imshow(mask, alpha=0.5, cmap='Reds_r')
        axes['exp{}_roi'.format(idx)].set_xlim(xmin-20, xmax+20)
        axes['exp{}_roi'.format(idx)].set_ylim(ymax+20, ymin-20)
        axes['exp{}_roi'.format(idx)].axis('off')
        axes['exp{}_roi'.format(idx)].set_title('zoom with mask overlaid', fontsize=10)

        # add a title
        title = '{}\nacquired_on {}\nroi_id {}\nfull model var explained = {:0.1f}%'.format(row['session_type'], row['date_of_acquisition'].split(' ')[0], row['cell_roi_id'], 100*row['variance_explained'])
        axes['exp{}_full'.format(idx)].set_title(title)
        
    for ii in range(idx+1,6):
        axes['exp{}_full'.format(ii)].axis('off')
        axes['exp{}_full'.format(ii)].set_title('no model fit')
        axes['exp{}_zoom'.format(ii)].axis('off')
        axes['exp{}_roi'.format(ii)].axis('off')

    # add a full figure title
    fig.suptitle('Cell Specimen ID = {}, Cre line = {}, rig = {}, GLM Version = {}'.format(cell_specimen_id, row['cre_line'], row['equipment_name'], glm_version))
    
    return fig, axes


def plot_regressor_heatmap_by_cre_line_sorted_by_MI(results_pivoted, session_numbers, regressor, model_output_type = 'adj_fraction_change_from_full',
                                                        limit_to_nonzero=True, save_dir=None):

    '''
    Plots modulation index for the same cells in two ophys sessions.

    INPUT:
    results_pivoted       glm output with matched cells in two sessions
    session_numbers       session numbers that the cells were matched across
    regressor             which regressor to use for MI
    model_output_type     this is for figure name purposes, default is 'adj_fraction_change_from_full'
    limit_to_nonzero      default True
    save_dir              default None, string of figure path

    '''
    figsize = (12, 8)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    cbar_ax = fig.add_axes([.95, .15, .03, .7])
    for i, cre_line in enumerate(results_pivoted['cre_lines'].unique()):
        results = results_pivoted[(results_pivoted.cre_line==cre_line)]
        reg_values = results.pivot(index='cell_specimen_id', columns='session_number', values=regressor)
        reg_values = reg_values.abs()
        reg_values['MI'] = [(reg_values.loc[row][session_numbers[1]]-reg_values.loc[row][session_numbers[0]])/(reg_values.loc[row][session_numbers[1]]+reg_values.loc[row][session_numbers[0]]) for row in reg_values.index.values]
        reg_values = reg_values.sort_values(by=['MI'])
        if limit_to_nonzero:
            reg_values = reg_values.dropna()
            suffix = '_nonzero'
        else:
            suffix = ''
        index_label = '('+str(session_numbers[1])+'-'+str(session_numbers[0])+')/('+str(session_numbers[1])+'+'+str(session_numbers[0])+')'
        ax[i] = sns.heatmap(reg_values.values, ax = ax[i], cmap='RdBu',
                            vmin=-1, vmax=1, cbar_kws={'label':model_output_type+'\n'+index_label}, cbar_ax=cbar_ax)
        ax[i].set_title(cre_line)
        ax[i].set_xticklabels(session_numbers+['MI'])
        ax[i].set_ylabel('cell number')
    fig.suptitle(regressor+', sessions '+str(session_numbers[0])+' - '+str(session_numbers[1]), x=0.51, y=0.99, fontsize=22)
    plt.subplots_adjust(wspace=0.5)
    if save_dir:
        if 'single' in regressor:
            utils.save_figure(fig, figsize, os.path.join(save_dir, 'regressor_heatmaps\single'),'sorted_by_MI'+suffix, model_output_type+'_'+regressor+'_'+str(session_numbers[0])+'_'+str(session_numbers[1])+'_MI')
        else:
            utils.save_figure(fig, figsize, os.path.join(save_dir, 'regressor_heatmaps\standard'), 'sorted_by_MI'+suffix, model_output_type+'_'+regressor+'_'+str(session_numbers[0])+'_'+str(session_numbers[1])+'_MI')

def plot_var_explained(results_summary, figsize=(10,6)):
    fig, ax = plt.subplots(figsize=figsize)

    cre_lines = np.sort(results_summary['cre_line'].unique())
    colors = project_colors()
    palette = [colors[cre_line] for cre_line in cre_lines]

    sns.boxplot(
        data = results_summary,
        y = 'Full__avg_cv_var_test',
        x = 'session_number',
        hue = 'cre_line',
        palette = palette, 
        whis = np.inf,
        ax = ax
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()

    
def plot_sample_cells(glm, cell_specimen_ids, t0, t1, figwidth=8, height_per_cell=1, title='cell_specimen_id'):
    '''
    makes a plot of each of the example cells in the list of cell_specimen_ids
    inputs:
        glm: an instance of the GLM class
        cell_specimen_ids: a list of cell specimen IDs. Must be valid IDs for the session.
        t0, t1: initial and final time for the plots
        figwidth: width of figure (default = 8)
        height_per_cell: height of each subfigure (default = 1). Total figure height will be height_per_cell*len(cell_specimen_ids)
        title: If 'cell_specimen_id' is specified (default), the cell specimen ID will be displayed in the title. Otherwise, it will say 'Example cell N'
    returns:
        fig, ax: matplotlib figure and axes
    '''
    fig, ax = plt.subplots(len(cell_specimen_ids), 1, figsize = (figwidth, height_per_cell*len(cell_specimen_ids)), sharex=True)

    # load the cell results df. This takes about 30 seconds.
    # If it's called as an attribute in the loop, it has to be reloaded on each call since it's not a cached attribute (at least for python < 3.8)
    cell_results_df = glm.cell_results_df

    # iterate over cell ids
    for row, cell_specimen_id in enumerate(cell_specimen_ids):
        # get the cell results for this cell between the desired times
        query_string = 'cell_specimen_id == {} and fit_trace_timestamps >= {} and fit_trace_timestamps <= {}'.format(
            cell_specimen_id,
            t0,
            t1,
        )
        local_df = cell_results_df.query(query_string)

        # determine whether the input model used dff or events, set some variables appropriately
        if glm.run_params['use_events']:
            y = 'events'
            ylabel = 'event\nmagnitude'
            color = 'cornflowerblue'
        else:
            y = 'dff'
            ylabel = r'$\Delta$F/F'
            color = 'green'
            
        # plot the data (be it dff or events)
        ax[row].plot(
            local_df['fit_trace_timestamps'],
            local_df[y],
            color = color,
            linewidth = 2,
        )

        # now plot the fit
        ax[row].plot(
            local_df['fit_trace_timestamps'],
            local_df['model_prediction'],
            color = 'black',
            linewidth = 2,
        )

        # determine what to display in title
        if title == 'cell_specimen_id':
            title_string = 'cell_specimen_id = '
            value = cell_specimen_id
        else:
            title_string = 'example cell '
            value = row + 1

        # now add the title
        ax[row].set_title(
            '{} {}, variance_explained = {:0.2f}'.format(
                title_string,
                value, 
                glm.results.loc[cell_specimen_id]['Full__avg_cv_var_test']
            )
        )
        # add the ylabel
        ax[row].set_ylabel(ylabel, rotation=0, labelpad=35, fontsize=12)
        
        # plot vertical bars for stimuli
        plot_stimuli(glm.session.stimulus_presentations, ax[row], t_span=(t0, t1), alpha=.25)
        
        # set the xlims
        ax[row].set_xlim(t0, t1)
        
    # add a legend
    ax[0].legend([y, 'model fit'], loc = 'upper left')

    # some final formatting
    ax[row].set_xlabel('time in session (s)', fontsize=12)
    sns.despine()
    fig.tight_layout()

    return fig, ax

def plot_sem_distribution(results_pivoted, cres=None):

    if cres is None:
        cres = results_pivoted.cre_line.unique()

    # Determine threshold
    thresholds = gat.get_sem_thresholds(results_pivoted)


    # Plot histograms
    plt.figure()
    plt.axvline(0.005, color='r',linestyle='--',label='Current Threshold')
    for cre in cres:
        plt.hist(results_pivoted.query('cre_line == @cre')['variance_explained_full_sem'], bins=100,alpha=.25,range=(0,.1),density=True,label=cre,color=project_colors()[cre])
        plt.axvline(thresholds[cre], color=project_colors()[cre],linestyle='--',label='95% Threshold')
    
    plt.ylabel('Density',fontsize=14)
    plt.xlabel('SEM: Full Model VE',fontsize=14)
    plt.legend()

def plot_sem_comparison(results_pivoted):
    plt.figure(figsize=(8,4))
    cres = results_pivoted.cre_line.unique()
    for cre in cres:
        cre_slice = results_pivoted.query('cre_line ==@cre')
        plt.plot(cre_slice['variance_explained_full'], cre_slice['variance_explained_full_sem'],'o', alpha=.1,color=project_colors()[cre])
    plt.ylabel('SEM')
    plt.xlabel('Full Model VE')
    plt.plot([0,0.1],[0,0.1], color='r',linestyle='--')
    plt.axvline(0.005, color='c', linestyle='--')
    plt.ylim(0,.1)
    plt.xlim(0,1)

def compare_events_and_dff(results_pivoted_dff, results_pivoted_events,savefig=False, versions=None):

    joint = pd.merge(
        results_pivoted_dff,
        results_pivoted_events,
        how='inner',
        on='identifier',
        suffixes=("_dff","_events"),
    )
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    cres = np.flip(joint.cre_line_dff.unique())
    cres = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    for cre in cres:
        cre_slice = joint.query('cre_line_dff ==@cre')
        ax[0].plot(cre_slice['variance_explained_full_dff'],cre_slice['variance_explained_full_events'],'o',alpha=.2,color=project_colors()[cre],label=cre)
    ax[0].set_ylabel('Variance Explained (events)')
    ax[0].set_xlabel('Variance Explained (df/f)')
    ax[0].plot([0,1],[0,1],'r--',alpha=.5)
    ax[0].set_aspect('equal')
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,1)
    ax[0].legend()
    
    joint['VE (df/f-events)'] = joint['variance_explained_full_dff'] - joint['variance_explained_full_events']
    ax[1] = sns.histplot(
        data = joint,
        x='VE (df/f-events)',
        hue='cre_line_dff',
        hue_order = cres, 
        palette = [project_colors()[cre_line] for cre_line in cres],
        kde=False,
        stat='density',
        common_norm=False,
        element='step',
    )
    ax[1].set_xlim(-.05,.3)
    plt.tight_layout()
    if savefig and (versions is not None):
        version_strings = '_'.join([x.split('_')[0] for x in versions])
        for v in versions:
            run_params = glm_params.load_run_json(v)
            filepath = os.path.join(run_params['figure_dir'], 'dff_events_comparison_'+version_strings+'.png')
            print(filepath)
            plt.savefig(filepath)

def compare_weight_index(weights_df, kernel='all-images'):
    '''
        Scatter plots the kernel dropout score against the kernel weight index. 
    '''
    plt.figure()
    cres = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    for cre in cres:
        temp = weights_df.query('cre_line == @cre')
        plt.plot(temp[kernel], temp[kernel+'_weights_index'], 'o',color=project_colors()[cre],alpha=.2,label=cre)
    plt.ylabel('Weight Index')
    plt.xlabel('Dropout Index')
    plt.ylim(0,weights_df[kernel+'_weights_index'].quantile(q=.995))
    plt.title(kernel)
    plt.legend()
    return None

def shuffle_analysis(results,run_params,bins=50,savefig=True,shuffle='time'):
    '''
        Plots a distribution of variance explained in a shuffle test compared to dropout threshold
        shuffle is the test to plot, should be either 'time', or 'cells'
    '''
    plt.figure()
    plt.hist(results.query('dropout=="Full"')['shuffle_'+shuffle],bins=bins,label='Shuffle {}'.format(shuffle),density=True)
    plt.ylabel('Density')
    plt.xlabel('Variance explained, shuffling across {}'.format(shuffle))
    maxve = results.query('dropout=="Full"')['shuffle_'+shuffle].max()
    plt.axvline(maxve, color='r',linestyle='-',alpha=1,label='Max shuffled VE')
    plt.axvline(run_params['dropout_threshold'],color='m',linestyle='--',alpha=1,label='Threshold for minimum VE (non shuffled)')
    plt.legend()
    plt.title(run_params['version'])
    if savefig:
        filepath = os.path.join(run_params['fig_overfitting_dir'],'shuffle_{}.png'.format(shuffle))
        print(filepath)
        plt.savefig(filepath)

def compare_dropout_thresholds(results_in):
    '''
        Plots summary dropouts for different filtering steps
    '''
    results = results_in.copy()

    fig,ax = plt.subplots(3,2,figsize=(12,10))
    plot_dropout_summary_population(results,ax=ax[0,0])
    ax[0,0].set_title('VE < 0.5% set dropout = 0 ')
    ax[0,0].get_legend().remove()

    plot_dropout_summary_population(results.query('variance_explained_full > 0.005'),ax=ax[0,1])
    ax[0,1].set_title('VE < 0.5% removed')        
    ax[0,1].get_legend().remove()

    plot_dropout_summary_population(results.query('variance_explained_full > 0.01'),ax=ax[1,1])
    ax[1,1].set_title('VE < 1.0% removed')        
    ax[1,1].get_legend().remove()

    plot_dropout_summary_population(results.query('variance_explained_full > 0.02'),ax=ax[2,1])
    ax[2,1].set_title('VE < 2.0% removed')        
    ax[2,1].get_legend().remove()

    results.loc[results['variance_explained_full'] <0.01,'adj_fraction_change_from_full'] = 0
    plot_dropout_summary_population(results,ax=ax[1,0])
    ax[1,0].set_title('VE < 1.0% set dropout = 0')        
    ax[1,0].get_legend().remove()

    results.loc[results['variance_explained_full'] <0.02,'adj_fraction_change_from_full'] = 0
    plot_dropout_summary_population(results,ax=ax[2,0])
    ax[2,0].set_title('VE < 2.0% set dropout = 0')        
    ax[2,0].get_legend().remove()
    plt.tight_layout()


def compare_L2_values(results,run_params):
    results = results.query('dropout == "Full"').copy()

    fig,ax = plt.subplots(1,2,figsize=(8,3))
    ax[0] = sns.histplot(
        results, 
        x='cell_L2_regularization',
        hue='cre_line',
        element='step',
        bins=40,
        stat='density',
        common_norm=False,
        ax=ax[0],
        palette=project_colors()
        )
    ax[0].set_title('All cells')
    results = results.query('variance_explained > 0.005')

    sns.histplot(
        results, 
        x='cell_L2_regularization',
        hue='cre_line',
        element='step',
        bins=40,
        legend=False,
        stat='density',
        common_norm=False,
        ax=ax[1],
        palette=project_colors()
        )
    ax[1].set_title('Cells > 0.005 VE')

    plt.tight_layout()

def clustering_kernels(weights_df, run_params, kernel,just_coding=False,pca_by_experience=True):
    problem_sessions = get_problem_sessions()
    colors = project_colors() 
    fig, ax = plt.subplots(3,5,figsize=(16,8))

    if kernel in ['preferred_image', 'all-images']:
        run_params['kernels'][kernel] = run_params['kernels']['image0'].copy()
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)
    if 'image' in kernel:
        time_vec = time_vec[:-1]
    if ('omissions' == kernel) & ('post-omissions' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('hits' == kernel) & ('post-hits' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('misses' == kernel) & ('post-misses' in run_params['kernels']):
        time_vec = time_vec[:-1]
    if ('passive_change' == kernel) & ('post-passive_change' in run_params['kernels']):
        time_vec = time_vec[:-1]



    cre_lines = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre'] 

    for index, cre_line in enumerate(cre_lines):
        if just_coding:
            weights = weights_df.query('(cre_line == @cre_line) & (ophys_session_id not in @problem_sessions) &(not passive)&({0} <0)'.format(kernel)).copy()       
        else:
            weights = weights_df.query('(cre_line == @cre_line) & (ophys_session_id not in @problem_sessions) &(not passive)').copy()
        weights = weights[~weights[kernel+'_weights'].isnull()]
        
        # Do PCA across experience levels
        x = np.vstack(weights[kernel+'_weights'].values) 
        pca = PCA()
        pca.fit(x) 
        y = pca.transform(x)
        weights['pc1'] = y[:,0]
        weights['pc2'] = y[:,1]   
        
        # Plot PCA summary
        if pca_by_experience:
            #ax[index,0].plot(pca.explained_variance_ratio_, 'o-', color=colors[cre_line], label='global')   
            ax[index,0].set_xlim(-.5,5)
            ax[index,0].set_title(kernel,fontsize=18)
            ax[index,0].tick_params(axis='both',labelsize=16)
            ax[index,0].set_ylabel(cre_line+'\nVariance Explained',fontsize=16)
            ax[index,0].set_xlabel('PC #',fontsize=16)
        else:
            ax[0,0].plot(pca.explained_variance_ratio_, 'o-', color=colors[cre_line], label=cre_line)
            ax[0,0].set_xlim(-.5,5)
            ax[0,0].set_title(kernel,fontsize=18)
            ax[0,0].tick_params(axis='both',labelsize=16)
            ax[0,0].set_ylabel('Variance Explained',fontsize=16)
            ax[0,0].set_xlabel('PC #',fontsize=16)

        #ax[index+1,0].plot(y[:,0],y[:,1], 'x',color=colors[cre_line])
        #ax[index+1,0].set_title(cre_line)
        #ax[index+1,0].set_ylabel('PC 2',fontsize=16)
        #ax[index+1,0].set_xlabel('PC 1',fontsize=16)
        #ax[index+1,0].tick_params(axis='both',labelsize=16)

        if pca_by_experience:
            #ax[index,1].plot(time_vec, pca.components_[0,:], 'k-',label='global')
            ax[index,1].set_ylabel('PC 1',fontsize=16)
        else:
            ax[index,1].plot(time_vec, pca.components_[0,:], 'r-',label='PC 1')
            ax[index,1].plot(time_vec,pca.components_[1,:], 'b-',label='PC 2')
            ax[index,1].legend()
            ax[index,1].set_ylabel('PC',fontsize=16)
        ax[index,1].set_xlabel('Time (s)',fontsize=16)
        ax[index,1].tick_params(axis='both',labelsize=16)

        weights = weights.sort_values(by=['pc1'])
        for eindex, experience_level in enumerate(['Familiar','Novel 1','Novel >1']):
            if pca_by_experience:
                eweights = weights.query('experience_level ==@experience_level').copy()
                x = np.vstack(eweights[kernel+'_weights'].values) 
                pca = PCA()
                pca.fit(x) 
                y = pca.transform(x)
                eweights['pc1'] = y[:,0]
                eweights['pc2'] = y[:,1]   
                sorted_x = np.vstack(eweights.query('experience_level ==@experience_level')[kernel+'_weights'].values)
                ax[index,0].plot(pca.explained_variance_ratio_, 'o-',label=experience_level,color=colors[experience_level])
                ax[index,1].plot(time_vec, pca.components_[0,:],label=experience_level,color=colors[experience_level])
                ax[index,0].legend()
            else:           
                sorted_x = np.vstack(weights.query('experience_level ==@experience_level')[kernel+'_weights'].values)

            cbar=ax[index,2+eindex].imshow(sorted_x,aspect='auto',extent=[time_vec[0], time_vec[-1],0,np.shape(sorted_x)[1]],cmap='bwr')    
            cbar.set_clim(-np.nanpercentile(np.abs(sorted_x),95),np.nanpercentile(np.abs(sorted_x),95))
            ax[0,2+eindex].set_title(experience_level,fontsize=18)
            ax[index,2+eindex].set_xlabel('Time (s)',fontsize=16)
            ax[index,2+eindex].set_ylabel('Cells sorted \nby PC1 weight',fontsize=16)
            ax[index,2+eindex].set_yticks([])
            ax[index,2+eindex].tick_params(axis='both',labelsize=16)

    plt.tight_layout()
    if pca_by_experience:
        filename = run_params['fig_clustering_dir']+'/'+kernel+'_by_experience.png'
    else:
        filename = run_params['fig_clustering_dir']+'/'+kernel+'.png'
    plt.savefig(filename)
    
def depth_heatmap(weights_df, run_params,metric='omission_responsive',just_coding=False,just_mesoscope=False):
    if just_mesoscope:
        df = weights_df.query('equipment_name == "MESO.1"').query('experience_level == "Familiar"').copy()    
    else:
        df = weights_df.query('experience_level in ["Novel 1"]').copy() 
    
    df['binned_depth'] = [bin_depth(x) for x in df['imaging_depth']]       
    df['change_responsive'] = df['misses'] < 0
    df['omissions_index'] = [np.argmax(x) for x in df['omissions_weights']]
    df['omission_responsive'] = df['omissions_index'] <=24
    df['omission_coding'] = df['omissions'] < 0 

    if just_coding:
        fraction = df.query('omissions < 0').groupby(['cre_line','targeted_structure','binned_depth'])[metric].mean()   
        fraction['n'] = df.query('omissions < 0').groupby(['cre_line','targeted_structure','binned_depth'])[metric].count()
    else:
        fraction = df.groupby(['cre_line','targeted_structure','binned_depth'])[[metric]].mean()
        fraction['n'] = df.groupby(['cre_line','targeted_structure','binned_depth'])[metric].count()
    fraction[metric+'_ci'] = 1.96*np.sqrt((fraction[metric]*(1-fraction[metric]))/fraction['n'])

    cre_lines = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre'] 
    mapper = {
        'Slc17a7-IRES2-Cre':'Excitatory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Vip-IRES-Cre':'Vip Inhibitory'
        }

    fig, ax = plt.subplots(2,3,figsize=(12,8))
    for index, cell in enumerate(cre_lines):    
        values = fraction.unstack().loc[cell][metric].values
        ci = fraction.unstack().loc[cell][metric+'_ci'].values
        cbar = ax[0,index].imshow(np.fliplr(values.T),cmap='plasma')
        cbar.set_clim(np.min(values),np.max(values))
        color_bar = fig.colorbar(cbar, ax = ax[0,index])
        if just_coding:
            color_bar.ax.set_ylabel('Fraction of coding cells\n that are '+metric)       
        else:
            color_bar.ax.set_ylabel('Fraction of all cells\n that are '+metric)
        ax[0,index].set_title(mapper[cell],fontsize=18)
        ax[0,index].set_yticks([0,1,2,3])
        ax[0,index].set_yticklabels(['75','175','275','375'],fontsize=16)
        ax[0,index].set_ylabel('Binned Depth',fontsize=18)
        ax[0,index].set_xticks([0,1])
        ax[0,index].set_xticklabels(['VISp','VISl'],fontsize=16)
       
        ax[1,index].plot([75,175,275,375],values[1,:],'ko-',label='VISp') 
        ax[1,index].plot([75,175,275,375],values[0,:],'bo-',label='VISl') 
        ax[1,index].errorbar([75,175,275,375], values[1,:],yerr=ci[1,:],color='k',alpha=.25)
        ax[1,index].errorbar([75,175,275,375], values[0,:],yerr=ci[0,:],color='b',alpha=.25)
        ax[1,index].set_xticks([75,175,275,375])
        ax[1,index].set_xticklabels(['75','175','275','375'],fontsize=16)
        ax[1,index].set_xlabel('Binned Depth',fontsize=18)
        ax[1,index].set_ylabel('Fraction of cells',fontsize=18)
        ax[1,index].tick_params(axis='both',labelsize=16)
        ax[1,index].legend()

    ax[0,1].set_xlabel('Area',fontsize=18)
    plt.tight_layout()
    plt.savefig(run_params['fig_coding_dir']+'/heatmap_'+metric+'.png')
    plt.savefig(run_params['fig_coding_dir']+'/heatmap_'+metric+'.svg')
    return fraction    

def bin_depth(x):
    if x < 100:
        return 75
    elif x< 200:
        return 175
    elif x<300:
        return 275
    else:
        return 375
 
def coarse_bin_depth(x):
    if x< 250:
        return 'upper'
    else:
        return 'lower'

