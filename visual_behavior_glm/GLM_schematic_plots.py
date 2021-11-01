import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import matplotlib
from mpl_toolkits.axes_grid1 import Divider, Size

def plot_glm_example(g,celldex=1,times=[908,918]):
    #oeid = 775614751
    style={
        'fs1':16,
        'fs2':14,
        'trace_linewidth':2,
        'dff':'k',
        'events':'b',
        'model':'r',
        }

    index_times = [np.where(g.fit['fit_trace_timestamps']>=times[0])[0][0],np.where(g.fit['fit_trace_timestamps']>times[1])[0][0]+1]
    plot_glm_example_trace(g,celldex,times,style)
    ##gvt.plot_kernel_support(g,plot_bands=False,start=index_times[0],end=index_times[1])
    plot_glm_example_inputs(g,times,style)
    return 

def plot_glm_example_inputs(g,times,style,ax=None):
    if ax is None:
        #fig,ax = plt.subplots(figsize=(12,6))
        fig = plt.figure(figsize=(12,6))
        h = [Size.Fixed(2.0),Size.Fixed(9.5)]
        v = [Size.Fixed(1.0),Size.Fixed(4.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))
    
    time_vec = (g.fit['fit_trace_timestamps'] > times[0])&(g.fit['fit_trace_timestamps'] < times[1])

    # plot stimulus and change bars
    stim = g.session.stimulus_presentations.query('start_time > @times[0] & start_time < @times[1]')
    top =100
    ticklabels={}
    for index, image in enumerate(range(0,9)):
        image_times = stim.query('image_index == @index')['start_time'].values
        ticklabels[top-index]='Image '+str(image)
        ecolor ='k'
        fcolor ='k'
        if index == 8:
            ticklabels[top-index]='Omission'
            fcolor='None'
        if len(image_times) > 0:
            #ax.plot(image_times, np.ones(np.shape(image_times))*(top-index),'|',markersize=20)
            for t in image_times:
                rect =patches.Rectangle((t,top-index-.5),0.25,1,edgecolor=ecolor,facecolor=fcolor,alpha=.5)
                ax.add_patch(rect)

    # Running Data
    run = g.session.running_speed.query('(timestamps > @times[0])&(timestamps < @times[1])').copy()
    run['normalized_speed'] = run['speed'].apply(lambda x: (x - run['speed'].min())/(run['speed'].max() - run['speed'].min()))
    run['normalized_speed'] = run['normalized_speed'] + top-10
    ax.plot(run.timestamps,run.normalized_speed,'k')
    ticklabels[top-9.5]='Running Speed'

    # Pupil
    eye = g.session.eye_tracking.query('(timestamps > @times[0])&(timestamps < @times[1])').copy()
    eye['pupil_radius'] = np.sqrt(eye['pupil_area']*(1/np.pi))
    eye['normalized_pupil_radius'] = eye['pupil_radius'].apply(lambda x: (x - eye['pupil_radius'].min())/(eye['pupil_radius'].max() - eye['pupil_radius'].min()))
    eye['normalized_pupil_radius'] = eye['normalized_pupil_radius'] + top-12
    ax.plot(eye.timestamps,eye.normalized_pupil_radius,'k')
    ticklabels[top-11.5]='Pupil Radius'

    # licking
    licks = g.session.licks.query('(timestamps > @times[0])&(timestamps < @times[1])').copy()
    ax.plot(licks.timestamps, np.ones((len(licks),))*(top-13),'k|',markersize=20)
    ticklabels[top-13]='Licking'

    # Trials
    trials = g.session.trials.query('(change_time >= @times[0])&(change_time <=@times[1])').copy()
    hits = trials.query('hit')
    ax.plot(hits.reward_time, np.ones((len(hits),))*(top-14),'r|',markersize=20)
    ticklabels[top-14]='Hit'

    miss = trials.query('miss')
    ax.plot(miss.change_time, np.ones((len(miss),))*(top-15),'r|',markersize=20)
    ticklabels[top-15]='Miss'

    fa = trials.query('false_alarm')
    ax.plot(fa.change_time, np.ones((len(fa),))*(top-16),'r|',markersize=20)
    ticklabels[top-16]='False Alarm'

    correct_reject = trials.query('correct_reject')
    ax.plot(correct_reject.change_time, np.ones((len(correct_reject),))*(top-17),'r|',markersize=20)
    ticklabels[top-17]='Correct Reject'

    ax.yaxis.set_ticks(list(ticklabels.keys()))
    ax.yaxis.set_ticklabels(list(ticklabels.values()),fontsize=style['fs2'])
    ax.set_xlabel('Time (s)',fontsize=style['fs1'])
    ax.tick_params(axis='x',labelsize=style['fs2'])
    ax.set_xlim(times)
    ax.set_ylim(top-17.5,top+.5)
    #plt.tight_layout()
    plt.savefig('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/figures/example_inputs.svg')

def plot_glm_example_trace(g,celldex,times,style,include_events=True, include_dff=True,ax=None):
    if ax is None:
        #fig,ax = plt.subplots(figsize=(12,3))
        fig = plt.figure(figsize=(12,3))
        h = [Size.Fixed(2.0),Size.Fixed(9.5)]
        v = [Size.Fixed(.7),Size.Fixed(2.25)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1,ny=1))

    time_vec = (g.fit['fit_trace_timestamps'] > times[0])&(g.fit['fit_trace_timestamps'] < times[1])

    # plot stimulus and change bars
    stim = g.session.stimulus_presentations.query('start_time > @times[0] & start_time < @times[1] & not omitted')
    for index, time in enumerate(stim['start_time'].values):
        plt.axvspan(time, time+0.25, color='k',alpha=.1)
    change = g.session.stimulus_presentations.query('start_time > @times[0] & start_time < @times[1] & is_change')
    for index, time in enumerate(change['start_time'].values):
        plt.axvspan(time, time+0.25, color='b',alpha=.2)

    # Plot df/f
    if include_dff:
        ax.plot(g.fit['fit_trace_timestamps'][time_vec], 
            g.fit['dff_trace_arr'][time_vec,celldex],
            style['dff'],label='df/f',
            linewidth=style['trace_linewidth'],alpha=.6)

    # Plot Filtered event trace
    if include_events:
        ax.plot(g.fit['fit_trace_timestamps'][time_vec], 
            g.fit['events_trace_arr'][time_vec,celldex],
            style['events'],label='Events',
            linewidth=style['trace_linewidth'])

    # Plot Model
    ax.plot(g.fit['fit_trace_timestamps'][time_vec],
        g.fit['dropouts']['Full']['full_model_train_prediction'][time_vec,celldex],
        style['model'],label='Model',linewidth=style['trace_linewidth'])

    # Clean up plot
    ax.legend()
    ax.set_ylabel('Neural activity',fontsize=style['fs1'])
    ax.set_xlabel('Time (s)',fontsize=style['fs1'])
    ax.tick_params(axis='x',labelsize=style['fs2'])
    ax.tick_params(axis='y',labelsize=style['fs2'])
    ax.set_xlim(times)
    #plt.tight_layout()
    plt.savefig('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/figures/example_trace.svg')
    return

def plot_all_dropouts(VERSION):
    '''
        Plots all kernels and all nested dropouts
    '''
    run_params = glm_params.load_run_json(VERSION)
    cd = plot_dropouts(run_params)
    return cd

def plot_high_level_dropouts(VERSION):
    '''
        Plots the full model, major and minor components, and the features.
        Ignores time and intercept. 
    '''
    run_params = glm_params.load_run_json(VERSION)
    run_params['kernels'].pop('time',None)
    run_params['kernels'].pop('intercept',None)
    run_params['levels'].pop('2',None)
    run_params['levels'].pop('3',None)
    run_params['levels']['2'] = run_params['levels'].pop('4')
    run_params['levels']['3'] = run_params['levels'].pop('5')
    run_params['levels']['4'] = run_params['levels'].pop('6')
    cd = plot_dropouts(run_params,num_levels=4,add_text=False,SAC=True)
    return cd

def plot_nice_dropouts(VERSION):
    '''
        Plots the full model, major and minor components, and the features.
        Ignores time and intercept.
        Removes behavioral model
        major components separate omissions and images  
    '''
    run_params = glm_params.load_run_json(VERSION)
    run_params['kernels'].pop('time',None)
    run_params['kernels'].pop('intercept',None)
    run_params['kernels'].pop('model_bias',None)
    run_params['kernels'].pop('model_task0',None)
    run_params['kernels'].pop('model_omissions1',None)
    run_params['kernels'].pop('model_timing1D',None)
    run_params['levels'].pop('2')
    run_params['levels'].pop('3')
    run_params['levels'].pop('4')
    run_params['levels']['2'] = run_params['levels'].pop('5')
    run_params['levels']['3'] = run_params['levels'].pop('6')
    run_params['levels']['2'] = ['all-images','expectation','behavioral','cognitive']
    cd = plot_dropouts_2(run_params,num_levels=3,add_text=False)
    return cd

def plot_dropouts_final(VERSION):
    run_params = glm_params.load_run_json(VERSION)
    run_params['kernels'].pop('time',None)
    run_params['kernels'].pop('intercept',None)
    run_params['kernels'].pop('model_bias',None)
    run_params['kernels'].pop('model_task0',None)
    run_params['kernels'].pop('model_omissions1',None)
    run_params['kernels'].pop('model_timing1D',None)
    run_params['levels'] = {
            '3':['Full'],
            '2':['all-images','expectation','behavioral','task'],
            }
    run_params['dropouts']['expectation'] = run_params['dropouts']['omissions'].copy()
    cd = plot_dropouts_3(run_params)
    return cd   

def plot_dropouts_3(run_params,save_results=True,add_text=False,num_levels=3):
    '''
        Makes a visual and graphic representation of how the kernels are nested inside dropout models
        save_results (bool) if True, saves the figure
        num_levels (int) number of levels in nested model to plot
        add_text (bool) if True, adds descriptive text to left hand side of plot for each kernel
    '''
    if add_text:
        plt.figure(figsize=(9,8))
    else:
        plt.figure(figsize=(5.5,8))
    w = 1/num_levels  
 
    # Get list of dropouts and kernels
    drops = set([x for x in run_params['dropouts'] if not run_params['dropouts'][x]['is_single'] ])
    kernels = run_params['kernels'].copy()
 
    # Build dataframe
    df = pd.DataFrame(index=kernels.keys())
    
    # Add the individual dropouts
    df['level-1']= df.index.values
    for k in kernels:
        if k in drops:
            drops.remove(k)
    
    # Add each grouping of dropouts
    levels = run_params['levels'].copy()
    keys = list(levels.keys())
    for dex, key in enumerate(keys):
        levels[int(key)] = levels.pop(key)

    #return (df,drops,levels)
    for level in np.arange(num_levels,1,-1):
        df,drops = make_level(df,drops, level,  levels[level],  run_params)
        
    # re-organized dataframe
    # All the renaming is for sorting the features
    df=df[['level-'+str(x) for x in range(1,num_levels+1)]]
    df['level-2'] = ['task' if x == 'task' else x for x in df['level-2']]
    df['level-2'] = ['ayexpectation' if x == 'expectation' else x for x in df['level-2']]
    df['level-1'] = ['z'+x if 'face' in x else x for x in df['level-1']]
    df['level-1'] = ['ahits' if x == 'hits' else x for x in df['level-1']]
    df['level-1'] = ['bmisses' if x == 'misses' else x for x in df['level-1']]
    df['level-1'] = ['bnpassive_change' if x == 'passive_change' else x for x in df['level-1']]
    df = df.sort_values(by=['level-'+str(x) for x in np.arange(num_levels,0,-1)])
    df['level-2'] = ['task' if x == 'task' else x for x in df['level-2']]
    df['level-2'] = ['expectation' if x == 'ayexpectation' else x for x in df['level-2']]
    df['level-1'] = [ x[1:] if 'zface' in x else x for x in df['level-1']]
    df['level-1'] = ['hits' if x == 'ahits' else x for x in df['level-1']]
    df['level-1'] = ['misses' if x == 'bmisses' else x for x in df['level-1']]
    df['level-1'] = ['passive_change' if x == 'bnpassive_change' else x for x in df['level-1']]
    df['text'] = [run_params['kernels'][k]['text'] for k in df.index.values]
    df['support'] = [(np.round(run_params['kernels'][k]['offset'],2), np.round(run_params['kernels'][k]['length'] +  run_params['kernels'][k]['offset'],2)) for k in df.index.values]

    # Rename stuff, purely for explanatory purposes

    df['level-2'] = ['behavioral_model' if x == 'beh_model' else x for x in df['level-2']]  
    df['level-2'] = ['licks' if x == 'licks' else x for x in df['level-2']]
    df['level-2'] = ['omissions' if x == 'expectation' else x for x in df['level-2']]
    df['level-2'] = ['task' if x == 'cognitive' else x for x in df['level-2']]

    df['level-1'] = ['bias strategy' if x == 'model_bias' else x for x in df['level-1']]
    df['level-1'] = ['task strategy' if x == 'model_task0' else x for x in df['level-1']]
    df['level-1'] = ['post omission strategy' if x == 'model_omissions1' else x for x in df['level-1']]
    df['level-1'] = ['timing strategy' if x == 'model_timing1D' else x for x in df['level-1']]

    # Make sure all dropouts were used
    if len(drops) > 0:
        print('Warning, dropouts not used')
        print(drops)

    # Make Color Dictionary
    labels=[]
    colors=[]
    for level in range(1,num_levels+1):
        new_labels = list(df['level-'+str(level)].unique())
        labels = labels + ['level-'+str(level)+'-'+ x for x in new_labels]
        colors = colors + sns.color_palette('hls', len(new_labels)) 
    color_dict = {x:y for (x,y) in  zip(labels,colors)}
    for level in range(1,num_levels+1):
        color_dict['level-'+str(level)+'--'] = (0.8,0.8,0.8)

    # add color of level-1 value to df['color']
    df['color'] = None
    # Get Project Colors
    proj_colors = gvt.project_colors() 
    for key in color_dict.keys():
        dropout = key.split('-')[2]
        if dropout == 'all':
            dropout = 'all-images'
        if dropout in proj_colors:
            color_dict[key] = proj_colors[dropout]
        if key.startswith('level-1'):
            dropout = key.split('level-1-')[1]
            if dropout in df.index.values.tolist():
                df.at[dropout,'color'] = color_dict[key]
    color_dict['level-2-behavioral'] = color_dict['level-1-licks']

    # Plot Squares
    uniques = set()
    maxn = len(df)
    last = {x:'null' for x in np.arange(1,num_levels+1,1)} 
    for index, k in enumerate(df.index.values):
        for level in range(1,num_levels+1):
            plt.axhspan(maxn-index-1,maxn-index,w*(level-1),w*level,color=color_dict['level-'+str(level)+'-'+df.loc[k]['level-'+str(level)]]) 
            # If this is a new group, add a line and a text label
            if (level > 1)&(not (df.loc[k]['level-'+str(level)] == '-')) & ('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)] not in uniques) :
                uniques.add('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)])
                text_str = df.loc[k]['level-'+str(level)].replace('_', ' ')
                plt.text(w*(level-1)+0.01,maxn-index-1+.25,text_str,fontsize=12)
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif (level > 1) & (not (df.loc[k]['level-'+str(level)] == last[level])):
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif level == 1:
                # For the individual regressors, just label, no lines
                text_str = df.loc[k]['level-'+str(level)].replace('_', ' ').replace('image', 'image ')
                plt.text(0.01,maxn-index-1+.25,text_str,fontsize=12)
            last[level] = df.loc[k]['level-'+str(level)]

    # Define some lines between levels   
    for level in range(1,num_levels): 
        plt.axvline(w*level,color='k') 
    
    # Make formated ylabels that include support and alignment event   
    max_name = np.max([len(x) for x in df.index.values])+3 
    max_support = np.max([len(str(x)) for x in df['support'].values])+3
    max_text = np.max([len(str(x)) for x in df['text'].values])
    aligned_names = [row[1].name.ljust(max_name)+str(row[1]['support']).ljust(max_support)+row[1]['text'].ljust(max_text) for row in df.iterrows()]

    # clean up axes
    plt.ylim(0,len(kernels))
    plt.xlim(0,1)
    labels = ['Features']+['Minor Component']*(num_levels-3)+['Major Component','Full Model']
    plt.xticks([w*x for x in np.arange(0.5,num_levels+0.5,1)],labels,fontsize=12)
    if add_text:
        plt.yticks(np.arange(len(kernels)-0.5,-0.5,-1),aligned_names,ha='left',family='monospace')
        plt.gca().get_yaxis().set_tick_params(pad=400)
    else:
        plt.yticks([])
    plt.title('Nested Models',fontsize=20)
    plt.tight_layout()
    if add_text:
        plt.text(-.255,len(kernels)+.35,'Alignment',fontsize=12)
        plt.text(-.385,len(kernels)+.35,'Support',fontsize=12)
        plt.text(-.555,len(kernels)+.35,'Kernel',fontsize=12)
        
    # Save results
    if save_results:
        fig_filename = os.path.join(run_params['figure_dir'],'nested_models_'+str(num_levels)+'_polished.png')
        plt.savefig(fig_filename)
        #df.to_csv(run_params['output_dir']+'/kernels_and_dropouts.csv')
    return df




def plot_dropouts_2(run_params,save_results=True,num_levels=3,add_text=True):
    '''
        Makes a visual and graphic representation of how the kernels are nested inside dropout models
        save_results (bool) if True, saves the figure
        num_levels (int) number of levels in nested model to plot
        add_text (bool) if True, adds descriptive text to left hand side of plot for each kernel
    '''
    if num_levels==3:
        if add_text:
            plt.figure(figsize=(9,8))
        else:
            plt.figure(figsize=(5.5,8))
    else:
        plt.figure(figsize=(16,8))
    w = 1/num_levels  
 
    # Get list of dropouts and kernels
    drops = set([x for x in run_params['dropouts'] if not run_params['dropouts'][x]['is_single'] ])
    kernels = run_params['kernels'].copy()
 
    # Build dataframe
    df = pd.DataFrame(index=kernels.keys())
    
    # Add the individual dropouts
    df['level-1']= df.index.values
    for k in kernels:
        if k in drops:
            drops.remove(k)
    
    # Add each grouping of dropouts
    if 'levels' in run_params:
        levels = run_params['levels'].copy()
        keys = list(levels.keys())
        for dex, key in enumerate(keys):
            levels[int(key)] = levels.pop(key)
    else:
        levels={
            num_levels:['Full'],
            num_levels-1:['visual','behavioral','cognitive'],
            num_levels-2:['licking','task','face_motion_energy','pupil_and_running','all-images','beh_model','expectation'],
            num_levels-3:['licking_bouts','licking_each_lick','pupil_and_omissions','trial_type','change_and_rewards'],
            num_levels-4:['running_and_omissions','hits_and_rewards'],
            }
    #return (df,drops,levels)
    for level in np.arange(num_levels,1,-1):
        df,drops = make_level(df,drops, level,  levels[level],  run_params)
        
    # re-organized dataframe
    # All the renaming is for sorting the features
    df=df[['level-'+str(x) for x in range(1,num_levels+1)]]
    df['level-2'] = ['atask' if x == 'task' else x for x in df['level-2']]
    df['level-2'] = ['azexpectation' if x == 'expectation' else x for x in df['level-2']]
    df['level-1'] = ['z'+x if 'face' in x else x for x in df['level-1']]
    df['level-1'] = ['ahits' if x == 'hits' else x for x in df['level-1']]
    df['level-1'] = ['bmisses' if x == 'misses' else x for x in df['level-1']]
    df['level-1'] = ['bnpassive_change' if x == 'passive_change' else x for x in df['level-1']]
    df = df.sort_values(by=['level-'+str(x) for x in np.arange(num_levels,0,-1)])
    df['level-2'] = ['task' if x == 'atask' else x for x in df['level-2']]
    df['level-2'] = ['expectation' if x == 'azexpectation' else x for x in df['level-2']]
    df['level-1'] = [ x[1:] if 'zface' in x else x for x in df['level-1']]
    df['level-1'] = ['hits' if x == 'ahits' else x for x in df['level-1']]
    df['level-1'] = ['misses' if x == 'bmisses' else x for x in df['level-1']]
    df['level-1'] = ['passive_change' if x == 'bnpassive_change' else x for x in df['level-1']]
    df['text'] = [run_params['kernels'][k]['text'] for k in df.index.values]
    df['support'] = [(np.round(run_params['kernels'][k]['offset'],2), np.round(run_params['kernels'][k]['length'] +  run_params['kernels'][k]['offset'],2)) for k in df.index.values]

    # Rename stuff, purely for explanatory purposes

    df['level-2'] = ['behavioral_model' if x == 'beh_model' else x for x in df['level-2']]  
    df['level-2'] = ['licks' if x == 'licks' else x for x in df['level-2']]
    df['level-2'] = ['omissions' if x == 'expectation' else x for x in df['level-2']]
    df['level-2'] = ['task' if x == 'cognitive' else x for x in df['level-2']]

    df['level-1'] = ['bias strategy' if x == 'model_bias' else x for x in df['level-1']]
    df['level-1'] = ['task strategy' if x == 'model_task0' else x for x in df['level-1']]
    df['level-1'] = ['post omission strategy' if x == 'model_omissions1' else x for x in df['level-1']]
    df['level-1'] = ['timing strategy' if x == 'model_timing1D' else x for x in df['level-1']]

    # Make sure all dropouts were used
    if len(drops) > 0:
        print('Warning, dropouts not used')
        print(drops)

    # Make Color Dictionary
    labels=[]
    colors=[]
    for level in range(1,num_levels+1):
        new_labels = list(df['level-'+str(level)].unique())
        labels = labels + ['level-'+str(level)+'-'+ x for x in new_labels]
        colors = colors + sns.color_palette('hls', len(new_labels)) 
    color_dict = {x:y for (x,y) in  zip(labels,colors)}
    for level in range(1,num_levels+1):
        color_dict['level-'+str(level)+'--'] = (0.8,0.8,0.8)

    # add color of level-1 value to df['color']
    df['color'] = None
    # Get Project Colors
    proj_colors = gvt.project_colors() 
    for key in color_dict.keys():
        dropout = key.split('-')[2]
        if dropout == 'all':
            dropout = 'all-images'
        if dropout in proj_colors:
            color_dict[key] = proj_colors[dropout]
        if key.startswith('level-1'):
            dropout = key.split('level-1-')[1]
            if dropout in df.index.values.tolist():
                df.at[dropout,'color'] = color_dict[key]
    color_dict['level-2-behavioral'] = color_dict['level-1-licks']

    # Plot Squares
    uniques = set()
    maxn = len(df)
    last = {x:'null' for x in np.arange(1,num_levels+1,1)} 
    for index, k in enumerate(df.index.values):
        for level in range(1,num_levels+1):
            plt.axhspan(maxn-index-1,maxn-index,w*(level-1),w*level,color=color_dict['level-'+str(level)+'-'+df.loc[k]['level-'+str(level)]]) 
            # If this is a new group, add a line and a text label
            if (level > 1)&(not (df.loc[k]['level-'+str(level)] == '-')) & ('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)] not in uniques) :
                uniques.add('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)])
                plt.text(w*(level-1)+0.01,maxn-index-1+.25,df.loc[k]['level-'+str(level)],fontsize=12)
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif (level > 1) & (not (df.loc[k]['level-'+str(level)] == last[level])):
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif level == 1:
                # For the individual regressors, just label, no lines
                plt.text(0.01,maxn-index-1+.25,df.loc[k]['level-'+str(level)],fontsize=12)
            last[level] = df.loc[k]['level-'+str(level)]

    # Define some lines between levels   
    for level in range(1,num_levels): 
        plt.axvline(w*level,color='k') 
    
    # Make formated ylabels that include support and alignment event   
    max_name = np.max([len(x) for x in df.index.values])+3 
    max_support = np.max([len(str(x)) for x in df['support'].values])+3
    max_text = np.max([len(str(x)) for x in df['text'].values])
    aligned_names = [row[1].name.ljust(max_name)+str(row[1]['support']).ljust(max_support)+row[1]['text'].ljust(max_text) for row in df.iterrows()]

    # clean up axes
    plt.ylim(0,len(kernels))
    plt.xlim(0,1)
    labels = ['Features']+['Minor Component']*(num_levels-3)+['Major Component','Full Model']
    plt.xticks([w*x for x in np.arange(0.5,num_levels+0.5,1)],labels,fontsize=12)
    if add_text:
        plt.yticks(np.arange(len(kernels)-0.5,-0.5,-1),aligned_names,ha='left',family='monospace')
        plt.gca().get_yaxis().set_tick_params(pad=400)
    else:
        plt.yticks([])
    plt.title('Nested Models',fontsize=20)
    plt.tight_layout()
    if add_text:
        plt.text(-.255,len(kernels)+.35,'Alignment',fontsize=12)
        plt.text(-.385,len(kernels)+.35,'Support',fontsize=12)
        plt.text(-.555,len(kernels)+.35,'Kernel',fontsize=12)
        
    # Save results
    if save_results:
        fig_filename = os.path.join(run_params['figure_dir'],'nested_models_'+str(num_levels)+'_polished.png')
        plt.savefig(fig_filename)
        #df.to_csv(run_params['output_dir']+'/kernels_and_dropouts.csv')
    return df

def plot_dropouts(run_params,save_results=True,num_levels=6,add_text=True, SAC=False):
    '''
        Makes a visual and graphic representation of how the kernels are nested inside dropout models
        save_results (bool) if True, saves the figure
        num_levels (int) number of levels in nested model to plot
        add_text (bool) if True, adds descriptive text to left hand side of plot for each kernel
    '''
    if num_levels==4:
        if add_text:
            plt.figure(figsize=(16,8))
        else:
            plt.figure(figsize=(12,8))
    elif num_levels==6:
        plt.figure(figsize=(19,8))
    else:
        plt.figure(figsize=(16,8))
    w = 1/num_levels  
 
    # Get list of dropouts and kernels
    drops = set([x for x in run_params['dropouts'] if not run_params['dropouts'][x]['is_single'] ])
    kernels = run_params['kernels'].copy()
 
    # Build dataframe
    df = pd.DataFrame(index=kernels.keys())
    
    # Add the individual dropouts
    df['level-1']= df.index.values
    for k in kernels:
        if k in drops:
            drops.remove(k)
    
    # Add each grouping of dropouts
    if 'levels' in run_params:
        levels = run_params['levels'].copy()
        keys = list(levels.keys())
        for dex, key in enumerate(keys):
            levels[int(key)] = levels.pop(key)
    else:
        levels={
            num_levels:['Full'],
            num_levels-1:['visual','behavioral','cognitive'],
            num_levels-2:['licking','task','face_motion_energy','pupil_and_running','all-images','beh_model','expectation'],
            num_levels-3:['licking_bouts','licking_each_lick','pupil_and_omissions','trial_type','change_and_rewards'],
            num_levels-4:['running_and_omissions','hits_and_rewards'],
            }
    #return (df,drops,levels)
    for level in np.arange(num_levels,1,-1):
        if SAC & level == 2:
            drops.add('expectation')
            drops.add('all-images')
        df,drops = make_level(df,drops, level,  levels[level],  run_params)
        
    # re-organized dataframe
    # All the renaming is for sorting the features
    df=df[['level-'+str(x) for x in range(1,num_levels+1)]]
    df['level-3'] = ['avisual' if x == 'visual' else x for x in df['level-3']]
    if SAC:
        df['level-3'] = ['aomissions' if x == 'expectation' else x for x in df['level-3']]
    df['level-2'] = ['atask' if x == 'task' else x for x in df['level-2']]
    df['level-2'] = ['zface' if x == 'face_motion_energy' else x for x in df['level-2']]
    df['level-1'] = ['ahits' if x == 'hits' else x for x in df['level-1']]
    df['level-1'] = ['bmisses' if x == 'misses' else x for x in df['level-1']]
    df['level-1'] = ['bnpassive_change' if x == 'passive_change' else x for x in df['level-1']]
    df = df.sort_values(by=['level-'+str(x) for x in np.arange(num_levels,0,-1)])
    df['level-3'] = ['visual' if x == 'avisual' else x for x in df['level-3']]
    if SAC:
        df['level-3'] = ['expectation' if x == 'aomissions' else x for x in df['level-3']]
    df['level-2'] = ['task' if x == 'atask' else x for x in df['level-2']]
    df['level-2'] = ['face_motion_energy' if x == 'zface' else x for x in df['level-2']]
    df['level-1'] = ['hits' if x == 'ahits' else x for x in df['level-1']]
    df['level-1'] = ['misses' if x == 'bmisses' else x for x in df['level-1']]
    df['level-1'] = ['passive_change' if x == 'bnpassive_change' else x for x in df['level-1']]
    df['text'] = [run_params['kernels'][k]['text'] for k in df.index.values]
    df['support'] = [(np.round(run_params['kernels'][k]['offset'],2), np.round(run_params['kernels'][k]['length'] +  run_params['kernels'][k]['offset'],2)) for k in df.index.values]

    # Rename stuff, purely for explanatory purposes
    if SAC:
        df['level-3'] = ['omissions' if x == 'expectation' else x for x in df['level-3']]
    df['level-2'] = ['behavioral_model' if x == 'beh_model' else x for x in df['level-2']]  
    df['level-2'] = ['licks' if x == 'licks' else x for x in df['level-2']]
    df['level-2'] = ['omissions' if x == 'expectation' else x for x in df['level-2']]

    df['level-1'] = ['bias strategy' if x == 'model_bias' else x for x in df['level-1']]
    df['level-1'] = ['task strategy' if x == 'model_task0' else x for x in df['level-1']]
    df['level-1'] = ['post omission strategy' if x == 'model_omissions1' else x for x in df['level-1']]
    df['level-1'] = ['timing strategy' if x == 'model_timing1D' else x for x in df['level-1']]

    # Make sure all dropouts were used
    if len(drops) > 0:
        print('Warning, dropouts not used')
        print(drops)

    # Make Color Dictionary
    labels=[]
    colors=[]
    for level in range(1,num_levels+1):
        new_labels = list(df['level-'+str(level)].unique())
        labels = labels + ['level-'+str(level)+'-'+ x for x in new_labels]
        colors = colors + sns.color_palette('hls', len(new_labels)) 
    color_dict = {x:y for (x,y) in  zip(labels,colors)}
    for level in range(1,num_levels+1):
        color_dict['level-'+str(level)+'--'] = (0.8,0.8,0.8)

    # add color of level-1 value to df['color']
    df['color'] = None
    # Get Project Colors
    proj_colors = gvt.project_colors() 
    for key in color_dict.keys():
        dropout = key.split('-')[2]
        if dropout == 'all':
            dropout = 'all-images'
        if dropout in proj_colors:
            color_dict[key] = proj_colors[dropout]
        if key.startswith('level-1'):
            dropout = key.split('level-1-')[1]
            if dropout in df.index.values.tolist():
                df.at[dropout,'color'] = color_dict[key]
 
    # Plot Squares
    uniques = set()
    maxn = len(df)
    last = {x:'null' for x in np.arange(1,num_levels+1,1)} 
    for index, k in enumerate(df.index.values):
        for level in range(1,num_levels+1):
            plt.axhspan(maxn-index-1,maxn-index,w*(level-1),w*level,color=color_dict['level-'+str(level)+'-'+df.loc[k]['level-'+str(level)]]) 
            # If this is a new group, add a line and a text label
            if (level > 1)&(not (df.loc[k]['level-'+str(level)] == '-')) & ('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)] not in uniques) :
                uniques.add('level-'+str(level)+'-'+df.loc[k]['level-'+str(level)])
                plt.text(w*(level-1)+0.01,maxn-index-1+.25,df.loc[k]['level-'+str(level)],fontsize=12)
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif (level > 1) & (not (df.loc[k]['level-'+str(level)] == last[level])):
                plt.plot([0,w*level],[maxn-index,maxn-index], 'k-',alpha=1)
                #plt.plot([w*(level-1),w*level],[maxn-index,maxn-index], 'k-',alpha=1)
            elif level == 1:
                # For the individual regressors, just label, no lines
                plt.text(0.01,maxn-index-1+.25,df.loc[k]['level-'+str(level)],fontsize=12)
            last[level] = df.loc[k]['level-'+str(level)]

    # Define some lines between levels   
    for level in range(1,num_levels): 
        plt.axvline(w*level,color='k') 
    
    # Make formated ylabels that include support and alignment event   
    max_name = np.max([len(x) for x in df.index.values])+3 
    max_support = np.max([len(str(x)) for x in df['support'].values])+3
    max_text = np.max([len(str(x)) for x in df['text'].values])
    aligned_names = [row[1].name.ljust(max_name)+str(row[1]['support']).ljust(max_support)+row[1]['text'].ljust(max_text) for row in df.iterrows()]

    # clean up axes
    plt.ylim(0,len(kernels))
    plt.xlim(0,1)
    labels = ['Features']+['Minor Component']*(num_levels-3)+['Major Component','Full Model']
    plt.xticks([w*x for x in np.arange(0.5,num_levels+0.5,1)],labels,fontsize=16)
    if add_text:
        plt.yticks(np.arange(len(kernels)-0.5,-0.5,-1),aligned_names,ha='left',family='monospace')
        plt.gca().get_yaxis().set_tick_params(pad=400)
    else:
        plt.yticks([])
    plt.title('Nested Models',fontsize=20)
    plt.tight_layout()
    if add_text:
        plt.text(-.255,len(kernels)+.35,'Alignment',fontsize=12)
        plt.text(-.385,len(kernels)+.35,'Support',fontsize=12)
        plt.text(-.555,len(kernels)+.35,'Kernel',fontsize=12)
        
    # Save results
    if save_results:
        fig_filename = os.path.join(run_params['figure_dir'],'nested_models_'+str(num_levels)+'.png')
        plt.savefig(fig_filename)
        df.to_csv(run_params['output_dir']+'/kernels_and_dropouts.csv')
    return df

def make_level(df, drops, this_level_num,this_level_drops,run_params):
    '''
        Helper function for plot_dropouts()
        Determines what dropout each kernel is a part of, as well as keeping track of which dropouts have been used. 
    '''
    df['level-'+str(this_level_num)] = [get_containing_dictionary(key, this_level_drops,run_params) for key in df.index.values]
    for d in this_level_drops:
        drops.remove(d)
    return df,drops

def get_containing_dictionary(key,dicts,run_params):
    '''
        Helper function for plot_dropouts()
        returns which dropout contains each kernel
    '''
    label='-'
    
    for d in dicts:
        found=False
        if (d == 'Full') & (key in run_params['dropouts']['Full']['kernels']):
            if found:
                print('WARNING DUPLICATE DROPOUT')
            found=True
            label= d
        elif key in run_params['dropouts'][d]['dropped_kernels']:
            if found:
                print('WARNING DUPLICATE DROPOUT')
            found=True
            label= d
    return label




