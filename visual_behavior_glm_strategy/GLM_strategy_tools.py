import os
from scipy import stats
import statsmodels.stats.multicomp as mc
import copy
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior_glm_strategy.GLM_visualization_tools as gvt
from scipy.stats import linregress
import seaborn as sns
from mpl_toolkits.axes_grid1 import Divider, Size

## General Utilities
################################################################################

def add_behavior_session_metrics(df,summary_df):
    '''
        Merges the behavioral summary table onto the dataframe passed in 
    ''' 
    behavior_columns = ['behavior_session_id','visual_strategy_session',
        'strategy_dropout_index','dropout_task0','dropout_omissions1',
        'dropout_omissions','dropout_timing1D','strategy_labels_with_none',
        'strategy_labels_with_mixed','strategy_labels']
    out_df = pd.merge(df, summary_df[behavior_columns],
        on='behavior_session_id',
        suffixes=('','_ophys_table'),
        validate='many_to_one')
    return out_df


def save_figure(fig,model_version, ymetric, filename):
    glm_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
    if not os.path.isdir(glm_dir + 'v_'+model_version +'/figures/strategy'):
        os.mkdir(glm_dir + 'v_'+model_version +'/figures/strategy')
    if not os.path.isdir(glm_dir + 'v_'+model_version +'/figures/strategy/'+ymetric):
        os.mkdir(glm_dir + 'v_'+model_version +'/figures/strategy/'+ymetric)
    plt.savefig(glm_dir + 'v_'+ model_version +'/figures/strategy/'+ymetric+\
        '/'+filename+".svg")
    plt.savefig(glm_dir + 'v_'+ model_version +'/figures/strategy/'+ymetric+\
        '/'+filename+".png")
    print('Figure saved to: '+filename)


def string_mapper(string):
    d = {
        'Vip-IRES-Cre':'Vip Inhibitory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Slc17a7-IRES2-Cre':'Excitatory',
    }
    return d[string]

## Kernel plots
################################################################################

def plot_kernels_by_strategy_by_session(weights_beh, run_params, ym='omissions',
    cre_line = 'Vip-IRES-Cre',compare=['strategy_labels'],savefig=False):

    # By Session number
    sessions = ['Familiar','Novel 1', 'Novel >1']
    filter_sessions_on ='experience_level'
    image_set = ['familiar','novel']

    fig, ax = plt.subplots(2,len(sessions),figsize=(len(sessions)*4,6),sharey=True)
    for dex, session in enumerate(sessions):
        show_legend = dex == len(sessions) - 1
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = [session], cell_filter = cre_line,
            area_filter=['VISp'], compare=compare, plot_errors=True,
            save_kernels=False,ax=ax[0,dex],fs1=16,fs2=12,
            show_legend=show_legend,filter_sessions_on = filter_sessions_on,
            image_set=image_set) 
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = [session], cell_filter = cre_line,
            area_filter=['VISl'], compare=compare, plot_errors=True,
            save_kernels=False,ax=ax[1,dex],fs1=16,fs2=12,show_legend=False,
            filter_sessions_on = filter_sessions_on,image_set=image_set) 
        ax[0,dex].set_title(str(session),fontsize=16)
        if dex == 0:
            ax[0,0].set_ylabel('V1\n'+ax[0,0].get_ylabel(),fontsize=16)
            ax[1,0].set_ylabel('LM\n'+ax[1,0].get_ylabel(),fontsize=16)

    plt.tight_layout()
    if savefig:
        filename = ym+'_by_'+filter_sessions_on+'_'+cre_line
        save_figure(fig,run_params['version'], ym, filename)

def kernels_by_cre(weights_beh, run_params, kernel='omissions',
    compare=['strategy_labels'],equipment_filter='all',area_filter=['VISp','VISl'],
    sessions=['Familiar'], image_set='familiar',filter_sessions_on='experience_level',
    savefig=False, sharey=False, depth_filter=[0,1000]):
     
    cres = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    limit_list = {
        'Slc17a7-IRES2-Cre':[-.0005,0.002],
        'Sst-IRES-Cre':[-.006,0.0125],
        'Vip-IRES-Cre':[-0.005,0.025],
        }
    for dex, cre in enumerate(cres):
        height = 4
        width=8
        pre_horz_offset = 1.5
        post_horz_offset = 2.5
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        
        duration = run_params['kernels'][kernel]['length']
        h = [Size.Fixed(pre_horz_offset),\
            Size.Fixed((width-pre_horz_offset-post_horz_offset)/3*duration)]

        #if kernel == 'all-images':
        #    h = [Size.Fixed(pre_horz_offset),\
        #        Size.Fixed((width-pre_horz_offset-post_horz_offset)/3*.75)]     
        #else:
        #    h = [Size.Fixed(pre_horz_offset),\
        #        Size.Fixed(width-pre_horz_offset-post_horz_offset)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(),\
            axes_locator=divider.new_locator(nx=1,ny=1))  
     
        show_legend = dex == len(cres) - 1
        out = strategy_kernel_comparison(weights_beh, run_params, kernel, 
            threshold=0, drop_threshold = 0, 
            session_filter = sessions, cell_filter = cre,area_filter=area_filter, 
            compare=compare, plot_errors=True,save_kernels=False,ax=ax,
            show_legend=show_legend,filter_sessions_on=filter_sessions_on,
            image_set=image_set,equipment_filter=equipment_filter,
            depth_filter=depth_filter) 
        ax.set_title(string_mapper(cre),fontsize=16)
        ax.set_ylabel(kernel+' weights\n(Ca$^{2+}$ events)',fontsize=16)
       
        #ylim=ax.get_ylim()
        ylim = limit_list[cre]
        ax.set_ylim(ylim) 
        if kernel =='omissions':
            out[2].plot(0,ylim[0],'co',zorder=10,clip_on=False)
        elif kernel =='hits':
            out[2].plot(0,ylim[0],'ro',zorder=10,clip_on=False)
        elif kernel == 'misses':
            out[2].plot(0,ylim[0],'rx',zorder=10,clip_on=False)
        else:
            out[2].plot(0,ylim[0],'ko',zorder=10,clip_on=False)
        if kernel in ['omissions','hits','misses']:
            out[2].plot(.75,ylim[0],'ko',zorder=10,clip_on=False)
            out[2].plot(1.5,ylim[0],'o',color='gray',zorder=10,clip_on=False)

        if savefig:
            #filename = kernel+'_by_cre_line_'+'_'.join(compare)+'_'+equipment_filter
            filename = kernel+'_'+cre+'_by_strategy'
            save_figure(fig,run_params['version'], kernel, filename)
    return ax 



def compare_cre_kernels(weights_beh, run_params, ym='omissions',
    compare=['strategy_labels'],equipment_filter='all',area_filter=['VISp','VISl'],
    sessions=['Familiar','Novel 1','Novel >1'],image_set='familiar',
    filter_sessions_on='experience_level',savefig=False,sharey=False,
    depth_filter=[0,1000]):

    cres = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    fig, ax = plt.subplots(2,len(cres),figsize=(12,6),sharey=sharey)
    for dex, cre in enumerate(cres):
        show_legend = dex == len(cres) - 1
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = sessions, cell_filter = cre,area_filter=area_filter[0], 
            compare=compare, plot_errors=True,save_kernels=False,ax=ax[0,dex],
            show_legend=show_legend,filter_sessions_on=filter_sessions_on,
            image_set=image_set,equipment_filter=equipment_filter,
            depth_filter=depth_filter) 
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = sessions, cell_filter = cre,area_filter=area_filter[1], 
            compare=compare, plot_errors=True,save_kernels=False,ax=ax[1,dex],
            show_legend=show_legend,filter_sessions_on=filter_sessions_on,
            image_set=image_set,equipment_filter=equipment_filter,
            depth_filter=depth_filter) 
        ax[0,dex].set_title(string_mapper(cre),fontsize=16)

    ax[0,0].set_ylabel('V1 Kernel Weights',fontsize=16)
    ax[1,0].set_ylabel('LM Kernel Weights',fontsize=16)
    plt.tight_layout()

    if savefig:
        filename = ym+'_by_cre_line_'+'_'.join(compare)+'_'+equipment_filter
        save_figure(fig,run_params['version'], ym, filename)
    return ax 

def strategy_kernel_comparison(weights_df, run_params, kernel, drop_threshold=0,
    session_filter=['Familiar','Novel 1','Novel >1'],equipment_filter="all",
    depth_filter=[0,1000],cell_filter="all",area_filter=['VISp','VISl'],
    compare=['cre_line'],plot_errors=False,save_kernels=False,fig=None, 
    ax=None,fs1=16,fs2=12,show_legend=True,filter_sessions_on='experience_level',
    image_set=['familiar','novel'],threshold=0,set_title=None): 
    '''
        Plots the average kernel across different comparisons groups of cells
        First applies hard filters, then compares across remaining cells

        INPUTS:
        run_params              = glm_params.load_run_params(<version>) 
        results_pivoted         = gat.build_pivoted_results_summary(
            'adj_fraction_change_from_full',results_summary=results)
        weights_df              = gat.build_weights_df(run_params, results_pivoted)
        kernel                  The name of the kernel to be plotted
        drop_threshold,         the minimum adj_fraction_change_from_full 
                                for the dropout model of just dropping this kernel
        session_filter,         The list of session numbers to include
        equipment_filter,       "scientifica" or "mesoscope" filter, anything 
                                else plots both 
        cell_filter,            "sst","vip","slc", anything else plots all types
        area_filter,            the list of targeted_structures to include
        compare (list of str)   list of categorical labels in weights_df to 
                                split on and compare
                                First entry of compare determines color of the 
                                line, second entry determines linestyle
        plot_errors (bool)      if True, plots a shaded error bar for each group of cells 
    '''
    version = run_params['version']
    filter_string = ''
    problem_sessions = gvt.get_problem_sessions()   
    weights_df = weights_df.copy()
     
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
    filename = os.path.join(run_params['fig_kernels_dir'],kernel+\
        '_comparison_by_'+'_and_'.join(compare)+filter_string+'.svg')

    # Set up time vectors.
    if kernel in ['preferred_image', 'all-images']:
        run_params['kernels'][kernel] = run_params['kernels']['image0'].copy()
    if kernel == 'all-omissions':
        run_params['kernels'][kernel] = run_params['kernels']['omissions'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['omissions']['length'] +\
            run_params['kernels']['post-omissions']['length']
    if kernel == 'all-hits':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['hits']['length'] + \
            run_params['kernels']['post-hits']['length']   
    if kernel == 'all-misses':
        run_params['kernels'][kernel] = run_params['kernels']['misses'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['misses']['length'] + \
            run_params['kernels']['post-misses']['length']   
    if kernel == 'all-passive_change':
        run_params['kernels'][kernel] = run_params['kernels']['passive_change'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['passive_change']['length'] + \
            run_params['kernels']['post-passive_change']['length']   
    if kernel == 'task':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()   
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], \
        run_params['kernels'][kernel]['offset'] + \
        run_params['kernels'][kernel]['length'],1/31)
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
        weights = weights_df.query('(not passive)&\
            (targeted_structure in @area_filter)&\
            (cre_line in @cell_list)&\
            (equipment_name in @equipment_list)&\
            ({0} in @session_filter)&\
            (ophys_session_id not in @problem_sessions)&\
            (imaging_depth < @depth_filter[1])&\
            (imaging_depth > @depth_filter[0])&\
            (variance_explained_full > @threshold)&\
            ({1} <= @drop_threshold)'.format(filter_sessions_on, kernel))
    else:
        weights = weights_df.query('(not passive)&\
            (targeted_structure in @area_filter)&\
            (cre_line in @cell_list)&\
            (equipment_name in @equipment_list)&\
            ({0} in @session_filter) &\
            (ophys_session_id not in @problem_sessions) &\
            (imaging_depth < @depth_filter[1]) &\
            (imaging_depth > @depth_filter[0])&\
            (variance_explained_full > @threshold)'.format(filter_sessions_on))


    # Plotting settings
    if ax is None:
        #fig,ax=plt.subplots(figsize=(8,4))
        height = 4
        width=8
        pre_horz_offset = 1.5
        post_horz_offset = 2.5
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        if kernel == 'all_images':
            h = [Size.Fixed(pre_horz_offset),\
                Size.Fixed((width-pre_horz_offset-post_horz_offset)/3*.75)]     
        else:
            h = [Size.Fixed(pre_horz_offset),\
                Size.Fixed(width-pre_horz_offset-post_horz_offset)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(),\
            axes_locator=divider.new_locator(nx=1,ny=1))  
    
    # Define color scheme for project
    colors = gvt.project_colors()

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
        k=strategy_kernel_comparison_inner(ax,weights_dfiltered,group,\
            color,linestyle, time_vec, plot_errors=plot_errors) 
        outputs[group]=k

    # Clean Plot, and add details
    ax.axhline(0, color='k',linestyle='--',alpha=0.25)

    if kernel == 'omissions':
        ax.set_xlabel('Time from omission (s)',fontsize=fs1)
    elif kernel in ['hits','misses']:
        ax.set_xlabel('Time from change (s)',fontsize=fs1)
    else:
        ax.set_xlabel('Time (s)',fontsize=fs1)

    ax.set_xlim(time_vec[0]-0.05,time_vec[-1])  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    gvt.add_stimulus_bars(ax,kernel,alpha=.1)
    ax.xaxis.set_tick_params(labelsize=fs2)
    ax.yaxis.set_tick_params(labelsize=fs2)
    if show_legend:
        ax.legend(title=' & '.join(compare).replace('_',' '),handlelength=4)
 
    return outputs, fig,ax

def strategy_kernel_comparison_inner(ax, df,label,color,linestyle,time_vec,\
    plot_errors=True,linewidth=4,alpha=.25):
    '''
        Plots the average kernel for the cells in df
        
        ax, the axis to plot on
        df, series of cells with column that is the kernel to plot
        label, what to label this group of cells
        color, the line color for this group of cells
        linestyle, the line style for this group of cells
        time_vec, the time basis to plot on
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
        ax.fill_between(time_vec, 
            df_norm.mean(axis=0)-df_norm.std(axis=0)/np.sqrt(df_norm.shape[0]), 
            df_norm.mean(axis=0)+df_norm.std(axis=0)/np.sqrt(df_norm.shape[0]),
            facecolor=color, alpha=alpha)   
    ax.plot(time_vec, df_norm.mean(axis=0),linestyle=linestyle,label=label,
        color=color,linewidth=linewidth)
    return df_norm.mean(axis=0)


def strategy_kernel_evaluation(weights_df, run_params, kernel, save_results=False, 
    drop_threshold=0,session_filter=['Familiar','Novel 1','Novel >1'],
    equipment_filter="all",cell_filter='all',area_filter=['VISp','VISl'],
    depth_filter=[0,1000],filter_sessions_on='experience_level',
    plot_dropout_sorted=True):  
    '''

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
    problem_sessions = gvt.get_problem_sessions()

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
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['omissions']['length'] +\
            run_params['kernels']['post-omissions']['length']
    if kernel == 'all-hits':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['hits']['length'] + \
            run_params['kernels']['post-hits']['length']   
    if kernel == 'all-misses':
        run_params['kernels'][kernel] = run_params['kernels']['misses'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['misses']['length'] + \
            run_params['kernels']['post-misses']['length']   
    if kernel == 'all-passive_change':
        run_params['kernels'][kernel] = run_params['kernels']['passive_change'].copy()
        run_params['kernels'][kernel]['length'] = \
            run_params['kernels']['passive_change']['length'] + \
            run_params['kernels']['post-passive_change']['length']   
    if kernel == 'task':
        run_params['kernels'][kernel] = run_params['kernels']['hits'].copy()   
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], \
        run_params['kernels'][kernel]['offset'] + \
        run_params['kernels'][kernel]['length'],1/31)
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
            'single-post-omissions':'single_post_omissions',
            'single-all-images':'single_all_omissions',
            })

        kernel = kernel.replace('-','_')

    # Applying hard thresholds to dataset
    # don't apply overall VE, or dropout threshold limits here, 
    # since we look at the effects of those criteria below. 
    # we do remove NaN dropouts here
    if kernel in weights_df:
        weights = weights_df.query('(not passive)&\
            (targeted_structure in @area_filter)&\
            (cre_line in @cell_list)&\
            (equipment_name in @equipment_list)&\
            ({0} in @session_filter) &\
            (ophys_session_id not in @problem_sessions) &\
            (imaging_depth < @depth_filter[1]) &\
            (imaging_depth > @depth_filter[0])&\
            (variance_explained_full > 0) &\
            ({1} <= 0)'.format(filter_sessions_on, kernel))
    else:
        weights = weights_df.query('(not passive)&\
            (targeted_structure in @area_filter)&\
            (cre_line in @cell_list)&\
            (equipment_name in @equipment_list)&\
            ({0} in @session_filter) &\
            (ophys_session_id not in @problem_sessions) &\
            (imaging_depth < @depth_filter[1]) &\
            (imaging_depth > @depth_filter[0])&\
            (variance_explained_full > 0)'.format(filter_sessions_on)) 

    # Have to do a manual filtering step here because weird things happen when combining
    # two kernels
    if kernel == 'task':
        weights = weights[~weights['task_weights'].isnull()]

    # Get all cells data
    sst_table=weights.query('cre_line=="Sst-IRES-Cre"')[[kernel+'_weights',\
        kernel,'strategy_dropout_index']]
    vip_table=weights.query('cre_line=="Vip-IRES-Cre"')[[kernel+'_weights',\
        kernel,'strategy_dropout_index']]
    slc_table=weights.query('cre_line=="Slc17a7-IRES2-Cre"')[[kernel+'_weights',\
        kernel,'strategy_dropout_index']]  
    ncells={
        'vip':len(vip_table),
        'sst':len(sst_table),
        'exc':len(slc_table),
        }

    zlims_test = strategy_kernel_heatmap_with_dropout(vip_table, sst_table, slc_table,
        time_vec, kernel, run_params,ncells,session_filter=session_filter,
        savefig=save_results)

def strategy_kernel_heatmap_with_dropout(vip_table, sst_table, slc_table, 
    time_vec,kernel, run_params, ncells = {},ax=None,extra='',zlims=None,
    session_filter=['Familiar','Novel 1','Novel >1'],savefig=False):

    if ax==None:
        #fig,ax = plt.subplots(figsize=(8,4))
        height = 4
        width=8
        pre_horz_offset = 1.5
        post_horz_offset = 2.5
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(pre_horz_offset),
            Size.Fixed(width-pre_horz_offset-post_horz_offset-.25)]
        v = [Size.Fixed(vertical_offset),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax3 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))
        v = [Size.Fixed(vertical_offset+(height-vertical_offset-.5)/3),
            Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax2 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))
        v = [Size.Fixed(vertical_offset+2*(height-vertical_offset-.5)/3),
            Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax1 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))
        
        h = [Size.Fixed(width-post_horz_offset-.25),Size.Fixed(.25)]
        v = [Size.Fixed(vertical_offset+2*(height-vertical_offset-.5)/3),
            Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        dax1 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))  
        v = [Size.Fixed(vertical_offset+(height-vertical_offset-.5)/3),
            Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        dax2 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))  
        v = [Size.Fixed(vertical_offset),Size.Fixed((height-vertical_offset-.5)/3)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        dax3 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))  

        h = [Size.Fixed(width-post_horz_offset+.25),Size.Fixed(.25)]
        v = [Size.Fixed(vertical_offset+(height-vertical_offset-.5)/2)+.125,
            Size.Fixed((height-vertical_offset-.5)/2-.125)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        cax1 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))  

        h = [Size.Fixed(width-post_horz_offset+.25),Size.Fixed(.25)]
        v = [Size.Fixed(vertical_offset/4),
            Size.Fixed((height-vertical_offset-.5)/2-.125)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        cax2 = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))  


    # Sort cells
    # convert kernels to columns
    ncols = len(vip_table[kernel+'_weights'].values[0])
    vip_df = pd.DataFrame(vip_table[kernel+'_weights'].to_list(),
                columns = ['w'+str(x) for x in range(0,ncols)])
    vip_df['dropout'] = vip_table.reset_index()['strategy_dropout_index']*-1
    vip_df = vip_df.sort_values(by=['dropout'],ascending=False)    
    sst_df = pd.DataFrame(sst_table[kernel+'_weights'].to_list(),
                columns = ['w'+str(x) for x in range(0,ncols)])
    sst_df['dropout'] = sst_table.reset_index()['strategy_dropout_index']*-1
    sst_df = sst_df.sort_values(by=['dropout'],ascending=False) 
    slc_df = pd.DataFrame(slc_table[kernel+'_weights'].to_list(),
                columns = ['w'+str(x) for x in range(0,ncols)])
    slc_df['dropout'] = slc_table.reset_index()['strategy_dropout_index']*-1
    slc_df = slc_df.sort_values(by=['dropout'],ascending=False) 

    weights_sorted = np.concatenate([slc_df.to_numpy(),sst_df.to_numpy(), 
        vip_df.to_numpy()])[:,0:-1].T
    drop_sorted = np.concatenate([slc_df.to_numpy(),sst_df.to_numpy(), 
        vip_df.to_numpy()])[:,-1].T
    slc_weights_sorted =slc_df.to_numpy()[:,0:-1].T
    slc_drop_sorted =   slc_df.to_numpy()[:,-1].T
    sst_weights_sorted =sst_df.to_numpy()[:,0:-1].T
    sst_drop_sorted =   sst_df.to_numpy()[:,-1].T
    vip_weights_sorted =vip_df.to_numpy()[:,0:-1].T
    vip_drop_sorted =   vip_df.to_numpy()[:,-1].T

    cbar1 = ax1.imshow(vip_weights_sorted.T,aspect='auto',
        extent=[time_vec[0], time_vec[-1], 0, 
        np.shape(slc_weights_sorted)[1]],cmap='bwr')
    if zlims is None:
        zlims =[-np.nanpercentile(np.abs(weights_sorted),97.5),
            np.nanpercentile(np.abs(weights_sorted),97.5)]
    cbar1.set_clim(zlims[0], zlims[1])
    cbar2 = ax2.imshow(sst_weights_sorted.T,aspect='auto',extent=[time_vec[0], 
        time_vec[-1], 0, np.shape(sst_weights_sorted)[1]],cmap='bwr')
    cbar2.set_clim(zlims[0], zlims[1])
    cbar3 = ax3.imshow(slc_weights_sorted.T,aspect='auto',extent=[time_vec[0], 
        time_vec[-1], 0, np.shape(vip_weights_sorted)[1]],cmap='bwr')
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


    if len(session_filter) ==1:
        extra=extra+'_'+session_filter[0].replace(' ','_').replace('>','p')
        title = title + ', '+session_filter[0]
    ax1.set_title(title,fontsize=20)

    cbar2=dax1.imshow(vip_drop_sorted[:,np.newaxis],aspect='auto',
        cmap='plasma') 
    dax1.set_yticks([])
    dax1.set_xticks([])
    color_bar=fig.colorbar(cbar2, cax=cax2,extend='min')
    color_bar.ax.set_title('S.I.',fontsize=16,loc='left') 
 
    color_bar.ax.tick_params(axis='both',labelsize=16)

    dax2.imshow(sst_drop_sorted[:,np.newaxis],aspect='auto',
        cmap='plasma') 
    dax2.set_yticks([])
    dax2.set_xticks([])
    dax3.imshow(slc_drop_sorted[:,np.newaxis],aspect='auto',
        cmap='plasma') 
    dax3.set_yticks([])
    dax3.set_xticks([])   

    if savefig:
        filename = os.path.join(run_params['fig_kernels_dir'],
            kernel+'_heatmap_with_dropout'+extra+'.svg')
        plt.savefig(filename) 
        print('Figure saved to: '+filename)
        filename = os.path.join(run_params['fig_kernels_dir'],
            kernel+'_heatmap_with_dropout'+extra+'.png')
        plt.savefig(filename) 
    return zlims




## Dropout Scatter functions 
################################################################################

def scatter_dataset(results_beh, run_params,threshold=0,ymetric_threshold=0, 
    xmetric='strategy_dropout_index', ymetric='omissions',depth=[0,1000],
    sessions=['Familiar','Novel 1','Novel >1'],area=['VISp','VISl'],
    use_prior_omissions=False,use_prior_image_set=False):

    fig, ax = plt.subplots(3,len(sessions)+1, figsize=(15,8))
    fits = {}
    cres = ['Slc17a7-IRES2-Cre', 'Vip-IRES-Cre','Sst-IRES-Cre']
    ymins = []
    ymaxs =[]
    for dex,cre in enumerate(cres):
        col_start = dex == 0
        fits[cre] = scatter_by_experience(results_beh, run_params, cre_line = cre, 
            threshold=threshold, ymetric_threshold=ymetric_threshold, 
            xmetric=xmetric, ymetric=ymetric, ax = ax[dex,:], depth=depth,
            col_start=col_start,use_prior_omissions=use_prior_omissions,
            area=area,use_prior_image_set=use_prior_image_set)
        ymins.append(fits[cre]['yrange'][0])
        ymaxs.append(fits[cre]['yrange'][1])
   
    # make consistent axes 
    for dex, cre in enumerate(cres):
        ax[dex,-1].set_ylim(np.min(ymins),np.max(ymaxs))
  
    return fits

def scatter_by_experience(results_beh, run_params, cre_line=None, threshold=0,
    ymetric_threshold=0,xmetric='strategy_dropout_index',ymetric='omissions',
    sessions=['Familiar','Novel 1','Novel >1'],ax=None,col_start=False,
    use_prior_omissions=False,area=['VISp','VISl'], depth=[0,1000],
    use_prior_image_set=False):

    if ax is None:
        fig, ax = plt.subplots(1,len(sessions)+1, figsize=(16,4))

    fits = {}
    for dex,s in enumerate(sessions):
        row_start = dex == 0
        fits[str(s)] = scatter_by_cell(results_beh,run_params, cre_line=cre_line,
            threshold=threshold,ymetric_threshold=ymetric_threshold, 
            xmetric=xmetric, ymetric=ymetric,title=str(s),
            experience_level=[s],
            ax=ax[dex],row_start=row_start,col_start=col_start,
            use_prior_omissions=use_prior_omissions,depth=depth,
            area=area,use_prior_image_set=use_prior_image_set)

    ax[-1].axhline(0, linestyle='--',color='k',alpha=.25)   
    for s in sessions:
        if fits[str(s)] is not None:
            ax[-1].plot(s,fits[str(s)][0],'ko')
            ax[-1].plot([s,s], [fits[str(s)][0]-fits[str(s)][4],
                fits[str(s)][0]+fits[str(s)][4]], 'k--')
    ax[-1].set_ylabel('Regression slope',fontsize=16)
    ax[-1].set_title(cre_line,fontsize=16)
    ax[-1].spines['top'].set_visible(False)
    ax[-1].spines['right'].set_visible(False)
    ax[-1].tick_params(axis='y',labelsize=12)
    ax[-1].tick_params(axis='x',labelsize=16)
    plt.tight_layout()

    fits['yrange'] = ax[-1].get_ylim()
    fits['xmetric'] = xmetric
    fits['ymetric'] = ymetric
    fits['cre_line'] = cre_line
    fits['threshold'] = threshold
    fits['glm_version'] = run_params['version'] 
    return fits 

def scatter_by_cell(results_beh, run_params, cre_line='Vip-IRES-Cre', threshold=0, 
    ymetric_threshold=0, sessions=[],xmetric='strategy_dropout_index',
    ymetric='omissions',title='',nbins=10,ax=None,row_start=False,
    col_start=False,use_prior_omissions = False,plot_single=False,
    experience_level=['Familiar'],area=['VISp','VISl'],use_prior_image_set=False,
    equipment=["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"],savefig=False,
    depth=[0,1000]):

    if plot_single:
        row_start=True
        col_start=True

    if use_prior_omissions: 
        g = results_beh.query('(cre_line == @cre_line)&\
            (variance_explained_full >= @threshold)&\
            (prior_exposures_to_omissions in @sessions)&\
            (experience_level in @experience_level)&\
            (targeted_structure in @area)&\
            (imaging_depth >= @depth[0])&\
            (imaging_depth <= @depth[1])&\
            (equipment_name in @equipment)').\
            dropna(axis=0, subset=[ymetric,xmetric]).copy()
    elif use_prior_image_set:
        g = results_beh.query('(cre_line == @cre_line)&\
            (variance_explained_full > @threshold)&\
            (prior_exposures_to_image_set in @sessions)&\
            (experience_level in @experience_level)&\
            (targeted_structure in @area)&\
            (imaging_depth >= @depth[0])&\
            (imaging_depth <= @depth[1])&\
            (equipment_name in @equipment)').\
            dropna(axis=0, subset=[ymetric,xmetric]).copy()
    else:
        g = results_beh.query('(cre_line == @cre_line)&\
            (variance_explained_full > @threshold)&\
            (experience_level in @experience_level)&\
            (targeted_structure in @area)&\
            (imaging_depth >= @depth[0])&\
            (imaging_depth <= @depth[1])&\
            (equipment_name in @equipment)').\
            dropna(axis=0, subset=[ymetric,xmetric]).copy()

    if ymetric_threshold != 0:
        print('filtering based on {}'.format(ymetric))
        g = g[g[ymetric] < ymetric_threshold]

    if len(g) == 0:
        print('No cells match criteria')
        return

    # Figure axis
    if ax is None:
        fig, ax =plt.subplots()
    
    # Plot Raw data
    ax.plot(g[xmetric], g[ymetric],'ko',alpha=.1,label='raw data')
    ax.set_xlim(results_beh[xmetric].min(), results_beh[xmetric].max())
    ax.set_ylim(-1,0)

    # Plot binned data
    g['binned_xmetric'] = pd.cut(g[xmetric],nbins,labels=False) 
    xpoints = g.groupby('binned_xmetric')[xmetric].mean()
    ypoints = g.groupby('binned_xmetric')[ymetric].mean()
    y_sem = g.groupby('binned_xmetric')[ymetric].sem()
    ax.plot(xpoints, ypoints, 'ro',label='binned data')
    ax.plot(np.array([xpoints,xpoints]), np.array([ypoints-y_sem,ypoints+y_sem]), 'r-')

    # Plot Linear regression
    x = linregress(g[xmetric], g[ymetric])
    label = 'r = '+str(np.round(x.rvalue,4))+'\np = '+str(np.round(x.pvalue,decimals=4))
    ax.plot(g[xmetric], x[1]+x[0]*g[xmetric],'r-',label=label)

    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both',labelsize=12)
    ax.set_xlabel(xmetric,fontsize=16)   
    if col_start:
        ax.set_title(title,fontsize=16)
    if row_start:
        ax.set_ylabel(cre_line +'\n'+ymetric,fontsize=16)

    return x    



def plot_dropout_summary_population(results, run_params,
    dropouts_to_show =  ['all-images','omissions','behavioral','task'],
    palette=None,use_violin=True,add_median=True,
    include_zero_cells=True,add_title=False,dropout_cleaning_threshold=None,
    exclusion_threshold=None,savefig=False): 
    '''
        Makes a bar plot that shows the population dropout summary by cre line 
            for different regressors 
        palette , color palette to use. If None, uses gvt.project_colors()
        use_violion (bool) if true, uses violin, otherwise uses boxplots
        add_median (bool) if true, adds a line at the median of each population
        include_zero_cells (bool) if true, uses all cells, otherwise uses a 
            threshold for minimum variance explained
    '''

 
    if palette is None:
        palette = gvt.project_colors()

    if include_zero_cells:
        threshold = 0
    else:
        threshold=exclusion_threshold


    cre_lines = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']

    if ('post-omissions' in results.dropout.unique())&('omissions' in dropouts_to_show):
        dropouts_to_show = ['all-omissions' if x == 'omissions' else x \
            for x in dropouts_to_show]
    if ('post-hits' in results.dropout.unique())&('hits' in dropouts_to_show):
        dropouts_to_show = ['all-hits' if x == 'hits' else x for x in dropouts_to_show]
    if ('post-misses' in results.dropout.unique())&('misses' in dropouts_to_show):
        dropouts_to_show = ['all-misses' if x == 'misses' else x \
            for x in dropouts_to_show]
    if ('post-passive_change' in results.dropout.unique())&\
        ('passive_change' in dropouts_to_show):
        dropouts_to_show = ['all-passive_change' if x == 'passive_change' \
            else x for x in dropouts_to_show]
 
    data_to_plot = results.query('not passive')\
        .query('dropout in @dropouts_to_show and variance_explained_full > {}'\
        .format(threshold)).copy()
    data_to_plot['explained_variance'] = -1*data_to_plot['adj_fraction_change_from_full']
    if dropout_cleaning_threshold is not None:
        print('Clipping dropout scores for cells with full model VE < '\
            +str(dropout_cleaning_threshold))
        data_to_plot.loc[data_to_plot['adj_variance_explained_full']\
            <dropout_cleaning_threshold,'explained_variance'] = 0 
   

    for index, cre in enumerate(cre_lines):
        height = 4
        width=12
        horz_offset = 2
        vertical_offset = .75
        fig = plt.figure(figsize=(width,height))
        h = [Size.Fixed(horz_offset),Size.Fixed(width-horz_offset-.5)]
        v = [Size.Fixed(vertical_offset),Size.Fixed(height-vertical_offset-.5)]
        divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
        ax = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1))  
        cre_data = data_to_plot.query('cre_line == @cre')

        if use_violin:
            plot1= sns.violinplot(
                data = cre_data,
                x='dropout',
                y='explained_variance',
                hue='strategy_labels',
                hue_order = ['visual','timing'],
                order=dropouts_to_show,
                fliersize=0,
                ax=ax,
                inner='quartile',
                linewidth=0,
                palette=palette,
                cut = 0,
                split=True,
                alpha=.7
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
                data = cre_data,
                x='dropout',
                y='explained_variance',
                hue='strategy_labels',
                hue_order = ['visual','timing'],
                order=dropouts_to_show,
                fliersize=0,
                ax=ax,
                palette=palette,
                width=.7,
            )
        plt.setp(ax.collections,alpha=.5)
        ax.set_ylim(0,1)
        h,labels =ax.get_legend_handles_labels()
        clean_labels={
            'Slc17a7-IRES2-Cre visual':'Exc visual',
            'Sst-IRES-Cre visual':'Sst visual',
            'Vip-IRES-Cre visual':'Vip visual',
            'Slc17a7-IRES2-Cre timing':'Exc timing',
            'Sst-IRES-Cre timing':'Sst timing',
            'Vip-IRES-Cre timing':'Vip timing'
            }
        #mylabels = [clean_labels[x] for x in labels]
        #ax.legend(h,mylabels,loc='upper right',fontsize=16)
        #ax.set_ylabel('Fraction reduction \nin explained variance',fontsize=20)
        ax.set_ylabel(cre+'\nCoding Score',fontsize=20)
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
            filename = run_params['figure_dir']+\
                '/strategy/'+cre+' dropout_summary_boxplot'+extra+'.svg'
            print('Figure saved to: '+filename)
            plt.savefig(filename)


def plot_fraction_summary_population(results_pivoted, run_params,sharey=True,
    kernel_excitation=False,kernel=None,savefig=False):
    if kernel_excitation:
        assert kernel is not None, "Need to name the excited kernel"
    else:
        assert kernel is None, "Kernel Excitation is False, you should \
            not provide a named kernel"

    # compute coding fractions
    results_pivoted = results_pivoted.query('not passive').copy()
    results_pivoted['code_anything'] = results_pivoted['variance_explained_full'] \
        > run_params['dropout_threshold'] 
    results_pivoted['code_images'] = results_pivoted['code_anything'] \
        & (results_pivoted['all-images'] < 0)
    results_pivoted['code_omissions'] = results_pivoted['code_anything'] \
        & (results_pivoted['omissions'] < 0)
    results_pivoted['code_behavioral'] = results_pivoted['code_anything'] \
        & (results_pivoted['behavioral'] < 0)
    results_pivoted['code_task'] = results_pivoted['code_anything'] \
        & (results_pivoted['task'] < 0)

    cre_lines = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    results_pivoted['cre_line_strat'] = [x[0]+' '+x[1] for x in \
        zip(results_pivoted['cre_line'], results_pivoted['strategy_labels'])]

    summary_df = results_pivoted.groupby(['cre_line_strat','experience_level'])\
        [['code_anything','code_images','code_omissions','code_behavioral',\
        'code_task']].mean()

    # Compute Confidence intervals
    summary_df['n'] = results_pivoted.groupby(['cre_line_strat','experience_level'])\
        [['code_anything','code_images','code_omissions','code_behavioral',\
        'code_task']].count()['code_anything']
    summary_df['code_images_ci'] = 1.96*np.sqrt((summary_df['code_images']\
        *(1-summary_df['code_images']))/summary_df['n'])
    summary_df['code_omissions_ci'] = 1.96*np.sqrt((summary_df['code_omissions']\
        *(1-summary_df['code_omissions']))/summary_df['n'])
    summary_df['code_behavioral_ci'] = 1.96*np.sqrt((summary_df['code_behavioral']\
        *(1-summary_df['code_behavioral']))/summary_df['n'])
    summary_df['code_task_ci'] = 1.96*np.sqrt((summary_df['code_task']\
        *(1-summary_df['code_task']))/summary_df['n'])

    if kernel_excitation:
        results_pivoted['code_'+kernel] = results_pivoted['code_anything'] \
            & (results_pivoted[kernel] < 0)
        results_pivoted['code_'+kernel+'_excited'] = results_pivoted['code_anything'] \
            & (results_pivoted[kernel] < 0) & (results_pivoted[kernel+'_excited'])    
        results_pivoted['code_'+kernel+'_inhibited'] = results_pivoted['code_anything'] \
            & (results_pivoted[kernel] < 0) & (results_pivoted[kernel+'_excited']==False)
        summary_df = results_pivoted.groupby(['cre_line_strat','experience_level'])\
            [['code_anything','code_'+kernel,'code_'+kernel+'_excited',\
            'code_'+kernel+'_inhibited']].mean()
        summary_df['n'] = results_pivoted.groupby(['cre_line_strat','experience_level'])\
            [['code_anything','code_'+kernel,'code_'+kernel+'_excited',\
            'code_'+kernel+'_inhibited']].count()['code_anything']
        summary_df['code_'+kernel+'_ci'] = 1.96*np.sqrt((summary_df['code_'+kernel]*\
            (1-summary_df['code_'+kernel]))/summary_df['n'])
        summary_df['code_'+kernel+'_excited_ci'] = 1.96*np.sqrt((summary_df['code_'+\
            kernel+'_excited']*(1-summary_df['code_'+kernel+'_excited']))\
            /summary_df['n'])
        summary_df['code_'+kernel+'_inhibited_ci'] = 1.96*np.sqrt((\
            summary_df['code_'+kernel+'_inhibited']*(1-summary_df['code_'+kernel\
            +'_inhibited']))/summary_df['n'])

    # plotting variables
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    colors = gvt.project_colors()

    if kernel_excitation:
        coding_groups = ['code_'+kernel,'code_'+kernel+'_excited','code_'\
            +kernel+'_inhibited']   
        titles = [kernel.replace('all-images','images'), 'excited','inhibited']
    else:
        coding_groups = ['code_images','code_omissions','code_behavioral','code_task']
        titles = ['images','omissions','behavioral','task']

    # make combined across cre line plot
    if kernel_excitation:
        fig, ax = plt.subplots(1,len(coding_groups),figsize=(8.1,4), sharey=sharey)
    else:
        fig, ax = plt.subplots(3,len(coding_groups),figsize=(10.8,8), sharey=sharey)
    for index, feature in enumerate(coding_groups):
        # plots three cre-lines in standard colors
        ax[0,index].plot([0,1,2], summary_df.loc['Vip-IRES-Cre visual'][feature],'-',\
            color=colors['visual'],label='Vip visual',linewidth=3)
        ax[0,index].plot([0,1,2], summary_df.loc['Vip-IRES-Cre timing'][feature],'-',\
            color=colors['timing'],label='Vip timing',linewidth=3)
        ax[1,index].plot([0,1,2], summary_df.loc['Sst-IRES-Cre visual'][feature],'-',\
            color=colors['visual'],label='Sst visual',linewidth=3)
        ax[1,index].plot([0,1,2], summary_df.loc['Sst-IRES-Cre timing'][feature],'-',\
            color=colors['timing'],label='Sst timing',linewidth=3)
        ax[2,index].plot([0,1,2], summary_df.loc['Slc17a7-IRES2-Cre visual'][feature],\
            '-',color=colors['visual'],label='Exc visual',linewidth=3)
        ax[2,index].plot([0,1,2], summary_df.loc['Slc17a7-IRES2-Cre timing'][feature],\
            '-',color=colors['timing'],label='Exc timing',linewidth=3)
        
        ax[0,index].errorbar([0,1,2], summary_df.loc['Vip-IRES-Cre visual'][feature],\
            yerr=summary_df.loc['Vip-IRES-Cre visual'][feature+'_ci'],\
            color=colors['visual'],linewidth=3)
        ax[0,index].errorbar([0,1,2], summary_df.loc['Vip-IRES-Cre timing'][feature],\
            yerr=summary_df.loc['Vip-IRES-Cre timing'][feature+'_ci'],\
            color=colors['timing'],linewidth=3)

        ax[1,index].errorbar([0,1,2], summary_df.loc['Sst-IRES-Cre visual'][feature],\
            yerr=summary_df.loc['Sst-IRES-Cre visual'][feature+'_ci'],\
            color=colors['visual'],linewidth=3)
        ax[1,index].errorbar([0,1,2], summary_df.loc['Sst-IRES-Cre timing'][feature],\
            yerr=summary_df.loc['Sst-IRES-Cre timing'][feature+'_ci'],\
            color=colors['timing'],linewidth=3)

        ax[2,index].errorbar([0,1,2], summary_df.loc['Slc17a7-IRES2-Cre visual'][feature],\
            yerr=summary_df.loc['Slc17a7-IRES2-Cre visual'][feature+'_ci'],\
            color=colors['visual'],linewidth=3)
        ax[2,index].errorbar([0,1,2], summary_df.loc['Slc17a7-IRES2-Cre timing'][feature],\
            yerr=summary_df.loc['Slc17a7-IRES2-Cre timing'][feature+'_ci'],\
            color=colors['timing'],linewidth=3)


        for f in [0,1,2]:
            ax[f,index].set_title(titles[index],fontsize=20)
            ax[f,index].set_ylabel('')
            ax[f,index].set_xlabel('')
            ax[f,index].set_xticks([0,1,2])
            if f == 2:
                ax[f,index].set_xticklabels(experience_levels, rotation=90)
            else:
                ax[f,index].set_xticklabels(['','',''])
            ax[f,index].tick_params(axis='x',labelsize=16)
            ax[f,index].tick_params(axis='y',labelsize=16)
            ax[f,index].spines['top'].set_visible(False)
            ax[f,index].spines['right'].set_visible(False)
            ax[f,index].set_xlim(-.5,2.5)
            ax[f,index].set_ylim(bottom=0)
            if index ==3:
                ax[f,index].legend()

    ax[1,0].set_ylabel('Fraction of cells coding for ',fontsize=20)
    plt.tight_layout()
    if savefig:
        if kernel_excitation:
            filename = run_params['figure_dir']+'/strategy/coding_fraction_'+kernel\
                +'_summary.svg'  
            plt.savefig(filename)  
            print('Figure saved to: '+filename) 
        else:
            filename = run_params['figure_dir']+'/strategy/coding_fraction_summary.svg'
            plt.savefig(filename)  
            print('Figure saved to: '+filename) 
    return summary_df 


def plot_population_averages(results_pivoted, run_params, 
    dropouts_to_show = ['all-images','omissions','behavioral','task'],
    sharey=True,include_zero_cells=True,boxplot=False,add_stats=True,
    extra='',strict_experience_matching=False,plot_by_cell_type=False,
    across_session=False,stats_on_across=True, matched_with_variance_explained=False,
    matched_ve_threshold=0,savefig=False):
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
        matched_ve_threshold (float) threshold used by matched_with_variance_explained
        savefig (bool) whether to save the figure or not
        
    ''' 
    
    if not sharey:
        extra = extra+'_untied'
    if include_zero_cells:
        extra = extra+'_with_zero_cells'
 
    # Filter for cells with low variance explained
    if include_zero_cells:
        results_pivoted = results_pivoted.query('not passive').copy()       
    else:
        extra = extra + '_no_zero_cells'
        results_pivoted = results_pivoted.query('(variance_explained_full > 0.005)\
            &(not passive)').copy()    

    # Convert dropouts to positive values
    for dropout in dropouts_to_show:
        if '_signed' in dropout:
            results_pivoted[dropout] = -results_pivoted[dropout]
        else:
            results_pivoted[dropout] = results_pivoted[dropout].abs()
    
    # Add additional columns about experience levels
    experiments_table = loading.get_platform_paper_experiment_table(\
        include_4x2_data=run_params['include_4x2_data'])
    experiment_table_columns = experiments_table.reset_index()\
        [['ophys_experiment_id','last_familiar_active','second_novel_active',\
        'cell_type','binned_depth']]
    results_pivoted = results_pivoted.merge(experiment_table_columns, \
            on='ophys_experiment_id')

    # plotting variables
    cell_types = ['Vip Inhibitory','Sst Inhibitory','Excitatory']
    experience_levels = np.sort(results_pivoted.experience_level.unique())
    colors = gvt.project_colors()

    # Repeat the plots but transposed
    # Iterate cell types and make a plot for each
    summary_data = {}
    for index, feature in enumerate(dropouts_to_show):   
        fig, ax = plt.subplots(1,3,figsize=(8.4,4), sharey=sharey) 

        summary_data[feature + ' data'] = {}
        stats = {}
        # Iterate dropouts and plot each by experience
        for cindex, cell_type in enumerate(cell_types):
            all_data = results_pivoted.query('cell_type ==@cell_type')

            visual_data = all_data.query('strategy_labels == "visual"')
            timing_data = all_data.query('strategy_labels == "timing"')

            stats_feature = feature
            stats[cell_type]= test_significant_dropout_averages_by_strategy(\
                    all_data,stats_feature)
            summary_data[feature+' data'][cell_type+' all data'] = \
                all_data.groupby(['experience_level'])[feature].describe()

            ax[cindex] = sns.pointplot(
                data = timing_data,
                x = 'experience_level',
                y= feature,
                order=experience_levels,
                color = colors['timing'],
                join=True,
                ax=ax[cindex]
                )

            ax[cindex] = sns.pointplot(
                data = visual_data,
                x = 'experience_level',
                y=feature,
                order=experience_levels,
                color=colors['visual'],
                join=True,
                ax=ax[cindex],
            )

            ax[cindex].set_title(cell_type,fontsize=20)
            ax[cindex].set_ylabel('')
            ax[cindex].set_xlabel('')
            ax[cindex].set_xticks([0,1,2])
            ax[cindex].set_xticklabels(experience_levels, rotation=90)
            ax[cindex].set_xlim(-.5,2.5)
            ax[cindex].tick_params(axis='x',labelsize=16)
            ax[cindex].tick_params(axis='y',labelsize=16)
            ax[cindex].spines['top'].set_visible(False)
            ax[cindex].spines['right'].set_visible(False)
            ax[cindex].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        if add_stats:
            ytop = ax[0].get_ylim()[1]
            y1h = ytop*1.05
            stats_color='k'
            for cindex, cell_type in enumerate(cell_types):
                for eindex, exp in enumerate(experience_levels):
                    ttest = stats[cell_type][exp]
                    if ttest.pvalue<0.001:
                            ax[cindex].plot(eindex,y1h, '*',color=stats_color)
                            ax[cindex].plot(eindex+.1,y1h, '*',color=stats_color)
                            ax[cindex].plot(eindex-.1,y1h, '*',color=stats_color)
                    elif ttest.pvalue<0.01:
                            ax[cindex].plot(eindex,y1h, '*',color=stats_color)
                            ax[cindex].plot(eindex+.1,y1h, '*',color=stats_color)
                    elif ttest.pvalue<0.05:
                            ax[cindex].plot(eindex,y1h, '*',color=stats_color)
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
        fig.tight_layout() 

        if savefig:
            filename = run_params['figure_dir']+'/strategy/dropout_average_'\
                +clean_feature.replace(' ','_')+extra+'.svg'
            plt.savefig(filename)
            print('Figure saved to: '+filename)
        summary_data[feature+' stats'] = stats

    return summary_data



def test_significant_dropout_averages_by_strategy(data,feature):
    data = data[~data[feature].isnull()].copy()
    ttests = {}
    for experience in data['experience_level'].unique():
       ttests[experience] = stats.ttest_ind(
            data.query('experience_level == @experience & strategy_labels == "visual"')[feature],  
            data.query('experience_level == @experience & strategy_labels == "timing"')[feature],
            )
    return ttests


