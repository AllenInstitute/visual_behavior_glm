import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import Divider, Size

## General Utilities
################################################################################

def add_behavior_metrics(df,summary_df):
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


def compare_cre_kernels(weights_beh, run_params, ym='omissions',
    compare=['strategy_labels'],equipment_filter='all',area_filter=['VISp','VISl'],
    sessions=['Familiar','Novel 1','Novel >1'],image_set='familiar',
    filter_sessions_on='experience_level',savefig=False,sharey=False,
    depth_filter=[0,1000]):

    cres = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    fig, ax = plt.subplots(2,len(cres),figsize=(len(sessions)*4,6),sharey=sharey)
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
    ax[1,0].set_ylabel('V1 Kernel Weights',fontsize=16)
    plt.tight_layout()

    if savefig:
        filename = ym+'_by_cre_line_'+'_'.join(compare)+'_'+equipment_filter
        save_figure(fig,run_params['version'], ym, filename)


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





