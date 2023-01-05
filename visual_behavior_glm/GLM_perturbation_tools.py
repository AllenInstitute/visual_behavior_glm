import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_strategy_tools as gst
import visual_behavior_glm.PSTH as psth
import os
from scipy.spatial import ConvexHull as ch
from mpl_toolkits.axes_grid1 import Divider, Size

def analysis(weights_beh, run_params, kernel,experience_level='Familiar',savefig=False):
    
    if kernel in ['hits','misses']:
        lims=[[-.0005,.002],[-.01,.005]]
    if kernel in ['omissions','all-images']:
        lims=[[-.0005,.001],[-.015,.03]]

    # Plot 1D summaries
    out1 = gst.strategy_kernel_comparison(weights_beh.query('visual_strategy_session'),
        run_params, kernel,session_filter=[experience_level])
    out2 =gst.strategy_kernel_comparison(weights_beh.query('not visual_strategy_session'),
        run_params, kernel,session_filter=[experience_level])

    out3 = gst.strategy_kernel_comparison(weights_beh.query('visual_strategy_session'),
        run_params, 'all-images',session_filter=[experience_level])
    out4 =gst.strategy_kernel_comparison(weights_beh.query('not visual_strategy_session'),
        run_params, 'all-images',session_filter=[experience_level])

    # Unify ylimits and add time markers
    ylim1 = out1[2].get_ylim()
    ylim2 = out2[2].get_ylim()
    ylim3 = out3[2].get_ylim()
    ylim4 = out4[2].get_ylim()
    ylim = [np.min([ylim1[0],ylim2[0],ylim3[0],ylim4[0]]), \
        np.max([ylim1[1],ylim2[1],ylim3[1],ylim4[1]])]

    out1[2].set_ylim(ylim)
    out2[2].set_ylim(ylim)
    out3[2].set_ylim(ylim)
    out4[2].set_ylim(ylim)
    out1[2].set_ylabel(kernel+' kernel\n(Ca$^{2+}$ events)',fontsize=16)
    out2[2].set_ylabel(kernel+' kernel\n(Ca$^{2+}$ events)',fontsize=16)
    out3[2].set_ylabel('image kernel\n(Ca$^{2+}$ events)',fontsize=16)
    out4[2].set_ylabel('image kernel\n(Ca$^{2+}$ events)',fontsize=16)

    out1[2].set_title('Visual strategy',fontsize=16)
    out2[2].set_title('Timing strategy',fontsize=16)
    out3[2].set_title('Visual strategy',fontsize=16)
    out4[2].set_title('Timing strategy',fontsize=16)
    if kernel =='omissions':
        out1[2].plot(0,ylim[0],'co',zorder=10,clip_on=False)
        out2[2].plot(0,ylim[0],'co',zorder=10,clip_on=False)
    elif kernel =='hits':
        out1[2].plot(0,ylim[0],'ro',zorder=10,clip_on=False)
        out2[2].plot(0,ylim[0],'ro',zorder=10,clip_on=False)
    elif kernel == 'misses':
        out1[2].plot(0,ylim[0],'rx',zorder=10,clip_on=False)
        out2[2].plot(0,ylim[0],'rx',zorder=10,clip_on=False)
    else:
        out1[2].plot(0,ylim[0],'ko',zorder=10,clip_on=False)
        out2[2].plot(0,ylim[0],'ko',zorder=10,clip_on=False)
    if kernel in ['omissions','hits','misses']:
        out1[2].plot(.75,ylim[0],'ko',zorder=10,clip_on=False)
        out1[2].plot(1.5,ylim[0],'o',color='gray',zorder=10,clip_on=False)
        out2[2].plot(.75,ylim[0],'ko',zorder=10,clip_on=False)
        out2[2].plot(1.5,ylim[0],'o',color='gray',zorder=10,clip_on=False)

    # Make 2D plot
    ax = plot_perturbation(weights_beh, run_params, kernel,experience_level=experience_level,
        savefig=savefig,lims=lims)

    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+kernel+'_visual_comparison_{}.svg'.format(experience_level)
        print('Figure saved to: '+filepath)
        out1[1].savefig(filepath) 
        filepath = run_params['figure_dir']+\
            '/strategy/'+kernel+'_timing_comparison_{}.svg'.format(experience_level)
        print('Figure saved to: '+filepath)
        out2[1].savefig(filepath) 
        filepath = run_params['figure_dir']+\
            '/strategy/images_visual_comparison_{}.svg'.format(experience_level)
        print('Figure saved to: '+filepath)
        out3[1].savefig(filepath) 
        filepath = run_params['figure_dir']+\
            '/strategy/images_timing_comparison_{}.svg'.format(experience_level)
        print('Figure saved to: '+filepath)
        out4[1].savefig(filepath) 

    return ax 

def get_kernel_averages(weights_df, run_params, kernel, drop_threshold=0,
    session_filter=['Familiar','Novel 1','Novel >1'],equipment_filter="all",
    depth_filter=[0,1000],cell_filter="all",area_filter=['VISp','VISl'],
    compare=['cre_line'],plot_errors=False,save_kernels=False,fig=None, 
    ax=None,fs1=16,fs2=12,show_legend=True,filter_sessions_on='experience_level',
    image_set=['familiar','novel'],threshold=0,set_title=None,active=True): 
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
       
    if active:
        active_str = 'not passive'
    else:
        active_str = 'passive' 

    # Applying hard thresholds to dataset
    if kernel in weights_df:
        weights = weights_df.query('({0})&\
            (targeted_structure in @area_filter)&\
            (cre_line in @cell_list)&\
            (equipment_name in @equipment_list)&\
            ({1} in @session_filter)&\
            (ophys_session_id not in @problem_sessions)&\
            (imaging_depth < @depth_filter[1])&\
            (imaging_depth > @depth_filter[0])&\
            (variance_explained_full > @threshold)&\
            ({2} <= @drop_threshold)'.format(active_str, filter_sessions_on, kernel))
    else:
        weights = weights_df.query('({0})&\
            (targeted_structure in @area_filter)&\
            (cre_line in @cell_list)&\
            (equipment_name in @equipment_list)&\
            ({1} in @session_filter) &\
            (ophys_session_id not in @problem_sessions) &\
            (imaging_depth < @depth_filter[1]) &\
            (imaging_depth > @depth_filter[0])&\
            (variance_explained_full > @threshold)'.format(active_str, filter_sessions_on))
    
    # Determine unique groups of cells by the categorical attributes in compare
    groups = list(weights.groupby(compare).groups.keys())
    if len(compare) >1:
        # Determine number of 2nd level attributes for linestyle definitions 
        num_2nd = len(list(weights[compare[1]].unique()))
   
    outputs={}
    outputs['time'] = time_vec
    # Iterate over groups of cells
    for dex,group in enumerate(groups):
        if len(compare) ==1:
            query_str = '({0} == @group)'.format(compare[0])
        else:
            query_str = '&'.join(['('+x[0]+'==\"'+x[1]+'\")' for x in zip(compare,group)])
        # Filter for this group, and plot
        weights_dfiltered = weights.query(query_str)[kernel+'_weights']
        k,sem=get_average_kernels_inner(weights_dfiltered) 
        outputs[group]=k
        outputs[group+'_sem']=sem
    return outputs

def get_error(T,x='Slc17a7-IRES2-Cre',y='y'):
    '''
        Generates a dataframe where each row is a point along the trajectory
        defined by T[x], and T[y]. Adds columns for the 4 error points
        defined by T[x_sem] and T[y_sem]
    '''
    T[x]    
    df = pd.DataFrame()
    df['time']=T['time']
    df['x']=T[x]
    df['y']=T[y]
    df['x1']=T[x]-T[x+'_sem']
    df['x2']=T[x]+T[x+'_sem']
    df['y1']=T[y]-T[y+'_sem']
    df['y2']=T[y]+T[y+'_sem']
    return df  

def plot_iterative_ch(ax,df,color,show_steps=False):
    '''
        For every pair of points t_i, t_i+1, finds the convex hull
        around the 8 error points (4 from t_i, 4 from t_i+1) and
        fills the polygon defined by that convex hull
        show_steps (bool) if True, plot the convex hull rather than fill it. 
    '''

    if color == 'lightgray':
        color=(.875,.875,.875)

    for index, row in df.iterrows():
        if index ==0:
            pass
        points = get_points(df.loc[index-1:index])
        hull = ch(points)
        if show_steps:
            for simplex in hull.simplices:
                ax.plot(points[simplex,0],points[simplex,1],'k-')
        else:
            ax.fill(points[hull.vertices,0],points[hull.vertices,1],color=color)
    
def get_points(df):
    '''
        Return a 2D array of error points for two subsequent points in df
    '''
    x=[]
    y=[]
    for index, row in df.iterrows():
        this_x= [row.x1,row.x,row.x2,row.x]
        this_y= [row.y,row.y2,row.y,row.y1]
        x=x+this_x
        y=y+this_y
    return np.array([x,y]).T

def demonstrate_iterative_ch(Fvisual,kernel='omissions',show_steps=True):
    '''
        A demonstration that shows the iterative convex hull solution
    '''
    time = Fvisual['time']
    offset = 0
    multiimage = kernel in ['hits','misses','omissions']
    if multiimage:
        pi= np.where(time > (.75+offset))[0][0]
        pi2 = np.where(time > (1.5+offset))[0][0]
    if time[-1] > 2.25:
        pi3 = np.where(time > 2.25)[0][0]
    else:
        pi3 = len(time)

    colors = gvt.project_colors()
    fig, ax = plt.subplots(figsize=(4,3.5))
    df = get_error(Fvisual)
    df = df.loc[0:pi3]
    plot_iterative_ch(ax,df,'lightgray',show_steps=show_steps)
    if show_steps:
        ax.errorbar(Fvisual['Slc17a7-IRES2-Cre'][0:pi3],Fvisual['y'][0:pi3],
            xerr=Fvisual['Slc17a7-IRES2-Cre_sem'][0:pi3],
            yerr=Fvisual['y_sem'][0:pi3],color='gray',alpha=.5)
    ax.plot(Fvisual['Slc17a7-IRES2-Cre'][0:pi3], Fvisual['y'][0:pi3],
        color=colors['visual'],label='Visual',linewidth=3)
    
    ax.set_ylabel('Vip - Sst',fontsize=16)
    ax.set_xlabel('Exc',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)  
    ax.set_title('Familiar, {}'.format(kernel),fontsize=16)
    ax.legend() 
    ax.axhline(0,color='k',linestyle='--',alpha=.25)
    ax.axvline(0,color='k',linestyle='--',alpha=.25)

def get_perturbation(weights_df, run_params, kernel,experience_level="Familiar"):
    visual = get_kernel_averages(weights_df.query('visual_strategy_session'), 
        run_params, kernel, session_filter=[experience_level])
    timing = get_kernel_averages(weights_df.query('not visual_strategy_session'), 
        run_params, kernel, session_filter=[experience_level])
    visual['y'] = -visual['Sst-IRES-Cre'] + visual['Vip-IRES-Cre']
    timing['y'] = -timing['Sst-IRES-Cre'] + timing['Vip-IRES-Cre']
    visual['y_sem'] = np.sqrt(np.sum([visual['Sst-IRES-Cre_sem']**2,
        visual['Vip-IRES-Cre_sem']**2],axis=0))
    timing['y_sem'] = np.sqrt(np.sum([timing['Sst-IRES-Cre_sem']**2,
        timing['Vip-IRES-Cre_sem']**2],axis=0))
    return visual, timing

def plot_multiple(weights_df, run_params,savefig=False):
    fig, ax = plt.subplots(3,4,sharey=True,sharex=True,figsize=(12,9))
    for edex, e in enumerate(['Familiar','Novel 1','Novel >1']):    
        for kdex, k in enumerate(['omissions','hits','misses','all-images']):
            plot_perturbation(weights_df, run_params,k,e,ax=ax[edex,kdex],
                col1=kdex==0,row1=edex==2,multi=True) 
    plt.tight_layout()
    return    

def plot_perturbation(weights_df, run_params, kernel,experience_level="Familiar",
    savefig=False,lims = None,show_steps=False,ax=None,col1=False,row1=False,multi=False):
    visual, timing = get_perturbation(weights_df, run_params,kernel,experience_level)
    time = visual['time']
    offset = 0
    multiimage = kernel in ['hits','misses','omissions']
    if multiimage:
        pi= np.where(time > (.75+offset))[0][0]
        pi2 = np.where(time > (1.5+offset))[0][0]
    if time[-1] > 2.25:
        pi3 = np.where(time > 2.25)[0][0]
    else:
        pi3 = len(time)

    if show_steps:
        demonstrate_iterative_ch(visual,kernel)

    colors = gvt.project_colors()
    if ax is None:
        fig, ax = plt.subplots(1,1,sharey=True,sharex=True,figsize=(4.5,4))
    
    df = get_error(visual).loc[0:pi3]
    plot_iterative_ch(ax,df,'lightgray')  

    df = get_error(timing).loc[0:pi3]
    plot_iterative_ch(ax,df,'lightgray')

    ax.plot(visual['Slc17a7-IRES2-Cre'][0:pi3], visual['y'][0:pi3],
        color=colors['visual'],label='Visual',linewidth=3)
    ax.plot(timing['Slc17a7-IRES2-Cre'][0:pi3], timing['y'][0:pi3],
        color=colors['timing'],label='Timing',linewidth=3)
    if kernel =='omissions':
        ax.plot(visual['Slc17a7-IRES2-Cre'][0],visual['y'][0],'co')
        ax.plot(timing['Slc17a7-IRES2-Cre'][0],timing['y'][0],'co')
    elif kernel == 'hits':
        ax.plot(visual['Slc17a7-IRES2-Cre'][0],visual['y'][0],'ro')
        ax.plot(timing['Slc17a7-IRES2-Cre'][0],timing['y'][0],'ro')
    elif kernel == 'misses':
        ax.plot(visual['Slc17a7-IRES2-Cre'][0],visual['y'][0],'rx')
        ax.plot(timing['Slc17a7-IRES2-Cre'][0],timing['y'][0],'rx')
    else:
        ax.plot(visual['Slc17a7-IRES2-Cre'][0],visual['y'][0],'ko')
        ax.plot(timing['Slc17a7-IRES2-Cre'][0],timing['y'][0],'ko')
    if multiimage:
        ax.plot(visual['Slc17a7-IRES2-Cre'][pi],visual['y'][pi],'ko')
        ax.plot(timing['Slc17a7-IRES2-Cre'][pi],timing['y'][pi],'ko')
        ax.plot(visual['Slc17a7-IRES2-Cre'][pi2],visual['y'][pi2],'o',color='gray')
        ax.plot(timing['Slc17a7-IRES2-Cre'][pi2],timing['y'][pi2],'o',color='gray')

    if col1:
        ax.set_ylabel(experience_level+'\nVip - Sst',fontsize=16)
    elif not multi:
        ax.set_ylabel('Vip - Sst',fontsize=16)
    if row1:
        ax.set_xlabel(kernel+'\nExc',fontsize=16)
    elif not multi:
        ax.set_xlabel('Exc',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0,.001, .002])
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)  
    if not multi:
        ax.set_title('{}'.format(kernel),fontsize=16)
    #ax.legend() 
    ax.axhline(0,color='k',linestyle='--',alpha=.25)
    ax.axvline(0,color='k',linestyle='--',alpha=.25)
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    plt.tight_layout()
    
    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+kernel+'_strategy_perturbation_{}.svg'.format(experience_level)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 

def plot_perturbation_3D(weights_df,run_params, kernel, session='Familiar',savefig=False):
    visual = get_kernel_averages(weights_df.query('visual_strategy_session'), 
        run_params, kernel, session_filter=[session])
    timing = get_kernel_averages(weights_df.query('not visual_strategy_session'), 
        run_params, kernel, session_filter=[session])
    time = visual['time']
    offset=0
    multiimage = kernel in ['hits','misses','omissions']
    if multiimage:
        pi= np.where(time > (.75+offset))[0][0]
        pi2 = np.where(time > (1.5+offset))[0][0]
    if time[-1] > 2.25:
        pi3 = np.where(time > 2.25)[0][0]
    else:
        pi3 = len(time)


    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(visual['Slc17a7-IRES2-Cre'][0:pi3],
        visual['Sst-IRES-Cre'][0:pi3], 
        visual['Vip-IRES-Cre'][0:pi3],
        color='darkorange',
        linewidth=3)
    ax.plot(timing['Slc17a7-IRES2-Cre'][0:pi3],
        timing['Sst-IRES-Cre'][0:pi3], 
        timing['Vip-IRES-Cre'][0:pi3],
        color='blue',
        linewidth=3)
    ax.set_xlabel('Exc',fontsize=16)
    ax.set_ylabel('Sst',fontsize=16)
    ax.set_zlabel('Vip',fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    ax.view_init(elev=15,azim=-115)
    ax.set_xlim(-.00045,.00095)
    ax.set_ylim(-.0055,.012)
    ax.set_zlim(-.0035,.022)
    plt.tight_layout()
 
    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+kernel+'_strategy_perturbation_3D_{}.svg'.format(session)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 

def get_average_kernels_inner(df):
    '''
        Plots the average kernel for the cells in df
        
        df, series of cells with column that is the kernel to plot
    '''

    # Normalize kernels, and interpolate to time_vec
    df_norm = [x for x in df[~df.isnull()].values]
    
    # Needed for stability
    if len(df_norm)>0:
        df_norm = np.vstack(df_norm)
    else:
        df_norm = np.empty((2,len(time_vec)))
        df_norm[:] = np.nan
    
    return df_norm.mean(axis=0),df_norm.std(axis=0)/np.sqrt(np.shape(df_norm)[0])


def plot_PSTH_perturbation_diff(dfs,labels,condition,min_time=-.75,
    run_params=None,savefig=False,experience_level="Familiar",
    x='Exc',y='y'):
    traces = get_PSTH_2D_traces(dfs,labels,condition)
    traces['diff_Exc'] = traces['visual_Exc'] - traces['timing_Exc']
    traces['diff_Sst'] = traces['visual_Sst'] - traces['timing_Sst']
    traces['diff_Vip'] = traces['visual_Vip'] - traces['timing_Vip']
    traces['diff_y'] = traces['visual_y'] - traces['timing_y']

    height = 4
    width=5.25
    pre_horz_offset = 2
    post_horz_offset = .5
    vertical_offset = .75
    fig = plt.figure(figsize=(width,height))
    h = [Size.Fixed(pre_horz_offset),\
        Size.Fixed((width-pre_horz_offset-post_horz_offset))]     
    v = [Size.Fixed(vertical_offset),
        Size.Fixed(height-vertical_offset-.5)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax = fig.add_axes(divider.get_position(),\
        axes_locator=divider.new_locator(nx=1,ny=1))  

    ts = np.where(traces['time'] >=min_time)[0]
    ax.plot(traces['diff_'+x][ts],traces['diff_'+y][ts],
        color='k',lw=3) 

    p1 = np.where(traces['time'] == 0)[0]
    p2 = np.argmin(np.abs(traces['time'] - .75)) 
    p3 = np.where(traces['time'] == 1.5)[0]
    if condition == 'omission':
        ax.plot(traces['diff_'+x][p1], traces['diff_'+y][p1],
        'co',zorder=10)
    elif condition == 'hit':
        ax.plot(traces['diff_'+x][p1], traces['diff_'+y][p1],
        'ro',zorder=10)
    elif condition == 'miss':
        ax.plot(traces['diff_'+x][p1], traces['diff_'+y][p1],
        'rx',zorder=10)
    if condition in ['omission','hit','miss']: 
        ax.plot(traces['diff_'+x][p2], traces['diff_'+y][p2],
        'ko',zorder=10)
        ax.plot(traces['diff_'+x][p3], traces['diff_'+y][p3],
        'o',color='gray',zorder=10)
    if condition =='image':
        ax.plot(traces['diff_'+x][p1], traces['diff_'+y][p1],
        'ko',zorder=10)
        ax.plot(traces['diff_'+x][p2], traces['diff_'+y][p2],
        'ko',zorder=10)
        ax.plot(traces['diff_'+x][p3], traces['diff_'+y][p3],
        'ko',zorder=10)


    if y == 'y':
        ax.set_ylabel('Vip - Sst',fontsize=16)
    else:
        ax.set_ylabel(y,fontsize=16)
    ax.set_xlabel(x,fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #if (x=='Exc'):
    #    if condition in ['hit','miss']:
    #        ax.set_xlim(0,.008)
    #    if condition in ['image','omission']:
    #        ax.set_xlim( 0,0.004)
    #if y=='y':
    #    ax.set_ylim(-0.03,0.045)
    ax.axhline(0,color='k',linestyle='--',alpha=.25)
    ax.axvline(0,color='k',linestyle='--',alpha=.25)
    ax.set_title(condition,fontsize=16)
    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+condition+'_strategy_perturbation_PSTH_2D_diff_{}_{}_{}.svg'.format(experience_level,x,y)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 
    return ax

def plot_PSTH_2D_diff(dfs,labels, condition, run_params, 
        experience_level="Familiar",savefig=True,times=[-.75,2]):

    traces = get_PSTH_2D_traces(dfs,labels,condition,
        experience_level=experience_level)
    traces['diff_Exc'] = traces['visual_Exc'] - traces['timing_Exc']
    traces['diff_Sst'] = traces['visual_Sst'] - traces['timing_Sst']
    traces['diff_Vip'] = traces['visual_Vip'] - traces['timing_Vip']
    traces['diff_y'] = traces['visual_y'] - traces['timing_y']
    x = np.vstack([traces['diff_Exc'],traces['diff_Sst'],traces['diff_Vip']])
    vec = [1,-1,1]
    vec = vec/np.linalg.norm(vec)
    #traces['diff_magnitude'] = np.linalg.norm(x,axis=0)
    traces['diff_magnitude'] = vec @ x 

    stamps = np.where((traces['time']>= times[0])&\
        (traces['time'] <=times[-1]))[0]
    time = traces['time'][stamps]

    height = 4
    width=8
    pre_horz_offset = 1.5
    post_horz_offset = 2.5
    vertical_offset = .75
    fig = plt.figure(figsize=(width,height))
    duration = time[-1] -time[0] 
    h = [Size.Fixed(pre_horz_offset),\
        Size.Fixed((width-pre_horz_offset-post_horz_offset)\
        /3*duration)]     
    v = [Size.Fixed(vertical_offset),
        Size.Fixed(height-vertical_offset-.5)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax = fig.add_axes(divider.get_position(),\
        axes_locator=divider.new_locator(nx=1,ny=1))  

    colors = gvt.project_colors()
    ax.plot(time,traces['diff_magnitude'][stamps],
        color='k',lw=4)

    omitted = 'omission' in condition
    change = (not omitted) and (('change' in condition) or \
        ('hit' in condition) or ('miss' in condition))
    psth.plot_flashes_on_trace(ax, time, 
        change=change, omitted=omitted)
    ax.set_xlabel('time from {} (s)'.format(condition),fontsize=16)
    ax.set_ylabel(condition+' response\n(Ca$^{2+}$ events)',
        fontsize=16)
    ax.set_ylim(-.004,.0225)
    ax.set_xlim(time[0],time[-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #ax.set_title(strategy+' strategy',fontsize=16)


    if condition =='omission':
        ax.plot(0,0,'co',zorder=10,clip_on=False)
    elif condition =='hit':
        ax.plot(0,0,'ro',zorder=10,clip_on=False)
    elif condition == 'miss':
        ax.plot(0,0,'rx',zorder=10,clip_on=False)
    else:
        ax.plot(0,0,'ko',zorder=10,clip_on=False)
    if condition in ['omission','hit','miss']:
        ax.plot(.75,0,'ko',zorder=10,clip_on=False)
        ax.plot(1.5,0,'o',color='gray',zorder=10,clip_on=False)

    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+condition+'_strategy_perturbation_PSTH_diff_{}_{}.svg'.format(strategy, experience_level)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 
    return ax





def plot_PSTH_perturbation(dfs,labels,condition,min_time=-.75,
    run_params=None,savefig=False,experience_level="Familiar",
    x='Exc',y='y'):
    traces = get_PSTH_2D_traces(dfs,labels,condition)

    height = 4
    width=5.25
    pre_horz_offset = 2
    post_horz_offset = .5
    vertical_offset = .75
    fig = plt.figure(figsize=(width,height))
    h = [Size.Fixed(pre_horz_offset),\
        Size.Fixed((width-pre_horz_offset-post_horz_offset))]     
    v = [Size.Fixed(vertical_offset),
        Size.Fixed(height-vertical_offset-.5)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax = fig.add_axes(divider.get_position(),\
        axes_locator=divider.new_locator(nx=1,ny=1))  

    ts = np.where(traces['time'] >=min_time)[0]
    ax.plot(traces['visual_'+x][ts],traces['visual_'+y][ts],
        color='darkorange',lw=3)
    ax.plot(traces['timing_'+x][ts],traces['timing_'+y][ts],
        color='blue',lw=3)

    p1 = np.where(traces['time'] == 0)[0]
    p2 = np.argmin(np.abs(traces['time'] - .75)) 
    p3 = np.where(traces['time'] == 1.5)[0]
    if condition == 'omission':
        ax.plot(traces['visual_'+x][p1], traces['visual_'+y][p1],
        'co',zorder=10)
        ax.plot(traces['timing_'+x][p1], traces['timing_'+y][p1],
        'co',zorder=10)
    elif condition == 'hit':
        ax.plot(traces['visual_'+x][p1], traces['visual_'+y][p1],
        'ro',zorder=10)
        ax.plot(traces['timing_'+x][p1], traces['timing_'+y][p1],
        'ro',zorder=10)
    elif condition == 'miss':
        ax.plot(traces['visual_'+x][p1], traces['visual_'+y][p1],
        'rx',zorder=10)
        ax.plot(traces['timing_'+x][p1], traces['timing_'+y][p1],
        'rx',zorder=10)
    if condition in ['omission','hit','miss']: 
        ax.plot(traces['visual_'+x][p2], traces['visual_'+y][p2],
        'ko',zorder=10)
        ax.plot(traces['timing_'+x][p2], traces['timing_'+y][p2],
        'ko',zorder=10)
        ax.plot(traces['visual_'+x][p3], traces['visual_'+y][p3],
        'o',color='gray',zorder=10)
        ax.plot(traces['timing_'+x][p3], traces['timing_'+y][p3],
        'o',color='gray',zorder=10)
    if condition =='image':
        ax.plot(traces['visual_'+x][p1], traces['visual_'+y][p1],
        'ko',zorder=10)
        ax.plot(traces['timing_'+x][p1], traces['timing_'+y][p1],
        'ko',zorder=10)   
        ax.plot(traces['visual_'+x][p2], traces['visual_'+y][p2],
        'ko',zorder=10)
        ax.plot(traces['timing_'+x][p2], traces['timing_'+y][p2],
        'ko',zorder=10)   
        ax.plot(traces['visual_'+x][p3], traces['visual_'+y][p3],
        'ko',zorder=10)
        ax.plot(traces['timing_'+x][p3], traces['timing_'+y][p3],
        'ko',zorder=10)   


    if y == 'y':
        ax.set_ylabel('Vip - Sst',fontsize=16)
    else:
        ax.set_ylabel(y,fontsize=16)
    ax.set_xlabel(x,fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    if (x=='Exc'):
        if condition in ['hit','miss']:
            ax.set_xlim(0,.008)
        if condition in ['image','omission']:
            ax.set_xlim( 0,0.004)
    if y=='y':
        ax.set_ylim(-0.03,0.045)
    ax.axhline(0,color='k',linestyle='--',alpha=.25)
    ax.axvline(0,color='k',linestyle='--',alpha=.25)
    ax.set_title(condition,fontsize=16)
    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+condition+'_strategy_perturbation_PSTH_2D_{}_{}_{}.svg'.format(experience_level,x,y)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 
    return ax

def plot_PSTH_2D(dfs,labels, condition, strategy, run_params, 
        experience_level="Familiar",savefig=True,times=[-.75,2]):
    traces = get_PSTH_2D_traces(dfs,labels,condition,
        experience_level=experience_level)

    stamps = np.where((traces['time']>= times[0])&\
        (traces['time'] <=times[-1]))[0]
    time = traces['time'][stamps]

    height = 4
    width=8
    pre_horz_offset = 1.5
    post_horz_offset = 2.5
    vertical_offset = .75
    fig = plt.figure(figsize=(width,height))
    duration = time[-1] -time[0] 
    h = [Size.Fixed(pre_horz_offset),\
        Size.Fixed((width-pre_horz_offset-post_horz_offset)\
        /3*duration)]     
    v = [Size.Fixed(vertical_offset),
        Size.Fixed(height-vertical_offset-.5)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax = fig.add_axes(divider.get_position(),\
        axes_locator=divider.new_locator(nx=1,ny=1))  

    colors = gvt.project_colors()
    ax.plot(time,traces[strategy+'_Exc'][stamps],
        color=colors['slc'],lw=4)
    ax.plot(time,traces[strategy+'_Vip'][stamps],
        color=colors['vip'],lw=4)
    ax.plot(time,traces[strategy+'_Sst'][stamps],
        color=colors['sst'],lw=4)


    omitted = 'omission' in condition
    change = (not omitted) and (('change' in condition) or \
        ('hit' in condition) or ('miss' in condition))
    psth.plot_flashes_on_trace(ax, time, 
        change=change, omitted=omitted)
    ax.set_xlabel('time from {} (s)'.format(condition),fontsize=16)
    ax.set_ylabel(condition+' response\n(Ca$^{2+}$ events)',
        fontsize=16)
    ax.set_ylim(0,.049)
    ax.set_xlim(time[0],time[-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_title(strategy+' strategy',fontsize=16)


    if condition =='omission':
        ax.plot(0,0,'co',zorder=10,clip_on=False)
    elif condition =='hit':
        ax.plot(0,0,'ro',zorder=10,clip_on=False)
    elif condition == 'miss':
        ax.plot(0,0,'rx',zorder=10,clip_on=False)
    else:
        ax.plot(0,0,'ko',zorder=10,clip_on=False)
    if condition in ['omission','hit','miss']:
        ax.plot(.75,0,'ko',zorder=10,clip_on=False)
        ax.plot(1.5,0,'o',color='gray',zorder=10,clip_on=False)

    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+condition+'_strategy_perturbation_PSTH_{}_{}.svg'.format(strategy, experience_level)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 
    return ax

def plot_PSTH_3D(dfs,labels, condition,run_params, 
    experience_level="Familiar",savefig=True):
    traces = get_PSTH_2D_traces(dfs,labels,condition,
        experience_level=experience_level)
    
    ax = plt.figure().add_subplot(projection='3d')
    ts = np.where((traces['time'] >=0)&(traces['time']<=.75))[0]
    ax.plot(traces['visual_Exc'][ts],
        traces['visual_Sst'][ts], 
        traces['visual_Vip'][ts],
        color='darkorange',
        linewidth=3)
    ax.plot(traces['timing_Exc'][ts],
        traces['timing_Sst'][ts], 
        traces['timing_Vip'][ts],
        color='blue',
        linewidth=3)

    p1 = np.where(traces['time'] == 0)[0]
    p2 = np.argmin(np.abs(traces['time'] - .75)) 
    p3 = np.where(traces['time'] == 1.5)[0]
    ax.plot(traces['visual_Exc'][p1],
        traces['visual_Sst'][p1], 
        traces['visual_Vip'][p1],
        'ko')
    ax.plot(traces['timing_Exc'][p1],
        traces['timing_Sst'][p1], 
        traces['timing_Vip'][p1],
        'ko')
    ax.plot(traces['visual_Exc'][p1+1],
        traces['visual_Sst'][p1+1], 
        traces['visual_Vip'][p1+1],
        'kx')
    ax.plot(traces['timing_Exc'][p1+1],
        traces['timing_Sst'][p1+1], 
        traces['timing_Vip'][p1+1],
        'kx')


    ax.set_xlabel('Exc',fontsize=16)
    ax.set_ylabel('Sst',fontsize=16)
    ax.set_zlabel('Vip',fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    ax.set_zlim(bottom=0)
    ax.set_xlim(left=0.001)
    ax.set_ylim(bottom=0)
    ax.view_init(elev=15,azim=-115)

    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+condition+'_strategy_perturbation_PSTH_3D_{}.svg'.format(experience_level)
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 

def get_PSTH_2D_traces(dfs,labels,condition,experience_level="Familiar"):
    
    traces={}
    for index, label in enumerate(labels):
        traces['visual_'+label[0:3]] = dfs[index]\
            .query('(visual_strategy_session)&'\
            +'(experience_level==@experience_level)&'\
            +'(condition==@condition)')['response'].mean()
        traces['timing_'+label[0:3]] = dfs[index]\
            .query('(not visual_strategy_session)&'\
            +'(experience_level==@experience_level)&'\
            +'(condition==@condition)')['response'].mean()
    traces['visual_y'] = traces['visual_Vip']-traces['visual_Sst']
    traces['timing_y'] = traces['timing_Vip']-traces['timing_Sst']
    traces['time'] = dfs[0].query('condition ==@condition').iloc[0]['time']
    return traces

def PSTH_analysis(dfs, labels,condition, run_params, 
    experience_level="Familiar",savefig=False):
   
    # error bars on 2D 
    ax1= plot_PSTH_2D(dfs,labels, condition,'visual',run_params, 
        experience_level,savefig)
    ax2= plot_PSTH_2D(dfs,labels, condition,'timing',run_params, 
        experience_level,savefig)

    ax = plot_PSTH_perturbation(dfs,labels, condition,
        run_params=run_params,savefig=savefig,
        experience_level=experience_level)
    




















