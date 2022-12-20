import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior_glm.GLM_strategy_tools as gst
import os
from scipy.spatial import ConvexHull as ch
from mpl_toolkits.axes_grid1 import Divider, Size

def analysis(weights_beh, run_params, kernel,session_filter=['Familiar'],savefig=False,
    lims=None):
    out1 = gst.strategy_kernel_comparison(weights_beh.query('visual_strategy_session'),
        run_params, kernel,session_filter=['Familiar'])
    out2 =gst.strategy_kernel_comparison(weights_beh.query('not visual_strategy_session'),
        run_params, kernel,session_filter=['Familiar'])
    out3 = gst.strategy_kernel_comparison(weights_beh.query('visual_strategy_session'),
        run_params, kernel,session_filter=['Novel 1'])
    out4 =gst.strategy_kernel_comparison(weights_beh.query('not visual_strategy_session'),
        run_params, kernel,session_filter=['Novel 1'])
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
    out1[2].set_title('Visual, Familiar',fontsize=16)
    out2[2].set_title('Timing, Familiar',fontsize=16)
    out3[2].set_title('Visual, Novel ',fontsize=16)
    out4[2].set_title('Timing, Novel ',fontsize=16)
    if savefig:
        filename = run_params['figure_dir']+\
                '/strategy/'+kernel+'_visual_familiar_dynamics.svg'
        out1[1].savefig(filename)
        print('Figure saved to: '+filename)

    ax = plot_perturbation(weights_beh, run_params, kernel,savefig=savefig,lims=lims)
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
    fig, ax = plt.subplots()
    df = get_error(Fvisual)
    df = df.loc[0:pi3]
    plot_iterative_ch(ax,df,'lightgray',show_steps=show_steps)
    if show_steps:
        ax.errorbar(Fvisual['Slc17a7-IRES2-Cre'][0:pi3],Fvisual['y'][0:pi3],
            xerr=Fvisual['Slc17a7-IRES2-Cre_sem'][0:pi3],
            yerr=Fvisual['y_sem'][0:pi3],color='gray',alpha=.5)
    ax.plot(Fvisual['Slc17a7-IRES2-Cre'][0:pi3], Fvisual['y'][0:pi3],
        color=colors['visual'],label='Visual',linewidth=3)

def get_perturbation(weights_df, run_params, kernel):
    Fvisual = get_kernel_averages(weights_df.query('visual_strategy_session'), 
        run_params, kernel, session_filter=['Familiar'])
    Ftiming = get_kernel_averages(weights_df.query('not visual_strategy_session'), 
        run_params, kernel, session_filter=['Familiar'])
    Nvisual = get_kernel_averages(weights_df.query('visual_strategy_session'), 
        run_params, kernel, session_filter=['Novel 1'])
    Ntiming = get_kernel_averages(weights_df.query('not visual_strategy_session'), 
        run_params, kernel, session_filter=['Novel 1'])
    Fvisual['y'] = -Fvisual['Sst-IRES-Cre'] + Fvisual['Vip-IRES-Cre']
    Ftiming['y'] = -Ftiming['Sst-IRES-Cre'] + Ftiming['Vip-IRES-Cre']
    Nvisual['y'] = -Nvisual['Sst-IRES-Cre'] + Nvisual['Vip-IRES-Cre']
    Ntiming['y'] = -Ntiming['Sst-IRES-Cre'] + Ntiming['Vip-IRES-Cre']
    Fvisual['y_sem'] = np.sum([Fvisual['Sst-IRES-Cre_sem'],Fvisual['Vip-IRES-Cre_sem']],axis=0)
    Ftiming['y_sem'] = np.sum([Ftiming['Sst-IRES-Cre_sem'],Ftiming['Vip-IRES-Cre_sem']],axis=0)
    Nvisual['y_sem'] = np.sum([Nvisual['Sst-IRES-Cre_sem'],Nvisual['Vip-IRES-Cre_sem']],axis=0)
    Ntiming['y_sem'] = np.sum([Ntiming['Sst-IRES-Cre_sem'],Ntiming['Vip-IRES-Cre_sem']],axis=0)
    return Fvisual, Ftiming, Nvisual, Ntiming

def plot_perturbation(weights_df, run_params, kernel,savefig=False,lims = None,show_steps=False):
    Fvisual, Ftiming, Nvisual, Ntiming = get_perturbation(weights_df, run_params,kernel)
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

    if show_steps:
        demonstrate_iterative_ch(Fvisual,kernel)

    colors = gvt.project_colors()
    fig, ax = plt.subplots(1,2,sharey=True,sharex=True,figsize=(7,3.5))
    
    df = get_error(Fvisual).loc[0:pi3]
    plot_iterative_ch(ax[0],df,'lightgray')  

    df = get_error(Ftiming).loc[0:pi3]
    plot_iterative_ch(ax[0],df,'lightgray')

    ax[0].plot(Fvisual['Slc17a7-IRES2-Cre'][0:pi3], Fvisual['y'][0:pi3],
        color=colors['visual'],label='Visual',linewidth=3)
    ax[0].plot(Ftiming['Slc17a7-IRES2-Cre'][0:pi3], Ftiming['y'][0:pi3],
        color=colors['timing'],label='Timing',linewidth=3)
    ax[0].plot(Fvisual['Slc17a7-IRES2-Cre'][0],Fvisual['y'][0],'co')
    ax[0].plot(Ftiming['Slc17a7-IRES2-Cre'][0],Ftiming['y'][0],'co')
    if multiimage:
        ax[0].plot(Fvisual['Slc17a7-IRES2-Cre'][pi],Fvisual['y'][pi],'ko')
        ax[0].plot(Ftiming['Slc17a7-IRES2-Cre'][pi],Ftiming['y'][pi],'ko')
        ax[0].plot(Fvisual['Slc17a7-IRES2-Cre'][pi2],Fvisual['y'][pi2],'o',color='gray')
        ax[0].plot(Ftiming['Slc17a7-IRES2-Cre'][pi2],Ftiming['y'][pi2],'o',color='gray')

    ax[0].set_ylabel('Vip - Sst',fontsize=16)
    ax[0].set_xlabel('Exc',fontsize=16)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].xaxis.set_tick_params(labelsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)  
    ax[0].set_title('Familiar, {}'.format(kernel),fontsize=16)
    ax[0].legend() 
    ax[0].axhline(0,color='k',linestyle='--',alpha=.25)
    ax[0].axvline(0,color='k',linestyle='--',alpha=.25)

    df = get_error(Nvisual).loc[0:pi3]
    plot_iterative_ch(ax[1],df,'lightgray')  

    df = get_error(Ntiming).loc[0:pi3]
    plot_iterative_ch(ax[1],df,'lightgray')
 
    ax[1].plot(Nvisual['Slc17a7-IRES2-Cre'][0:pi3], Nvisual['y'][0:pi3],
        color=colors['visual'],linewidth=3)
    ax[1].plot(Ntiming['Slc17a7-IRES2-Cre'][0:pi3], Ntiming['y'][0:pi3],
        color=colors['timing'],linewidth=3)   
    ax[1].plot(Nvisual['Slc17a7-IRES2-Cre'][0],Nvisual['y'][0],'co')
    ax[1].plot(Ntiming['Slc17a7-IRES2-Cre'][0],Ntiming['y'][0],'co')
    if multiimage:
        ax[1].plot(Nvisual['Slc17a7-IRES2-Cre'][pi],Nvisual['y'][pi],'ko')
        ax[1].plot(Ntiming['Slc17a7-IRES2-Cre'][pi],Ntiming['y'][pi],'ko')
        ax[1].plot(Nvisual['Slc17a7-IRES2-Cre'][pi2],Nvisual['y'][pi2],'o',color='gray')
        ax[1].plot(Ntiming['Slc17a7-IRES2-Cre'][pi2],Ntiming['y'][pi2],'o',color='gray')
    ax[1].set_ylabel('Vip - Sst',fontsize=16)
    ax[1].set_xlabel('Exc',fontsize=16)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].xaxis.set_tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)  
    ax[1].set_title('Novel, {}'.format(kernel),fontsize=16)
    ax[1].axhline(0,color='k',linestyle='--',alpha=.25)
    ax[1].axvline(0,color='k',linestyle='--',alpha=.25)
    
    if lims is not None:
        ax[1].set_xlim(lims[0])
        ax[1].set_ylim(lims[1])
    plt.tight_layout()
    
    if savefig:
        filepath = run_params['figure_dir']+\
            '/strategy/'+kernel+'_strategy_perturbation.svg'
        print('Figure saved to: '+filepath)
        plt.savefig(filepath) 

def plot_perturbation_3D(weights_df,run_params, kernel, session='Familiar'):
    visual = get_kernel_averages(weights_df.query('visual_strategy_session'), 
        run_params, kernel, session_filter=[session])
    timing = get_kernel_averages(weights_df.query('not visual_strategy_session'), 
        run_params, kernel, session_filter=[session])
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(visual['Slc17a7-IRES2-Cre'],
        visual['Sst-IRES-Cre'], 
        visual['Vip-IRES-Cre'],
        color='darkorange',
        linewidth=3)
    ax.plot(timing['Slc17a7-IRES2-Cre'],
        timing['Sst-IRES-Cre'], 
        timing['Vip-IRES-Cre'],
        color='blue',
        linewidth=3)
    ax.set_xlabel('Exc',fontsize=16)
    ax.set_ylabel('Sst',fontsize=16)
    ax.set_zlabel('Vip',fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)


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


