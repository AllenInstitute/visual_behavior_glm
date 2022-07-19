import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def add_behavior_metrics(df,summary_df):
    '''
        Merges the behavioral summary table onto the dataframe passed in 
    ''' 
    behavior_columns = ['behavior_session_id','visual_strategy_session',
        'strategy_dropout_index','dropout_task0','dropout_omissions1',
        'dropout_omissions','dropout_timing1D']
    out_df = pd.merge(df, summary_df[behavior_columns],
        on='behavior_session_id',
        suffixes=('','_ophys_table'),
        validate='many_to_one')
    out_df['strategy'] = ['visual' if x else 'timing' \
        for x in out_df['visual_strategy_session']]
    return out_df


##### DEV BELOW HERE

# TODO, need to update
def plot_kernels_by_strategy_by_session(weights_beh, run_params, ym='omissions',
    cre_line = 'Vip-IRES-Cre',compare=['strategy'],savefig=False):
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


# TODO, need to update
def plot_kernels_by_strategy_by_omission_exposure(weights_beh, run_params, 
    ym='omissions',cre_line = 'Vip-IRES-Cre',compare=['strategy']):
    # by omission exposures
    sessions = [[0,1,2,3],[0],[1,2,3,4]]
    filter_sessions_on =['prior_exposures_to_omissions',\
        'prior_exposures_to_image_set',\
        'prior_exposures_to_image_set']
    image_set = [['familiar'],['novel'],['novel']]
    fig, ax = plt.subplots(2,len(sessions),figsize=(len(sessions)*3,6),sharey=True)
    for dex, session in enumerate(sessions):
        show_legend = dex == len(sessions) - 1
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = session, cell_filter = cre_line,area_filter=['VISp'], 
            compare=compare, plot_errors=True,save_kernels=False,
            ax=ax[0,dex],fs1=14,fs2=12,show_legend=show_legend,
            filter_sessions_on = filter_sessions_on[dex],image_set=image_set[dex]) 
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = session, cell_filter = cre_line,area_filter=['VISl'], 
            compare=compare, plot_errors=True,save_kernels=False,
            ax=ax[1,dex],fs1=14,fs2=12,show_legend=False,
            filter_sessions_on = filter_sessions_on[dex],image_set=image_set[dex])
        if dex == 0:
            ax[0,0].set_ylabel('V1, '+cre_line+'\nAvg. Kernel ($\Delta f/f)$')
            ax[1,0].set_ylabel('LM\nAvg. Kernel ($\Delta f/f)$')
        else:
            ax[0,dex].set_ylabel('Avg. Kernel ($\Delta f/f)$')
            ax[1,dex].set_ylabel('Avg. Kernel ($\Delta f/f)$')
        if dex in [0]:
            ax[0,dex].set_title('Familiar Images')
        elif dex == 1:
            ax[0,dex].set_title('Novel 1')       
        else:
            ax[0,dex].set_title('Novel > 1')
    if 'binned_strategy' in compare:
        ax[0,3].legend(labels=['Most Timing','Middle','Most Visual'],
            loc='upper left',bbox_to_anchor=(1.05,1),title=' & '.join(compare),
            handlelength=4)
        plt.tight_layout()
    filename = ym+'_by_exposure_'+cre_line+'_'+'_'.join(compare)
    save_figure(fig,run_params['version'], ym, filename)
    plt.tight_layout()
    return ax

# TODO, need to update
def compare_cre_kernels(weights_beh, run_params, ym='omissions',
    compare=['strategy'],equipment_filter='all',title='',
    sessions=['Familiar','Novel 1','Novel >1'],image_set='familiar',
    filter_sessions_on='experience_level',savefig=False):

    cres = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    fig, ax = plt.subplots(1,len(cres),figsize=(len(sessions)*4,4),sharey=True)
    for dex, cre in enumerate(cres):
        show_legend = dex == len(cres) - 1
        out = strategy_kernel_comparison(weights_beh, run_params, ym, 
            threshold=0, drop_threshold = 0, 
            session_filter = sessions, cell_filter = cre,area_filter=['VISp'], 
            compare=compare, plot_errors=True,save_kernels=False,ax=ax[dex],
            show_legend=show_legend,filter_sessions_on=filter_sessions_on,
            image_set=image_set,equipment_filter=equipment_filter) 
        ax[dex].set_title(string_mapper(cre),fontsize=16)

    if (equipment_filter == 'all')&(title==""):
        ax[0].set_ylabel('V1\n'+ax[0].get_ylabel(),fontsize=16)
    elif title != '':
        ax[0].set_ylabel('V1 - '+title+'\n'+ax[0].get_ylabel(),fontsize=16)
    else:
        ax[0].set_ylabel('V1 - '+equipment_filter+'\n'+ax[0].get_ylabel(),fontsize=16)
    if 'binned_strategy' in compare:
        ax[2].legend(labels=['Most Timing','Partial Timing','Partial Visual',\
            'Most Visual'],loc='upper left',bbox_to_anchor=(1.05,1),
            title=' & '.join(compare),handlelength=4)
    plt.tight_layout()

    if savefig:
        filename = ym+'_by_cre_line_'+'_'.join(compare)+'_'+equipment_filter
        save_figure(fig,run_params['version'], ym, filename)


def plot_strategy(results_beh, run_params,ym='omissions'):
    cres = ['Vip-IRES-Cre','Slc17a7-IRES2-Cre','Sst-IRES-Cre']
    for cre in cres:
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym, 
            sessions=[0,1,2,3], use_prior_omissions=True, plot_single=True, 
            title=cre+', familiar')
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym, 
            sessions=[0,1,2,3], use_prior_omissions=True, plot_single=True, 
            title=cre+', familiar',area = ['VISp'])
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym, 
            sessions=[0,1,2,3], use_prior_omissions=True, plot_single=True, 
            title=cre+', familiar',area = ['VISl'])

    for cre in cres:
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre,ymetric=ym,
            sessions=[0,1,2,3,4,5,6,7,8],use_prior_omissions=True,plot_single=True,
            image_set='novel',title=cre)
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre,ymetric=ym,
            sessions=[0,1,2,3,4,5,6,7,8],use_prior_omissions=True,plot_single=True,
            image_set='novel',title=cre,area=['VISp'])
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre,ymetric=ym,
            sessions=[0,1,2,3,4,5,6,7,8],use_prior_omissions=True,plot_single=True,
            image_set='novel',title=cre,area=['VISl'])

    fits=scatter_dataset(results_beh, run_params,sessions=[0,1,2,3], 
        use_prior_omissions=True, ymetric=ym,area=['VISp'],image_set='familiar')

    for cre in cres:
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            threshold=0, sessions=[0,1,2,3], use_prior_omissions=True, 
            plot_single=True, title=cre+', familiar')
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            threshold=0.001, sessions=[0,1,2,3], use_prior_omissions=True, 
            plot_single=True, title=cre+', familiar',area = ['VISp'])
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            threshold=0.01, sessions=[0,1,2,3], use_prior_omissions=True, 
            plot_single=True, title=cre+', familiar',area = ['VISp'])

    for cre in cres:
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            ymetric_threshold=0, sessions=[0,1,2,3], use_prior_omissions=True, 
            plot_single=True, title=cre+', familiar')
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            ymetric_threshold=-0.000001, sessions=[0,1,2,3], 
            use_prior_omissions=True, plot_single=True, title=cre+', familiar',
            area = ['VISp'])
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            ymetric_threshold=-0.001, sessions=[0,1,2,3], 
            use_prior_omissions=True, plot_single=True, title=cre+', familiar',
            area = ['VISp'])
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            ymetric_threshold=-0.01, sessions=[0,1,2,3], use_prior_omissions=True, 
            plot_single=True, title=cre+', familiar',area = ['VISp'])
        fits=scatter_by_cell(results_beh,run_params,cre_line=cre, ymetric=ym,
            ymetric_threshold=-0.1, sessions=[0,1,2,3], use_prior_omissions=True, 
            plot_single=True, title=cre+', familiar',area = ['VISp'])


def scatter_dataset(results_beh, run_params,threshold=0,ymetric_threshold=0, 
    xmetric='strategy_dropout_index', ymetric='omissions',sessions=[1,3,4,6],
    use_prior_omissions=False,image_set='familiar',area=['VISp','VISl'],
    use_prior_image_set=False):

    fig, ax = plt.subplots(3,len(sessions)+1, figsize=(18,10))
    fits = {}
    cres = ['Slc17a7-IRES2-Cre', 'Vip-IRES-Cre','Sst-IRES-Cre']
    ymins = []
    ymaxs =[]
    for dex,cre in enumerate(cres):
        col_start = dex == 0
        fits[cre] = scatter_by_session(results_beh, run_params, cre_line = cre, 
            threshold=threshold, ymetric_threshold=ymetric_threshold, 
            xmetric=xmetric, ymetric=ymetric, sessions=sessions, ax = ax[dex,:],
            col_start=col_start,use_prior_omissions=use_prior_omissions,
            image_set=image_set,area=area,use_prior_image_set=use_prior_image_set)
        ymins.append(fits[cre]['yrange'][0])
        ymaxs.append(fits[cre]['yrange'][1])
    
    for dex, cre in enumerate(cres):
        ax[dex,-1].set_ylim(np.min(ymins),np.max(ymaxs))
  
    if use_prior_omissions:
       filename = xmetric+'_by_'+ymetric+'_dataset_'+'_'.join(area)+\
        '_by_omission_exposures_'+''.join([str(x) for x in sessions])+\
        '_threshold_'+str(threshold)+'_ymetric_threshold_'+\
        str(ymetric_threshold)+'_image_set_'+image_set
    else:
        filename = xmetric+'_by_'+ymetric+'_dataset_'+'_'.join(area)+\
        '_by_session_number_'+''.join([str(x) for x in sessions])+\
        '_threshold_'+str(threshold)+'_ymetric_threshold_'+str(ymetric_threshold)
    save_figure(fig,run_params['version'], ymetric, filename)
    return fits

def scatter_by_session(results_beh, run_params, cre_line=None, threshold=0,
    ymetric_threshold=0,xmetric='strategy_dropout_index',ymetric='omissions',
    sessions=[1,3,4,6],ax=None,col_start=False,use_prior_omissions=False,
    image_set='familiar',area=['VISp','VISl'],use_prior_image_set=False):

    if ax is None:
        fig, ax = plt.subplots(1,len(sessions)+1, figsize=(18,3.25))

    fits = {}
    for dex,s in enumerate(sessions):
        row_start = dex == 0
        fits[str(s)] = scatter_by_cell(results_beh,run_params, cre_line=cre_line,
            threshold=threshold,ymetric_threshold=ymetric_threshold, 
            xmetric=xmetric, ymetric=ymetric, sessions=[s],title='Session '+str(s),
            ax=ax[dex],row_start=row_start,col_start=col_start,
            use_prior_omissions=use_prior_omissions,image_set = image_set, 
            area=area,use_prior_image_set=use_prior_image_set)

    ax[-1].axhline(0, linestyle='--',color='k',alpha=.25)   
    for s in sessions:
        if fits[str(s)] is not None:
            ax[-1].plot(s,fits[str(s)][0],'ko')
            ax[-1].plot([s,s], [fits[str(s)][0]-fits[str(s)][4],
                fits[str(s)][0]+fits[str(s)][4]], 'k--')
    ax[-1].set_ylabel('Regression slope')
    ax[-1].set_xlabel('Session Number')
    ax[-1].set_title(cre_line)
    plt.tight_layout()

    fits['yrange'] = ax[-1].get_ylim()
    fits['xmetric'] = xmetric
    fits['ymetric'] = ymetric
    fits['cre_line'] = cre_line
    fits['threshold'] = threshold
    fits['glm_version'] = run_params['version'] 
    return fits 

def scatter_by_cell(results_beh, run_params, cre_line=None, threshold=0, 
    ymetric_threshold=0, sessions=[1],xmetric='strategy_dropout_index',
    ymetric='omissions',title='',nbins=10,ax=None,row_start=False,
    col_start=False,use_prior_omissions = False,plot_single=False,
    image_set='familiar',area=['VISp','VISl'],use_prior_image_set=False,
    equipment=['mesoscope','scientifica']):

    if use_prior_omissions: 
        g = results_beh.query('(cre_line == @cre_line)& \
            (variance_explained_full > @threshold)&\
            (prior_exposures_to_omissions in @sessions)&\
            (familiar == @image_set)&(targeted_structure in @area)&\
            (equipment_name in @equipment)').\
            dropna(axis=0, subset=[ymetric,xmetric]).copy()
    elif use_prior_image_set:
        g = results_beh.query('(cre_line == @cre_line)&\
            (variance_explained_full > @threshold)&\
            (prior_exposures_to_image_set in @sessions)&\
            (familiar == @image_set)&(targeted_structure in @area)&\
            (equipment_name in @equipment)').\
            dropna(axis=0, subset=[ymetric,xmetric]).copy()
    else:
        g = results_beh.query('(cre_line == @cre_line)&\
            (variance_explained_full > @threshold)&\
            (session_number in @sessions)&(targeted_structure in @area)&\
            (equipment_name in @equipment)').\
            dropna(axis=0, subset=[ymetric,xmetric]).copy()
    if ymetric_threshold != 0:
        print('filtering')
        g = g[g[ymetric] < ymetric_threshold]

    if len(g) == 0:
        return

    # Figure axis
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    # Plot Raw data
    ax.plot(g[xmetric], g[ymetric],'ko',alpha=.1,label='raw data')
    ax.set_xlabel(xmetric)
    ax.set_xlim(results_beh[xmetric].min(), results_beh[xmetric].max())
    if row_start:
        ax.set_ylabel(cre_line +'\n'+ymetric)
    else:
        ax.set_ylabel(ymetric)

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
    if col_start:
        ax.set_title(title)
    if plot_single:
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        if use_prior_omissions:
            filename = run_params['version']+'_'+xmetric+'_by_'+ymetric+'_'+\
                cre_line+'_'+'_'.join(area)+'_by_omission_exposures_'+\
                ''.join([str(x) for x in sessions])+'_threshold_'+str(threshold)+\
                '_ymetric_threshold_'+str(ymetric_threshold)+'_image_set_'+image_set
        elif use_prior_image_set:
            filename = run_params['version']+'_'+xmetric+'_by_'+ymetric+'_'+\
                cre_line+'_'+'_'.join(area)+'_by_image_exposures_'+\
                ''.join([str(x) for x in sessions])+'_threshold_'+\
                str(threshold)+'_ymetric_threshold_'+str(ymetric_threshold)+\
                '_image_set_'+image_set
        else:
            filename = run_params['version']+'_'+xmetric+'_by_'+ymetric+'_'+\
                cre_line+'_'+'_'.join(area)+'_by_session_number_'+\
                ''.join([str(x) for x in sessions])+'_threshold_'+\
                str(threshold)+'_ymetric_threshold_'+str(ymetric_threshold)
        if len(equipment) == 1:
            filename += '_'+equipment[0]
        save_figure(plt.gcf(),run_params['version'], ymetric, filename)
    return x    

def string_mapper(string):
    d = {
        'Vip-IRES-Cre':'Vip Inhibitory',
        'Sst-IRES-Cre':'Sst Inhibitory',
        'Slc17a7-IRES2-Cre':'Excitatory',
    }
    return d[string]

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
        use_dropouts=True
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


