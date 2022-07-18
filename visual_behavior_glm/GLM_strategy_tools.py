import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_visualization_tools as gvt
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
    cre_line = 'Vip-IRES-Cre',compare=['strategy']):
    # By Session number
    sessions = ['1','3','4','6']
    filter_sessions_on ='session_number'

    image_set = ['familiar','novel']
    fig, ax = plt.subplots(2,len(sessions),figsize=(len(sessions)*3,6),sharey=True)
    for dex, session in enumerate(sessions):
        show_legend = dex == len(sessions) - 1
        out = gvt.plot_kernel_comparison(weights_beh, run_params, ym, 
            save_results = False,threshold=0, drop_threshold = 0, 
            session_filter = [session], cell_filter = cre_line,
            area_filter=['VISp'], compare=compare, plot_errors=True,
            save_kernels=False,ax=ax[0,dex],fs1=14,fs2=12,
            show_legend=show_legend,filter_sessions_on = filter_sessions_on,
            image_set=image_set) 
        out = gvt.plot_kernel_comparison(weights_beh, run_params, ym, 
            save_results = False,threshold=0, drop_threshold = 0, 
            session_filter = [session], cell_filter = cre_line,
            area_filter=['VISl'], compare=compare, plot_errors=True,
            save_kernels=False,ax=ax[1,dex],fs1=14,fs2=12,show_legend=False,
            filter_sessions_on = filter_sessions_on,image_set=image_set) 
        ax[0,dex].set_title('Session '+str(session))
        if dex == 0:
            ax[0,0].set_ylabel('V1 '+cre_line+'\n'+ax[0,0].get_ylabel())
            ax[1,0].set_ylabel('LM\n'+ax[1,0].get_ylabel())
    filename = ym+'_by_session_number_'+cre_line
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
        out = gvt.plot_kernel_comparison(weights_beh, run_params, ym, 
            save_results = False,threshold=0, drop_threshold = 0, 
            session_filter = session, cell_filter = cre_line,area_filter=['VISp'], 
            compare=compare, plot_errors=True,save_kernels=False,
            ax=ax[0,dex],fs1=14,fs2=12,show_legend=show_legend,
            filter_sessions_on = filter_sessions_on[dex],image_set=image_set[dex]) 
        out = gvt.plot_kernel_comparison(weights_beh, run_params, ym, 
            save_results = False,threshold=0, drop_threshold = 0, 
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
    return ax

# TODO, need to update
def compare_cre_kernels(weights_beh, run_params, ym='omissions',
    compare=['strategy'],equipment_filter='all',title='',
    sessions=['Familiar','Novel 1','Novel >1'],image_set='familiar',
    filter_sessions_on='experience_level'):

    cres = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    fig, ax = plt.subplots(1,len(cres),figsize=(len(sessions)*3,4),sharey=True)
    for dex, cre in enumerate(cres):
        show_legend = dex == len(cres) - 1
        out = gvt.plot_kernel_comparison(weights_beh, run_params, ym, 
            save_results = False,threshold=0, drop_threshold = 0, 
            session_filter = sessions, cell_filter = cre,area_filter=['VISp'], 
            compare=compare, plot_errors=True,save_kernels=False,ax=ax[dex],
            fs1=14,fs2=12,show_legend=show_legend,filter_sessions_on=filter_sessions_on,
            image_set=image_set,equipment_filter=equipment_filter) 
        ax[dex].set_title(cre)

    if (equipment_filter == 'all')&(title==""):
        ax[0].set_ylabel('V1\n'+ax[0].get_ylabel())
    elif title != '':
        ax[0].set_ylabel('V1 - '+title+'\n'+ax[0].get_ylabel())
    else:
        ax[0].set_ylabel('V1 - '+equipment_filter+'\n'+ax[0].get_ylabel())
    if 'binned_strategy' in compare:
        ax[2].legend(labels=['Most Timing','Partial Timing','Partial Visual',\
            'Most Visual'],loc='upper left',bbox_to_anchor=(1.05,1),
            title=' & '.join(compare),handlelength=4)
    plt.tight_layout()
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
    print(filename)
