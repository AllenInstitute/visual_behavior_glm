import visual_behavior.plotting as vbp
import visual_behavior.utilities as vbu
import visual_behavior.data_access.loading as loading
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior.database as db
import matplotlib as mpl
import seaborn as sns
import scipy
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import gc
from scipy import ndimage
from scipy import stats

def plot_kernel_support(glm,include_cont = False,plot_bands=True,plot_ticks=True,start=10000,end=11000):
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

    # Basic figure set up
    if plot_bands:
        plt.figure(figsize=(12,10))
    else:
        plt.figure(figsize=(12,6))
    time_vec = glm.fit['dff_trace_timestamps'][start:end]
    start_t = time_vec[0]
    end_t = time_vec[-1]
    ones = np.ones(np.shape(time_vec))
    colors = sns.color_palette('hls', len(discrete)+len(continuous)) 

    # Plot the kernels
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
    stim_points = {}
    for index, d in enumerate(discrete):
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
    all_k = discrete

    # Plot Rewards
    if 'rewards' in glm.run_params['kernels']:
        reward_dex = stim_points['rewards'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['rewards']['offset'])*31)
    elif 'hits' in glm.run_params['kernels']:
        reward_dex = stim_points['hits'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['hits']['offset'])*31)
    else:
        reward_dex = 0
    if plot_bands:
        reward_dex += -.4
    if plot_ticks:
        rewards =glm.session.dataset.rewards.query('timestamps < @end_t & timestamps > @start_t')['timestamps']
        plt.plot(rewards, reward_dex*np.ones(np.shape(rewards)),'k|')
    
    # Stimulus Presentations
    stim = glm.session.dataset.stimulus_presentations.query('start_time > @start_t & start_time < @end_t & not omitted')
    for index, time in enumerate(stim['start_time'].values):
        plt.axvspan(time, time+0.25, color='k',alpha=.1)
    if plot_ticks:
        for index in range(0,8):
            image = glm.session.dataset.stimulus_presentations.query('start_time >@start_t & start_time < @end_t & image_index == @index')['start_time']
            image_dex = stim_points['image'+str(index)][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['image'+str(index)]['offset'])*31)
            plt.plot(image, image_dex*np.ones(np.shape(image)),'k|')

    # Stimulus Changes
    change = glm.session.dataset.stimulus_presentations.query('start_time > @start_t & start_time < @end_t & change')
    if plot_ticks:
        if 'change' in glm.run_params['kernels']:
            change_dex = stim_points['change'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['change']['offset'])*31)
            plt.plot(change['start_time'], change_dex*np.ones(np.shape(change['start_time'])),'k|')
    for index, time in enumerate(change['start_time'].values):
        plt.axvspan(time, time+0.25, color='b',alpha=.2)

    # Stimulus Omissions
    if plot_ticks:
        omitted = glm.session.dataset.stimulus_presentations.query('start_time >@start_t & start_time < @end_t & omitted')['start_time']
        omitted_dex = stim_points['omissions'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['omissions']['offset'])*31)
        plt.plot(omitted, omitted_dex*np.ones(np.shape(omitted)),'k|')

    # Image Expectation
    if plot_ticks:
        expectation = glm.session.dataset.stimulus_presentations.query('start_time >@start_t & start_time < @end_t & not omitted')['start_time']
        expectation_dex = stim_points['image_expectation'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['image_expectation']['offset'])*31)
        plt.plot(expectation, expectation_dex*np.ones(np.shape(expectation)),'k|')

    # Licks
    if plot_ticks:
        licks = glm.session.dataset.licks.query('timestamps < @end_t & timestamps > @start_t')['timestamps']
        
        if 'pre_lick_bouts' in glm.run_params['kernels']:
            bouts = glm.session.dataset.licks.query('timestamps < @end_t & timestamps > @start_t & bout_start')['timestamps']
            pre_dex = stim_points['pre_lick_bouts'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['pre_lick_bouts']['offset'])*31)
            post_dex = stim_points['post_lick_bouts'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['post_lick_bouts']['offset'])*31)
            plt.plot(bouts, pre_dex*np.ones(np.shape(bouts)),'k|')
            plt.plot(bouts, post_dex*np.ones(np.shape(bouts)),'k|')
        if 'lick_bouts' in glm.run_params['kernels']:
            bouts = glm.session.dataset.licks.query('timestamps < @end_t & timestamps > @start_t & bout_start')['timestamps']
            dex = stim_points['lick_bouts'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['lick_bouts']['offset'])*31)
            plt.plot(bouts, dex*np.ones(np.shape(bouts)),'k|')

        if 'pre_licks' in glm.run_params['kernels']:
            pre_dex = stim_points['pre_licks'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['pre_licks']['offset'])*31)
            post_dex = stim_points['post_licks'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['post_licks']['offset'])*31)
            plt.plot(licks, pre_dex*np.ones(np.shape(licks)),'k|')
            plt.plot(licks, post_dex*np.ones(np.shape(licks)),'k|')
        if 'licks' in glm.run_params['kernels']:
            dex = stim_points['licks'][0] + dt*np.ceil(np.abs(glm.run_params['kernels']['licks']['offset'])*31)
            plt.plot(licks, dex*np.ones(np.shape(licks)),'k|')


    # Trials
    if plot_ticks:
        types = ['hit','miss','false_alarm','correct_reject']
        ks = ['hits','misses','false_alarms','correct_rejects']
        trials = glm.session.dataset.trials.query('change_time < @end_t & change_time > @start_t')
        for index, t in enumerate(types):
            try:
                change_time = trials[trials[t]]['change_time'] 
                trial_dex = stim_points[ks[index]][0] + dt*np.ceil(np.abs(glm.run_params['kernels'][ks[index]]['offset'])*31)
                plt.plot(change_time, trial_dex*np.ones(np.shape(change_time)),'k|')
            except:
                print('error plotting - '+t)

    plt.xlabel('Time (s)')
    plt.yticks(ticks,all_k)
    plt.xlim(stim.iloc[0].start_time, stim.iloc[-1].start_time+.75)
    plt.tight_layout()
    return

def plot_glm_version_comparison(comparison_table=None, versions_to_compare=None):
    '''
    makes a scatterplot comparing cellwise performance on two GLM versions

    if a comparison table is not passed, the versions to compare must be passed (as a list of strings)
    comparison table will be built using GLM_analysis_tools.get_glm_version_comparison_table, which takes about 2 minutes
    '''
    assert not (comparison_table is None and versions_to_compare is None), 'must pass either a comparison table or a list of two versions to compare'
    if comparison_table is None:
        comparison_table = gat.get_glm_version_comparison_table(versions_to_compare)

    jointplot = sns.jointplot(
        data = comparison_table,
        x='6_L2_optimize_by_session',
        y='7_L2_optimize_by_session',
        hue='cre_line',
        hue_order=np.sort(comparison_table['cre_line'].unique()),
        alpha=0.15,
        marginal_kws={'common_norm':False},
    )

    jointplot.ax_joint.plot([0,1],[0,1],color='k',linewidth=2,alpha=0.5,zorder=np.inf)
    return jointplot

def plot_significant_cells(results_pivoted,dropout, dropout_threshold=-0.10,save_fig=False,filename=None):
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
        plt.savefig(filename+dropout+".png")

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
        plt.savefig('discrete.png')

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
        plt.savefig('continuous.png') 

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
        plt.savefig('continuous_events.png') 

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

def compare_var_explained(results=None, fig=None, ax=None, figsize=(15,12), outlier_threshold=1.5):
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
    if results is None:
        results_dict = gat.retrieve_results()
        results = results_dict['full']
    if fig is None and ax is None:
        fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True, sharex='col')

    cre_line_order = np.sort(results['cre_line'].unique())
    glm_version_order = np.sort(results['glm_version'].unique())

    for row,dataset in enumerate(['train','test']):
        plot1 = sns.boxplot(
            data=results,
            x='glm_version',
            y='Full_avg_cv_var_{}'.format(dataset),
            order = glm_version_order,
            hue='cre_line',
            hue_order=cre_line_order,
            fliersize=0,
            whis=outlier_threshold,
            ax=ax[row,0],
        )

        plot2 = sns.boxplot(
            data=results,
            x='cre_line',
            y='Full_avg_cv_var_{}'.format(dataset),
            order = cre_line_order,
            hue='glm_version',
            hue_order=glm_version_order,
            fliersize=0,
            whis=outlier_threshold,
            ax=ax[row,1],
            palette='brg',
        )
        ax[row, 0].set_ylabel('variance explained')
        ax[row, 0].set_xlabel('GLM version')
        ax[row, 0].set_title('{} data full model performance\ngrouped by version'.format(dataset))
        ax[row, 1].set_title('{} data full model performance\ngrouped by cre line'.format(dataset))

        # calculate interquartile ranges
        grp = results.groupby(['glm_version','cre_line'])['Full_avg_cv_var_{}'.format(dataset)]
        IQR = grp.quantile(0.75) - grp.quantile(0.25)


        lower_bounds = grp.quantile(0.25) - 1.5*IQR
        upper_bounds = grp.quantile(0.75) + 1.5*IQR

        for i in range(2):
            ax[row, i].legend(loc='upper left',bbox_to_anchor=(1.01, 1),borderaxespad=0)
            ax[row, i].set_ylim(lower_bounds.min()-0.05 ,upper_bounds.max()+0.05)
            ax[row, i].axhline(0, color='black', linestyle=':')
            ax[row, i].set_xticklabels(ax[row, i].get_xticklabels(),rotation=30, ha='right')

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('variance explained by GLM version and cre_line (outliers removed from visualization)')

    return fig, ax


def plot_licks(session, ax, y_loc=0, t_span=None):
    if t_span:
        df = session.dataset.licks.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        df = session.dataset.licks
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
        df = session.dataset.rewards.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        df = session.dataset.licks
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
        running_df = session.dataset.running_data_df.reset_index().query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        running_df = session.dataset.running_data_df.reset_index()
    ax.plot(
        running_df['timestamps'],
        running_df['speed'],
        color='skyblue',
        linewidth=3
    )
    ax.set_ylim(
        session.dataset.running_data_df['speed'].min(),
        session.dataset.running_data_df['speed'].max(),
    )

def plot_pupil(session, ax, t_span=None):
    '''shares axis with running'''
    vbp.initialize_legend(ax=ax, colors=['skyblue','LemonChiffon'],linewidth=3)
    if t_span:
        pupil_df = session.dataset.eye_tracking.query(
            'time >= {} and time <= {}'.format(t_span[0], t_span[1]))
    else:
        pupil_df = session.dataset.eye_tracking
    ax.plot(
        pupil_df['time'],
        pupil_df['pupil_area'],
        color='LemonChiffon',
        linewidth=3
    )
    ax.set_ylim(
        0,
        np.percentile(session.dataset.eye_tracking['pupil_area'].fillna(0), 95)
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


def plot_stimuli(session, ax, t_span=None,alpha=.35):
    buffer = 0.25
    images = session.dataset.stimulus_presentations['image_name'].unique()
    colors = {image: color for image, color in zip(
        np.sort(images), sns.color_palette("Set2", 8))}

    if t_span:
        query_string = 'start_time >= {0} - {2} and stop_time <= {1} + {2}'.format(
            t_span[0], t_span[1], buffer)
        visual_stimuli = session.dataset.stimulus_presentations.query(
            'omitted == False').query(query_string).copy()
    else:
        visual_stimuli = session.dataset.stimulus_presentations.query(
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

    arr = np.zeros_like(session.dataset.max_projection)
    for ii, cell_specimen_id in enumerate(session.dataset.cell_specimen_ids):

        F_cell = F_dataframe.loc[cell_specimen_id][column]
        # arr += session.cell_specimen_table.loc[cell_specimen_id]['image_mask']*F_cell
        arr += session.dataset.get_roi_masks().loc[{'cell_specimen_id':cell_specimen_id}].values*F_cell

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
        legend=False,
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

def plot_dropout_summary(results_summary, cell_specimen_id, ax):
    '''
    makes bar plots of results summary
    inputs:
        glm -- glm object
        cell_specimen_id -- cell to plot
        ax -- axis on which to plot
    '''
    data_to_plot = (
        results_summary
        .query('cell_specimen_id == @cell_specimen_id')
        .sort_values(by='adj_fraction_change_from_full', ascending=False)
    ).copy().reset_index(drop=True)

    dropouts = data_to_plot.dropout.unique()
    single_dropouts = [d for d in dropouts if d.startswith('single-')]
    combined_dropouts = [d.split('single-')[1] for d in single_dropouts]

    for idx,row in data_to_plot.iterrows():
        if row['dropout'] in single_dropouts:
            data_to_plot.at[idx,'dropout_type']='single'
            data_to_plot.at[idx,'dropout_simple']=row['dropout'].split('single-')[1]
        elif row['dropout'] in combined_dropouts:
            data_to_plot.at[idx,'dropout_type']='combined'
            data_to_plot.at[idx,'dropout_simple']=row['dropout']

    yorder = (
        data_to_plot
        .query('dropout_type == "single"')
        .sort_values(by='adj_fraction_change_from_full',ascending=False)['dropout_simple']
        .values
    )
    
    sns.barplot(
        data = data_to_plot.sort_values(by='adj_fraction_change_from_full', ascending=False),
        x = 'adj_fraction_change_from_full',
        y = 'dropout_simple',
        ax=ax,
        hue='dropout_type',
        hue_order=['combined','single'],
        order=yorder,
        palette=['magenta','cyan']
    )
    ax.set_ylabel('Dropout')
    ax.set_title('Fraction Change\nin Variance Explained')


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
                t = np.linspace(
                    0,
                    glm.design.kernel_dict[kernel_name]['kernel_length_samples']/glm.fit['ophys_frame_rate'],
                    glm.design.kernel_dict[kernel_name]['kernel_length_samples']
                )
                t += glm.design.kernel_dict[kernel_name]['offset_seconds']

                kernel_weight_names = [w for w in all_weight_names if w.startswith(kernel_name)]
                w_kernel = glm.W.loc[dict(weights=kernel_weight_names, cell_specimen_id=cell_specimen_id)]
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
    title = '{}__specimen_id={}__exp_id={}__{}__{}__depth={}__cell_id={}__glm_version={}'.format(
        row['cre_line'],
        row['specimen_id'],
        row['ophys_experiment_id'],
        row['session_type'],
        row['targeted_structure'],
        row['imaging_depth'],
        cell_specimen_id,
        glm_version,
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

    def __init__(self, glm, cell_specimen_id, start_frame, end_frame, frame_interval=1, fps=10, destination_folder=None):

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

        self.model_timestamps = glm.fit['dff_trace_arr']['dff_trace_timestamps'].values
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


        self.results_summary = gat.generate_results_summary(self.glm).reset_index()
        self.dropout_summary_plotted = False
        self.cell_roi_plotted = False

        self.fig, self.ax = self.set_up_axes()
        self.writer = self.set_up_writer()

    def make_cell_movie_frame(self, ax, glm, F_index, cell_specimen_id, t_before=10, t_after=10):
        # ti = time.time()
        this_cell = glm.df_full.query('cell_specimen_id == @cell_specimen_id')
        cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

        t_now = self.model_timestamps[F_index]
        t_span = [t_now - t_before, t_now + t_after]
        # print('setup done at {} seconds'.format(time.time() - ti))
        if not self.dropout_summary_plotted:
            plot_dropout_summary(self.results_summary, self.cell_specimen_id, ax['dropout_summary'])
            self.dropout_summary_plotted = True

        for axis_name in ax.keys():
            if axis_name != 'dropout_summary' and axis_name != 'cell_roi':
                ax[axis_name].cla()

        F_this_frame = glm.df_full.query('frame_index == @F_index').set_index('cell_specimen_id')

        
        # 2P ROI images:
        if not self.cell_roi_plotted:
            # ax['cell_roi'].imshow(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'],cmap='gray')
            self.com = ndimage.measurements.center_of_mass(glm.session.dataset.get_roi_masks().loc[{'cell_specimen_id':cell_specimen_id}].values)
            self.cell_roi_plotted = True


        for movie_name in ['behavior','eye']:
            # what follows is an attempt at adjusting the contrast, but keeping it somewhat constant in a local window
            # if I adjust contrast as the 99th percentile for every frame, it looks flickery
            frame = self.tracking_movies[movie_name].get_frame(time=t_now)[:,:,0]
            frame_n1 = self.tracking_movies[movie_name].get_frame(time=1710-0.035)[:,:,0]
            frame_p1 = self.tracking_movies[movie_name].get_frame(time=1710+0.035)[:,:,0]
            frame_n2 = self.tracking_movies[movie_name].get_frame(time=1710-0.070)[:,:,0]
            frame_p2 = self.tracking_movies[movie_name].get_frame(time=1710+0.070)[:,:,0]
            p99 = np.percentile(
                np.hstack((
                    frame.flatten(),
                    frame_n1.flatten(),
                    frame_p1.flatten(),
                    frame_n2.flatten(),
                    frame_p2.flatten()
                    )),
                99
            )
            ax['{}_movie'.format(movie_name)].imshow(
                frame ,
                # clim=[0,p99], 
                cmap='gray'
            )
            ax['{}_movie'.format(movie_name)].axis('off')
            ax['{}_movie'.format(movie_name)].set_title('{} tracking movie'.format(movie_name))

        real_fov = self.real_2p_movie[F_index]
        cmax = np.percentile(real_fov, 95) #set cmax to 95th percentile of this image
        ax['real_fov'].imshow(real_fov, cmap='gray', clim=[0, cmax])

        ax['real_fov'].set_title('Real FOV')

        for axis_name in ['real_fov']: #,'reconstructed_fov','simulated_fov']:
            ax[axis_name].set_xticks([])
            ax[axis_name].set_yticks([])
            ax[axis_name].axvline(self.com[1],color='MediumAquamarine',alpha=0.5)
            ax[axis_name].axhline(self.com[0],color='MediumAquamarine',alpha=0.5)

        # time series plots:
        query_string = 'dff_trace_timestamps >= {} and dff_trace_timestamps <= {}'.format(
            t_span[0],
            t_span[1]
        )
        local_df = this_cell.query(query_string)

        ax['cell_response'].plot(
            local_df['dff_trace_timestamps'],
            local_df['dff'],
            alpha=0.9,
            color='lightgreen',
            linewidth=3,
        )

        ax['cell_response'].plot(
            local_df['dff_trace_timestamps'],
            local_df['dff_predicted'],
            alpha=1,
            color='white',
            linewidth=3,
        )
        qs = 'dff_trace_timestamps >= {} and dff_trace_timestamps <= {}'.format(
            self.initial_time,
            self.final_time
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

        plot_rewards(glm.session, ax['licks'], t_span=t_span)
        plot_licks(glm.session, ax['licks'], t_span=t_span)
        
        plot_running(glm.session, ax['running'], t_span=t_span)
        plot_pupil(glm.session, ax['pupil'], t_span=t_span)
        plot_kernels(self.kernel_df, ax['kernel_contributions'], self.palette_df, t_span)

        # some axis formatting: 
        for axis_name in ['licks', 'cell_response', 'running','kernel_contributions']:
            ax[axis_name].axvline(t_now, color='white', linewidth=3, alpha=0.5)
            plot_stimuli(glm.session, ax[axis_name], t_span=t_span)
            if axis_name != 'kernel_contributions':
                ax[axis_name].set_xticklabels([])

        ax['cell_response'].set_title('Time series plots for cell {}'.format(cell_specimen_id))
        ax['licks'].set_xlim(t_span[0], t_span[1])
        ax['licks'].set_yticks([])

        ax['cell_response'].set_xticklabels('')

        ax['licks'].set_xlabel('time')

        ax['licks'].set_ylabel('licks/rewards       ', rotation=0,ha='right', va='center')
        ax['cell_response'].set_ylabel('$\Delta$F/F', rotation=0, ha='right', va='center')
        ax['running'].set_ylabel('Running\nSpeed\n(cm/s)', rotation=0, ha='right', va='center')
        ax['pupil'].set_ylabel('Pupil\nDiameter\n(pix^2)', rotation=0, ha='left', va='center')
        ax['kernel_contributions'].set_ylabel('kernel\ncontributions\nto predicted\nsignal\n($\Delta$F/F)', rotation=0, ha='right', va='center')


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
            'dropout_summary':vbp.placeAxesOnGrid(fig, xspan=[0,0.2], yspan=[0,1]),
            'cell_response': vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.30, 0.5]),
            'licks': vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.5, 0.525]),
            'running': vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.525, 0.625]),
            'kernel_contributions':vbp.placeAxesOnGrid(fig, xspan=[0.3, 1], yspan=[0.625, 1]),
            
        }
        ax['pupil'] = ax['running'].twinx()

        ax['licks'].get_shared_x_axes().join(ax['licks'], ax['cell_response'])
        ax['running'].get_shared_x_axes().join(ax['running'], ax['cell_response'])
        ax['kernel_contributions'].get_shared_x_axes().join(ax['kernel_contributions'], ax['cell_response'])

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

def make_level(df, drops, this_level_num,this_level_drops,run_params):
    '''
        Helper function for plot_dropouts()
        Determines what dropout each kernel is a part of, as well as keeping track of which dropouts have been used. 
    '''
    df['level-'+str(this_level_num)] = [get_containing_dictionary(key, this_level_drops,run_params) for key in df.index.values]
    for d in this_level_drops:
        drops.remove(d)
    return df,drops

def plot_dropouts(run_params,save_results=True,num_levels=6):
    '''
        Makes a visual and graphic representation of how the kernels are nested inside dropout models
    '''
    if num_levels==4:
        plt.figure(figsize=(16,8))
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
    for level in np.arange(num_levels,1,-1):
        df,drops = make_level(df,drops, level,  levels[level],  run_params)
        
    # re-organized dataframe
    df=df[['level-'+str(x) for x in range(1,num_levels+1)]]
    df = df.sort_values(by=['level-'+str(x) for x in np.arange(num_levels,0,-1)])
    df['text'] = [run_params['kernels'][k]['text'] for k in df.index.values]
    df['support'] = [(np.round(run_params['kernels'][k]['offset'],2), np.round(run_params['kernels'][k]['length'] +  run_params['kernels'][k]['offset'],2)) for k in df.index.values]

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
    for key in color_dict.keys():
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
    labels = ['Individual Model']+['Minor Component']*(num_levels-3)+['Major Component','Full Model']
    plt.xticks([w*x for x in np.arange(0.5,num_levels+0.5,1)],labels,fontsize=12)
    plt.yticks(np.arange(len(kernels)-0.5,-0.5,-1),aligned_names,ha='left',family='monospace')
    plt.gca().get_yaxis().set_tick_params(pad=400)
    plt.title('Nested Models')
    plt.tight_layout()
    plt.text(-.255,len(kernels)+.35,'Alignment',fontsize=12)
    plt.text(-.385,len(kernels)+.35,'Support',fontsize=12)
    plt.text(-.555,len(kernels)+.35,'Kernel',fontsize=12)
        
    # Save results
    if save_results:
        plt.savefig(run_params['output_dir']+'/nested_models_'+str(num_levels)+'.png')
        df.to_csv(run_params['output_dir']+'/kernels_and_dropouts.csv')
    return df

def kernel_evaluation(weights_df, run_params, kernel, save_results=True,threshold=0.01, drop_threshold=-0.10,normalize=True,drop_threshold_single=False,session_filter=[1,2,3,4,5,6],equipment_filter="all",mode='science',interpolate=True,depth_filter=[0,1000],problem_9c=False,problem_9d=False):
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
        threshold,              the minimum variance explained by the full model
        drop_threshold,         the minimum adj_fraction_change_from_full for the dropout model of just dropping this kernel
        normalize,              if True, normalizes each cell to np.max(np.abs(x))
        drop_threshold_single,  if True, applies drop_threshold to single-<kernel> instead of <kernel> dropout model
        session_filter,         The list of session numbers to include
        equipment_filter,       "scientifica" or "mesoscope" filter, anything else plots both
        mode,                   if "diagnostic" then it plots marina's suggestions for kernel length in red. Otherwise does nothing
        interpolate,            if True, then interpolates mesoscope data onto scientifica timebase. This value is forced to True if plotting a mix of the two datasets. 
        
    '''

    # Filter out Mesoscope and make time basis 
    # Filtering out that one session because something is wrong with it, need to follow up TODO
    version = run_params['version']
    filter_string = ''
    problem_sessions = [962045676, 1048363441,1050231786,1051107431,1051319542,1052096166,1052512524,1052752249,1049240847,1050929040,1052330675]
    if problem_9c:
        problem_sessions = [962045676, 1048363441,1050231786,1051107431,1051319542,1052096166,1052512524,1052752249,1049240847,1050929040,1052330675, 822734832,843871375]
    if problem_9d:
        problem_sessions = [962045676, 1048363441,1050231786,1051107431,1051319542,1052096166,1052512524,1052752249,1049240847,1050929040,1052330675, 873653940, 878436988,883509540,822734832]
    if equipment_filter == "scientifica": 
        weights = weights_df.query('(equipment_name in ["CAM2P.3","CAM2P.4","CAM2P.5"]) & (session_number in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0]) ')
        filter_string+='_scientifica'
    elif equipment_filter == "mesoscope":
        weights = weights_df.query('(equipment_name in ["MESO.1"]) & (session_number in @session_filter) & (ophys_session_id not in @problem_sessions)& (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0]) ')   
        filter_string+='_mesoscope'
    else:
        weights = weights_df.query('(session_number in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])')
        if not interpolate:
            print('Forcing interpolate=True because we have mixed scientifica and mesoscope data')
            interpolate=True

    # Set up time vectors.
    # Mesoscope sessions have not been interpolated onto the right time basis yet
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)
    meso_time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/10.725)
    if (equipment_filter == "mesoscope") & (not interpolate):
        time_vec = meso_time_vec

    if mode == 'diagnostic':
        suggestions = pd.read_csv('/allen/programs/braintv/workgroups/nc-ophys/alex.piet/glm_figs/kernels_and_dropouts_MG_suggestions.csv',engine='python').set_index('Unnamed: 0')
        suggestions.index.name=None
        if isinstance(suggestions.loc[kernel]['MG suggestion'],str):
            x = suggestions.loc[kernel]['MG suggestion']
            start = float(x[1:-1].split(',')[0].replace(' ',''))
            end = float(x[1:-1].split(',')[1].replace(' ',''))
        else: 
            start = time_vec[0]
            end = time_vec[-1]
 
    # Plotting settings
    colors=['C0','C1','C2']
    line_alpha = 0.25
    width=0.25

    # Determine filename
    if not normalize:
        filter_string+='_unnormalized'
    if session_filter != [1,2,3,4,5,6]:
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if mode == "diagnostic":
        filter_string+='_suggestions' 
    if depth_filter !=[0,1000]:
        filter_string+='_depth_'+str(depth_filter[0])+'_'+str(depth_filter[1])
    filename = run_params['output_dir']+'/figures/'+kernel+'_analysis'+filter_string+'.png'

    # Get all cells data and plot Average Trajectories
    fig,ax=plt.subplots(3,3,figsize=(12,9))
    sst_weights = weights.query('cre_line == "Sst-IRES-Cre"')[kernel+'_weights']
    vip_weights = weights.query('cre_line == "Vip-IRES-Cre"')[kernel+'_weights']
    slc_weights = weights.query('cre_line == "Slc17a7-IRES2-Cre"')[kernel+'_weights']
    if normalize:
        sst = [x/np.max(np.abs(x)) for x in sst_weights[~sst_weights.isnull()].values if np.max(np.abs(x)) > 0]
        vip = [x/np.max(np.abs(x)) for x in vip_weights[~vip_weights.isnull()].values if np.max(np.abs(x)) > 0]
        slc = [x/np.max(np.abs(x)) for x in slc_weights[~slc_weights.isnull()].values if np.max(np.abs(x)) > 0]
    else:
        sst = [x for x in sst_weights[~sst_weights.isnull()].values if np.max(np.abs(x)) > 0]
        vip = [x for x in vip_weights[~vip_weights.isnull()].values if np.max(np.abs(x)) > 0]
        slc = [x for x in slc_weights[~slc_weights.isnull()].values if np.max(np.abs(x)) > 0]

    # Interpolate Mesoscope
    # Doing interpolation step here because we have removed the NaN results
    if interpolate:
        sst = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in sst]
        vip = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in vip]
        slc = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in slc]
 
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
    ax[0,0].fill_between(time_vec, sst.mean(axis=0)-sst.std(axis=0), sst.mean(axis=0)+sst.std(axis=0),facecolor=colors[0], alpha=0.1)   
    ax[0,0].fill_between(time_vec, vip.mean(axis=0)-vip.std(axis=0), vip.mean(axis=0)+vip.std(axis=0),facecolor=colors[1], alpha=0.1)    
    ax[0,0].fill_between(time_vec, slc.mean(axis=0)-slc.std(axis=0), slc.mean(axis=0)+slc.std(axis=0),facecolor=colors[2], alpha=0.1)    
    ax[0,0].plot(time_vec, sst.mean(axis=0),label='SST',color=colors[0])
    ax[0,0].plot(time_vec, vip.mean(axis=0),label='VIP',color=colors[1])
    ax[0,0].plot(time_vec, slc.mean(axis=0),label='SLC',color=colors[2])
    ax[0,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[0,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    if mode == 'diagnostic':
        ax[0,0].axvline(start, color='r',linestyle='-')
        ax[0,0].axvline(end, color='r',linestyle='-')
    if normalize:
        ax[0,0].set_ylabel('Weights (Normalized df/f)')
    else:
        ax[0,0].set_ylabel('Weights (df/f)')
    ax[0,0].set_xlabel('Time (s)')
    ax[0,0].legend()
    ax[0,0].set_title('Average kernel')
    if mode == 'diagnostic':
        ax[0,0].set_xlim(np.min([start-.1,time_vec[0]]),np.max([end+.1,time_vec[-1]]))
    else:
        ax[0,0].set_xlim(time_vec[0],time_vec[-1])   
    add_stimulus_bars(ax[0,0],kernel)
    sst = sst.T
    vip = vip.T
    slc = slc.T

    # Get Full model filtered data, and plot average kernels
    sst_weights_filtered = weights.query('cre_line == "Sst-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    vip_weights_filtered = weights.query('cre_line == "Vip-IRES-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    slc_weights_filtered = weights.query('cre_line == "Slc17a7-IRES2-Cre" & variance_explained_full > @threshold')[kernel+'_weights']
    if normalize: 
        sst_f = [x/np.max(np.abs(x)) for x in sst_weights_filtered[~sst_weights_filtered.isnull()].values if np.max(np.abs(x)) > 0]
        vip_f = [x/np.max(np.abs(x)) for x in vip_weights_filtered[~vip_weights_filtered.isnull()].values if np.max(np.abs(x)) > 0]
        slc_f = [x/np.max(np.abs(x)) for x in slc_weights_filtered[~slc_weights_filtered.isnull()].values if np.max(np.abs(x)) > 0]
    else:
        sst_f = [x for x in sst_weights_filtered[~sst_weights_filtered.isnull()].values if np.max(np.abs(x)) > 0]
        vip_f = [x for x in vip_weights_filtered[~vip_weights_filtered.isnull()].values if np.max(np.abs(x)) > 0]
        slc_f = [x for x in slc_weights_filtered[~slc_weights_filtered.isnull()].values if np.max(np.abs(x)) > 0]

    # Interpolate Mesoscope
    if interpolate:
        sst_f = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in sst_f]
        vip_f = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in vip_f]
        slc_f = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in slc_f] 

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
    ax[1,0].fill_between(time_vec, sst_f.mean(axis=0)-sst_f.std(axis=0), sst_f.mean(axis=0)+sst_f.std(axis=0),facecolor=colors[0], alpha=0.1)   
    ax[1,0].fill_between(time_vec, vip_f.mean(axis=0)-vip_f.std(axis=0), vip_f.mean(axis=0)+vip_f.std(axis=0),facecolor=colors[1], alpha=0.1)    
    ax[1,0].fill_between(time_vec, slc_f.mean(axis=0)-slc_f.std(axis=0), slc_f.mean(axis=0)+slc_f.std(axis=0),facecolor=colors[2], alpha=0.1)    
    ax[1,0].plot(time_vec, sst_f.mean(axis=0),label='SST',color=colors[0])
    ax[1,0].plot(time_vec, vip_f.mean(axis=0),label='VIP',color=colors[1])
    ax[1,0].plot(time_vec, slc_f.mean(axis=0),label='SLC',color=colors[2])
    ax[1,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[1,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    if mode == 'diagnostic':
        ax[1,0].axvline(start, color='r',linestyle='-')
        ax[1,0].axvline(end, color='r',linestyle='-')
    if normalize:
        ax[1,0].set_ylabel('Weights (Normalized df/f)')
    else:
        ax[1,0].set_ylabel('Weights (df/f)')
    ax[1,0].set_xlabel('Time (s)')
    ax[1,0].legend()
    ax[1,0].set_title('Filtered on Full Model')
    if mode == 'diagnostic':
        ax[1,0].set_xlim(np.min([start-.1,time_vec[0]]),np.max([end+.1,time_vec[-1]]))
    else:
        ax[1,0].set_xlim(time_vec[0],time_vec[-1])   
    add_stimulus_bars(ax[1,0],kernel)
    sst_f = sst_f.T
    vip_f = vip_f.T
    slc_f = slc_f.T

    # Get Dropout filtered data, and plot average kernels
    if drop_threshold_single:
        sst_weights_dfiltered = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold)')
        vip_weights_dfiltered = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold)')
        slc_weights_dfiltered = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold)')
        sst_weights_dfiltered = sst_weights_dfiltered[sst_weights_dfiltered['single-'+kernel] < drop_threshold][kernel+'_weights']
        vip_weights_dfiltered = vip_weights_dfiltered[vip_weights_dfiltered['single-'+kernel] < drop_threshold][kernel+'_weights']
        slc_weights_dfiltered = slc_weights_dfiltered[slc_weights_dfiltered['single-'+kernel] < drop_threshold][kernel+'_weights']
    else:
        sst_weights_dfiltered = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel+'_weights']
        vip_weights_dfiltered = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel+'_weights']
        slc_weights_dfiltered = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))[kernel+'_weights']

    if normalize:
        sst_df = [x/np.max(np.abs(x)) for x in sst_weights_dfiltered[~sst_weights_dfiltered.isnull()].values]

        vip_df = [x/np.max(np.abs(x)) for x in vip_weights_dfiltered[~vip_weights_dfiltered.isnull()].values]
        slc_df = [x/np.max(np.abs(x)) for x in slc_weights_dfiltered[~slc_weights_dfiltered.isnull()].values]
    else:
        sst_df = [x for x in sst_weights_dfiltered[~sst_weights_dfiltered.isnull()].values]
        vip_df = [x for x in vip_weights_dfiltered[~vip_weights_dfiltered.isnull()].values]
        slc_df = [x for x in slc_weights_dfiltered[~slc_weights_dfiltered.isnull()].values] 

    # Interpolate Mesoscope
    if interpolate:
        sst_df = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in sst_df]
        vip_df = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in vip_df]
        slc_df = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in slc_df] 
 
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

    ax[2,0].fill_between(time_vec, sst_df.mean(axis=0)-sst_df.std(axis=0), sst_df.mean(axis=0)+sst_df.std(axis=0),facecolor=colors[0], alpha=0.1)   
    ax[2,0].fill_between(time_vec, vip_df.mean(axis=0)-vip_df.std(axis=0), vip_df.mean(axis=0)+vip_df.std(axis=0),facecolor=colors[1], alpha=0.1)    
    ax[2,0].fill_between(time_vec, slc_df.mean(axis=0)-slc_df.std(axis=0), slc_df.mean(axis=0)+slc_df.std(axis=0),facecolor=colors[2], alpha=0.1)    
    ax[2,0].plot(time_vec, sst_df.mean(axis=0),label='SST',color=colors[0])
    ax[2,0].plot(time_vec, vip_df.mean(axis=0),label='VIP',color=colors[1])
    ax[2,0].plot(time_vec, slc_df.mean(axis=0),label='SLC',color=colors[2])
    ax[2,0].axhline(0, color='k',linestyle='--',alpha=line_alpha)
    ax[2,0].axvline(0, color='k',linestyle='--',alpha=line_alpha)
    if mode == 'diagnostic':
        ax[2,0].axvline(start, color='r',linestyle='-')
        ax[2,0].axvline(end, color='r',linestyle='-')
    if normalize:
        ax[2,0].set_ylabel('Weights (Normalized df/f)')   
    else:
        ax[2,0].set_ylabel('Weights (df/f)')
    ax[2,0].set_xlabel('Time (s)')
    ax[2,0].legend()
    ax[2,0].set_title('Filtered on Dropout')
    if mode == 'diagnostic':
        ax[2,0].set_xlim(np.min([start-.1,time_vec[0]]),np.max([end+.1,time_vec[-1]]))
    else:
        ax[2,0].set_xlim(time_vec[0],time_vec[-1])   
    add_stimulus_bars(ax[2,0],kernel)
    sst_df = sst_df.T
    vip_df = vip_df.T
    slc_df = slc_df.T

    # Plot Heat maps
    sst_sorted = sst[:,np.argsort(np.argmax(sst,axis=0))]
    vip_sorted = vip[:,np.argsort(np.argmax(vip,axis=0))]
    slc_sorted = slc[:,np.argsort(np.argmax(slc,axis=0))]
    weights_sorted = np.hstack([slc_sorted,sst_sorted, vip_sorted])
    cbar = ax[0,1].imshow(weights_sorted.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted)[1]],cmap='bwr')
    cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted),95),np.nanpercentile(np.abs(weights_sorted),95))
    color_bar=fig.colorbar(cbar, ax=ax[0,1])
    if normalize:
        color_bar.ax.set_ylabel('Normalized Weights')
    else:
        color_bar.ax.set_ylabel('Weights')   
    ax[0,1].set_ylabel('{0} Cells'.format(np.shape(weights_sorted)[1]))
    ax[0,1].set_xlabel('Time (s)')
    ax[0,1].axhline(np.shape(vip)[1],color='k',linewidth='1')
    ax[0,1].axhline(np.shape(vip)[1] + np.shape(sst)[1],color='k',linewidth='1')
    ax[0,1].set_yticks([np.shape(vip)[1]/2,np.shape(vip)[1]+np.shape(sst)[1]/2, np.shape(vip)[1]+np.shape(sst)[1]+np.shape(slc)[1]/2])
    ax[0,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[0,1].set_title(kernel)

    # Plot Heatmap of filtered cells
    sst_sorted_f = sst_f[:,np.argsort(np.argmax(sst_f,axis=0))]
    vip_sorted_f = vip_f[:,np.argsort(np.argmax(vip_f,axis=0))]
    slc_sorted_f = slc_f[:,np.argsort(np.argmax(slc_f,axis=0))]
    weights_sorted_f = np.hstack([slc_sorted_f,sst_sorted_f, vip_sorted_f])
    cbar = ax[1,1].imshow(weights_sorted_f.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_f)[1]],cmap='bwr')
    cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted_f),95),np.nanpercentile(np.abs(weights_sorted_f),95))
    color_bar = fig.colorbar(cbar, ax=ax[1,1])
    if normalize:
        color_bar.ax.set_ylabel('Normalized Weights')
    else:
        color_bar.ax.set_ylabel('Weights')   
    ax[1,1].set_ylabel('{0} Cells'.format(np.shape(weights_sorted_f)[1]))
    ax[1,1].set_xlabel('Time (s)')
    ax[1,1].axhline(np.shape(vip_f)[1],color='k',linewidth='1')
    ax[1,1].axhline(np.shape(vip_f)[1] + np.shape(sst_f)[1],color='k',linewidth='1')
    ax[1,1].set_yticks([np.shape(vip_f)[1]/2,np.shape(vip_f)[1]+np.shape(sst_f)[1]/2, np.shape(vip_f)[1]+np.shape(sst_f)[1]+np.shape(slc_f)[1]/2])
    ax[1,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[1,1].set_title('Filtered on Full Model')

    # Plot Heatmap of filtered cells
    sst_sorted_df = sst_df[:,np.argsort(np.argmax(sst_df,axis=0))]
    vip_sorted_df = vip_df[:,np.argsort(np.argmax(vip_df,axis=0))]
    slc_sorted_df = slc_df[:,np.argsort(np.argmax(slc_df,axis=0))]
    weights_sorted_df = np.hstack([slc_sorted_df,sst_sorted_df, vip_sorted_df])
    cbar = ax[2,1].imshow(weights_sorted_df.T,aspect='auto',extent=[time_vec[0], time_vec[-1], 0, np.shape(weights_sorted_df)[1]],cmap='bwr')
    cbar.set_clim(-np.nanpercentile(np.abs(weights_sorted_df),95),np.nanpercentile(np.abs(weights_sorted_df),95))
    color_bar = fig.colorbar(cbar, ax=ax[2,1])
    if normalize:
        color_bar.ax.set_ylabel('Normalized Weights')
    else:
        color_bar.ax.set_ylabel('Weights')   
    ax[2,1].set_ylabel('{0} Cells'.format(np.shape(weights_sorted_df)[1]))
    ax[2,1].set_xlabel('Time (s)')
    ax[2,1].axhline(np.shape(vip_df)[1],color='k',linewidth='1')
    ax[2,1].axhline(np.shape(vip_df)[1] + np.shape(sst_df)[1],color='k',linewidth='1')
    ax[2,1].set_yticks([np.shape(vip_df)[1]/2,np.shape(vip_df)[1]+np.shape(sst_df)[1]/2, np.shape(vip_df)[1]+np.shape(sst_df)[1]+np.shape(slc_df)[1]/2])
    ax[2,1].set_yticklabels(['Vip','Sst','Slc'])
    ax[2,1].set_title('Filtered on Dropout')

    ## Right Column, Dropout Scores 
    # Make list of dropouts that contain this kernel
    drop_list = [d for d in run_params['dropouts'].keys() if (
                    (run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['kernels'])) 
                    or ((not run_params['dropouts'][d]['is_single']) & (kernel in run_params['dropouts'][d]['dropped_kernels']))]
    medianprops = dict(color='k')
    
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
        for patch, color in zip(drops['boxes'],colors):
            patch.set_facecolor(color)

    # Clean up plot
    num_cells = len(drop_sst)+len(drop_slc)+len(drop_vip)
    ax[0,2].set_ylabel('Adj. Fraction from Full \n'+str(num_cells)+' cells')
    ax[0,2].set_xticks(np.arange(0,len(drop_list)))
    ax[0,2].set_xticklabels(drop_list,rotation=60,fontsize=8,ha='right')
    ax[0,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
    ax[0,2].set_ylim(-1.05,.05)
    ax[0,2].set_title('Dropout Scores')

    # Filtered by Full model
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
        for patch, color in zip(drops['boxes'],colors):
            patch.set_facecolor(color)

    # Clean up plot
    num_cells = len(drop_sst)+len(drop_slc)+len(drop_vip)
    ax[1,2].set_ylabel('Adj. Fraction from Full \n'+str(num_cells)+' cells')
    ax[1,2].set_xticks(np.arange(0,len(drop_list)))
    ax[1,2].set_xticklabels(drop_list,rotation=60,fontsize=8,ha='right')
    ax[1,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
    ax[1,2].set_ylim(-1.05,.05)
    ax[1,2].set_title('Filter on Full Model')

    # Filtered by Dropout Score
    # For each dropout, plot score
    for index, dropout in enumerate(drop_list):
        if drop_threshold_single:
            drop_sst = weights.query('(cre_line == "Sst-IRES-Cre") & (variance_explained_full > @threshold)')
            drop_vip = weights.query('(cre_line == "Vip-IRES-Cre") & (variance_explained_full > @threshold)')
            drop_slc = weights.query('(cre_line == "Slc17a7-IRES2-Cre") & (variance_explained_full > @threshold)')
            drop_sst = drop_sst[drop_sst['single-'+kernel] < drop_threshold][dropout].values
            drop_vip = drop_vip[drop_vip['single-'+kernel] < drop_threshold][dropout].values
            drop_slc = drop_slc[drop_slc['single-'+kernel] < drop_threshold][dropout].values
        else:
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
        for patch, color in zip(drops['boxes'],colors):
            patch.set_facecolor(color)

    # Clean Up Plot
    num_cells = len(drop_sst)+len(drop_slc)+len(drop_vip)
    ax[2,2].set_ylabel('Adj. Fraction from Full \n'+str(num_cells)+' cells')
    ax[2,2].set_xticks(np.arange(0,len(drop_list)))
    ax[2,2].set_xticklabels(drop_list,rotation=60,fontsize=8,ha='right')
    ax[2,2].axhline(0,color='k',linestyle='--',alpha=line_alpha)
    ax[2,2].set_ylim(-1.05,.05)
    ax[2,2].set_title('Filter on Dropout Score')
    ax[2,2].axhline(drop_threshold, color='r',linestyle='--', alpha=line_alpha)

    
    ## Final Clean up and Save
    plt.tight_layout()
    if save_results:
        print('Figure Saved to: '+filename)
        plt.savefig(filename)

def all_kernels_evaluation(weights_df, run_params,threshold=0.01, drop_threshold=-0.10,normalize=True, drop_threshold_single=False,session_filter=[1,2,3,4,5,6],equipment_filter="all",mode='science',depth_filter=[0,1000]):
    '''
        Makes the analysis plots for all kernels in this model version. Excludes intercept and time kernels
                
        INPUTS:
        Same as kernel_evaluation
        
        SAVES:
        a figure for each kernel    

    '''
    kernels = set(run_params['kernels'].keys())
    kernels.remove('intercept')
    kernels.remove('time')
    crashed = set()
    for k in kernels:
        try:
            kernel_evaluation(weights_df, run_params, k, save_results=True,
                threshold=threshold, drop_threshold=drop_threshold,
                normalize=normalize,drop_threshold_single=drop_threshold_single,
                session_filter=session_filter, equipment_filter=equipment_filter,mode=mode,depth_filter=depth_filter)
            plt.close(plt.gcf().number)
        except:
            crashed.add(k)
            plt.close(plt.gcf().number)

    for k in crashed:
        print('Crashed - '+k) 

def add_stimulus_bars(ax, kernel):
    '''
        Adds stimulus bars to the given axis, but only for certain kernels 
    '''
    # Check if this is an image aligned kernel
    if kernel in ['change','hits','misses','false_alarms','omissions','image_expectation','image0','image1','image2','image3','image4','image5','image6','image7']:
        # Define timepoints of stimuli
        lims = ax.get_xlim()
        times = set(np.concatenate([np.arange(0,lims[1],0.75),np.arange(-0.75,lims[0]-0.001,-0.75)]))
        if kernel == 'omissions':
            # For omissions, remove omitted stimuli
            times.remove(0.0)
        if kernel in ['change','hits','misses','false_alarms']:
            # For change aligned kernels, plot the two stimuli different colors
            for flash_start in times:
                if flash_start < 0:
                    ax.axvspan(flash_start,flash_start+0.25,color='green',alpha=0.25,zorder=-np.inf)                   
                else:
                    ax.axvspan(flash_start,flash_start+0.25,color='blue',alpha=0.25,zorder=-np.inf)                   
        else:
            # Normal case, just plot all the same color
            for flash_start in times:
                ax.axvspan(flash_start,flash_start+0.25,color='blue',alpha=0.25,zorder=-np.inf)
         
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
        fig, ax = plt.subplots(1,2,figsize=(8,4))   
    else:
        fig, ax = plt.subplots(1,3,figsize=(12,4))
    
    # First panel, relationship between variance explained and overfitting proportion
    ax[0].plot(full_results[dropout+'__avg_cv_var_test'], full_results[dropout+'__over_fit'],'ko',alpha=.1)
    ax[0].set_xlim(0,1)
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Overfitting Proportion: '+dropout)
    ax[0].set_xlabel('Test Variance Explained')
    
    # Second panel, histogram of overfitting proportion, with mean/median marked
    hist_output = ax[1].hist(full_results[dropout+'__over_fit'],100)
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,1.25*np.max(hist_output[0][:-1]))
    ax[1].plot(np.mean(full_results[dropout+'__over_fit']), 1.1*np.max(hist_output[0][:-1]),'rv',markerfacecolor='none',label='Mean All Cells')
    ax[1].plot(np.mean(full_results[dropout+'__over_fit'][full_results[dropout+'__over_fit']<1]), 1.1*np.max(hist_output[0][:-1]),'rv',label='Mean Exclude overfit=1 cells')
    ax[1].plot(np.median(full_results[dropout+'__over_fit']), 1.1*np.max(hist_output[0][:-1]),'bv',markerfacecolor='none',label='Median All Cells')
    ax[1].plot(np.median(full_results[dropout+'__over_fit'][full_results[dropout+'__over_fit']<1]), 1.1*np.max(hist_output[0][:-1]),'bv',label='Median Exclude overfit=1 cells')
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel('Overfitting Proportion: '+dropout)
    ax[1].legend(loc='lower right')
    
    # Third panel, distribution of dropout_overfitting_proportion compared to full model
    if dropout != "Full":
        ax[2].hist(full_results[dropout+'__dropout_overfit_proportion'].where(lambda x: (x<1)&(x>-1)),100)
        ax[2].axvline(full_results[dropout+'__dropout_overfit_proportion'].where(lambda x: (x<1)&(x>-1)).median(),color='r',linestyle='--')
        ax[2].set_xlim(-1,1)

    # Clean up and save
    plt.tight_layout()
    if save_file !="":
        plt.savefig(save_file+dropout+'.png')

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
    plt.savefig(run_params['output_dir']+'/figures/over_fitting_figures/over_fitting_summary.png')

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
            plot_over_fitting(full_results, d,save_file=run_params['output_dir']+'/figures/over_fitting_figures/')
        except:
            # Plot crashed for some reason, print error and move on
            print('crashed - '+d)

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
        filename = run_params['output_dir']+'/figures/nested_dropouts/pie'+str(num_levels)
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


def cosyne_make_dropout_summary_plot(dropout_summary):
    '''
        Top level function for cosyne summary plot 
        Plots the distribution of dropout scores by cre-line for the visual, behavioral, and cognitive nested models
    '''
    fig, ax = plt.subplots(figsize=(10,8))
    dropouts_to_show = ['visual','behavioral','cognitive']
    plot_dropout_summary_cosyne(dropout_summary, ax, dropouts_to_show)
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

def plot_dropout_summary_cosyne(dropout_summary, ax, dropouts_to_show):
    '''
    makes bar plots of results summary
    aesthetics specific to cosyne needs

    '''

    cre_lines = np.sort(dropout_summary['cre_line'].unique())
    
    data_to_plot = dropout_summary.query('dropout in @dropouts_to_show and absolute_change_from_full < -0.01').copy()
    data_to_plot['explained_variance'] = -1*data_to_plot['adj_fraction_change_from_full']
    sns.boxplot(
        data = data_to_plot,
        x='dropout',
        y='adj_fraction_change_from_full',
        hue='cre_line',
        order=dropouts_to_show,
        hue_order=cre_lines,
        fliersize=0,
        ax=ax
    )

    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_ylabel('Fraction change\nin variance explained')

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



def make_cosyne_summary_figure(glm, cell_specimen_id, t_span,alpha=0.35):
    '''
    makes a summary figure for cosyne abstract
    inputs:
        glm: glm object
        cell_specimen_id
        time_to_plot: time to show in center of plot for time-varying axes
        t_span: time range to show around time_to_plot, in seconds
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
    dropout_df = plot_dropouts(run_params)
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
        plot_stimuli(glm.session, ax['{}_kernels'.format(regressor_category)], t_span=t_span)
        ax['{}_kernels'.format(regressor_category)].set_title('{}'.format(regressor_category))
        ax['{}_kernels'.format(regressor_category)].set_ylim(
            kernel_df.query('timestamps >= @t0 and timestamps <= @t1')['kernel_outputs'].min()-0.2,
            kernel_df.query('timestamps >= @t0 and timestamps <= @t1')['kernel_outputs'].max()+0.2
        )


    # cell df/f plots:

    this_cell = glm.df_full.query('cell_specimen_id == @cell_specimen_id')
    cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

    query_string = 'dff_trace_timestamps >= {} and dff_trace_timestamps <= {}'.format(
        t_span[0],
        t_span[1]
    )
    local_df = this_cell.query(query_string)

    ax['cell_response'].plot(
        local_df['dff_trace_timestamps'],
        local_df['dff'],
        alpha=0.9,
        color='darkgreen',
        linewidth=3,
    )

    ax['cell_response'].plot(
        local_df['dff_trace_timestamps'],
        local_df['dff_predicted'],
        alpha=1,
        color='black',
        linewidth=3,
    )
    qs = 'dff_trace_timestamps >= {} and dff_trace_timestamps <= {}'.format(
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

    plot_stimuli(glm.session, ax['cell_response'], t_span=t_span,alpha=alpha)

    return fig, ax


def plot_all_coding_fraction(results_pivoted, run_params,threshold=-.1):
    
    fail = []
    
    # Set up which sessions to plot
    active_only  = ['licks','hits','misses','false_alarms','correct_rejects', 'model_bias','model_task0','model_omissions1','model_timing1D','beh_model','licking']
    passive_only = ['passive_change']
    active_only  = active_only+['single-'+x for x in active_only]
    passive_only = passive_only+['single-'+x for x in passive_only]

    # Iterate over list of dropouts
    for dropout in run_params['dropouts']:
        try:
            if dropout == 'Full':
                continue

            session ='all'
            if dropout in active_only:
                session = 'active'
            elif dropout in passive_only:
                session = 'passive'

            plot_coding_fraction(results_pivoted, dropout,threshold=threshold,savefile=run_params['output_dir']+'/figures/',sessions=session)
        except:
            fail.append(dropout)
        plt.close(plt.gcf().number)
    
    if len(fail) > 0:
        print(fail)

def plot_coding_fraction(results_pivoted, dropout,threshold=-.1,savefig=True,savefile='',sessions='all'):
    
    # Dumb stability thing because pandas doesnt like '-' in column names
    if '-' in dropout:
        old_dropout = dropout
        dropout = dropout.replace('-','_')
        if old_dropout in results_pivoted:
            results_pivoted = results_pivoted.rename({old_dropout:dropout},axis=1)

    # Set up indexing
    conditions  =['cre_line','session_number']

    # Get Total number of cells
    num_cells   = results_pivoted.groupby(conditions)['Full'].count()

    # Get number of cells with significant coding
    filter_str  = dropout +' < @threshold'
    sig_cells   = results_pivoted.query(filter_str).groupby(conditions)['Full'].count()

    # Get fraction significant
    fraction    = sig_cells/num_cells
    
    # Build dataframe
    fraction    = fraction.rename('fraction')
    sig_cells   = sig_cells.rename('num_sig')
    num_cells   = num_cells.rename('num_cells')
    df = pd.concat([num_cells, sig_cells, fraction],axis=1)

    # plot
    plt.figure(figsize=(8,4))
    plt.ylabel('% of cells with \n '+dropout+' coding',fontsize=18)
    plt.xlabel('Session',fontsize=18)
    plt.tick_params(axis='both',labelsize=16)

    # Determine what sessions to plot
    if sessions == 'active':
        # Active only
        plt.xticks([0,1,2,3],['F1','F3','N1','N3'],fontsize=18)
        df = df.drop(index=[2.0,5.0], level=1)
    elif sessions == 'passive':
        # Passive only
        plt.xticks([0,1],['F2','N2'],fontsize=18)
        df = df.drop(index=[1.0,3.0,4.0,6.0], level=1)
    else:
        # All sessions
        plt.xticks([0,1,2,3,4,5],['F1','F2','F3','N1','N2','N3'],fontsize=18)

    # Set up color scheme for each cre line
    cre_lines = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre'] 
    colors = {
        'Sst-IRES-Cre':(158/255,218/255,229/255),
        'Slc17a7-IRES2-Cre':(255/255,152/255,150/255),
        'Vip-IRES-Cre':(197/255,176/255,213/255)
        }
    
    # Iterate over cre-lines
    for dex, cre in enumerate(cre_lines):
        plot_coding_fraction_inner(plt.gca(), df.loc[cre], colors[cre],cre)
    
    # Clean up plot
    plt.legend(loc='upper left',bbox_to_anchor=(1.05,1),title='Cre Line')
    plt.tight_layout()
    
    # Save figure
    if savefig:
        savefile = savefile+'coding_fraction_'+dropout+'.png'
        plt.savefig(savefile)
    
    # return coding dataframe
    return df 

def plot_coding_fraction_inner(ax,df,color,label):
    '''
        plots the fraction of significant cells with 95% binomial error bars    
    '''   
    # unpack fraction and get confidence interval
    frac = df['fraction'].values 
    num  = df['num_cells'].values
    se   = 1.98*np.sqrt(frac*(1-frac)/num)
    
    # convert to percentages
    frac = frac*100
    se   = se*100
    plt.plot(range(0,len(frac)), frac,'o-',color=color,linewidth=4,label=label)
    for dex, val in enumerate(zip(frac,se)):
        plt.plot([dex,dex],[val[0]+val[1],val[0]-val[1]], 'k',linewidth=1)



