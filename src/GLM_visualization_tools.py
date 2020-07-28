import visual_behavior.utilities as vbu
import visual_behavior.plotting as vbp
import visual_behavior.visualization.utils as vis_utils
import visual_behavior.data_access.loading as loading

import visual_behavior_glm.src.GLM_analysis_tools as gat

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from tqdm import tqdm

from matplotlib import animation, rc
import matplotlib.pyplot as plt
import gc
from scipy import ndimage


def compare_var_explained(results=None, fig=None, ax=None, figsize=(8,6), outlier_threshold=1.5):
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
        fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=results,
        x='glm_version',
        y='Full_avg_cv_var_test',
        hue='cre_line',
        fliersize=0,
        whis=outlier_threshold,
        ax=ax,
    )
    ax.set_ylabel('variance explained')
    ax.set_xlabel('GLM version')
    ax.set_title('variance explained by GLM version and cre_line (outliers removed)')

    # calculate interquartile ranges
    grp = results.groupby(['glm_version','cre_line'])['Full_avg_cv_var_test']
    IQR = grp.quantile(0.75) - grp.quantile(0.25)


    lower_bounds = grp.quantile(0.25) - 1.5*IQR
    upper_bounds = grp.quantile(0.75) + 1.5*IQR

    ax.set_ylim(lower_bounds.min()-0.05 ,upper_bounds.max()+0.05)
    ax.axhline(0, color='black', linestyle=':')

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
        'ok',
        alpha=0.5
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
        color='blue',
        linewidth=3
    )

def plot_pupil(session, ax, t_span=None):
    '''shares axis with running'''
    vbp.initialize_legend(ax=ax, colors=['blue','black'],linewidth=3)
    if t_span:
        pupil_df = session.dataset.eye_tracking.query(
            'time >= {} and time <= {}'.format(t_span[0], t_span[1]))
    else:
        pupil_df = session.dataset.eye_tracking
    ax.plot(
        pupil_df['time'],
        pupil_df['pupil_area'],
        color='black',
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


def plot_stimuli(session, ax, t_span=None):
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
            alpha=0.5
        )


def build_simulated_FOV(session, F_dataframe, column):

    assert len(session.cell_specimen_table) == len(F_dataframe)

    arr = np.zeros_like(session.dataset.max_projection)
    for ii, cell_specimen_id in enumerate(session.dataset.cell_specimen_ids):

        F_cell = F_dataframe.loc[cell_specimen_id][column]
        arr += session.cell_specimen_table.loc[cell_specimen_id]['image_mask']*F_cell

    return arr


def plot_kernels(kernel_df,ax,t_span=None):
    # kernels_to_exclude_from_plot = []#['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    # kernels_to_exclude_from_plot = ['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    kernels_to_exclude_from_plot = ['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']

    if t_span:
        t0,t1 = t_span
        data_to_plot = kernel_df.query('timestamps >= @t0 and timestamps <= @t1 and kernel_name not in @kernels_to_exclude_from_plot')
    else:
        data_to_plot = kernel_df.query('kernel_name not in @kernels_to_exclude_from_plot')
    palette = vbp.generate_random_colors(
        len(data_to_plot['kernel_name'].unique()), 
        lightness_range=(0.1,.65), 
        saturation_range=(0.5,1), 
        random_seed=3, 
        order_colors=False
    )
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
    ax.legend(
        data_to_plot['kernel_name'].unique(),
        loc='upper left',
        ncol=10, 
        mode="expand", 
        framealpha = 0.5,
    )
    # plt.setp(ax.lines,linewidth=4)


def plot_dropout_summary(results_summary, cell_specimen_id, ax):
    '''
    makes bar plots of results summary
    inputs:
        glm -- glm object
        cell_specimen_id -- cell to plot
        ax -- a vector of three matplotlib axis handles
    '''
    data_to_plot = (
        results_summary
        .query('cell_specimen_id == @cell_specimen_id')
        .sort_values(by='fraction_change_from_full', ascending=False)
    )

    mixed_dropout_color = 'DimGray'
    special_dropout_colors = {
        'Full':'DarkGreen',
        'beh_model':mixed_dropout_color,
        'all-images':mixed_dropout_color,
        'visual':mixed_dropout_color,
        
    }
    palette = [special_dropout_colors[key] if key in special_dropout_colors else 'black' for key in data_to_plot['dropout']]

    sns.barplot(
        data = data_to_plot,
        x = 'variance_explained',
        y = 'dropout',
        ax=ax[0],
        palette=palette
    )
    sns.barplot(
        data = data_to_plot,
        x = 'absolute_change_from_full',
        y = 'dropout',
        ax=ax[1],
        palette=palette
    )
    sns.barplot(
        data = data_to_plot,
        x = 'fraction_change_from_full',
        y = 'dropout',
        ax=ax[2],
        palette=palette
    )
    ax[0].set_title('variance explained\nfor each model dropout')
    ax[1].set_title('absolute change\nin variance explained')
    ax[2].set_title('fractional change\nin variance explained')
    for col in [1,2]:
        ax[col].set_yticklabels([])
        ax[col].set_ylabel('')

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
                t = np.arange(
                    0,
                    glm.design.kernel_dict[kernel_name]['kernel_length_samples']/glm.fit['ophys_frame_rate'],
                    1/glm.fit['ophys_frame_rate']
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


def get_title(ophys_experiment_id, cell_specimen_id):
    '''
    generate a standardized figure title containing identifying information
    '''
    experiments_table = loading.get_filtered_ophys_experiment_table().reset_index()

    row = experiments_table.query('ophys_experiment_id == @ophys_experiment_id').iloc[0].to_dict()
    title = '{}__specimen_id={}__exp_id={}__{}__{}__depth={}__cell_id={}'.format(
        row['cre_line'],
        row['specimen_id'],
        row['ophys_experiment_id'],
        row['session_type'],
        row['targeted_structure'],
        row['imaging_depth'],
        cell_specimen_id,
    )
    return title

class GLM_Movie(object):

    def __init__(self, glm, cell_specimen_id, start_frame, end_frame, frame_interval=1, fps=10):

        # note that ffmpeg must be installed on your system
        # this is tested on linux (not sure if it works on windows)
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

        plt.style.use('ggplot')

        self.glm = glm
        self.cell_specimen_id = cell_specimen_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_interval = frame_interval

        self.title = get_title(self.glm.oeid, self.cell_specimen_id)

        self.kernel_df = gat.build_kernel_df(self.glm, self.cell_specimen_id)

        self.real_2p_movie = loading.load_motion_corrected_movie(self.glm.oeid)

        self.frames = np.arange(self.start_frame, self.end_frame, self.frame_interval)
        self.fps = fps

        self.results_summary = gat.generate_results_summary(self.glm)
        self.dropout_summary_plotted = False
        self.cell_roi_plotted = False

        self.fig, self.ax = self.set_up_axes()
        self.writer = self.set_up_writer()

    def make_cell_movie_frame(self, ax, glm, F_index, cell_specimen_id, t_before=10, t_after=10):
        # ti = time.time()
        this_cell = glm.df_full.query('cell_specimen_id == @cell_specimen_id')
        cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

        model_timestamps = glm.fit['dff_trace_arr']['dff_trace_timestamps'].values
        t_now = model_timestamps[F_index]
        t_span = [t_now - t_before, t_now + t_after]
        # print('setup done at {} seconds'.format(time.time() - ti))
        if not self.dropout_summary_plotted:
            plot_dropout_summary(self.results_summary, self.cell_specimen_id, ax['dropout_summary'])
            self.dropout_summary_plotted = True

        for axis_name in ax.keys():
            if axis_name != 'dropout_summary' and axis_name != 'cell_roi':
                ax[axis_name].cla()

        # print('setup done at {} seconds'.format(time.time() - ti))
        F_this_frame = glm.df_full.query('frame_index == @F_index').set_index('cell_specimen_id')
        # dff_actual = dft.loc[glm.W['cell_specimen_id'].values]['dff'].values
        # dff_pred = dft.loc[glm.W['cell_specimen_id'].values]['dff_predicted'].values
        
        # 2P ROI images:
        if not self.cell_roi_plotted:
            ax['cell_roi'].imshow(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'],cmap='gray')
            self.com = ndimage.measurements.center_of_mass(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'])
            self.cell_roi_plotted = True

        reconstructed_fov = build_simulated_FOV(glm.session, F_this_frame, 'dff')
        ax['reconstructed_fov'].imshow(reconstructed_fov, cmap='seismic', clim=[-0.5, .5])

        simulated_fov = build_simulated_FOV(glm.session, F_this_frame, 'dff_predicted')
        ax['simulated_fov'].imshow(simulated_fov, cmap='seismic', clim=[-0.5, .5])

        real_fov = self.real_2p_movie[F_index]
        ax['real_fov'].imshow(real_fov, cmap='gray', clim=[0, 15000])

        ax['cell_roi'].set_title('ROI mask for cell {}'.format(cell_specimen_id))
        ax['reconstructed_fov'].set_title('Reconstructed FOV')
        ax['simulated_fov'].set_title('Simulated FOV')
        ax['real_fov'].set_title('Real FOV')

        for axis_name in ['cell_roi','real_fov','reconstructed_fov','simulated_fov']:
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
            alpha=0.5,
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

        ax['cell_response'].legend(
            ['Actual $\Delta$F/F','Model Predicted $\Delta$F/F'],
            loc='upper left',
            ncol=2, 
            framealpha = 0.2,
        )

        plot_licks(glm.session, ax['licks'], t_span=t_span)
        plot_running(glm.session, ax['running'], t_span=t_span)
        plot_pupil(glm.session, ax['pupil'], t_span=t_span)
        plot_kernels(self.kernel_df, ax['kernel_contributions'], t_span)

        # some axis formatting: 
        for axis_name in ['licks', 'cell_response', 'running','kernel_contributions']:
            ax[axis_name].axvline(t_now, color='black', linewidth=3, alpha=0.5)
            plot_stimuli(glm.session, ax[axis_name], t_span=t_span)
            if axis_name is not 'kernel_contributions':
                ax[axis_name].set_xticklabels([])

        # ax['running'].set_ylim(
        #     self.glm.session.dataset.running_data_df['speed'].min(),
        #     self.glm.session.dataset.running_data_df['speed'].max()
        # )
        # ax['pupil'].set_ylim(
        #     self.glm.session.dataset.eye_tracking['pupil_area'].min(),
        #     self.glm.session.dataset.eye_tracking['pupil_area'].max()
        # )

        # ax['cell_response'].set_ylim(
        #     glm.df_full['dff_predicted'].min(),
        #     glm.df_full['dff_predicted'].max()
        # )

        ax['cell_response'].set_title('Time series plots for cell {}'.format(cell_specimen_id))
        ax['licks'].set_xlim(t_span[0], t_span[1])
        ax['licks'].set_yticks([])

        ax['cell_response'].set_xticklabels('')

        ax['licks'].set_xlabel('time')

        ax['licks'].set_ylabel('licks       ', rotation=0,ha='right', va='center')
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
        fig = plt.figure(figsize=(24, 18))
        ax = {
            'cell_roi': vbp.placeAxesOnGrid(fig, xspan=(0, 0.25), yspan=(0, 0.25)),
            'real_fov': vbp.placeAxesOnGrid(fig, xspan=(0.25, 0.5), yspan=(0, 0.25)),
            'reconstructed_fov': vbp.placeAxesOnGrid(fig, xspan=(0.5, 0.75), yspan=(0, 0.25)),
            'simulated_fov': vbp.placeAxesOnGrid(fig, xspan=(0.75, 1), yspan=(0, 0.25)),
            'cell_response': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.3, 0.45]),
            'licks': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.45, 0.475]),
            'running': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.475, 0.575]),
            'kernel_contributions':vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.575, 0.70]),
            'dropout_summary':vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.775, 1], dim=[1,3], wspace=0.01),
        }
        ax['pupil'] = ax['running'].twinx()

        ax['licks'].get_shared_x_axes().join(ax['licks'], ax['cell_response'])
        ax['running'].get_shared_x_axes().join(ax['running'], ax['cell_response'])
        ax['kernel_contributions'].get_shared_x_axes().join(ax['kernel_contributions'], ax['cell_response'])

        variance_explained_string = 'Variance explained (full model) = {:0.1f}%'.format(100*self.glm.results.loc[self.cell_specimen_id]['Full_avg_cv_var_test'])
        fig.suptitle(self.title+'\n'+variance_explained_string)

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

        base_path = self.glm.run_params['output_dir'].split('/v_')[0]
        save_folder = os.path.join(base_path, 'output_files')
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)

        filename = self.title+'_frame_{}_to_{}.mp4'.format(self.start_frame, self.end_frame)

        with tqdm(total=len(self.frames)) as self.pbar:
            a.save(
                os.path.join(save_folder, filename),
                writer=self.writer
            )
