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
        running_df = session.dataset.running_speed.query(
            'timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
    else:
        running_df = session.dataset.running_speed
    ax.plot(
        running_df['timestamps'],
        running_df['speed'],
        color='blue'
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
    kernels_to_exclude_from_plot = ['intercept','time',]#['intercept','time','model_task0','model_timing1D','model_bias','model_omissions1']
    
    if t_span:
        t0,t1 = t_span
        data_to_plot = kernel_df.query('timestamps >= @t0 and timestamps <= @t1 and kernel_name not in @kernels_to_exclude_from_plot')
    else:
        data_to_plot = kernel_df.query('kernel_name not in @kernels_to_exclude_from_plot')
    palette = vbp.generate_random_colors(
        len(data_to_plot['kernel_name'].unique()), 
        lightness_range=(0.1,.65), 
        saturation_range=(0.5,1), 
        random_seed=0, 
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
        legend=False
    )
    ax.legend(
        data_to_plot['kernel_name'].unique(),
        loc='upper left',
        ncol=10, 
        mode="expand", 
        framealpha = 0.2,
    )
    plt.setp(ax.lines,linewidth=2)


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

        self.fig, self.ax = self.set_up_axes()
        self.writer = self.set_up_writer()

    def make_cell_movie_frame(self, ax, glm, F_index, cell_specimen_id, t_before=10, t_after=10):

        this_cell = glm.df_full.query('cell_specimen_id == @cell_specimen_id')
        cell_index = np.where(glm.W['cell_specimen_id'] == cell_specimen_id)[0][0]

        model_timestamps = glm.fit['dff_trace_arr']['dff_trace_timestamps'].values
        t_now = model_timestamps[F_index]
        t_span = [t_now - t_before, t_now + t_after]

        for axis in ax.keys():
            ax[axis].cla()

        F_this_frame = glm.df_full.query('frame_index == @F_index').set_index('cell_specimen_id')
        # dff_actual = dft.loc[glm.W['cell_specimen_id'].values]['dff'].values
        # dff_pred = dft.loc[glm.W['cell_specimen_id'].values]['dff_predicted'].values
        
        from scipy import ndimage
        # 2P ROI images:
        ax['cell_roi'].imshow(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'],cmap='gray')
        com = ndimage.measurements.center_of_mass(glm.session.dataset.cell_specimen_table.loc[cell_specimen_id]['image_mask'])

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
            ax[axis_name].axvline(com[1],color='MediumAquamarine',alpha=0.5)
            ax[axis_name].axhline(com[0],color='MediumAquamarine',alpha=0.5)

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
        )

        ax['cell_response'].plot(
            local_df['dff_trace_timestamps'],
            local_df['dff_predicted'],
            alpha=1,
            color='black',
        )

        ax['cell_response'].legend(
            ['Actual $\Delta$F/F','Model Predicted $\Delta$F/F'],
            loc='upper left',
            ncol=2, 
            framealpha = 0.2,
        )

        plot_licks(glm.session, ax['licks'], t_span=t_span)
        plot_running(glm.session, ax['running'], t_span=t_span)
        plot_kernels(self.kernel_df, ax['kernel_contributions'], t_span)

        # some axis formatting: 
        for axis_name in ['licks', 'cell_response', 'running','kernel_contributions']:
            ax[axis_name].axvline(t_now, color='black', linewidth=3, alpha=0.5)
            plot_stimuli(glm.session, ax[axis_name], t_span=t_span)
            if axis_name is not 'kernel_contributions':
                ax[axis_name].set_xticklabels([])

        ax['cell_response'].set_title('Time series plots for cell {}'.format(cell_specimen_id))
        ax['licks'].set_xlim(t_span[0], t_span[1])
        ax['licks'].set_yticks([])

        ax['cell_response'].set_xticklabels('')

        ax['licks'].set_xlabel('time')

        ax['licks'].set_ylabel('licks       ', rotation=0,ha='right', va='center')
        ax['cell_response'].set_ylabel('$\Delta$F/F', rotation=0, ha='right', va='center')
        ax['running'].set_ylabel('Running\nSpeed\n(cm/s)', rotation=0, ha='right', va='center')
        ax['kernel_contributions'].set_ylabel('kernel\ncontributions\nto predicted\nsignal', rotation=0, ha='right', va='center')


    def update(self, frame_number):
        '''
        method to update figure
        animation class will call this

        the print statement is there to help track progress
        '''
        self.make_cell_movie_frame(
            self.ax, self.glm, F_index=frame_number, cell_specimen_id=self.cell_specimen_id)

        self.pbar.update(1)

    def set_up_axes(self):
        fig = plt.figure(figsize=(18, 12))
        ax = {
            'cell_roi': vbp.placeAxesOnGrid(fig, xspan=(0, 0.25), yspan=(0, 0.4)),
            'real_fov': vbp.placeAxesOnGrid(fig, xspan=(0.25, 0.5), yspan=(0, 0.4)),
            'reconstructed_fov': vbp.placeAxesOnGrid(fig, xspan=(0.5, 0.75), yspan=(0, 0.4)),
            'simulated_fov': vbp.placeAxesOnGrid(fig, xspan=(0.75, 1), yspan=(0, 0.4)),
            'cell_response': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.45, 0.6]),
            'licks': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.6, 0.625]),
            'running': vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.625, 0.75]),
            'kernel_contributions':vbp.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0.75, 1]),
        }

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
