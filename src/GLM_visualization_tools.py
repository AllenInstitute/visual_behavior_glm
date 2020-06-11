import visual_behavior.utilities as vbu
import visual_behavior.plotting as vbp
import visual_behavior.visualization.utils as vis_utils
import visual_behavior.data_access.loading as loading

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

from matplotlib import animation, rc
import matplotlib.pyplot as plt

def plot_licks(session, ax, y_loc=0, t_span=None):
    if t_span:
        df = session.dataset.licks.query('timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
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
        running_df = session.dataset.running_speed.query('timestamps >= {} and timestamps <= {}'.format(t_span[0], t_span[1]))
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
    colors = {image:color for image,color in zip(np.sort(images),sns.color_palette("Set2", 8))}

    if t_span:
        query_string = 'start_time >= {0} - {2} and stop_time <= {1} + {2}'.format(t_span[0], t_span[1], buffer)
        visual_stimuli = session.dataset.stimulus_presentations.query('omitted == False').query(query_string).copy()
    else:
        visual_stimuli = session.dataset.stimulus_presentations.query('omitted == False').copy()
    
    visual_stimuli['color'] = visual_stimuli['image_name'].map(lambda i:colors[i])
    visual_stimuli['change'] = visual_stimuli['image_name'] != visual_stimuli['image_name'].shift()
    for idx,stimulus in visual_stimuli.iterrows(): 
        ax.axvspan(
            stimulus['start_time'],
            stimulus['stop_time'],
            color=stimulus['color'],
            alpha=0.5
        )

def build_simulated_FOV(session, F_values):

    assert len(session.cell_specimen_table) == len(F_values)
    
    arr = np.zeros_like(session.dataset.max_projection)
    for ii,cell_specimen_id in enumerate(session.dataset.cell_specimen_ids):
    
        arr += session.cell_specimen_table.loc[cell_specimen_id]['image_mask']*F_values[ii]
        
    return arr

def plot_weights(glm, ax, cell_index, F_index):

    v = glm.X[F_index,:].values * glm.W[:,cell_index].values

    ax.plot(v, color='black')
    ax.set_ylim(-0.1,0.25)

def make_cell_summary_plot(session, fit, design, cell_index):
    model_timestamps = fit['dff_trace_arr']['dff_trace_timestamps'].values
    dff_arr = fit['dff_trace_arr'].values
    
    X = design.get_X(kernels=fit['dropouts']['Full']['kernels'])
    W = fit['dropouts']['Full']['weights']
    
    y = X.T @ W
    
    fig = plt.figure(figsize=(15,4))

    ax = {
        'cell_response': vis_utils.placeAxesOnGrid(fig, xspan=[0,1], yspan = [0,0.85]),
        'licks': vis_utils.placeAxesOnGrid(fig, xspan=[0,1], yspan = [0.85,1]),
    }

    ax['cell_response'].plot(model_timestamps, dff_arr[:,cell_index],alpha=0.5,color='darkgreen')
    ax['cell_response'].plot(model_timestamps, y[:,cell_index],color='black')

    plot_licks(session,ax['licks'])
    plot_omissions(session, ax['cell_response'], y_loc=0)

    ax['cell_response'].set_xticklabels('')
    ax['licks'].set_yticks([])
    ax['licks'].set_xlabel('time')

    ax['licks'].set_ylabel('licks')
    ax['cell_response'].set_ylabel('$\Delta$F/F')

    ax['licks'].get_shared_x_axes().join(ax['licks'], ax['cell_response'])

    plot_stimuli(session, ax['cell_response'])

class GLM_Movie(object):

    def __init__(self, glm, cell_to_plot, start_frame, end_frame, fps=10):

        # note that ffmpeg must be installed on your system
        # this is tested on linux (not sure if it works on windows)
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

        plt.style.use('fivethirtyeight')

        self.glm = glm
        self.cell_to_plot = cell_to_plot
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.real_2p_movie = loading.load_motion_corrected_movie(self.glm.oeid)

        self.frames = np.arange(self.start_frame, self.end_frame)
        self.fps = fps

        self.fig, self.ax = self.set_up_axes()
        self.writer = self.set_up_writer()


    def make_cell_movie_frame(self, ax, glm, F_index, cell_to_plot, t_before=10, t_after=10):

        this_cell = glm.df_full.query('cell_specimen_id == @cell_to_plot')
        cell_index = np.where(glm.W['cell_specimen_id']==cell_to_plot)[0][0]

        model_timestamps = glm.fit['dff_trace_arr']['dff_trace_timestamps'].values
        t_now = model_timestamps[F_index]

        for axis in ax.keys():
            ax[axis].cla()
            
        for axis in ['licks','cell_response']:
            ax[axis].axvline(t_now,color='black',linewidth=3,alpha=0.5)

        dft = glm.df_full.query('frame_index == @F_index').set_index('cell_specimen_id')
        dff_pred = dft.loc[glm.W['cell_specimen_id'].values]['dff_predicted'].values

        simulated_fov = build_simulated_FOV(glm.session, dff_pred)
        ax['simulated_fov'].imshow(simulated_fov,cmap='gray',clim=[0,.25])
        ax['simulated_fov'].set_xticks([])
        ax['simulated_fov'].set_yticks([])
        
        real_fov = self.real_2p_movie[F_index]
        ax['real_fov'].imshow(real_fov,cmap='gray',clim=[0,15000])
        ax['real_fov'].set_xticks([])
        ax['real_fov'].set_yticks([])
        
        plot_weights(glm, ax['model_weights'], cell_index = cell_index, F_index=F_index)

        t_span = [t_now-t_before, t_now+t_after]
        query_string = 'dff_trace_timestamps >= {} and dff_trace_timestamps <= {}'.format(t_span[0], t_span[1])
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

        plot_licks(glm.session, ax['licks'], t_span=t_span)
        plot_running(glm.session, ax['running'], t_span=t_span)
        for axis_name in ['cell_response','licks','running']:
            plot_stimuli(glm.session, ax[axis_name], t_span=t_span)

        ax['licks'].set_xlim(t_span[0],t_span[1])
        ax['licks'].set_yticks([])

        ax['cell_response'].set_xticklabels('')
        
        ax['licks'].set_xlabel('time')

        ax['licks'].set_ylabel('licks       ', rotation=0,ha='right',va='center')
        ax['cell_response'].set_ylabel('$\Delta$F/F', rotation=0,ha='right',va='center')
        ax['running'].set_ylabel('Running\nSpeed\n(cm/s)', rotation=0,ha='right',va='center')
        ax['licks'].ticklabel_format(useOffset=False, style='plain')

        ax['simulated_fov'].set_title('Simulated FOV')
        ax['real_fov'].set_title('Real FOV')


    def update(self, frame_number):
        '''
        method to update figure
        animation class will call this
        
        the print statement is there to help track progress
        '''
        self.make_cell_movie_frame(self.ax, self.glm, F_index=frame_number, cell_to_plot=self.cell_to_plot)
        print('on frame {}'.format(frame_number), end='\r')

    def set_up_axes(self):
        fig = plt.figure(figsize=(11,8.5))
        ax = {
            'real_fov':vbp.placeAxesOnGrid(fig, xspan=(0,0.25), yspan=(0,0.4)),
            'simulated_fov':vbp.placeAxesOnGrid(fig, xspan=(0.25,0.5), yspan=(0,0.4)),
            'model_weights':vbp.placeAxesOnGrid(fig, xspan=(0.6,0.95), yspan=(0.05,0.4)),
            'cell_response': vbp.placeAxesOnGrid(fig, xspan=[0,1], yspan = [0.45,0.75]),
            'licks': vbp.placeAxesOnGrid(fig, xspan=[0,1], yspan = [0.75,0.8]),
            'running': vbp.placeAxesOnGrid(fig, xspan=[0,1], yspan = [0.8,1]),
        }


        ax['licks'].get_shared_x_axes().join(ax['licks'], ax['cell_response'])
        ax['running'].get_shared_x_axes().join(ax['running'], ax['cell_response'])



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
        filename = 'oeid={}_cell_id={}_frame_{}_to_{}.mp4'.format(self.glm.oeid, self.cell_to_plot, self.start_frame, self.end_frame)
        a.save(
            os.path.join(save_folder, filename),
            writer=self.writer
        )