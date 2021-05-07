import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import pandas as pd
import visual_behavior.database as db
import visual_behavior.utilities as vbu
import visual_behavior.plotting as vbp
import seaborn as sns
import numpy as np
import os

import matplotlib as mpl
from matplotlib import animation, rc
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def get_behavior_movie_filepath(session_id, session_type='OphysSession'):
    well_known_files = db.get_well_known_files(session_id, session_type)
    behavior_video_path = ''.join(well_known_files.loc['RawBehaviorTrackingVideo'][[
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


def get_movie(osid):
    sync_data = get_sync_data(osid)
    movie = vbu.Movie(
        get_behavior_movie_filepath(osid), 
        sync_timestamps=sync_data['cam1_exposure_rising'],
        lazy_load=False,
    )
    return movie

def make_fig_ax(pc_display='heatmap'):
    fig = plt.figure(figsize=(20,10))
    ax = {
        'frame':vbp.placeAxesOnGrid(fig, xspan=[0,0.3], yspan=[0,0.5]),
        'frame_diff':vbp.placeAxesOnGrid(fig, xspan=[0,0.3], yspan=[0.5,1]),
        'PC_masks':vbp.placeAxesOnGrid(fig, xspan=[0.325,0.43], yspan=[0, 0.95], dim=(15,1), sharey=False,wspace=0.05,hspace=0.05),
    }
    if pc_display == 'timeseries':
        ax.update({'PC_plot':vbp.placeAxesOnGrid(fig, xspan=[0.45,1], yspan=[0, 1], dim=(15,1), sharey=False,wspace=0.05,hspace=0.05)}),
    elif pc_display == 'heatmap':
        ax.update({'PC_heatmap':vbp.placeAxesOnGrid(fig, xspan=[0.41,0.89], yspan=[0, 0.95], sharey=False)}),
        ax.update({'PC_heatmap_cbar':vbp.placeAxesOnGrid(fig, xspan=[0.9,0.925], yspan=[0, 0.95], sharey=False)}),
    return fig,ax

def plot_frame(mask, nan_mask, prediction_df, movie, session, frame, fig, ax, t_range=3, frame_range = 5, pc_display='heatmap', 
               pc_min=None, pc_max=None, face_diff_min=None, face_diff_max=None, replot_pc_masks=True, mask_lims_lrtb=None):
    n_pcs = 15
    tnow = prediction_df.loc[frame]['timestamps']
    if mask_lims_lrtb is None:
        left_clip,right_clip = np.where(mask==1)[1].min(),np.where(mask==1)[1].max()
        top_clip,bottom_clip = np.where(mask==1)[0].max(),np.where(mask==1)[0].min()
    else:
        left_clip,right_clip,top_clip,bottom_clip = mask_lims_lrtb
    
    for key in ax.keys():
        if key == 'PC_plot' or (key == 'PC_masks' and replot_pc_masks==True):
            ax[key] = np.array(ax[key])
            for ii in range(n_pcs):
                ax[key].flatten()[ii].cla()
        elif key not in ['PC_plot','PC_masks']:
            ax[key].cla()

    ax['frame'].imshow(movie.array[frame, :,:], cmap='gray')
    ax['frame'].axis('off')
    ax['frame'].imshow(nan_mask,alpha=0.25,cmap='magma_r')
    ax['frame'].set_title('raw video with binary mask overlaid')
    
    arr = movie.array[frame - frame_range:frame + frame_range, :,:]*mask
    if face_diff_min is None:
        ax['frame_diff'].imshow(arr.max(axis=0) - arr.min(axis=0), cmap='magma')
    else:
        ax['frame_diff'].imshow(arr.max(axis=0) - arr.min(axis=0), cmap='magma',clim=(face_diff_min, face_diff_max))
    ax['frame_diff'].axis('off')
    ax['frame_diff'].set_xlim(left_clip, right_clip)
    ax['frame_diff'].set_ylim(top_clip, bottom_clip)
    ax['frame_diff'].set_title('range (max - min) of each pixel\nin the mask for current frame +/- {} frames'.format(frame_range))

    df_local = prediction_df.query('timestamps >= (@tnow - @t_range) and timestamps <= (@tnow + @t_range)')
    pc_cols = [col for col in df_local if 'PC' in col]
    if pc_min is None:
        pc_min = -1*df_local[pc_cols].abs().max().max()
    if pc_max is None:
        pc_max = df_local[pc_cols].abs().max().max()
    
    if pc_display == 'heatmap':
        dft = df_local[pc_cols].T
        dft.columns=df_local['timestamps']
        sns.heatmap(
            dft,
            ax=ax['PC_heatmap'],
            vmin=pc_min,
            vmax=pc_max,
            cmap='seismic',
            cbar_ax=ax['PC_heatmap_cbar']
        )
        ax['PC_heatmap'].axvline((df_local['timestamps'] - tnow).abs().idxmin() - df_local.index.min(),zorder=np.inf,color='black',linewidth=2,alpha=0.5)
        ax['PC_heatmap'].set_yticklabels([])
        ax['PC_heatmap'].set_yticks([])
        ax['PC_heatmap'].set_title('PC values vs. time')
#         ax['PC_heatmap'].set_xticklabels(['{:0.2f}'.format(l) for l in list(ax['PC_heatmap'].get_xticklabels())])
    

    for row in range(n_pcs):
        pc = row
        if pc_display == 'timeseries':
            ax['PC_plot'][row].plot(
                df_local['timestamps'],
                df_local['PC{}'.format(pc)]
            )

            ax['PC_plot'][row].axvline(tnow)
            ax['PC_plot'][row].set_xlim(tnow - t_range, tnow + t_range)
            ax['PC_plot'][row].set_ylim(minval, maxval)

            if row <= n_pcs-1:
                ax['PC_plot'][row].set_xticklabels([])
            else:
                ax['PC_plot'][row].set_xlabel('time (s)')
        
        if replot_pc_masks == True:
            arr = session.behavior_movie_pc_masks[:,:,row] #*prediction_df.loc[frame]['PC{}'.format(pc)]
            lim = np.abs(arr).max()
            ax['PC_masks'][row].imshow(arr,clim=[-1*lim, lim],cmap='seismic')
            ax['PC_masks'][row].set_xlim(left_clip, right_clip)
            ax['PC_masks'][row].set_ylim(top_clip, bottom_clip)
            ax['PC_masks'][row].axis('off')
            ax['PC_masks'][row].text(left_clip+1,bottom_clip+3,'PC{}'.format(pc),ha='left',va='top',color='black')
            if pc == 0:
                ax['PC_masks'][row].set_title('PC masks')
        
    sns.despine()

def make_movie(oeid, start_frame, end_frame, filename, fps=30):

    dataset = loading.get_ophys_dataset(oeid)
    session = ResponseAnalysis(dataset)

    query = 'select ophys_session_id from ophys_experiments where id = {} limit 1'
    osid = db.lims_query(query.format(oeid))

    print('loading movie...')
    movie = get_movie(osid)
    print('done loading movie')
    
    fig,ax = make_fig_ax()
    mask = np.zeros(session.behavior_movie_pc_masks[:,:,0].shape, dtype=np.uint8)
    mask[session.behavior_movie_pc_masks[:,:,0]!=0]=1
    nan_mask = mask.copy().astype(float)
    nan_mask[nan_mask==0]=np.nan
    
    left_clip,right_clip = np.where(mask==1)[1].min(),np.where(mask==1)[1].max()
    top_clip,bottom_clip = np.where(mask==1)[0].max(),np.where(mask==1)[0].min()
    mask_lims_lrtb = (left_clip,right_clip,top_clip,bottom_clip)
    
    prediction_df = pd.DataFrame({'timestamps': session.behavior_movie_timestamps})
    for pc in range(15):
        prediction_df['PC{}'.format(pc)] = session.behavior_movie_pc_activations[:,pc]
    prediction_df = prediction_df.merge(
        session.behavior_movie_predictions,
        left_on='timestamps',
        right_on='timestamps',
        how='left'
    )
    
    pc_cols = [col for col in prediction_df if 'PC' in col]
    pc_min = -1*prediction_df.loc[start_frame:end_frame][pc_cols].abs().max().max()
    pc_max = prediction_df.loc[start_frame:end_frame][pc_cols].abs().max().max()
    
    arr = movie.array[start_frame:end_frame, :,:]*mask
    full_range = arr.max(axis=0) - arr.min(axis=0)
    face_diff_min = full_range[full_range>0].min()
    face_diff_max = full_range[full_range>0].max()

    def update(frame):
        replot_pc_masks = frame == start_frame
        fraction_done = (frame-start_frame)/(end_frame-start_frame)
        print('on frame {} of {} ({:0.2f}%)'.format(
            frame, end_frame, 100*fraction_done), end='\r')
        plot_frame(
            mask, 
            nan_mask, 
            prediction_df, 
            movie, 
            session,
            frame, 
            fig, 
            ax, 
            pc_min=pc_min, 
            pc_max=pc_max, 
            face_diff_min=face_diff_min, 
            face_diff_max=face_diff_max, 
            replot_pc_masks=replot_pc_masks,
            mask_lims_lrtb=mask_lims_lrtb,
        )

    mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    print('fig = {}'.format(fig))
    print('filename = {}'.format(filename))

    writer = animation.FFMpegWriter(
        fps=fps,
        codec='mpeg4',
        bitrate=-1,
        extra_args=['-pix_fmt', 'yuv420p', '-q:v', '5']
    )

    a = animation.FuncAnimation(
        fig,
        update,
        frames=range(start_frame, end_frame),
        interval=1/fps*1000,
        repeat=False,
        blit=False
    )

    a.save(
        filename,
        writer=writer
    )


if __name__ == "__main__":

    et = loading.get_filtered_ophys_experiment_table()
    oeids = et.sample(100).index

    for oeid in oeids:
        for start_frame, end_frame in [(20000,21000), (120000, 121000)]:

            filename = 'oeid={}_frame_{}_to_{}.mp4'.format(oeid, start_frame, end_frame)
            make_movie(
                oeid,
                start_frame=start_frame, 
                end_frame=end_frame, 
                filename=os.path.join('/allen/programs/braintv/workgroups/nc-ophys/Doug/face_motion_movies', filename), 
                fps=15
            )