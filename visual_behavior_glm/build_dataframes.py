import numpy as np
import pandas as pd
from tqdm import tqdm

import mindscope_utilities as m
import visual_behavior.data_access.loading as loading

import psy_tools as ps


'''
    Generates two dataframes for each cell
    1.  response_df
        Contains a 4 second response trace aligned to each image presentation, 
        and relevant metrics (engaged, strategy weights, licked, running).
        the peak response_df is a strict subset of response_df
    2.  peak_response_df
        Just the peak response of each cell to each image presentation in the 
        (100ms,800ms) window, and relevant metrics

'''

BEHAVIOR_VERSION = 21

def load_data(oeid,include_invalid_rois=False):
    '''
        Loads the sdk object for this experiment
    '''
    session = loading.get_ophys_dataset(oeid, include_invalid_rois=include_invalid_rois)
    return session


def load_behavior_summary(session):
    '''
        Loads the behavior_session_df summary file and adds to the SDK object
    '''
    bsid = session.metadata['behavior_session_id']
    session_df = ps.load_session_strategy_df(bsid, BEHAVIOR_VERSION)
    session.behavior_df = session_df 
    temporary_engagement_updates(session)


def temporary_engagement_updates(session):
    '''
        Adds a second engagement definition because I am still looking at that issue
    '''
    session.behavior_df['engagement_v1'] = session.behavior_df['engaged']
    session.behavior_df['engagement_v2'] = session.behavior_df['engaged'] & session.behavior_df['lick_bout_rate'] > 0.1


def build_response_df_experiment(session):
    '''
        For each cell in this experiment
    '''

    # get session level behavior metrics
    load_behavior_summary(session)

    # loop over cells 
    cell_specimen_ids = session.cell_specimen_table.index.values
    image_dfs = []
    full_dfs = []
    for index, cell_id in tqdm(enumerate(cell_specimen_ids),
        total=len(cell_specimen_ids),desc='Iterating Cells'):
        this_image, this_full = build_response_df_cell(session, cell_id)
        image_dfs.append(this_image)
        full_dfs.append(this_full)


    # Build aggregate for this experiment and save
    image_df = pd.concat(image_dfs) 
    oeid = session.metadata['ophys_experiment_id']
    path = get_path(oeid, 'experiment','image_df')
    image_df.to_hdf(path, key='df')

    # Build aggregate for this experiment and save
    full_df = pd.concat(full_dfs) 
    oeid = session.metadata['ophys_experiment_id']
    path = get_path(oeid, 'experiment','full_df')
    full_df.to_hdf(path, key='df')

    # return 
    return image_df, full_df

def get_path(this_id, filetype,df_type):
    root = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
    filepath = root+df_type+'s/'+filetype+'s/'+str(this_id)+'.h5'
   
    return filepath


def build_response_df_cell(session, cell_specimen_id):

    # Get neural activity
    cell_df = get_cell_df(session,cell_specimen_id)
    
    # Get the max response to each image presentation   
    image_df = get_image_df(cell_df, session, cell_specimen_id) 

    # Get full trace for each event    
    full_df = get_full_df(cell_df, session, cell_specimen_id)

    return image_df, full_df


def get_cell_df(session, cell_specimen_id, data_type='filtered_events'):
    '''
        Builds a dataframe of the neural activity
    '''
    timestamps = session.ophys_timestamps
    traces = session.events.loc[cell_specimen_id,data_type]
    df = pd.DataFrame()
    df['t'] = timestamps
    df['response'] = traces
    return df


def get_cell_etr(df,session,time = [0.15,0.85]):
    etr = m.event_triggered_response(
        data = df,
        t = 't',
        y='response',
        event_times = session.stimulus_presentations.start_time,
        t_start = time[0],
        t_end = time[1],
        output_sampling_rate = 30,
        interpolate=True
        )
    return etr


def get_full_df(cell_df, session,cell_specimen_id):
    
    # Interpolate, then align to all images with long window
    full_df = get_cell_etr(cell_df, session, time = [-2,2])
    
    # add annotations
    full_df = pd.merge(full_df, session.behavior_df, on='stimulus_presentations_id')
    full_df['cell_specimen_id'] = cell_specimen_id
    full_df['mouse_id'] = session.metadata['mouse_id']
    full_df['behavior_session_id'] = session.metadata['behavior_session_id']   
    full_df['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
    full_df['cre_line'] = session.metadata['cre_line']
    
    # Save
    path = get_path(cell_specimen_id, 'cell','full_df')
    full_df.to_hdf(path,key='df')

    return full_df

   
def get_image_df(cell_df,session,cell_specimen_id):

    # Interpolate neural activity onto stimulus timestamps
    # then align to stimulus times
    etr = get_cell_etr(cell_df, session)

    # Get max response for each image
    image_df = etr.groupby('stimulus_presentations_id')['response'].max()
    image_df = pd.merge(image_df, session.behavior_df, on='stimulus_presentations_id')
    image_df['cell_specimen_id'] = cell_specimen_id
    image_df['mouse_id'] = session.metadata['mouse_id']
    image_df['behavior_session_id'] = session.metadata['behavior_session_id']   
    image_df['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
    image_df['cre_line'] = session.metadata['cre_line']

    # Save
    path = get_path(cell_specimen_id, 'cell','image_df')
    image_df.to_hdf(path,key='df')

    return image_df

    



