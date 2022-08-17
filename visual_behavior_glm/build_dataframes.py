import numpy as np
import pandas as pd
from tqdm import tqdm
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
    session.behavior_df['engagement_v1'] = session.behavior_df['engaged']
    session.behavior_df['engagement_v2'] = session.behavior_df['engaged'] & session.behavior_df['lick_bout_rate'] > 0.1

def build_response_df_cell(session, cell_specimen_id):
    # Interpolate neural activity onto stimulus timestamps
    # 
    # Build response dataframe
    # Save response_df, peak_response_df
    return 


def build_response_df_session(session):

    # get session level behavior metrics
    load_behavior_summary(session)

    # loop over cells 
    cell_specimen_ids = session.cell_specimen_table.index.values
    for index, cell_id in tqdm(enumerate(cell_specimen_ids),
        total=len(cell_specimen_ids),desc='Iterating Cells'):
        df = build_response_df_cell(session, cell_id)
        # build_response_df

    return


def aggregate_df(df_request, df_type='peak'):
    # df_type should be peak or full
    # df_request should be a df that contains either the cells, or sessions to aggregate (make a choice)
    return    
    
### DEV
################################################################################
def get_cell_df(session):
    timestamps = session.ophys_timestamps
    filtered_event_traces = session.events.iloc[0]['filtered_events']
    df = pd.DataFrame()
    df['t'] = timestamps
    df['response'] = filtered_event_traces
    return df


def get_cell_eta(df,session,session_df):
    eta = m.event_triggered_response(
        data = df,
        t = 't',
        y='response',
        event_times = session.stimulus_presentations.start_time,
        t_start = 0,
        t_end = 0.75,
        output_sampling_rate = 30,
        interpolate=True
        )
    # Add image identity labels
    eta = pd.merge(eta, session_df,
        on='stimulus_presentations_id')
    # add engagement labels
    # add strategy weights
    return eta


def get_cell_response(eta,session_df):
    responses = eta.groupby('stimulus_presentations_id')['response'].max()
    responses = pd.merge(responses, session_df, on='stimulus_presentations_id')
    return responses

