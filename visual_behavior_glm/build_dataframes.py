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

def load_population_df(df_type,cre):
    path ='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'\
        +df_type+'s/summary_'+cre+'.h5'
    df = pd.read_hdf(path)
    return df

def build_population_df(results_pivoted,df_type='image_df',savefile=True,
    cre='Vip-IRES-Cre'):

    # get list of experiments
    results_pivoted = results_pivoted.query('not passive')
    results_pivoted = results_pivoted.query('cre_line == @cre')
    oeids = results_pivoted['ophys_experiment_id'].unique()

    # load
    dfs = []
    num_rows = len(oeids)
    for idx,value in tqdm(enumerate(oeids),total = num_rows):
        try:
            path=get_path('',value, 'experiment',df_type)
            this_df = pd.read_hdf(path)
            dfs.append(this_df)
        except:
            pass 

    # combine    
    print('concatenating dataframes')
    population_df = pd.concat(dfs)

    # save
    if savefile:
        print('saving')
        path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'+df_type+'s/summary_'+cre+'.h5'
        population_df.to_hdf(path,key='df')

    return population_df,dfs


def load_data(oeid,include_invalid_rois=False):
    '''
        Loads the sdk object for this experiment
    '''
    print('loading sdk object')
    session = loading.get_ophys_dataset(oeid, include_invalid_rois=include_invalid_rois)
    return session


def load_behavior_summary(session):
    '''
        Loads the behavior_session_df summary file and adds to the SDK object
    '''
    print('loading session strategy df')
    bsid = session.metadata['behavior_session_id']
    session_df = ps.load_session_strategy_df(bsid, BEHAVIOR_VERSION)
    session.behavior_df = session_df 
    temporary_engagement_updates(session)


def temporary_engagement_updates(session):
    '''
        Adds a second engagement definition because I am still looking at that issue
    '''
    session.behavior_df['engagement_v1'] = session.behavior_df['engaged']
    session.behavior_df['engagement_v2'] = session.behavior_df['engaged'] \
        & session.behavior_df['lick_bout_rate'] > 0.1


def build_response_df_experiment(session):
    '''
        For each cell in this experiment
    '''

    # get session level behavior metrics
    load_behavior_summary(session)

    # loop over cells 
    cell_specimen_ids = session.cell_specimen_table.index.values
    print('Iterating over cells for this experiment to build image by image dataframes')
    image_dfs = []
    for index, cell_id in tqdm(enumerate(cell_specimen_ids),
        total=len(cell_specimen_ids),desc='Iterating Cells'):
        try:
            this_image = build_response_df_cell(session, cell_id)
            image_dfs.append(this_image)
        except Exception as e:
            print('error with '+str(cell_id))
            print(e)

    print('saving combined image df')
    path = get_path('',session.metadata['ophys_experiment_id'],'experiment','image_df')
    image_df = pd.concat(image_dfs)
    image_df.to_hdf(path, key='df')

    print('Iterating over cells for this experiment to build full dataframes')
    full_dfs = []
    for index, cell_id in tqdm(enumerate(cell_specimen_ids),
        total=len(cell_specimen_ids),desc='Iterating Cells'):
        try:
            this_full = build_full_df_cell(session, cell_id)
            full_dfs.append(this_full)
        except Exception as e:
            print('error with '+str(cell_id))
            print(e)

    print('saving combined full df')
    path = get_path('',session.metadata['ophys_experiment_id'],'experiment','full_df')
    full_df = pd.concat(full_dfs)
    full_df.to_hdf(path, key='df')

    print('Finished!')

def get_path(cell_id, oeid, filetype,df_type):
    root = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
    filepath = root+df_type+'s/'+filetype+'s/'+str(oeid)+'_'+str(cell_id)+'.h5'
   
    return filepath


def build_response_df_cell(session, cell_specimen_id):

    # Get neural activity
    cell_df = get_cell_df(session,cell_specimen_id)

    # get running speed
    try:
        run = get_running_etr(session)
        run_df = run.groupby('stimulus_presentations_id')['speed'].mean()
    except Exception as e:
        print('error procesing running '+str(cell_specimen_id))
        print(e)
        run_df = None

    # get pupil
    try:
        pupil = get_pupil_etr(session)
        pupil_df = pupil.groupby('stimulus_presentations_id')['pupil_width'].mean()  
    except Exception as e:
        print('error processing pupil '+str(cell_specimen_id))
        print(e)
        pupil_df = None
 
    # Get the max response to each image presentation   
    image_df = get_image_df(cell_df, run_df, pupil_df, session, cell_specimen_id) 
    return image_df


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


def get_running_etr(session, time=[0.05,.8]):
    etr = m.event_triggered_response(
        data = session.running_speed,
        t='timestamps',
        y='speed',
        event_times = session.stimulus_presentations.start_time,
        t_start = time[0],
        t_end = time[1],
        output_sampling_rate=30,
        interpolate=True
        )
    return etr


def get_pupil_etr(session, time=[0.05,.8]):
    etr = m.event_triggered_response(
        data = session.eye_tracking,
        t='timestamps',
        y='pupil_width',
        event_times = session.stimulus_presentations.start_time,
        t_start = time[0],
        t_end = time[1],
        output_sampling_rate=30,
        interpolate=True
        )
    return etr


def get_cell_etr(df,session,time = [0.05,0.8]):
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

   
def get_image_df(cell_df,run_df, pupil_df, session,cell_specimen_id):

    # Interpolate neural activity onto stimulus timestamps
    # then align to stimulus times
    etr = get_cell_etr(cell_df, session)

    # Get max response for each image
    image_df = etr.groupby('stimulus_presentations_id')['response'].mean()
    image_df = pd.merge(image_df, session.behavior_df, on='stimulus_presentations_id')
    image_df['cell_specimen_id'] = cell_specimen_id
    image_df['mouse_id'] = session.metadata['mouse_id']
    image_df['behavior_session_id'] = session.metadata['behavior_session_id']   
    image_df['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
    image_df['cre_line'] = session.metadata['cre_line']

    # Add running speed
    if run_df is not None:
        image_df = pd.merge(image_df, run_df, on='stimulus_presentations_id')
        image_df = image_df.rename(columns={'speed':'running_speed'})

    # Add pupil speed
    if pupil_df is not None:
        image_df = pd.merge(image_df, pupil_df, on='stimulus_presentations_id')

    # Save
    ophys_experiment_id = session.metadata['ophys_experiment_id']
    path = get_path(cell_specimen_id, ophys_experiment_id, 'cell','image_df')
    image_df.to_hdf(path,key='df')

    return image_df

def build_full_df_cell(session, cell_specimen_id):

    # Get neural activity
    cell_df = get_cell_df(session,cell_specimen_id)
    
    # Get the max response to each image presentation   
    full_df = get_full_df(cell_df, session, cell_specimen_id) 
    return full_df

    
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

    averages = pd.DataFrame()
    conditions = get_conditions()
    for c in conditions:
        averages = get_full_average(averages, full_df,conditions[c])

    averages['cell_specimen_id'] = cell_specimen_id
    averages['mouse_id'] = session.metadata['mouse_id']
    averages['behavior_session_id'] = session.metadata['behavior_session_id']   
    averages['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
    averages['cre_line'] = session.metadata['cre_line']
 
    # Save
    ophys_experiment_id = session.metadata['ophys_experiment_id']
    path = get_path(cell_specimen_id, ophys_experiment_id, 'cell','full_df')
    averages.to_hdf(path,key='df')

    return averages

def get_full_average(averages, full_df, condition):
   
    # Get conditional average
    if condition[1]=='':
        x = full_df.groupby('time')['response'].mean()
    else:
        x = full_df.query(condition[1]).groupby('time')['response'].mean()

    # Add to dataframe   
    if len(x) == 0:
        t = np.sort(full_df['time'].unique())
        r = np.empty(np.shape(t))
        r[:] = np.nan
        temp = {'condition':condition[0],
            'query':condition[1],
            'time':t,
            'response':r}
    else:
        temp = {'condition':condition[0],
                'query':condition[1],
                'time':x.index.values,
                'response':x.values}
    averages = averages.append(temp,ignore_index=True)
    
    # return
    return averages

def get_conditions():
    conditions = {
        'image':['image',''],
        'change':['change','is_change'],
        'omission':['omission','omitted'],
        'hit':['hit','is_change & rewarded'],
        'miss':['miss','is_change & not rewarded'],
        'licked':['licked','lick_bout_start'],
        'engaged_v1_image':['engaged_v1_image','engagement_v1'],
        'engaged_v2_image':['engaged_v2_image','engagement_v2'],
        'disengaged_v1_image':['disengaged_v1_image','(not engagement_v1)'],
        'disengaged_v2_image':['disengaged_v2_image','(not engagement_v2)'],
        'engaged_v1_change':['engaged_v1_change','engagement_v1 & is_change'],
        'engaged_v2_change':['engaged_v2_change','engagement_v2 & is_change'],
        'disengaged_v1_change':['disengaged_v1_change','(not engagement_v1) & is_change'],
        'disengaged_v2_change':['disengaged_v2_change','(not engagement_v2) & is_change'],
        'engaged_v1_omission':['engaged_v1_omission','engagement_v1 & omitted'],
        'engaged_v2_omission':['engaged_v2_omission','engagement_v2 & omitted'],
        'disengaged_v1_omission':['disengaged_v1_omission','(not engagement_v1) & omitted'],
        'disengaged_v2_omission':['disengaged_v2_omission','(not engagement_v2) & omitted'],
        'engaged_v1_hit':['engaged_v1_hit','engagement_v1 & is_change & rewarded'],
        'engaged_v2_hit':['engaged_v2_hit','engagement_v2 & is_change & rewarded'],
        'disengaged_v1_hit':['disengaged_v1_hit','(not engagement_v1) & is_change & rewarded'],
        'disengaged_v2_hit':['disengaged_v2_hit','(not engagement_v2) & is_change & rewarded'],
        'engaged_v1_miss':['engaged_v1_miss','engagement_v1 & is_change & (not rewarded)'],
        'engaged_v2_miss':['engaged_v2_miss','engagement_v2 & is_change & (not rewarded)'],
        'disengaged_v1_miss':['disengaged_v1_miss','(not engagement_v1) & is_change & (not rewarded)'],
        'disengaged_v2_miss':['disengaged_v2_miss','(not engagement_v2) & is_change & (not rewarded)'],       
        'engaged_v1_licked':['engaged_v1_licked','engagement_v1 & lick_bout_start'],
        'engaged_v2_licked':['engaged_v2_licked','engagement_v2 & lick_bout_start'],
        'disengaged_v1_licked':['disengaged_v1_licked','(not engagement_v1) & lick_bout_start'],
        'disengaged_v2_licked':['disengaged_v2_licked','(not engagement_v2) & lick_bout_start'],       
    }
    return conditions
