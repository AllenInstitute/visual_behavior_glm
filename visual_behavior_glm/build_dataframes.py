import numpy as np
import pandas as pd
from tqdm import tqdm

import mindscope_utilities as m
import visual_behavior.data_access.loading as loading

import psy_tools as ps
import psy_output_tools as po

BEHAVIOR_VERSION = 21

def add_area_depth(df,experiment_table):
    '''
        Adds targeted_structure, and layer columns from experiment table 
        index on oeid
    '''
    df = pd.merge(df, 
        experiment_table.reset_index()[[\
            'ophys_experiment_id',
            'targeted_structure',
            'layer']],
        on='ophys_experiment_id')
    return df

def load_population_df(data,df_type,cre,summary_df=None,first=False,second=False,
    image=False,experience_level='Familiar'):
    '''
        Loads a summary dataframe
        data should be 'events', 'filtered_events', or 'dff'
        df_type should be 'full_df', or 'image_df'
    '''
    if first:
        extra = '_first_half'
    elif second:
        extra = '_second_half'
    elif image:
        extra = '_image_period'
    else:
        extra =  ''

    if experience_level == 'Novel 1':
        extra +='_novel'
    elif experience_level == "Novel >1":
        extra +='_novelp'

    # load summary file
    path ='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'\
        +df_type+'s/'+data+'/summary_'+cre+extra+'.feather'
    df = pd.read_feather(path)

    # Add columsn from summary_df
    if (df_type =='image_df') and (summary_df is not None):
        cols = ['behavior_session_id','visual_strategy_session',
            'experience_level']
        df = pd.merge(df, summary_df[cols],
            on='behavior_session_id')
    return df


def build_population_df(summary_df,df_type='image_df',cre='Vip-IRES-Cre',
    data='filtered_events',savefile=True,first=False,second=False,image=False,
    experience_level='Familiar'):
    '''
        Generates the summary data files by aggregating over ophys experiment
    '''


    batch_size=50
    batch=False
    if df_type=='image_df' and cre=='Slc17a7-IRES2-Cre':
        print('Batch processing with chunk size={} for Exc cells'.format(batch_size))
        batch=True
    
    print('Generating population {} for {} cells'.format(df_type,cre))

    # get list of experiments
    summary_df = summary_df.query('cre_line == @cre')
    summary_df = summary_df.query('experience_level == @experience_level')
    oeids = np.concatenate(summary_df['ophys_experiment_id'].values) 

    # make list of columns to drop for space
    cols_to_drop = [
        'image_name',
        'cre_line',
        'bout_number',
        'num_licks',
        'lick_rate',
        'reward_rate',
        'lick_bout_rate',
        'lick_hit_fraction',
        'hit_rate',
        'miss_rate',
        'image_false_alarm',
        'image_correct_reject',
        'correct_reject_rate',
        'd_prime',
        'criterion',
        'licked',
        'engaged',
        'RT',
        'rewarded',
        'false_alarm_rate'
        ]

    # load
    dfs = []
    num_rows = len(oeids)
    if batch:
        batch_num=0

    failed_to_load = 0
    for idx,value in tqdm(enumerate(oeids),total = num_rows):
        try:
            path=get_path('',value, 'experiment',df_type,data,first,second,image)
            this_df = pd.read_hdf(path)
            if df_type == 'image_df':
                this_df = this_df.drop(columns=cols_to_drop)
            dfs.append(this_df)
        except:
            failed_to_load += 1
            print(value)
            pass 
        if batch:
            if np.mod(idx,batch_size) == batch_size-1:
                temp = pd.concat(dfs, ignore_index=True, sort=False)
                path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'\
                    +df_type+'s/'+data+'/temp_'+experience_level+str(batch_num)+'_'+cre+'.feather'
                temp.to_feather(path)
                batch_num+=1
                dfs = []

    print('Experiments that did not load: {}'.format(failed_to_load))

    if batch:
        temp = pd.concat(dfs, ignore_index=True, sort=False)
        path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'\
            +df_type+'s/'+data+'/temp_'+experience_level+str(batch_num)+'_'+cre+'.feather'
        temp.to_feather(path)

        n = list(range(0,batch_num+1))
        dfs = []
        for i in n:
            path='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'\
                +df_type+'s/'+data+'/temp_'+experience_level+str(i)+'_'+cre+'.feather'
            temp = pd.read_feather(path)
            dfs.append(temp)
 
    # combine    
    print('concatenating dataframes')
    population_df = pd.concat(dfs,ignore_index=True,sort=False)
    del dfs

    # save
    if savefile:
        print('saving')
        if first:
            extra = '_first_half'
        elif second:
            extra = '_second_half'  
        elif image:
            extra = '_image_period' 
        else:
            extra = ''
        if experience_level == 'Novel 1':
            extra +='_novel'
        elif experience_level == "Novel >1":
            extra +='_novelp'

        path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'\
            +df_type+'s/'+data+'/summary_'+cre+extra+'.feather'
        try:
            population_df.to_feather(path)
        except Exception as e:
            print(e)


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


def build_behavior_df_experiment(session,first=False,second=False,image=False):
    '''
        For each cell in this experiment
    '''

    # get session level behavior metrics
    load_behavior_summary(session)

    # Get summary table
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION) 

    data_types = ['running','pupil']
    good = True
    for data in data_types:
        try:
            print('Generating {} dataset'.format(data))
            if data == 'running':
                full_df = get_running_etr(session, time=[-2,2])
                full_df = full_df.rename(columns={'speed':'response'})
            else:
                full_df = get_pupil_etr(session, time=[-2,2])
                full_df = full_df.rename(columns={'pupil_width':'response'})
    
            full_df = pd.merge(full_df, session.behavior_df, 
                on='stimulus_presentations_id')
            full_df['mouse_id'] = session.metadata['mouse_id']
            full_df['behavior_session_id'] = session.metadata['behavior_session_id']   
            full_df['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
            full_df['cre_line'] = session.metadata['cre_line']
    
            averages = pd.DataFrame()
            conditions = get_conditions()
            for c in conditions:
                averages = get_full_average(session, averages, full_df, conditions[c])
    
            bsid = session.metadata['behavior_session_id']
            row = summary_df.set_index('behavior_session_id').loc[bsid]
            averages['experience_level'] = row['experience_level']
            averages['visual_strategy_session'] = row['visual_strategy_session']
     
            # Save
            ophys_experiment_id = session.metadata['ophys_experiment_id']
            path = get_path('', ophys_experiment_id, 'experiment','full_df',
                data,first,second,image)
            averages.to_hdf(path,key='df')
        except Exception as e:
            print(e)
            good = False
    if good:
        print('Finished!')
    else:
        print('errors')


def build_response_df_experiment(session,data,first=False,second=False,image=False):
    '''
        For each cell in this experiment
    '''
    
    if first:
        time = [0.05, 0.425]
    elif second:
        time = [0.425, 0.8]
    elif image:
        time = [.150, .250]
    else:
        time = [0.05, 0.8]

    # get session level behavior metrics
    load_behavior_summary(session)

    # Get summary table
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION) 

    # loop over cells 
    cell_specimen_ids = session.cell_specimen_table.index.values
    print('Iterating over cells for this experiment to build image by image dataframes')
    image_dfs = []
    for index, cell_id in tqdm(enumerate(cell_specimen_ids),
        total=len(cell_specimen_ids),desc='Iterating Cells'):
        try:
            this_image = build_response_df_cell(session, cell_id,data,time,
                first,second,image)
            image_dfs.append(this_image)
        except Exception as e:
            print('error with '+str(cell_id))
            print(e)

    print('saving combined image df')
    path = get_path('',session.metadata['ophys_experiment_id'],'experiment',
        'image_df',data,first,second,image)
    image_df = pd.concat(image_dfs)
    image_df.to_hdf(path, key='df')

    if first:
        print('skipping full_df because first=True')
        print('Finished!')
        return
    if second:
        print('skipping full_df because second=True')
        print('Finished!')
        return
    if image:
        print('skipping full_df because image=True')
        print('Finished!')
        return

    print('Iterating over cells for this experiment to build full dataframes')
    full_dfs = []
    for index, cell_id in tqdm(enumerate(cell_specimen_ids),
        total=len(cell_specimen_ids),desc='Iterating Cells'):
        try:
            this_full = build_full_df_cell(session, cell_id,summary_df,data)
            full_dfs.append(this_full)
        except Exception as e:
            print('error with '+str(cell_id))
            print(e)

    print('saving combined full df')
    path = get_path('',session.metadata['ophys_experiment_id'],'experiment','full_df',
        data,first=False,second=False,image=False)
    full_df = pd.concat(full_dfs)
    full_df.to_hdf(path, key='df')

    print('Finished!')


def get_path(cell_id, oeid, filetype,df_type,data,first=False,second=False,image=False):
    root = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
    if first:
        extra = '_1'
    elif second:
        extra = '_2'
    elif image:
        extra = '_image'
    else:
        extra = ''
    filepath = root+df_type+'s/'+data+'/'+filetype+'s/'+str(oeid)+'_'+str(cell_id)+\
        extra+'.h5'
   
    return filepath


def build_response_df_cell(session, cell_specimen_id,data,time=[0.05,0.8],
    first=False,second=False,image=False):

    # Get neural activity
    cell_df = get_cell_df(session,cell_specimen_id,data)

    # get running speed
    try:
        run = get_running_etr(session,time=time)
        run_df = run.groupby('stimulus_presentations_id')['speed'].mean()
    except Exception as e:
        print('error procesing running '+str(cell_specimen_id))
        print(e)
        run_df = None

    # get pupil
    try:
        pupil = get_pupil_etr(session,time=time)
        pupil_df = pupil.groupby('stimulus_presentations_id')['pupil_width'].mean()  
    except Exception as e:
        print('error processing pupil '+str(cell_specimen_id))
        print(e)
        pupil_df = None
 
    # Get the max response to each image presentation   
    image_df = get_image_df(cell_df, run_df, pupil_df, session, cell_specimen_id,
        data,time,first,second,image) 
    return image_df


def get_cell_df(session, cell_specimen_id, data='filtered_events'):
    '''
        Builds a dataframe of the neural activity
    '''
    timestamps = session.ophys_timestamps
    if data in ['filtered_events','events']:
        traces = session.events.loc[cell_specimen_id,data]
    elif data in ['dff']:
        traces = session.dff_traces.loc[cell_specimen_id, data]
        
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

   
def get_image_df(cell_df,run_df, pupil_df, session,cell_specimen_id,data,
    time=[0.05,0.8],first=False,second=False,image=False):

    # Interpolate neural activity onto stimulus timestamps
    # then align to stimulus times
    etr = get_cell_etr(cell_df, session,time=time)

    # Get max response for each image
    image_df = etr.groupby('stimulus_presentations_id')['response'].mean()
    image_df = pd.merge(image_df, session.behavior_df, on='stimulus_presentations_id')
    image_df['cell_specimen_id'] = cell_specimen_id
    image_df['mouse_id'] = session.metadata['mouse_id']
    image_df['behavior_session_id'] = session.metadata['behavior_session_id']   
    image_df['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
    image_df['cre_line'] = session.metadata['cre_line']

    # Add post omission
    image_df['post_omitted_1'] = image_df['omitted'].shift(1,fill_value=False)
    image_df['post_omitted_2'] = image_df['omitted'].shift(2,fill_value=False)

    # Add post change
    image_df['post_miss_1'] = image_df['miss'].shift(1)
    image_df['post_miss_2'] = image_df['miss'].shift(2)
    image_df['post_hit_1'] = image_df['hit'].shift(1)
    image_df['post_hit_2'] = image_df['hit'].shift(2)

    # Add pre change
    image_df['pre_miss_1'] = image_df['miss'].shift(-1)
    image_df['pre_hit_1'] = image_df['hit'].shift(-1)

    # Add pre omission licking
    image_df['pre_omission_lick'] = image_df['omitted'] & \
        image_df['lick_bout_start'].shift(-1)
    image_df['pre_omission_no_lick'] = image_df['omitted'] & \
        (~image_df['lick_bout_start']).shift(-1) 

    # Add running speed
    if run_df is not None:
        image_df = pd.merge(image_df, run_df, on='stimulus_presentations_id')
        image_df = image_df.rename(columns={'speed':'running_speed'})

    # Add pupil speed
    if pupil_df is not None:
        image_df = pd.merge(image_df, pupil_df, on='stimulus_presentations_id')

    # Save
    ophys_experiment_id = session.metadata['ophys_experiment_id']
    path = get_path(cell_specimen_id, ophys_experiment_id, 'cell','image_df',\
        data,first=first,second=second,image=image)
    image_df.to_hdf(path,key='df')

    return image_df


def build_full_df_cell(session, cell_specimen_id,summary_df,data):

    # Get neural activity
    cell_df = get_cell_df(session,cell_specimen_id,data)
    
    # Get the max response to each image presentation   
    full_df = get_full_df(cell_df, session, cell_specimen_id,summary_df,data) 
    return full_df

    
def get_full_df(cell_df, session,cell_specimen_id,summary_df,data,
    first=False,second=False,image=False):
    
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
        averages = get_full_average(session, averages, full_df,conditions[c])

    averages['cell_specimen_id'] = cell_specimen_id
    averages['mouse_id'] = session.metadata['mouse_id']
    averages['behavior_session_id'] = session.metadata['behavior_session_id']   
    averages['ophys_experiment_id'] = session.metadata['ophys_experiment_id']
    averages['cre_line'] = session.metadata['cre_line']

    bsid = session.metadata['behavior_session_id']
    row = summary_df.set_index('behavior_session_id').loc[bsid]
    averages['experience_level'] = row['experience_level']
    averages['visual_strategy_session'] = row['visual_strategy_session']
 
    # Save
    ophys_experiment_id = session.metadata['ophys_experiment_id']
    path = get_path(cell_specimen_id, ophys_experiment_id, 'cell','full_df',data,\
        first=first,second=second,image=image)
    averages.to_hdf(path,key='df')

    return averages


def get_engagement_check(session, condition):
    engaged = ('engaged' in condition[0]) & ('disengaged' not in condition[0])
    disengaged = 'disengaged' in condition[0]
    neither = (not engaged) & (not disengaged)
    min_engaged_fraction= 0.05
    engaged_fraction = np.nanmean(session.behavior_df['engaged'])

    if neither:
        #print('  '+condition[0])
        return True
    if engaged:
        more_than_threshold_engaged = engaged_fraction > min_engaged_fraction
        #print('E '+condition[0]+' '+str(more_than_threshold_engaged))
        return more_than_threshold_engaged 
    if disengaged:
        more_than_threshold_disengaged = engaged_fraction < (1-min_engaged_fraction)
        #print('D '+condition[0]+' '+str(more_than_threshold_disengaged))
        return more_than_threshold_disengaged 


def get_full_average(session, averages, full_df, condition):
 
    # Check to see if this session had sufficient time in the relevant
    # engagement state 
    engagement_check = get_engagement_check(session, condition)
 
    # Get conditional average
    if condition[1]=='':
        x = full_df.groupby('time')['response'].mean()
    else:
        x = full_df.query(condition[1]).groupby('time')['response'].mean()

    # Add to dataframe   
    if (len(x) == 0) or (not engagement_check):
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
        'image':['image','(not omitted) & (not is_change)'],
        'change':['change','is_change'],
        'omission':['omission','omitted'],
        'hit':['hit','is_change & rewarded'],
        'miss':['miss','is_change & not rewarded'],
        'licked':['licked','lick_bout_start'],
        'engaged_v1_image':['engaged_v1_image','(not omitted) & (not is_change) & engagement_v1'],
        'engaged_v2_image':['engaged_v2_image','(not omitted) & (not is_change) & engagement_v2'],
        'disengaged_v1_image':['disengaged_v1_image','(not omitted) & (not is_change) & (not engagement_v1)'],
        'disengaged_v2_image':['disengaged_v2_image','(not omitted) & (not is_change) & (not engagement_v2)'],
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
