import numpy as np
import pandas as pd
import mindscope_utilities as m

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
