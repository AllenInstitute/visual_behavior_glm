import numpy as np
import pandas as pd

'''
    Generates two dataframes for each cell
    1.  response_df
        Contains a 4 second response trace aligned to each image presentation, and relevant metrics (engaged, strategy weights, licked, running)
        the peak response_df is a strict subset of response_df
    2.  peak_response_df
        Just the peak response of each cell to each image presentation in the (100ms,800ms) window, and relevant metrics

'''


def build_response_df(session, cell_specimen_id):
    # Interpolate neural activity onto stimulus timestamps
    # 
    # Build response dataframe
    # Save response_df, peak_response_df
    return full_df

def build_response_df_session(session):
    # get session level behavior metrics
    # loop over cells 
        # build_response_df
    return

def aggregate_df(df_request, df_type='peak'):
    # df_type should be peak or full
    # df_request should be a df that contains either the cells, or sessions to aggregate (make a choice)
    
    


