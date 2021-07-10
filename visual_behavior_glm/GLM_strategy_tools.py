import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_visualization_tools as gvt
from scipy.stats import linregress

BEH_STRATEGY_OUTPUT = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/behavior_model_output/_summary_table.pkl'
# TODO
# Separate computation and plotting code
# analyze engaged/disengaged separatation 
# set up folder for saving figures
# set up automatic figure saving
# save fits dictionary somewhere
# on scatter plot, add binned values on regression
# on scatter plot, include regression values (r^2 and slope)
# disengaged regression has nans
# set up regression by exposure number
# maybe try regressing against hit/miss difference?
# what filtering do we need to do on cells and sessions?

def make_strategy_figures(VERSION=None,run_params=None, results=None, results_pivoted=None, full_results=None, weights_df = None):
    
    # Analysis Dataframes 
    #####################
    if run_params is None:
        print('loading data')
        run_params, results, results_pivoted, weights_df, full_results = get_analysis_dfs(VERSION)
        print('making figues')
    results_beh = add_behavior_metrics(results_pivoted.copy())
    weights_beh = add_behavior_metrics(weights_df.copy())
  
    scatter_by_session(results_beh, run_params, cre_line ='Slc17a7-IRES2-Cre',ymetric='hits') 
    scatter_by_session(results_beh, run_params, cre_line ='Vip-IRES-Cre',ymetric='omissions') 

def get_ophys_summary_table():
    '''
        Loads the behavior summary file
    '''
    return pd.read_pickle(BEH_STRATEGY_OUTPUT) 

def add_behavior_metrics(df):
    '''
        Merges the behavioral summary table onto the dataframe passed in 
    '''  
    ophys = get_ophys_summary_table()
    out_df = pd.merge(df, ophys, on='behavior_session_id',suffixes=('','_ophys_table'))
    out_df['strategy'] = ['visual' if x else 'timing' for x in out_df['visual_strategy_session']]
    return out_df

def scatter_by_session(results_beh, run_params, cre_line=None, threshold=0.01,xmetric='strategy_dropout_index',ymetric='omissions',sessions=[1,3,4,6]):
    fits = {}
    for s in sessions:
        fits[str(s)] = scatter_by_cell(results_beh,cre_line=cre_line,xmetric=xmetric, ymetric=ymetric, sessions=[s],title='Session '+str(s))
    
    plt.figure()
    for s in sessions:
        plt.plot(s,fits[str(s)][0],'ko')
        plt.plot([s,s], [fits[str(s)][0]-fits[str(s)][4],fits[str(s)][0]+fits[str(s)][4]], 'k--')
    plt.ylabel('Regression slope')
    plt.xlabel('Session Number')  
    plt.tight_layout()
 
    fits['xmetric'] = xmetric
    fits['ymetric'] = ymetric
    fits['cre_line'] = cre_line
    fits['threshold'] = threshold
    fits['glm_version'] = run_params['version'] 
    return fits 

def scatter_by_cell(results_beh, cre_line=None, threshold=0.01, sessions=[1],xmetric='strategy_dropout_index',ymetric='omissions',title=''):
    g = results_beh.query('cre_line == @cre_line').query('variance_explained_full > @threshold').query('session_number in @sessions')

    plt.figure()
    plt.plot(g[xmetric], g[ymetric],'ko')
    plt.xlabel(xmetric)
    plt.ylabel(ymetric)
    x = linregress(g[xmetric], g[ymetric])
    plt.plot(g[xmetric], x[1]+x[0]*g[xmetric],'r-')
    plt.title(title)
    return x    

