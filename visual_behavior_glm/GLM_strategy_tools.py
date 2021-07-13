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

def scatter_dataset(results_beh, run_params,threshold=0.01, xmetric='strategy_dropout_index', ymetric='omissions',sessions=[1,3,4,6]):
    fig, ax = plt.subplots(3,len(sessions)+1, figsize=(18,10))
    fits = {}
    cres = ['Slc17a7-IRES2-Cre', 'Vip-IRES-Cre','Sst-IRES-Cre']
    ymins = []
    ymaxs =[]
    for dex,cre in enumerate(cres):
        col_start = dex == 0
        fits[cre] = scatter_by_session(results_beh, run_params, cre_line = cre, threshold=threshold, xmetric=xmetric, ymetric=ymetric, sessions=sessions, ax = ax[dex,:],col_start=col_start)
        ymins.append(fits[cre]['yrange'][0])
        ymaxs.append(fits[cre]['yrange'][1])
    
    for dex, cre in enumerate(cres):
        ax[dex,-1].set_ylim(np.min(ymins),np.max(ymaxs))
    return fits

def scatter_by_session(results_beh, run_params, cre_line=None, threshold=0.01,xmetric='strategy_dropout_index',ymetric='omissions',sessions=[1,3,4,6],ax=None,col_start=False):
    if ax is None:
        fig, ax = plt.subplots(1,len(sessions)+1, figsize=(18,3.25))

    fits = {}
    for dex,s in enumerate(sessions):
        row_start = dex == 0
        fits[str(s)] = scatter_by_cell(results_beh,cre_line=cre_line,xmetric=xmetric, ymetric=ymetric, sessions=[s],title='Session '+str(s),ax=ax[dex],row_start=row_start,col_start=col_start)

    ax[-1].axhline(0, linestyle='--',color='k',alpha=.25)   
    for s in sessions:
        ax[-1].plot(s,fits[str(s)][0],'ko')
        ax[-1].plot([s,s], [fits[str(s)][0]-fits[str(s)][4],fits[str(s)][0]+fits[str(s)][4]], 'k--')
    ax[-1].set_ylabel('Regression slope')
    ax[-1].set_xlabel('Session Number')
    ax[-1].set_title(cre_line)
    plt.tight_layout()

    fits['yrange'] = ax[-1].get_ylim()
    fits['xmetric'] = xmetric
    fits['ymetric'] = ymetric
    fits['cre_line'] = cre_line
    fits['threshold'] = threshold
    fits['glm_version'] = run_params['version'] 
    return fits 

def scatter_by_cell(results_beh, cre_line=None, threshold=0.01, sessions=[1],xmetric='strategy_dropout_index',ymetric='omissions',title='',nbins=10,ax=None,row_start=False,col_start=False):
    g = results_beh.query('cre_line == @cre_line').query('variance_explained_full > @threshold').query('session_number in @sessions').dropna(axis=0, subset=[ymetric,xmetric]).copy()

    # Figure axis
    if ax is None:
        plt.figure()
        ax = plt.gca()
    
    # Plot Raw data
    ax.plot(g[xmetric], g[ymetric],'ko',alpha=.1,label='raw data')
    ax.set_xlabel(xmetric)
    if row_start:
        ax.set_ylabel(cre_line +'\n'+ymetric)
    else:
        ax.set_ylabel(ymetric)

    # Plot binned data
    g['binned_xmetric'] = pd.cut(g[xmetric],nbins,labels=False) 
    xpoints = g.groupby('binned_xmetric')[xmetric].mean()
    ypoints = g.groupby('binned_xmetric')[ymetric].mean()
    y_sem = g.groupby('binned_xmetric')[ymetric].sem()
    ax.plot(xpoints, ypoints, 'ro',label='binned data')
    ax.plot(np.array([xpoints,xpoints]), np.array([ypoints-y_sem,ypoints+y_sem]), 'r-')

    # Plot Linear regression
    x = linregress(g[xmetric], g[ymetric])
    ax.plot(g[xmetric], x[1]+x[0]*g[xmetric],'r-',label='r^2 = '+str(np.round(x.slope,4)))

    # Clean up
    if col_start:
        ax.set_title(title)
    #ax.legend()

    return x    

