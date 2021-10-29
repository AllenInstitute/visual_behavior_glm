import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior.ophys.response_analysis.cell_metrics as cell_metrics

def compute_all(results_pivoted_dff, results_pivoted_events,groups=['cre_line','equipment','targeted_structure','active']):
    '''
        Pass in the results_pivoted dataframes for the model fit to dff and events
        
        Returns a table of r2 values for each verison of the model compared against metrics computed on the dff and events traces
        
        computes r2 values for each group defined in groups
    '''
    metrics_dff = get_metrics(use_events=False)
    metrics_events = get_metrics(use_events=True)
    results_dff_dff = merge_cell_metrics_table(metrics_dff,results_pivoted_dff)
    results_events_dff = merge_cell_metrics_table(metrics_dff,results_pivoted_events)
    results_events_events = merge_cell_metrics_table(metrics_events,results_pivoted_events)
    r2_dff_dff = compute_r2(results_dff_dff,groups=groups).rename(columns={'r2':'glm_dff__metrics_dff'})
    r2_events_dff = compute_r2(results_events_dff,groups=groups).rename(columns={'r2':'glm_events__metrics_dff'})
    r2_events_events = compute_r2(results_events_events,groups=groups).rename(columns={'r2':'glm_events__metrics_events'})
    r2_dff_dff['glm_events__metrics_dff'] = r2_events_dff['glm_events__metrics_dff']
    r2_dff_dff['glm_events__metrics_events'] = r2_events_events['glm_events__metrics_events']
    return r2_dff_dff

def compute_r2(results_metrics,groups=['cre_line']):
    '''
        Computes a table of r2 values for each group defined in groups
        Computes r2 between the trace_mean_over_std column and variance_explained_full
        trace_mean_over_std can be computed on either dff or events, and should already be in the results_metrics table
    '''
    nlevels = len(groups)
    g = results_metrics.groupby(groups)
    r2 = g[['variance_explained_full','trace_mean_over_std']].corr().pow(2).round(decimals=3)
    r2 = r2.drop('trace_mean_over_std',level=nlevels,axis=0).drop(columns=['variance_explained_full']).droplevel(nlevels)
    r2 = r2.rename(columns={'trace_mean_over_std':'r2'})
    r2['count'] = g.size()
    return r2

def get_metrics(use_events=True,filter_events=False):
    metrics_df = cell_metrics.load_metrics_table_for_experiments('all_experiments','traces','full_session','full_session',use_events=use_events,filter_events=filter_events)
    return metrics_df

def merge_cell_metrics_table(metrics_df, results_pivoted):
    metrics_df['identifier'] = [str(x[0])+'_'+str(x[1]) for x in zip(metrics_df['ophys_experiment_id'],metrics_df['cell_specimen_id'])]
    results_metrics = pd.merge(results_pivoted, metrics_df, on='identifier')
    results_metrics['equipment'] = ['Mesoscope' if x=="MESO.1" else "Scientifica" for x in results_metrics['equipment_name']]
    results_metrics['active'] =['Active' if not x else 'Passive' for x in results_metrics['passive']]
    return results_metrics

def plot_all(results_metrics,version,metric='trace_mean_over_std',savefig=False):
    evaluate_against_metrics(results_metrics,xmetric=metric,version=version,savefig=savefig,label='all_events',title='All Cells')
    evaluate_against_metrics(results_metrics.query('equipment_name == "MESO.1"'),xmetric=metric,version=version,savefig=savefig,label='mesoscope_events',title='Mesoscope')
    evaluate_against_metrics(results_metrics.query('equipment_name != "MESO.!"'),xmetric=metric,version=version,savefig=savefig,label='scientifica_events',title='Scientifica')

def evaluate_against_metrics(results_metrics, ymetric='variance_explained_full',xmetric='trace_mean_over_std',savefig=False,version=None,label='',title=None,ylim=(0,1)):
    fig,ax = plt.subplots()
    cre_lines = np.sort(results_metrics['cre_line'].dropna().unique())
    jointplot = sns.scatterplot(
        data = results_metrics,
        x = xmetric, 
        y = ymetric,
        hue = 'cre_line',
        hue_order = cre_lines,
        alpha = 0.05,
        palette = [gvt.project_colors()[cre_line] for cre_line in cre_lines],
    )
    for cre in cre_lines:
        x = results_metrics.query('cre_line ==@cre')[xmetric].values.reshape((-1,1))
        y = results_metrics.query('cre_line ==@cre')[ymetric].values
        model = LinearRegression(fit_intercept=True).fit(x,y)
        sortx = np.sort(x).reshape((-1,1))
        y_pred = model.predict(sortx)
        score = round(model.score(x,y),2)
        ax.plot(sortx,y_pred, linestyle='--', color=gvt.project_colors()[cre],label=cre+', r^2='+str(score),linewidth=2)
       
    plt.ylabel(ymetric,fontsize=14)
    plt.xlabel(xmetric,fontsize=14)
    plt.ylim(ylim[0],ylim[1])

    plt.legend()
    if title is not None:
        plt.title(title)

    if savefig:
        run_params = glm_params.load_run_json(version)
        filepath = os.path.join(run_params['figure_dir'], 'performance_vs_'+xmetric+'_'+label+'.png') 
        plt.savefig(filepath)

  
 

