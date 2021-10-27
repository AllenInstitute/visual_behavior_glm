import visual_behavior_glm.GLM_visualization_tools as gvt
import visual_behavior.ophys.response_analysis.cell_metrics as cell_metrics
import visual_behavior.data_access.loading as loading
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
import visual_behavior_glm.GLM_params as glm_params

def plot_all(results_metrics,version):
    evaluate_against_metrics(results_metrics,xmetric='trace_mean_over_std',version=version,savefig=True,label='all_events',title='All Cells')
    evaluate_against_metrics(results_metrics.query('equipment_name == "MESO.1"'),xmetric='trace_mean_over_std',version=version,savefig=True,label='mesoscope_events',title='Mesoscope')
    evaluate_against_metrics(results_metrics.query('equipment_name != "MESO.!"'),xmetric='trace_mean_over_std',version=version,savefig=True,label='scientifica_events',title='Scientifica')

def make_main(results_pivoted, run_params):
    metrics_df, label = get_metrics()
    results_metrics = merge_cell_metrics_table(metrics_df, results_pivoted)
    evaluate_against_metrics(results_metrics,savefig=True, version=run_params['version'], label=label)

def make_dropout(results,run_params,dropout):
    metrics_df,label=get_metrics()
    results_metrics = merge_cell_metrics_table(metrics_df, results.query('dropout ==@dropout'))
    results_metrics = results_metrics[~results_metrics['variance_explained'].isnull()]
    evaluate_against_metrics(results_metrics,ymetric='variance_explained',title='All-images dropout vs. images reliability')

def get_metrics():
    #experiments_table = loading.get_platform_paper_experiment_table()
    #oeids = experiments_table.index.values
    #metrics_df = cell_metrics.load_cell_metrics_table_for_experiments(oeids,'images','all_images','full_session',use_events=False, filter_events=False)
    metrics_df = cell_metrics.load_metrics_table_for_experiments('all_experiments','traces','full_session','full_session',use_events=True,filter_events=True)
    label='full_trace'
    return metrics_df,label

def merge_cell_metrics_table(metrics_df, results_pivoted):
    metrics_df['identifier'] = [str(x[0])+'_'+str(x[1]) for x in zip(metrics_df['ophys_experiment_id'],metrics_df['cell_specimen_id'])]
    results_metrics = pd.merge(results_pivoted, metrics_df, on='identifier')
    return results_metrics

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
    #plt.xlim(left=0) 
    plt.legend()
    if title is not None:
        plt.title(title)

    if savefig:
        run_params = glm_params.load_run_json(version)
        filepath = os.path.join(run_params['figure_dir'], 'performance_vs_'+xmetric+'_'+label+'.png') 
        plt.savefig(filepath)

  
 

