import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.build_dataframes as bd

def metrics_by_strategy(avg,metric,bins=35,cell='Exc'):
    plt.figure()
    bins = plt.hist(avg.query('visual_strategy_session')[metric],
        alpha=.3,density=True,bins=bins,color='darkorange',label=cell+' Visual')
    plt.hist(avg.query('not visual_strategy_session')[metric],
        alpha=.3, density=True,bins=bins[1],color='blue',label=cell+' Timing')
    plt.ylim(top=1.1*np.max(bins[0][1:-1]))
    #plt.xlim(bins[1][1],bins[1][-2])
    plt.xlim(-.95,.95)
    ax = plt.gca()
    plt.axvline(0,linestyle='--',color='gray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.ylabel('density',fontsize=16)
    plt.xlabel(metric.replace('_',' '),fontsize=16)
    plt.legend()
    plt.tight_layout()

def compute_metrics(summary_df, cre, data='events',first=False, second=False):

    # Load the image_df    
    df = bd.load_population_df(data,'image_df',cre,first=first,second=second)
    
    # limit to familiar
    familiar_summary_df = summary_df.query('experience_level == "Familiar"')
    familiar_bsid = familiar_summary_df['behavior_session_id'].unique()
    df.drop(df[~df['behavior_session_id'].isin(familiar_bsid)].index, inplace=True)

    # Annotate image types
    df['type'] = 'image'
    df.at[df['omitted'],'type'] = 'omission'
    df.at[df['hit']==1,'type'] = 'hit'
    df.at[df['miss']==1,'type'] = 'miss'
    df.at[df['post_omitted_1'],'type'] = 'post_omitted'

    # Get average value for each image type for each cell
    avg = df.groupby(['ophys_experiment_id','cell_specimen_id','type'])['response'].mean().unstack().reset_index()

    # Compute metrics
    avg['omission_vs_image']=(avg['omission']-avg['image'])/(avg['omission']+avg['image'])
    avg['post_omitted_vs_image']=(avg['post_omitted']-avg['image'])/(avg['post_omitted']+avg['image'])
    avg['hit_vs_image'] = (avg['hit']-avg['image'])/(avg['hit']+avg['image'])   
    avg['miss_vs_image'] = (avg['miss']-avg['image'])/(avg['miss']+avg['image'])   
    avg['hit_vs_miss'] = (avg['hit']-avg['miss'])/(avg['hit']+avg['miss'])   
    
    # Add information
    experiment_table = glm_params.get_experiment_table()
    cols = ['ophys_experiment_id','behavior_session_id','targeted_structure',\
        'binned_depth']
    avg = pd.merge(avg, experiment_table.reset_index()[cols],on='ophys_experiment_id')
    avg = pd.merge(avg, summary_df[['behavior_session_id','visual_strategy_session']],
        on='behavior_session_id')
    return avg




