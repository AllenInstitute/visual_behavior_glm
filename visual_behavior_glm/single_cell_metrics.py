import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.build_dataframes as bd

def metrics_by_strategy(avg,metric):
    plt.figure()
    plt.hist(avg.query('visual_strategy_session')[metric],
        alpha=.3,density=True,bins=50,color='darkorange')
    plt.hist(avg.query('not visual_strategy_session')[metric],
        alpha=.3, density=True,bins=50,color='blue')


def compute_metrics(summary_df, cre, data='events',first=False, second=False):

    # Load the image_df    
    df = bd.load_population_df(data,'image_df',cre,first=first,second=second)
    
    #experiment_table = glm_params.get_experiment_table()
    #df = bd.add_area_depth(df,experiment_table)
    #df = pd.merge(df, experiment_table.reset_index()[['ophys_experiment_id',
    #'binned_depth']],on='ophys_experiment_id')

    # limit to familiar
    df = pd.merge(df, summary_df[['behavior_session_id','experience_level']],
        on='behavior_session_id')
    df = df.query('experience_level == "Familiar"').copy()

    # Annotate image types
    df['type'] = 'image'
    df.at[df['omitted'],'type'] = 'omission'
    df.at[df['hit']==1,'type'] = 'hit'
    df.at[df['miss']==1,'type'] = 'miss'

    # Get average value for each image type for each cell
    avg = df.groupby(['ophys_experiment_id','cell_specimen_id','type'])['response'].mean().unstack().reset_index()

    # Compute metrics
    avg['omission_vs_image']=(avg['omission']-avg['image'])/(avg['omission']+avg['image'])
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




