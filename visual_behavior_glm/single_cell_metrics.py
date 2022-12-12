import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.build_dataframes as bd

PSTH_DIR = '/home/alex.piet/codebase/behavior/PSTH/'

def metrics_by_strategy(avg,metric,bins=35,cell='Exc',in_ax=None,remove_1=False):
    if in_ax is None:
        fig, ax = plt.subplots()
    else:
        ax = in_ax   

    if remove_1:
        avg=avg.query('({} > -1)&({} < 1)'.format(metric, metric))
        vis = avg.query('visual_strategy_session')[metric].dropna()
        tim = avg.query('not visual_strategy_session')[metric].dropna()   
    else:
        vis = avg.query('visual_strategy_session')[metric].dropna()
        tim = avg.query('not visual_strategy_session')[metric].dropna()
    vis_mean = vis.mean()
    tim_mean = tim.mean()

    bins = ax.hist(vis,alpha=.3,density=True,bins=bins,color='darkorange',
        label=cell+' Visual')
    ax.hist(tim,alpha=.3, density=True,bins=bins[1],color='blue',label=cell+' Timing')
    ymax = np.max(bins[0][1:-1])
    ax.set_ylim(top=1.1*ymax)
    ax.plot(vis_mean, 1.05*ymax,'v',color='darkorange')
    ax.plot(tim_mean, 1.05*ymax,'v',color='blue')
    ax.set_xlim(-1,1)
    ax.axvline(0,linestyle='--',color='gray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylabel('density',fontsize=16)
    ax.set_xlabel(metric.replace('_',' '),fontsize=16)
    #ax.legend()

    if in_ax is None:
        plt.tight_layout()

def plot_all(avg,bins=35,cell='Exc',savefig=False,remove_1=False):
    fig, ax = plt.subplots(2,3,figsize=(10,6))

    metrics_by_strategy(avg, 'omission_vs_image', bins=bins, cell=cell,in_ax=ax[0,0],remove_1=remove_1)
    metrics_by_strategy(avg, 'post_omitted_vs_image', bins=bins, cell=cell,in_ax=ax[1,0],remove_1=remove_1)
    metrics_by_strategy(avg, 'hit_vs_image', bins=bins, cell=cell,in_ax=ax[0,1],remove_1=remove_1)   
    metrics_by_strategy(avg, 'post_hit_vs_image', bins=bins, cell=cell,in_ax=ax[1,1],remove_1=remove_1)
    metrics_by_strategy(avg, 'miss_vs_image', bins=bins, cell=cell,in_ax=ax[0,2],remove_1=remove_1)
    metrics_by_strategy(avg, 'hit_vs_miss', bins=bins, cell=cell,in_ax=ax[1,2],remove_1=remove_1)
    plt.tight_layout()    

    if savefig:
        filepath = PSTH_DIR + 'events/cell_metrics/'+cell+'_all_metrics.svg'
        plt.savefig(filepath)
        print('Figure saved to: {}'.format(filepath))

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
    df.at[df['post_hit_1']==1,'type'] = 'post_hit'

    # Get average value for each image type for each cell
    avg = df.groupby(['ophys_experiment_id','cell_specimen_id','type'])\
        ['response'].mean().unstack().reset_index()

    # Compute metrics
    avg['omission_vs_image']=(avg['omission']-avg['image'])/(avg['omission']+avg['image'])
    avg['post_omitted_vs_image']=(avg['post_omitted']-avg['image'])/\
        (avg['post_omitted']+avg['image'])
    avg['post_hit_vs_image']=(avg['post_hit']-avg['image'])/(avg['post_hit']+avg['image'])
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




