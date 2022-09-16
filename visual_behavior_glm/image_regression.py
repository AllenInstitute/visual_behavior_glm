import numpy as np
import pandas as pd
import mindscope_utilities as m
import matplotlib.pyplot as plt
from sklearn.linear_model import GammaRegressor

def omission_analysis(vip_image_df,condition='omitted', experience_level='Familiar'):
    df = vip_image_df.query('{} & (experience_level == @experience_level)'.format(condition)).copy()
    df['had_response'] = df['response'] > 0
    response_prob = df\
        .groupby(['cell_specimen_id'])[['visual_strategy_session','had_response']]\
        .mean()
    response_prob['visual_strategy_session'] = response_prob['visual_strategy_session'].astype(bool)

    fig, ax = plt.subplots(1,2,figsize=(6,3))     
    ax[0].hist(response_prob.query('visual_strategy_session')['had_response'],
        density=True, bins=np.linspace(0,1,50), color='orange',alpha=.5,label='Visual')
    ax[0].hist(response_prob.query('not visual_strategy_session')['had_response'],
        density=True, bins=np.linspace(0,1,50), color='blue',alpha=.5,label='Timing')
    ax[0].set_ylabel('Prob. (cells)',fontsize=16)
    ax[0].set_xlabel('Response probability',fontsize=16)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].xaxis.set_tick_params(labelsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)  
    ax[0].legend() 

    #df = df.query('had_response').copy()
    
    ax[1].hist(df.query('visual_strategy_session')['response'], density=True, 
        bins=np.linspace(0,1,100),color='orange', alpha=.5,label='Visual')
    ax[1].hist(df.query('not visual_strategy_session')['response'], density=True, 
        bins=np.linspace(0,1,100),color='blue',alpha=.5,label='Timing')
    ax[1].set_ylabel('Prob. (omissions)',fontsize=16)
    ax[1].set_xlabel('Response amplitude',fontsize=16)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].xaxis.set_tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)  
    #ax[1].set_xlim(0,1)
    ax[1].set_ylim(top=5)
    ax[1].legend() 

    plt.suptitle('Vip, Omission, Familiar',fontsize=16)
    plt.tight_layout()

def gamma(df):
    df = df.query('omitted &(experience_level == "Familiar") & (response > 0.015)').copy()
     

def histogram_response(df,condition,cell='r'):
    fig, ax =plt.subplots()
    bins =np.linspace(0,1,400)
    df = df.query('omitted & (experience_level == "Familiar")')
    #plt.hist(df['response'],bins=bins)
    plt.hist(df.query('visual_strategy_session')['response'],
        bins=bins,alpha=.5,label='visual',density=True)
    plt.hist(df.query('not visual_strategy_session')['response'],
        bins=bins,alpha=.5,label='timing',density=True)
    plt.ylim(top=5)
    plt.xlim(0,1)
    plt.legend()
    plt.xlabel('Response')
    plt.ylabel('count')

def plot_QQ_engagement(image_df,cre,condition,experience_level,savefig=False,quantiles=200,ax=None,
    data='filtered_events'):
    
    # Prep data
    e_condition = 'engaged_v2_'+condition
    d_condition = 'disengaged_v2_'+condition
    df_e = image_df\
            .query('(condition==@e_condition)&(experience_level==@experience_level)')\
            .copy()     
    df_e['max'] = [np.nanmax(x) for x in df_e['response']] 

    df_d = image_df\
            .query('(condition==@d_condition)&(experience_level==@experience_level)')\
            .copy()     
    df_d['max'] = [np.nanmax(x) for x in df_d['response']] 
 
    y = df_e['max'].values
    x = df_d['max'].values
    quantiles = np.linspace(start=0,stop=1,num=int(quantiles))
    x_quantiles = np.nanquantile(x,quantiles, interpolation='linear')[1:-1]
    y_quantiles = np.nanquantile(y,quantiles, interpolation='linear')[1:-1]
    max_val = np.max([np.max(x_quantiles),np.max(y_quantiles)])
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_quantiles, y_quantiles, 'o-',alpha=.5)
    ax.plot([0,1.05*max_val],[0,1.05*max_val],'k--',alpha=.5)
    ax.set_ylabel('engaged quantiles',fontsize=16)
    ax.set_xlabel('disengage quantiles',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_aspect('equal')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title('{}, {}, {}'.format(cre, condition, experience_level),fontsize=16)

    # Save figure
    if savefig:
        filename = PSTH_DIR + data+'/QQ/' +\
            'QQ_{}_{}_{}.png'.format(cre,condition,experience_level)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 

    return ax


def plot_QQ_strategy(image_df,cre,condition,experience_level,savefig=False,quantiles=200,ax=None,
    data='filtered_events'):
    
    # Prep data
    if condition == "omission":
        df = image_df.query('omitted').copy()
        #    .query('(omitted)&(experience_level==@experience_level)')\
        #    .copy()     
    else:
        df = image_df.copy()
    #df['max'] = [np.nanmax(x) for x in df['response']] 
 
    y = df.query('visual_strategy_session')['response'].values
    x = df.query('not visual_strategy_session')['response'].values
    quantiles = np.linspace(start=0,stop=1,num=int(quantiles))
    x_quantiles = np.nanquantile(x,quantiles, interpolation='linear')[1:-1]
    y_quantiles = np.nanquantile(y,quantiles, interpolation='linear')[1:-1]
    max_val = np.max([np.max(x_quantiles),np.max(y_quantiles)])
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_quantiles, y_quantiles, 'o-',alpha=.5)
    ax.plot([0,1.05*max_val],[0,1.05*max_val],'k--',alpha=.5)
    ax.set_ylabel('visual session cell quantiles',fontsize=16)
    ax.set_xlabel('timing session cell quantiles',fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_aspect('equal')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title('{}, {}, {}'.format(cre, condition, experience_level),fontsize=16)

    # Save figure
    if savefig:
        filename = PSTH_DIR + data+'/QQ/' +\
            'QQ_{}_{}_{}.png'.format(cre,condition,experience_level)
        print('Figure saved to {}'.format(filename))
        plt.savefig(filename) 

    return ax





