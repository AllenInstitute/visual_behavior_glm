import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_data():
    level_1_diff = 1
    level_2_var = 1
    level_3_var = 1

    n1 = 2
    n2 = 3
    n3 = 100
    dfs = []
    
    for group in range(0,n1):
        for subject in range(0,n2):
            for cell in range(0, n3):
                mean = group*level_1_diff + \
                    np.random.rand()*level_2_var + \
                    np.random.rand()*level_3_var
                this_df = {
                    'group':group,
                    'subject':subject+group*n2,
                    'cell':cell,
                    'response':mean
                    }
                dfs.append(pd.DataFrame(this_df,index=[0]))
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def bootstrap(df,metric='response', levels=['group','subject','cell'],nboots=100):
    groups = df[levels[0]].unique()
    means = {}
    for g in groups:
        means[g]= []

    for i in range(0,nboots):
        sample = sample_hierarchically(df, metric, levels)            
    return

def sample_hierarchically(df, metric, levels):
    if len(levels) = 1:
        return df[metric].sample(n=len(df),replace=True)
    else:
        items = df[levels[0]].unique()     
        n = len(items)
        vals = []
        for i in range(0,n):
            choice = np.random.choice(items)
            vals.append(sample_hierarchically(df.query('{} == @choice'.format(levels[0]



def plot_data(df):
    fig, ax = plt.subplots(1,3,figsize=(8,4),sharey=True)
    mean = df.groupby(['group'])['response'].mean()
    sem = df.groupby(['group'])['response'].sem()
    ax[0].plot(0, mean.loc[0],'o',color='r',alpha=.5)
    ax[0].plot(1, mean.loc[1],'o',color='b',alpha=.5)
    ax[0].errorbar(0, mean.loc[0],sem.loc[0],color='r',alpha=.5)
    ax[0].errorbar(1, mean.loc[1],sem.loc[1],color='b',alpha=.5)
    ax[0].set_ylabel('response')
    ax[0].set_xlabel('groups')
    ax[0].set_title('group mean + sem')
    ax[0].set_xlim(-.5,1.5)
    
    mean = df.groupby(['group','subject'])['response'].mean().to_frame().reset_index()
    sem = df.groupby(['group','subject'])['response'].sem().to_frame().reset_index()
    mean['sem'] = sem['response']
    mapper = {
        0:'r',
        1:'b'
        }
    for index, row in mean.iterrows():
        color = mapper[row.group]
        ax[1].plot(row.group+index*.05, row.response,'o',color=color,alpha=.5)
        ax[1].errorbar(row.group+index*.05, row.response, row['sem'],color=color,alpha=.5)
    ax[1].set_ylabel('response')
    ax[1].set_xlabel('groups')
    ax[1].set_title('subject mean + sem')
    ax[1].set_xlim(-.5,1.5)

    mean = df.groupby(['group','subject','cell'])['response']\
        .mean().to_frame().reset_index()
    mapper = {
        0:'r',
        1:'b'
        }

    for index, row in mean.iterrows():
        xoffset = np.random.rand()*.1
        color = mapper[row.group]
        ax[2].plot(row.group+xoffset, row.response,'o',color=color,alpha=.1)
    ax[2].set_ylabel('response')
    ax[2].set_xlabel('groups')
    ax[2].set_title('cell mean')
    ax[2].set_xlim(-.5,1.5)




    plt.tight_layout()
