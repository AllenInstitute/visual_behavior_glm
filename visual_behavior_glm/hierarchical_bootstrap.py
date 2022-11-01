import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

'''
    df = hb.make_data()
    means = hb.bootstrap(df)
    hb.plot_data(df,means) 
    
'''
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
            subject_mean = np.random.randn()*level_2_var
            for cell in range(0, n3):
                mean = group*level_1_diff + \
                    subject_mean + \
                    np.random.randn()*level_3_var
                this_df = {
                    'group':group,
                    'subject':subject+group*n2,
                    'cell':cell,
                    'response':mean
                    }
                dfs.append(pd.DataFrame(this_df,index=[0]))
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def make_data_2():
    level_1_diff = 1
    level_2_var = 1
    level_3_var = 1
    level_4_var = 1

    n1 = 2
    n2 = 3
    n3 = 100
    n4 = 10
    dfs = []
    
    for group in range(0,n1):
        for subject in range(0,n2):
            subject_mean = np.random.randn()*level_2_var
            for cell in range(0, n3):
                mean = group*level_1_diff + \
                    subject_mean + \
                    np.random.randn()*level_3_var
                for image in range(0,n4):
                    response = mean+np.random.randn()*level_4_var
                    this_df = {
                        'group':group,
                        'subject':subject+group*n2,
                        'cell':cell,
                        'image':image,
                        'response':response
                        }
                    dfs.append(pd.DataFrame(this_df,index=[0]))
    df = pd.concat(dfs).reset_index(drop=True)
    return df

def compare_hierarchy(df):
    bootstrap_subjects = bootstrap(df, levels=['group','subject'])
    bootstrap_cells = bootstrap(df, levels=['group','subject','cell'])
    bootstrap_images = bootstrap(df, levels=['group','subject','cell','image'])

def bootstrap(df,metric='response', levels=['group','subject','cell'],nboots=100):
    groups = df[levels[0]].unique()
    means = {}
    for g in groups:
        means[g]= []
        temp = df.query('{} == @g'.format(levels[0]))
        for i in tqdm(range(0,nboots)):
            sum_val, count = sample_hierarchically(temp, metric, levels[1:])
            means[g].append(sum_val/count)            
    return means

def sample_hierarchically(df, metric, levels):
    if len(levels) == 1:
        sum_val = df[metric].sample(n=len(df),replace=True).sum()
        count = len(df)
        return sum_val, count  
    else:
        items = df[levels[0]].unique()     
        n = len(items)
        sum_val = 0
        count = 0
        for i in range(0,n):
            choice = np.random.choice(items)
            temp = df.query('{} == @choice'.format(levels[0]))
            temp_sum_val, temp_count = sample_hierarchically(temp, metric, levels[1:])
            sum_val +=temp_sum_val
            count += temp_count
        return sum_val, count
        
def test(df,means):
    diff = np.array(means[1]) - np.array(means[0])
    p_boot = np.sum(diff <=0)/len(diff)

    a = df.query('group == 0')['response']
    b = df.query('group == 1')['response']
    p_ttest = stats.ttest_ind(a,b).pvalue

    return p_boot, p_ttest

def plot_data(df,means):
    stats = test(df,means)
    fig, ax = plt.subplots(1,4,figsize=(8,4),sharey=True)
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
    ax[0].text(0,2,'p value: {:0.3f}'.format(stats[1]))

    ax[3].plot(0, mean.loc[0],'o',color='r',alpha=.5)
    ax[3].plot(1, mean.loc[1],'o',color='b',alpha=.5)
    ax[3].errorbar(0, mean.loc[0],np.std(means[0]),color='r',alpha=.5)
    ax[3].errorbar(1, mean.loc[1],np.std(means[1]),color='b',alpha=.5)
    ax[3].set_ylabel('response')
    ax[3].set_xlabel('groups')
    ax[3].set_title('bootstrap')
    ax[3].set_xlim(-.5,1.5)
    ax[3].text(0,2,'p value: {:0.3f}'.format(stats[0]))

    mean = df.groupby(['group','subject'])['response'].mean().to_frame().reset_index()
    sem = df.groupby(['group','subject'])['response'].sem().to_frame().reset_index()
    mean['sem'] = sem['response']
    tab20 = plt.get_cmap('tab10')
    mapper = {
        0:tab20(0),
        1:tab20(1),
        2:tab20(2),
        3:tab20(4),
        4:tab20(5),
        5:tab20(6)
        }
    for index, row in mean.iterrows():
        color = mapper[row.subject]
        ax[1].plot(row.group+index*.05, row.response,'o',color=color,alpha=.5)
        ax[1].errorbar(row.group+index*.05, row.response, row['sem'],color=color,alpha=.5)
    ax[1].set_ylabel('response')
    ax[1].set_xlabel('groups')
    ax[1].set_title('subject mean')
    ax[1].set_xlim(-.5,1.5)

    mean = df.groupby(['group','subject','cell'])['response']\
        .mean().to_frame().reset_index()

    for index, row in mean.iterrows():
        xoffset = np.random.rand()*.1
        color = mapper[row.subject]
        ax[2].plot(row.group+xoffset, row.response,'o',color=color,alpha=.1)
    ax[2].set_ylabel('response')
    ax[2].set_xlabel('groups')
    ax[2].set_title('cell mean')
    ax[2].set_xlim(-.5,1.5)

    plt.tight_layout()



def plot_data_2(df,means):
    stats = test(df,means)
    fig, ax = plt.subplots(1,5,figsize=(10,4),sharey=True)
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
    ax[0].text(0,2,'p value: {:0.3f}'.format(stats[1]))

    ax[4].plot(0, mean.loc[0],'o',color='r',alpha=.5)
    ax[4].plot(1, mean.loc[1],'o',color='b',alpha=.5)
    ax[4].errorbar(0, mean.loc[0],np.std(means[0]),color='r',alpha=.5)
    ax[4].errorbar(1, mean.loc[1],np.std(means[1]),color='b',alpha=.5)
    ax[4].set_ylabel('response')
    ax[4].set_xlabel('groups')
    ax[4].set_title('bootstrap')
    ax[4].set_xlim(-.5,1.5)
    ax[4].text(0,2,'p value: {:0.3f}'.format(stats[0]))

    mean = df.groupby(['group','subject'])['response'].mean().to_frame().reset_index()
    sem = df.groupby(['group','subject'])['response'].sem().to_frame().reset_index()
    mean['sem'] = sem['response']
    tab20 = plt.get_cmap('tab10')
    mapper = {
        0:tab20(0),
        1:tab20(1),
        2:tab20(2),
        3:tab20(4),
        4:tab20(5),
        5:tab20(6)
        }
    for index, row in mean.iterrows():
        color = mapper[row.subject]
        ax[1].plot(row.group+index*.05, row.response,'o',color=color,alpha=.5)
        ax[1].errorbar(row.group+index*.05, row.response, row['sem'],color=color,alpha=.5)
    ax[1].set_ylabel('response')
    ax[1].set_xlabel('groups')
    ax[1].set_title('subject mean')
    ax[1].set_xlim(-.5,1.5)

    mean = df.groupby(['group','subject','cell'])['response']\
        .mean().to_frame().reset_index()
    for index, row in mean.iterrows():
        xoffset = np.random.rand()*.1
        color = mapper[row.subject]
        ax[2].plot(row.group+xoffset, row.response,'o',color=color,alpha=.1)
    ax[2].set_ylabel('response')
    ax[2].set_xlabel('groups')
    ax[2].set_title('cell mean')
    ax[2].set_xlim(-.5,1.5)

    ax[3].plot(df.group+np.random.rand(len(df))*.2, df.response,'ko',alpha=.01)
    ax[3].set_ylabel('response')
    ax[3].set_xlabel('groups')
    ax[3].set_title('image responses')
    ax[3].set_xlim(-.5,1.5)

    plt.tight_layout()



