import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import multinomial_proportions_confint as mpc
K = 5

'''
- if more than one session type, just taking the first
- how to deal with NaNs, there could always be more cells "lurking"
    - Can I just assume that all cells show up at least once?
    - repeat with and without nan cells?
- am I computing the estimate correctly? Right math, right code?
    should use np.dot(), not *
    order of operations
- filter by cre line
- Need to incorporate NaNs in transition matrices
- Need to flexibly incorporate multiple estimates
- Add NaNs to distribution
- Add error bars to measured distribution
- Add error bars to estimated distribution
- remove tracked cells from estimate?
'''

def load_table():
    filepath = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/glm/kmeans_glm_novel.hdf'
    df = pd.read_hdf(filepath)
    return df

def clean_table(df):
    df['identifier'] = [str(x[0])+'_'+str(x[1]) for x in zip(df['cell_specimen_id'], df['session_number'])]
    df = df.drop_duplicates(subset='identifier').copy()
    return df

def build_pivot_table(cre='all'):
    df = load_table()
    if cre != 'all':
        df = df.query('cre_line == @cre').copy()
    df = clean_table(df)
    df_pivot = pd.pivot_table(df, index='cell_specimen_id',columns='session_number',values='K_cluster')
    df_pivot = df_pivot.rename(columns={1:'1',2:'2',3:'3',4:'4',5:'5',6:'6'})
    df_pivot['tracked_13'] = df_pivot[['1','3']].isnull().sum(axis=1) ==0
    df_pivot['tracked_12'] = df_pivot[['1','2']].isnull().sum(axis=1) ==0
    df_pivot['tracked_34'] = df_pivot[['3','4']].isnull().sum(axis=1) ==0
    return df_pivot

def compute_distribution(df_pivot, session,session_dists={},dropna=False):
    session_dists[str(session)] = df_pivot[str(session)].value_counts(dropna=dropna)
    for val in range(0,K):
        if val not in session_dists[str(session)].index:
            session_dists[str(session)].loc[val] = 0
        if (not dropna) & (np.nan not in session_dists[str(session)].index):
            session_dists[str(session)].loc[np.nan] = 0
    session_dists[str(session)] = session_dists[str(session)].sort_index(na_position='first')
    session_dists['normalized_'+str(session)] = session_dists[str(session)]/sum(session_dists[str(session)])
    session_dists['ci_'+str(session)] = mpc(session_dists[str(session)])
    return session_dists

def compute_all_distributions(df_pivot):
    session_dists = {}
    session_dists_no_nans = {}
    for i in range(1,7):
        session_dists = compute_distribution(df_pivot, i, session_dists = session_dists)
        session_dists_no_nans = compute_distribution(df_pivot, i, session_dists = session_dists_no_nans, dropna=True)
    return session_dists, session_dists_no_nans 

def check_distributions(cre='all'):
    df_pivot = build_pivot_table(cre=cre)
    session_dists, session_dists_no_nans = compute_all_distributions(df_pivot)
    session_dists_13, session_dists_no_nans_13 = compute_all_distributions(df_pivot.query('tracked_13'))
    session_dists_12, session_dists_no_nans_12 = compute_all_distributions(df_pivot.query('tracked_12'))
    session_dists_34, session_dists_no_nans_34 = compute_all_distributions(df_pivot.query('tracked_34'))
    session_dists_n13, session_dists_no_nans_n13 = compute_all_distributions(df_pivot.query('not tracked_13'))
    session_dists_n12, session_dists_no_nans_n12 = compute_all_distributions(df_pivot.query('not tracked_12'))
    session_dists_n34, session_dists_no_nans_n34 = compute_all_distributions(df_pivot.query('not tracked_34'))

    ## Result 1, are clusters different across sessions?
    plot_distribution_by_session(session_dists, [str(x) for x in range(1,7)],title=cre)
    ## Result 2, novelty clusters are different without nans!
    plot_distribution_by_session(session_dists_no_nans, [str(x) for x in range(1,7)],dropna=True,title=cre)

    #plot_distribution_by_group([session_dists, session_dists_13],1,labels=['all','tracked 1,3'])
    plot_distribution_by_group([session_dists_no_nans, session_dists_no_nans_13, session_dists_no_nans_n13],1,labels=['all','tracked 1,3','not tracked 1,3'],dropna=True,title='Session 1')
    plot_distribution_by_group([session_dists_no_nans, session_dists_no_nans_13, session_dists_no_nans_n13],3,labels=['all','tracked 1,3','not tracked 1,3'],dropna=True,title='Session 3')


    #plot_distribution_by_group([session_dists, session_dists_12],1,labels=['all','tracked 1,2'])
    plot_distribution_by_group([session_dists_no_nans, session_dists_no_nans_12,session_dists_no_nans_n12],1,labels=['all','tracked 1,2','not tracked 1,2'],dropna=True,title='Session 1')
    plot_distribution_by_group([session_dists_no_nans, session_dists_no_nans_12,session_dists_no_nans_n12],2,labels=['all','tracked 1,2','not tracked 1,2'],dropna=True,title='Session 2')

    plot_distribution_by_group([session_dists_no_nans, session_dists_no_nans_34,session_dists_no_nans_n34],3,labels=['all','tracked 3,4','not tracked 3,4'],dropna=True,title='Session 3')
    plot_distribution_by_group([session_dists_no_nans, session_dists_no_nans_34,session_dists_no_nans_n34],4,labels=['all','tracked 3,4','not tracked 3,4'],dropna=True,title='Session 4')


def plot_distribution_by_session(session_dists,sessions,dropna=False,title=''):
    plt.figure()
    ax = plt.gca()
    if dropna:
        x = np.arange(0,K)
    else:
        x = np.arange(0,K+1)
    width=0.7/len(sessions)
    for dex,s in enumerate(sessions):
        xvec = x+width*(dex+1)-.35
        ax.bar(xvec,session_dists['normalized_'+str(s)], width, label=s)
        ax.plot(np.array([xvec,xvec]),session_dists['ci_'+str(s)].T,'k-')
    ax.set_ylabel('Fraction',fontsize=16)
    ax.set_xlabel('Cluster', fontsize=16)
    ax.set_ylim(0,1)
    ax.set_xticks(x)
    ax.set_xticklabels(session_dists['normalized_1'].index.values)
    ax.tick_params(axis='both',labelsize=14)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()

def plot_distribution_by_group(session_dists_lists, session, dropna=False, title='',labels=[]):
    plt.figure()
    ax = plt.gca()
    if dropna:
        x = np.arange(0,K)
    else:
        x = np.arange(0,K+1)
    width=0.7/len(session_dists_lists)
    for dex,s in enumerate(session_dists_lists):
        xvec = x+width*(dex+1)-.35
        ax.bar(xvec,s['normalized_'+str(session)], width, label=labels[dex])
        ax.plot(np.array([xvec,xvec]),s['ci_'+str(session)].T,'k-')
    ax.set_ylabel('Fraction',fontsize=16)
    ax.set_xlabel('Cluster', fontsize=16)
    ax.set_ylim(0,1)
    ax.set_xticks(x)
    ax.set_xticklabels(session_dists_lists[0]['normalized_1'].index.values)
    ax.tick_params(axis='both',labelsize=14)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()

#####################
def make_all(df):
    Ts = {}
    sessions = [1,2,3,4,5,6]
    for sdex, s in enumerate(sessions):
        for edex, e in enumerate(sessions[sdex+1:]):
            print('Computing '+str(s)+' '+str(e))
            Ts[str(s)+str(e)] = make_matrix(df, sessions=[s,e])
    normTs = {}
    for m in Ts:
        normTs[m] = normalize_matrix(Ts[m])
    return Ts, normTs

def build_estimates(normTs):
    sessions = [1,2,3,4,5,6]
    for sdex, s in enumerate(sessions):
        for edex, e in enumerate(sessions[sdex+2:]):
            print('Computing '+str(s)+' '+str(e))
            if e-s ==2:
                # we shouldn't need to renormalize
                normTs[str(s)+str(e)] = normTs[str(s)+str(e-1)]*normTs[str(s-1)+str(e)]
            else:
                # can we rely on subestimates already been computed? depends on order
                normTs[str(s)+str(e)] = normTs[str(s)+str(e-1)]*normTs[str(s-1)+str(e)]


def make_matrix(df, sessions=[1,3]):
    # assert sessions is length 2

    # Make empty matrix
    T = np.empty((K+1,K+1))
    T[:] = 0 


    dfs = df.query('session_number in @sessions')
    cells = dfs.cell_specimen_id.unique()

    # So slow!
    # Can probably speed up by making table of cells, columns as sessions, then doing a groupby
    for cell in cells:
        sA = dfs.query('(cell_specimen_id == @cell) & (session_number == @sessions[0])')
        sB = dfs.query('(cell_specimen_id == @cell) & (session_number == @sessions[1])')
        if len(sA) == 0:
            clusterA = K
        else:
            clusterA = sA.iloc[0]['K_cluster']
        if len(sB) == 0:
            clusterB = K
        else:
            clusterB = sB.iloc[0]['K_cluster']
        T[clusterB,clusterA] +=1
        # T(2nd session, 1st session)
    return T 

def normalize_matrix(T):
    normT = T/T.sum(axis=0)
    return normT

def sanity_check_matrix():
    T = np.empty((3,3))
    T[:] = 0
    T[0,0] = 1
    T[1,1] = 2
    T[2,2] = 3
    T[0,2] = 3 
    # makes a everything stay in its cluster, except for three cells that go from
    # cluster 2 to cluster 0
    return T

def plot_sanity_check():
    T = sanity_check_matrix()
    normT = normalize_matrix(T)
    plot_matrix(T,[1,2], k=2,normalized=False)
    plot_matrix(normT,[1,2], k=2)   
 
def plot_matrix(T,sessions,k=None,normalized=True,ax=None):
    if k is None:
        k=K
    if ax is None:
        plt.figure()
        ax = plt.gca()
    if normalized:
        cmap = 'Blues'
    else:
        cmap = 'Reds'
    im = ax.imshow(T,cmap=cmap)
    # triple check axis order, and value
    ax.set_ylabel('Session '+str(sessions[1]),fontsize=16)
    ax.set_xlabel('Session '+str(sessions[0]),fontsize=16)
    labels = [str(x) for x in list(range(0,k))]+['NaN']
    ax.set_xticks(range(0,k+1))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticks(range(0,k+1))
    ax.set_yticklabels(labels, fontsize=14)
    cbar = plt.colorbar(im,ax=ax)
    cbar.ax.tick_params(labelsize=14)
    if normalized:
        cbar.set_label('Prob('+str(sessions[1])+'|'+str(sessions[0])+')',fontsize=14)
    else:
         cbar.set_label('Count '+str(sessions[0])+'-> '+str(sessions[1])+'',fontsize=14)   
    plt.tight_layout()
    
def plot_normalization(Ts, normTs, sessions):
    key = ''.join([str(x) for x in sessions])
    fig, axes = plt.subplots(1,2,figsize=(8,3))
    plot_matrix(Ts[key],sessions, normalized=False, ax=axes[0])
    plot_matrix(normTs[key], sessions, ax=axes[1])
    plt.tight_layout()
    
def OLD_compute_distribution(df,session):
    session_dist = df.query('session_number == @session').groupby('K_cluster').count()[['cell_specimen_id']]
    cell_count = len(df.cell_specimen_id.unique()) - session_dist['cell_specimen_id'].sum()
    session_dist.at[K] = [cell_count]
    session_dist['normalized'] = session_dist['cell_specimen_id']/sum(session_dist['cell_specimen_id'])
    return session_dist

def compute_full_estimate(df,T, in_session,out_session):
    in_dist = compute_distribution(df,in_session)
    out_dist= compute_distribution(df, out_session)
    out_dist= compute_estimate(T,in_dist, out_dist)
    plot_estimate_distribution(out_dist)

def compute_estimate(T,in_session_dist,out_session_dist):
    out_session_dist['estimate_count'] = np.dot(T,in_session_dist['cell_specimen_id'])
    out_session_dist['estimate_normalized'] = out_session_dist['estimate_count']/sum(out_session_dist['estimate_count'])
    return out_session_dist
    


