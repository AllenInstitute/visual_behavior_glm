import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

K = 5

'''
- if more than one session type, just taking the first
- how to deal with NaNs, there could always be more cells "lurking"
    - Can I just assume that all cells show up at least once?
    - repeat with and without nan cells?
- am I computing the estimate correctly? Right math, right code?
    should use np.dot(), not *
    order of operations

'''

def load_table():
    filepath = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/glm/kmeans_glm_novel.hdf'
    df = pd.read_hdf(filepath)
    return df

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
    plot_matrix(T,[1,2], k=2)
    
def plot_matrix(T,sessions,k=None):
    if k is None:
        k=K
    plt.figure()
    im = plt.imshow(T,cmap='Blues')
    # triple check axis order, and value
    plt.ylabel('Session '+str(sessions[1]),fontsize=16)
    plt.xlabel('Session '+str(sessions[0]),fontsize=16)
    labels = [str(x) for x in list(range(0,k))]+['NaN']
    plt.xticks(range(0,k+1),labels, fontsize=14)
    plt.yticks(range(0,k+1),labels, fontsize=14)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Prob(Session '+str(sessions[1])+'| Session '+str(sessions[0])+')',fontsize=14)
    plt.tight_layout()
    


