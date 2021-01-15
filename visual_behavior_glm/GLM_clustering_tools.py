import os
import scipy
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans # Dev
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as sch
import visual_behavior_glm.GLM_visualization_tools as gvt

def notes():
    raise Exception("dont call this function, alex Dev")
    
    

def plot_top_level_dropouts_gmm(results_pivoted, filter_cre=False, cre='Slc17a7-IRES2-Cre',bins=150, cmax=10,n_clusters=10):
    '''
         IN DEVELOPMENT
    '''
    if filter_cre:
        rsp = results_pivoted.query('(variance_explained_full > 0.01) & (cre_line == @cre)').copy()
    else:
        rsp = results_pivoted.query('variance_explained_full > 0.01').copy()
        cre='All'
    rsp.fillna(value=0,inplace=True)

    pca = PCA()
    pca.fit(rsp[['visual','behavioral','cognitive']].values) 
    transformed = pca.transform(rsp[['visual','behavioral','cognitive']].values)
    rsp['pc1'] = transformed[:,0] 
    rsp['pc2'] = transformed[:,1] 

    # Determine best cluster size
    bic = []
    es = []
    for i in np.arange(1,15,1):
        e = fit_and_plot_gmm(transformed[:,0:2],i)
        bic.append(e.bic(transformed[:,0:2]))
        es.append(e)
        rsp['cluster_num_gmm_'+str(i)] = e.predict(transformed[:,0:2])
        plot_top_level_clustering(rsp, 'gmm',i)

    plt.figure()
    plt.plot(np.arange(1,15,1),bic,'ko-')
    plt.ylabel('BIC')
    plt.xlabel('# Clusters (k)')
    n_clusters = np.argmin(bic)
    plt.savefig('nested_gmm_elbow.png') 
    return rsp
    
def fit_and_plot_gmm(data,n,cmax=10, bins=150,offset=0,h=0.01,plot=True):
    estimator = GaussianMixture(n_components=n,covariance_type='full') 
    estimator.fit(data)

    plt.figure()
    plt.axis('equal')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.hist2d(data[:,0],data[:,1],bins=bins,density=True, cmap='inferno',cmax=cmax)  
    xmin,xmax = data[:,0].min()-offset,data[:,0].max()+offset
    ymin,ymax = data[:,1].min()-offset,data[:,1].max()+offset
    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
    Z = estimator.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(Z,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.tab10,origin='lower',alpha=0.35,zorder=9)
    centroids = estimator.means_
    cmap= plt.get_cmap("tab10")
    outer_colors = cmap(np.arange(0,10))
    for i in range(0,len(centroids)):
        plt.scatter(centroids[i,0],centroids[i,1],marker='o',s=35,color=outer_colors[np.mod(i,10)],zorder=10)
    plt.title(n)
    if plot:
        plt.savefig('nested_gmm_'+str(n)+'.png')
    return estimator


def fit_and_plot_kmeans(data,n,cmax=10, bins=150,offset=0,h=0.01):
    estimator = KMeans(init='k-means++',n_clusters=n,n_init=10)
    estimator.fit(data)
    plt.figure()
    plt.axis('equal')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.hist2d(data[:,0],data[:,1],bins=bins,density=True, cmap='inferno',cmax=cmax)  
    xmin,xmax = data[:,0].min()-offset,data[:,0].max()+offset
    ymin,ymax = data[:,1].min()-offset,data[:,1].max()+offset
    xx,yy = np.meshgrid(np.arange(xmin,xmax,h),np.arange(ymin,ymax,h))
    Z = estimator.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(Z,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.tab10,origin='lower',alpha=0.35,zorder=9)
    centroids = estimator.cluster_centers_
    cmap= plt.get_cmap("tab10")
    outer_colors = cmap(np.arange(0,10))
    for i in range(0,len(centroids)):
        plt.scatter(centroids[i,0],centroids[i,1],marker='o',s=35,color=outer_colors[np.mod(i,10)],zorder=10)
    plt.title(n)
    plt.savefig('nested_kmeans_'+str(n)+'.png')
    return estimator

def plot_top_level_dropouts_kmeans(results_pivoted, filter_cre=False, cre='Slc17a7-IRES2-Cre',bins=150, cmax=10,max_clusters=15):
    '''
         IN DEVELOPMENT
    '''
    if filter_cre:
        rsp = results_pivoted.query('(variance_explained_full > 0.01) & (cre_line == @cre)').copy()
    else:
        rsp = results_pivoted.query('variance_explained_full > 0.01').copy()
        cre='All'
    rsp.fillna(value=0,inplace=True)

    pca = PCA()
    pca.fit(rsp[['visual','behavioral','cognitive']].values) 
    transformed = pca.transform(rsp[['visual','behavioral','cognitive']].values)
   
    # PCA Results
    plt.figure()
    plt.plot(pca.explained_variance_ratio_,'ko-')
    plt.plot(np.cumsum(pca.explained_variance_ratio_),'ro-')
    plt.xticks([0,1,2],labels=['PC1','PC2','PC3'])
    plt.ylabel('Variance Explained')
    plt.ylim(0,1)
    plt.savefig('nested_pca_var_expl.png')   
 
    # Determine best kmeans cluster size
    sse = []
    for i in np.arange(1,max_clusters,1):
        kmeans = fit_and_plot_kmeans(transformed[:,0:2],i) 
        sse.append(kmeans.inertia_)
        rsp['cluster_num_kmeans_'+str(i)] = kmeans.predict(transformed[:,0:2])
        plot_top_level_clustering(rsp, 'kmeans',i) 

    plt.figure()
    plt.plot(np.arange(1,max_clusters,1),sse,'ko-')
    plt.ylabel('SSE')
    plt.xlabel('# Clusters (k)')
    n_clusters = np.argmin(sse)
    plt.savefig('nested_kmeans_elbow.png')   
 
    # 2d Histograms 
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    ax[0].hist2d(rsp['visual'],rsp['behavioral'],bins=bins,density=True, cmax=cmax,cmap='inferno')
    ax[1].hist2d(rsp['visual'],rsp['cognitive'],bins=bins,density=True, cmax=cmax,cmap='inferno')
    ax[2].hist2d(rsp['cognitive'],rsp['behavioral'],bins=bins,density=True, cmax=cmax,cmap='inferno')
    ax[0].set_ylabel('behavioral')
    ax[0].set_xlabel('visual')
    ax[1].set_ylabel('cognitive')
    ax[1].set_xlabel('visual')
    ax[2].set_ylabel('behavioral')
    ax[2].set_xlabel('cognitive')
    ax[2].set_title(cre)
    plt.tight_layout()
    plt.savefig('nested_2dhist.png')
    return rsp

def plot_top_level_clustering(rsp,method,n):
    fig, ax = plt.subplots(1,3, figsize=(12,4)) 
    size = .6
    radius = 1
    cmap= plt.get_cmap("tab10")
    outer_colors = cmap(np.arange(0,10))
    cres = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    g = rsp.groupby(['cluster_num_'+method+'_'+str(n),'cre_line']).size().unstack()
    for i, cre in enumerate(cres): 
        props = g[cre]
        props = props/np.sum(props) 
        wedges, texts= ax[i].pie(props,radius=radius,colors=outer_colors,wedgeprops=dict(width=size,edgecolor='w'))
        ax[i].legend(wedges,np.arange(0,n))
        ax[i].set_title(cre)
    plt.savefig('nested_clustering_'+method+'_'+str(n)+'.png')


def plot_second_level(results_pivoted):
    rsp = results_pivoted.query('variance_explained_full > 0.01').copy()
    rsp.fillna(value=0,inplace=True)
    pca = PCA()
    data = rsp[['all-images','expectation','omissions','face_motion_energy','licking','pupil_and_running','beh_model','task']].values
    pca.fit(data)
    transformed = pca.transform(data)
   
    # PCA Results
    plt.figure()
    plt.plot(pca.explained_variance_ratio_,'ko-')
    plt.plot(np.cumsum(pca.explained_variance_ratio_),'ro-')
    plt.xticks(np.arange(0,len(pca.explained_variance_ratio_))+1)
    plt.ylabel('Variance Explained')
    plt.ylim(0,1)

    # Determine best cluster size
    bic = []
    es = []
    for i in np.arange(1,15,1):
        #e = fit_and_plot_gmm(transformed[:,0:2],i,plot=False) # make this more general to pass in number of dimensions
        e = GaussianMixture(n_components=i,covariance_type='full') 
        e.fit(transformed[:,0:2])
        bic.append(e.bic(transformed[:,0:2]))
        es.append(e)
        #rsp['cluster_num_gmm_'+str(i)] = e.predict(transformed[:,0:2])
        #plot_top_level_clustering(rsp, 'gmm',i)

    plt.figure()
    plt.plot(np.arange(1,15,1),bic,'ko-')
    plt.ylabel('BIC')
    plt.xlabel('# Clusters (k)')
    n_clusters = np.argmin(bic)
    #plt.savefig('nested_gmm_elbow.png') 
    #return rsp
    return transformed 
  
def plot_clustering_by_session(rsp,n=3,method='gmm'):
    fig, ax = plt.subplots(1,3,figsize=(10,4))
    group = rsp.groupby(['cre_line','session_number','cluster_num_'+method+'_'+str(n)]).size().unstack() 
    cmap= plt.get_cmap("tab10")
    colors = cmap(np.arange(0,10))
    cre_lines = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']

    for cre_dex, cre in enumerate(cre_lines):
        for session in range(1,7):
            props = group.loc[cre,session].values
            props = props/np.sum(props)
            for cluster in range(0,n):
                ax[cre_dex].bar(session,[props[cluster]],width=.9,color=colors[cluster],bottom = np.sum(props[0:cluster]))
        ax[cre_dex].set_title(cre)
        ax[cre_dex].set_ylabel('Fraction cells in cluster')
        ax[cre_dex].set_xlabel('Session #')
        ax[cre_dex].set_xticks(np.arange(1,7))
    plt.tight_layout()


def plot_nested_dropouts(results_pivoted,run_params, num_levels=2,size=0.3,force_nesting=True,filter_cre=False, cre='Slc17a7-IRES2-Cre',invert=False,mixing=True,thresh=-.2,savefig=True,force_subsets=True):

    if filter_cre:
        rsp = results_pivoted.query('(variance_explained_full > 0.01) & (cre_line == @cre)').copy()
    else:
        rsp = results_pivoted.query('variance_explained_full > 0.01').copy()

    fig, ax = plt.subplots(1,num_levels+1,figsize=((num_levels+1)*3+1,4))
    cmap= plt.get_cmap("tab20c")
    outer_colors = cmap(np.array([0,4,8,12]))
    inner_colors = cmap(np.array([1,2,3,5,6,7,9,10,11]))
    
    if num_levels==1:
        size=size*2

    if invert:
        r = [1-size,1]
    else:
        r = [1,1-size]   
 
    # Compute Level 1 clusters
    if mixing:
        rsp['level1'] = [np.argmin(x) for x in zip(rsp['visual'],rsp['behavioral'],rsp['cognitive'])]
        rsp['level1'] = [3 if (x[0]<thresh)&(x[1]<thresh) else x[2] for x in zip(rsp['visual'],rsp['behavioral'],rsp['level1'])]
    else:
        rsp['level1'] = [np.argmin(x) for x in zip(rsp['visual'],rsp['behavioral'],rsp['cognitive'])]
    level_1_props = rsp.groupby('level1')['level1'].count()
    if 0 not in level_1_props.index:
        level_1_props.loc[0] = 0
    if 1 not in level_1_props.index:
        level_1_props.loc[1] = 0
    if 2 not in level_1_props.index:
        level_1_props.loc[2] = 0
    level_1_props = level_1_props.sort_index(inplace=False)
    level_1_props = level_1_props/np.sum(level_1_props)

    # Compute Level 2 clusters
    if force_nesting:
        rsp['level2_0'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'])]
        rsp['level2_1'] = [np.argmin(x) for x in zip(rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'])]
        rsp['level2_2'] = [np.argmin(x) for x in zip(rsp['beh_model'],rsp['task'])]
        if mixing:
            rsp['level2_3'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'])]
            rsp['level2'] = [x[x[0]+1]+3*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'],rsp['level2_3'])]
        else:
            rsp['level2'] = [x[x[0]+1]+3*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'])]
        level_2_props = rsp.groupby('level2')['level2'].count()
        for i in range(0,9):    
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        if mixing:
            for i in range(9,15):    
                if i not in level_2_props.index:
                    level_2_props.loc[i] = 0
        level_2_props = level_2_props.sort_index(inplace=False)       
        level_2_props = level_2_props/np.sum(level_2_props)

    elif force_subsets:
        rsp['level2_0'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        rsp['level2_1'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        rsp['level2_2'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        if mixing:
            rsp['level2_3'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
            rsp['level2'] = [x[x[0]+1]+100*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'],rsp['level2_3'])] 
        else:
            rsp['level2'] = [x[x[0]+1]+100*x[0] for x in zip(rsp['level1'], rsp['level2_0'],rsp['level2_1'],rsp['level2_2'])] 
        level_2_props = rsp.groupby('level2')['level2'].count()
        for i in range(0,9):
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        for i in range(100,109):
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        for i in range(200,209):
            if i not in level_2_props.index:
                level_2_props.loc[i] = 0
        if mixing:
            for i in range(300,309):
                if i not in level_2_props.index:
                    level_2_props.loc[i] = 0       
        level_2_props = level_2_props.sort_index(inplace=False)       
        level_2_props = level_2_props/np.sum(level_2_props)

    else:
        rsp['level2'] = [np.argmin(x) for x in zip(rsp['all-images'],rsp['expectation'],rsp['omissions'],rsp['face_motion_energy'],rsp['licking'],rsp['pupil_and_running'],rsp['beh_model'],rsp['task'])]
        level_2_props = rsp.groupby('level2')['level2'].count()
        level_2_props.loc[8] = 0 # Add third category for cognitive
        level_2_props = level_2_props.sort_index(inplace=False)       
        level_2_props = level_2_props/np.sum(level_2_props)

    # Plot Layer 1 for legend
    wedges, texts= ax[0].pie(level_1_props,radius=0,colors=outer_colors,wedgeprops=dict(width=size,edgecolor='w'))
    ax[0].legend(wedges, ['Visual','Behavioral','Cognitive','Mixed'],loc='center')#,bbox_to_anchor=(0,-.25,1,2))
    ax[0].set_title('Level 1')

    # Plot Layer 2 for legend
    if num_levels ==2:
        wedges, texts = ax[1].pie(level_2_props,radius=0,colors=inner_colors,wedgeprops=dict(width=size,edgecolor='w'))
        ax[1].legend(wedges,['all-images','expectation','omissions','face_motion_energy','licking','pupil_and_running','beh_model','task'],loc='center')#,bbox_to_anchor=(0,-.4,1,2))
        if force_nesting:
            ax[1].set_title('Level 2\nForced Hierarchy')   
        else:
            ax[1].set_title('Level 2')
        final_ax = 2
    else:
        final_ax = 1

    # Plot Full chart
    wedges, texts = ax[final_ax].pie(level_1_props,radius=r[0],colors=outer_colors,wedgeprops=dict(width=size,edgecolor='w'))
    if num_levels ==2:
        wedges, texts = ax[final_ax].pie(level_2_props,radius=r[1],colors=inner_colors,wedgeprops=dict(width=size,edgecolor='w'))
    if filter_cre:
        ax[final_ax].set_title(cre)
    else:
        ax[final_ax].set_title('All cells')

    plt.tight_layout()
    if savefig:
        filename = os.path.join(run_params['fig_clustering_dir'], 'pie_'+str(num_levels))
        if filter_cre:
            filename+='_'+cre[0:3]
        if num_levels ==2:
            if force_nesting:
                filename+='_forced'
        if mixing:
            filename+='_mixing'
        plt.savefig(filename+'.png')
    return level_1_props, level_2_props, rsp

def plot_all_nested_dropouts(results_pivoted, run_params):
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=False, force_nesting=False, num_levels=1)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=1,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=1,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=1,cre='Sst-IRES-Cre')

    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=True, force_nesting=False, num_levels=1)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=1,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=1,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=1,cre='Sst-IRES-Cre')

    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=False, force_nesting=False, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=False, num_levels=2,cre='Sst-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=True, force_nesting=False, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=False, num_levels=2,cre='Sst-IRES-Cre')

    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=False, force_nesting=True, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=True, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=True, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=False, force_nesting=True, num_levels=2,cre='Sst-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=False, mixing=True, force_nesting=True, num_levels=2)
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=True, num_levels=2,cre='Slc17a7-IRES2-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=True, num_levels=2,cre='Vip-IRES-Cre')
    plot_nested_dropouts(results_pivoted, run_params,filter_cre=True,  mixing=True, force_nesting=True, num_levels=2,cre='Sst-IRES-Cre')


def plot_dendrogram(results_pivoted,regressors='all', method = 'ward', metric = 'euclidean', ax = 'none'):
    '''
    Clusters and plots dendrogram of glm regressors using the dropout scores from glm output. 
    More info: https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
    Note: filling NaNs with 0 might affect the result
    
    INPUTS:
    results_pivoted - pandas dataframe of dropout scores from GLM_analysis_tools.build_pivoted_results_summary
    regressors - list of regressors to cluster; default is all
    method - string, linckage method ('centroid', 'single', etc); default = 'ward', which minimizes within cluster variance
    metric - string, metric of space in which the data is clustered; default = 'euclidean'
    ax - where to plot
    
    '''
    if regressors == 'all':
         regressors = results_pivoted.columns.to_numpy()

    if ax =='none':
        fig, ax = plt.subplots(1,1,figsize=(10,10))
            
    X = results_pivoted[regressors].fillna(0).to_numpy()
    Z = sch.linkage(X.T, method = method, metric = metric)
    dend = sch.dendrogram(Z, orientation = 'right',labels = regressors,\
                          color_threshold=None, leaf_font_size = 15, leaf_rotation=0, ax = ax)
    plt.tight_layout()
    
    return

def cluster_weights_df(weights_df, run_params, kernel, threshold=0.01,drop_threshold=-0.10,session_filter=[1,2,3,4,5,6], equipment_filter="all", depth_filter=[0,1000], cell_filter="all", area_filter=["VISp","VISl"]):
    '''
    dev notes 
    '''
    version = run_params['version']
    filter_string = ''
    problem_sessions = [962045676, 1048363441,1050231786,1051107431,1051319542,1052096166,1052512524,1052752249,1049240847,1050929040,1052330675]

    equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5","MESO.1"]
    if equipment_filter == "scientifica": 
        equipment_list = ["CAM2P.3","CAM2P.4","CAM2P.5"]
        filter_string += '_scientifica'
    elif equipment_filter == "mesoscope":
        equipment_list = ["MESO.1"]
        filter_string += '_mesoscope'
 
    # Filter by Cell Type    
    cell_list = ['Sst-IRES-Cre','Slc17a7-IRES2-Cre','Vip-IRES-Cre']     
    if cell_filter == "sst":
        cell_list = ['Sst-IRES-Cre']
        filter_string += '_sst'
    elif cell_filter == "vip":
        cell_list = ['Vip-IRES-Cre']
        filter_string += '_vip'
    elif cell_filter == "slc":
        cell_list = ['Slc17a7-IRES2-Cre']
        filter_string += '_slc'

    # Determine filename
    if session_filter != [1,2,3,4,5,6]:
        filter_string+= '_sessions_'+'_'.join([str(x) for x in session_filter])   
    if depth_filter !=[0,1000]:
        filter_string+='_depth_'+str(depth_filter[0])+'_'+str(depth_filter[1])
    if area_filter != ['VISp','VISl']:
        filter_string+='_area_'+'_'.join(area_filter)

    # Apply filters
    weights = weights_df.query('(targeted_structure in @area_filter)& (cre_line in @cell_list)&(equipment_name in @equipment_list)&(session_number in @session_filter) & (ophys_session_id not in @problem_sessions) & (imaging_depth < @depth_filter[1]) & (imaging_depth > @depth_filter[0])& (variance_explained_full > @threshold) & ({0} < @drop_threshold)'.format(kernel))

    # Set up matrix of data
    df = weights[kernel+"_weights"]

    # Set up time vectors.
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)
    meso_time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/10.725)

    # Normalize kernels, and interpolate to time_vec
    data = [x/np.max(np.abs(x)) for x in df[~df.isnull()].values]
    data = [x if len(x) == len(time_vec) else scipy.interpolate.interp1d(meso_time_vec, x, fill_value="extrapolate", bounds_error=False)(time_vec) for x in data]
    
    # Needed for stability
    if len(data)>0:
        data = np.vstack(data)
    else:
        data = np.empty((2,len(time_vec)))
        data[:] = np.nan
   
    # do some clustering
    #labels, cluster = do_clustering(data)   
 
    # apply cluster labels onto dataframe
    #weights['clusters'] = labels   

    # return
    return weights,data

def cluster_weights_dendrogram(data):
    method = 'ward' 
    metric = 'euclidean'
    plt.figure()
    ax = plt.gca()
    Z = sch.linkage(data, method = method, metric = metric)
    dend = sch.dendrogram(Z, orientation = 'right',color_threshold=None, leaf_font_size = 15, leaf_rotation=0, ax = ax)
    plt.tight_layout()
    return Z, dend

def cluster_weights_GMM(data,n=4):
    estimator = GaussianMixture(n_components=n,covariance_type='full') 
    estimator.fit(data)
    return estimator

def cluster_weights_UMAP(weights, data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data) 
    return embedding

def plot_weights_UMAP(weights,run_params, kernel,embedding,s=1):
    project_colors = gvt.project_colors()
    filename1 = os.path.join(run_params['fig_clustering_dir'],kernel+'_weights_UMAP_cre.png') 
    filename2 = os.path.join(run_params['fig_clustering_dir'],kernel+'_weights_UMAP_cre_and_session.png') 
    
    fig, axes = plt.subplots(2,2)
    colors = [project_colors[x] for x in weights['cre_line']]
    axes[0,0].scatter(embedding[:,0], embedding[:,1], c=colors,s=s)
    axes[0,0].set_title(kernel)
    vip = (weights['cre_line'] == 'Vip-IRES-Cre').values
    sst = (weights['cre_line'] == 'Sst-IRES-Cre').values
    slc = (weights['cre_line'] == 'Slc17a7-IRES2-Cre').values
    axes[0,1].scatter(embedding[vip,0], embedding[vip,1], c=np.array(colors)[vip],s=s)
    axes[1,0].scatter(embedding[slc,0], embedding[slc,1], c=np.array(colors)[slc],s=s)
    axes[1,1].scatter(embedding[sst,0], embedding[sst,1], c=np.array(colors)[sst],s=s)
    axes[0,1].set_title('VIP')
    axes[1,0].set_title('SLC')
    axes[1,1].set_title('SST')
    for i in [0,1]:
        for j in [0,1]:
            axes[i,j].set_aspect('equal','datalim')
            axes[i,j].set_xlabel('UMAP 1')
            axes[i,j].set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(filename1)


    fig, axes = plt.subplots(3,6,figsize=(14,6))
    cres = ['Vip-IRES-Cre','Sst-IRES-Cre','Slc17a7-IRES2-Cre']
    for i,cre in enumerate(cres):
        for j,session in enumerate([1,2,3,4,5,6]):
            dex = ((weights['cre_line'] == cre)&(weights['session_number'] == session)).values    
            axes[i,j].scatter(embedding[dex,0],embedding[dex,1],s=s,color=project_colors[str(session)])
            axes[i,j].set_aspect('equal','datalim')
            axes[i,j].set_xlabel('UMAP 1')
            axes[i,j].set_ylabel('UMAP 2')
            axes[i,j].set_title(cre[0:3]+'-'+str(session))
    plt.tight_layout()
    plt.savefig(filename2)

def plot_all_UMAP_clustering(run_params, weights_df):
    for kernel in run_params['kernels']:
        try:
            weights, data = cluster_weights_df(weights_df, run_params, kernel)
            embedding = cluster_weights_UMAP(weights, data)
            plot_weights_UMAP(weights, run_params, kernel, embedding)
        except:
            print('error-'+kernel)





 
