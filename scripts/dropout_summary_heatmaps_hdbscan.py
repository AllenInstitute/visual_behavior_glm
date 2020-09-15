#!/usr/bin/env python
# coding: utf-8

what_cre_all = 'Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre', 'allCre' # # what cre line to analyze (umap/clustering)
dosavefig = 1 #0 #1
do_single = 1 # if 1, use single models in the clustering analysis: one regressor is remained in the model, how much is the explained variance compared to the full model? (-V_i) / V_F
# if 0, use dropouts in the clustering analysis: one regressor is dropped from the full model, how much does it change the explained variance (compared to the full model)? (V_i - V_F) / V_F


#%% modify this for your own directory
# directory to save figures: os.path.join(dir0, dir_now)
dir0 = '/home/farzaneh/OneDrive/Analysis' # from def_paths import * 
dir_now = 'umap_cluster'

fmt = '.pdf'

import datetime
now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    
    

#%% 
import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_visualization_tools as gvt
from visual_behavior_glm.glm import GLM
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
import visual_behavior.database as db
import plotly.express as px

import pandas as pd
import numpy as np
import os
import copy 
import seaborn as sns

import visual_behavior.plotting as vbp

import plotly.express as px
from sklearn.cluster import KMeans
import umap
import hdbscan # First install the package on the terminal: pip install hdbscan
from importlib import reload
import plot_hist_parameter_bins
# reload(plot_hist_parameter_bins)
import random



#%%
get_ipython().magic(u'matplotlib inline')   
# get_ipython().run_line_magic('matplotlib', 'notebook')
# get_ipython().run_line_magic('widescreen', '')


### Gather/organize data

#%% load the results summary for a single GLM version from mongo

glm_model ='6_L2_optimize_by_cell' # '4_L2_optimize_by_cell'
rs = gat.retrieve_results(search_dict = {'glm_version': glm_model}, results_type='summary')
rs #.sample(10)

rs.shape, rs.keys(), list(rs['dropout'].unique())
# rs[['dropout', 'adj_fraction_change_from_full']]

# used for figure names
if do_single==1: # use single models in the clustering analysis: one regressor is remained in the model, how much is the explained variance compared to the full model? (-V_i) / V_F
    gmn = f'single_glm{glm_model}'
else:
    gmn = f'glm{glm_model}'
    

'''
# rs_now: rs for a single experiment, a single cell
rs_now = rs[np.logical_and(rs['ophys_experiment_id'].values==991852008, rs['cell_specimen_id'].values==1017215561)] #.shape (128, 37) # unique dropout: 32

print(np.shape(rs_now['dropout'].values))
print(list(rs_now['dropout'].values))
'''


#%% get a list of columns to use for clustering. Defining this list now makes it easier to avoid accidentally clustering on some identifying feature (e.g., cre_line, session_type, etc.)

cols_for_clustering = list(rs['dropout'].unique())
# cols_for_clustering = [col for col in rspm.columns if col not in ['identifier','cre_line','session_type','equipment_name', 
#                                                                   'session_id', 'imaging_depth','project_code','session_number','exposure_number']]
cols_for_clustering = [col for col in cols_for_clustering if col not in ['image0','image1','image2','image3',
                                                                         'image4','image5','image6','image7',
                                                                         'visual', 'Full']]

cols_for_clustering = [col for col in cols_for_clustering if col not in ['beh_model', 'intercept']]

# check this: https://github.com/AllenInstitute/visual_behavior_glm/blob/b15f5f2bbfacca1d2e85ec99f4ff2e8afc02afbd/notebooks/Visual_Behavior___Neural_GLM_Dropout_Adjustment.pdf    
if do_single==0: # use dropouts in the clustering analysis: one regressor is dropped from the full model, how much does it change the explained variance (compared to the full model)? (V_i - V_F) / V_F
    cols_for_clustering = [col for col in cols_for_clustering if 'single' not in col]

elif do_single==1: # use single models in the clustering analysis: one regressor is remained in the model, how much is the explained variance compared to the full model? (-V_i) / V_F
    cols_for_clustering = [col for col in cols_for_clustering if 'single' in col]
    cols_for_clustering = [col for col in cols_for_clustering if col not in ['single-beh_model', 'single-visual']]
    cols_for_clustering = [col for col in cols_for_clustering if col not in ['single-image0', 'single-image1', 'single-image2', 'single-image3', 'single-image4', 'single-image5', 'single-image6', 'single-image7']]

'''
# I'm removing the 'visual' dropout, since it's actually a combination of the omission and all-images dropouts.

cols_for_clustering.remove('visual')

# cols_for_clustering.remove('model_bias')
# cols_for_clustering.remove('model_task0')
# cols_for_clustering.remove('model_omissions1')be
# cols_for_clustering.remove('model_timing1D')
# cols_for_clustering.remove('beh_model')

# cols_for_clustering.remove('pre_lick_bouts')
# cols_for_clustering.remove('post_lick_bouts')
'''
print(np.shape(cols_for_clustering)), cols_for_clustering



#%% build a pivoted version of the results summary, using the `fraction_change_from_full` column as the values.

rsp = gat.build_pivoted_results_summary(results_summary=rs, cutoff=0.01, value_to_use='fraction_change_from_full')

'''
## codes below copied from notebook: 200908_characterize_glm_results.ipynb

# make a new column called 'identifier' that is {OPHYS_EXPERIMENT_ID}_{CELL_SPECIMEN_ID}. This will give each a unique ID
rs['identifier'] = rs['ophys_experiment_id'].astype(str) + '_' +  rs['cell_specimen_id'].astype(str)

# pivot the table to get it into a form for clustering
# first get variance explained for full model to filter out poorly fit cells
model_output_type = 'variance_explained'
ve = rs.pivot(index='identifier',columns='dropout',values=model_output_type).reset_index()
ve

# plot histogram of variance explained for the full model; mark cutoff .01
plt.hist(ve['Full'].values, 100); plt.vlines(.01, 0, 2e4, color='r', linestyle='dashed'); 
# plt.xlim([-.01, .2])


# apply a cutoff of .01 to variance explained, in order to include a neuron into analysis
cells_to_include = ve[ve['Full']>0.01].identifier.values

order = np.argsort(ve[ve.identifier.isin(cells_to_include)==True]['Full'])
cell_order = cells_to_include[order]

len(cells_to_include)


# model_output_type = 'fraction_change_from_full'
model_output_type = 'adj_fraction_change_from_full'
rsp = rs.pivot(index='identifier',columns='dropout',values=model_output_type).reset_index()
rsp


tmp = ve.rename(columns={'Full':'varience_explained_full_model'})
rsp = rsp.merge(tmp[['identifier','varience_explained_full_model']], on=['identifier'])

rsp = rsp[rsp.identifier.isin(cells_to_include)==True]

rspm = rsp.merge(rs[['identifier','cell_specimen_id','ophys_experiment_id','cre_line','session_type','imaging_depth','equipment_name','project_code','session_number','exposure_number','container_id']].drop_duplicates(),left_on='identifier',right_on='identifier',how='inner')
rspm
'''



#%% check what fraction of cells are kept after applying the cutoff above.

np.unique(rs['cell_specimen_id']).shape, np.unique(rsp['cell_specimen_id']).shape, np.unique(rsp['cell_specimen_id']).shape[0]/np.unique(rs['cell_specimen_id']).shape[0]

# rs.columns, rsp.columns, rs.shape, rsp.shape


#%% add a `session_id` column with a numeric value for the session_type (to lump together sessions by order, regardless of image set)

def map_session_types(session_type):
    session_id = int(session_type[6:7])
    
    if session_id==4:
        session_id=1 # novel
    else:
        session_id=0 # not novel
    
    return session_id

rsp['session_id'] = rsp['session_type'].map(lambda st:map_session_types(st))

sum(rsp['session_id']==0), sum(rsp['session_id']==1)


#%% Turn some categorical columns into numerical columns

def make_categorical(df, column):
    df['{}_categorical'.format(column)] = pd.Categorical(df[column], ordered=True).codes

for column in ['cre_line','equipment_name','targeted_structure','session_id']:
    make_categorical(rsp, column)

# ['CAM2P.3', 'CAM2P.4', 'CAM2P.5', 'MESO.1']  --> [0, 1, 2, 3]


#%% keep a copy of rsp before merging it with "data" that includes the behavioral model parameters

rsp00 = copy.deepcopy(rsp)


#%% Merge the behavioral glm df with rsp

# load the behavioral model for each session
# for now we care about 'task_dropout_index'
# task_weight_index is the average value of the task weight for that session. 
# task_dropout_index is the value of the (task dropout - timing dropout). 
# task_only_dropout_index is just task dropout

data = loading.get_behavior_model_summary_table()
# data['task_dropout_index']
data.shape


#%% compare number of sessions for the behavioral and neural glm models

np.unique(data['ophys_session_id'].values).shape, np.unique(rsp00['ophys_session_id'].values).shape



#%% merge rsp and data on session_id

# rsp = rsp.merge(data, on=['ophys_session_id'])
rsp = rsp00.merge(data, on=['ophys_session_id', 'cre_line', 'equipment_name']) #, how='outer') # , 'equipment_name'

print(rsp00.shape, rsp.shape)
print(np.unique(rsp['ophys_session_id'].values).shape)


#%% take care of column renaming after the merge

list(rsp.columns) #, rsp.shape
rsp = rsp.rename(columns = {'imaging_depth_x': 'imaging_depth'}, inplace = False)

# rsp[rsp['ophys_experiment_id_x'].values==795948257][['cell_specimen_id', 'beh_model']]

# average number of cells per experiment:
# rsp.shape[0]/np.unique(rsp['ophys_experiment_id_x']).shape[0]



#%% Redefine cols_for_clustering
# add some features from behavioral glm to rsp 

# make sure to not add features that get repeated across cells (eg task_dropout_index will be repeated for all cells of the same session)
# cols_for_clustering.append('task_dropout_index')
# np.shape(cols_for_clustering), cols_for_clustering


#####################################
#%% Choose what cre line goes into umap/clustering

# cres = np.unique(rsp['cre_line'])
# print(cres) # ['Slc17a7-IRES2-Cre' 'Sst-IRES-Cre' 'Vip-IRES-Cre']
# icre = 2
# rsp = rsp[rsp['cre_line']==cres[icre]]

rsp_main = copy.deepcopy(rsp)
print(rsp_main.shape)

for what_cre in what_cre_all: # what_cre = what_cre_all[0]
    
    print('____________________________')
    print(f'{what_cre}')
    print('____________________________')
    
    if what_cre != 'allCre':

        rsp = rsp_main[rsp_main['cre_line']==what_cre]
        print(rsp.shape)

        what_cre_name = what_cre[:3]
        
    else:
        rsp = copy.deepcopy(rsp_main)
        what_cre_name = what_cre

        
    #####################################
    #####################################
    # Dimensionality reduction 
    #####################################
    #####################################

    # Remember umaps is performed on: 
    # rsp[cols_for_clustering]
        
        
    #%% UMAP, set parameters

    neigh_vals = np.concatenate(([5, 10, 15, 50], np.arange(200, int(rsp.shape[0]/10), 500)))
    print(neigh_vals)


    #%% umap relies on stochastic methods so we need to set the seed for it.

    rand_state = 42
    np.random.seed(rand_state)
    os.environ['PYTHONHASHSEED'] = str(rand_state)
    random.seed(rand_state)                                

    print(cols_for_clustering)
    print(rsp.shape, rsp[cols_for_clustering].shape)



    #%% Take care of NaNs before doing UMAP; change them to 0s.
    # Note: after pivoting some rows of rsp have nans for 'correct_rejects', this is because some cells dont have 'correct_rejects' as a dropout in the dataframe rs.

    rsp_nan = copy.deepcopy(rsp)

    rsp = rsp.fillna(0)
    print(sum(np.isnan(rsp['correct_rejects'].values)), sum(np.isnan(rsp_nan['correct_rejects'].values)))
    print(rsp.shape)


    
    #%% UMAP

    n_components = 3 # dimensions of original data: np.shape(cols_for_clustering)[0]

    mindist = 0.1 # default: .1 
    neigh = neigh_vals[0]

    reducer = umap.UMAP(n_components=n_components, min_dist = mindist, n_neighbors = neigh)
    embedding = reducer.fit_transform(rsp[cols_for_clustering])
    print(embedding.shape)

    # reducer_2d = umap.UMAP(n_components=2)
    # reducer_3d = umap.UMAP(n_components=3)

    # embedding_2d = reducer_2d.fit_transform(rsp[cols_for_clustering])
    # embedding_3d = reducer_3d.fit_transform(rsp[cols_for_clustering])

    # keep these dimensions for plotting
    rsp['umap_3d_embedding_0'] = embedding[:, 0]
    rsp['umap_3d_embedding_1'] = embedding[:, 1]
    rsp['umap_3d_embedding_2'] = embedding[:, 2]

    # rsp['umap_2d_embedding_0'] = embedding_2d[:, 0]
    # rsp['umap_2d_embedding_1'] = embedding_2d[:, 1]

    print(embedding.shape, neigh, mindist)



    #####################################
    #####################################
    # Clustering
    #####################################
    #####################################

    #%% k-means clustering on 3d umap
    '''
    kmeans = KMeans(n_clusters=20)
    umap_cols = ['umap_3d_embedding_0','umap_3d_embedding_1','umap_3d_embedding_2']
    rsp['clusterer_labels'] = kmeans.fit_predict(rsp[umap_cols])
    rsp['clusterer_labels'].value_counts()
    '''


    #%% Hdbscan clustrering

    # try a range of parameters
    '''
    min_cluster_size_all = [5, 20, 30, 50, 100, 200] #100 #50 # (default=5)
    min_samples_all = [10] #10 #min_cluster_size #18 # default: same as min_cluster_size
    cluster_selection_epsilon = .5 # set to nan, if you don't want to set it.

    clusterer_all = []
    clusterer_labels_all = []
    clusterer_params = []
    cluster_size_all = []
    cluster_ids_all = []

    for min_cluster_size in min_cluster_size_all:
        for min_samples in np.unique(np.concatenate((min_samples_all, [min_cluster_size]))):

            print('______________________')
            min_cluster_size = int(min_cluster_size)
            min_samples = int(min_samples)
            print([min_cluster_size, min_samples])


            if np.isnan(cluster_selection_epsilon):   ### don't set cluster_selection_epsilon
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

            else:    ### set cluster_selection_epsilon to .5
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon) #(min_cluster_size=9, gen_min_span_tree=True)


            clusterer.fit(embedding) # rsp[umap_cols]

            clusterer_labels_ = clusterer.labels_
            cluster_size = [np.sum(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))] # number of points per cluster
            cluster_ids = np.unique(clusterer.labels_)

            print(cluster_ids[[0,-1]])
            print(cluster_size)

            clusterer_labels_all.append(clusterer_labels_) # cluster_labels = clusterer.fit_predict(embedding)
            clusterer_all.append(clusterer)
            clusterer_params.append([min_cluster_size, min_samples])
            cluster_size_all.append(cluster_size)
            cluster_ids_all.append(cluster_ids)
    '''

    # rsp = rspall
    # rsp.shape
    # embedding.shape


    #%% Pick certain parameter values informed by the search above, and rerun the clustering

    min_cluster_size = 20 #30 #100 #50 # (default=5)
    min_samples = 20 #10 #10 #100 #10 #min_cluster_size #18 # default: same as min_cluster_size
    cluster_selection_epsilon = .5 #.3 #

    if np.isnan(cluster_selection_epsilon):   ### don't set cluster_selection_epsilon
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples) #(min_cluster_size=9, gen_min_span_tree=True)
    else: 
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)


    clusterer.fit(embedding) # rsp[umap_cols]
    # clusterer.labels_ # cluster_labels = clusterer.fit_predict(embedding)
    # clusterer.probabilities_
    rsp['clusterer_labels'] = clusterer.labels_
    # cluster_size = rsp['clusterer_labels'].value_counts().values # number of points per cluster # this is sorted by the number of cells per cluster
    # below is sorted by the cluster id (i prefer below) (ie number of cells per cluster, sorted by cluster id, ie from cluster -1 to end)
    cluster_size = [np.sum(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))] # number of points per cluster


    #%% Print some clustering results

    # fract_per_clust = [np.mean(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))]
    print('number of clusters: ', len(np.unique(clusterer.labels_)))

    print(rsp['clusterer_labels'].value_counts())
#     print(np.sort(cluster_size))

    # noise points
    print('noise points:', sum(clusterer.labels_ == -1), np.mean(clusterer.labels_ == -1))
    
    # # number of clusters with <th_neur neurons in them.
    # th_n = 20 # we dont want to have less than 20 neurons in a cluster, maybe.
    # print(sum(cluster_size < th_n), np.mean(cluster_size < th_n))



    #%% Plot clusterer probabilities for each cell, and number of cells in each cluster.

    plt.figure(figsize=(8,3))

    plt.subplot(121), plt.plot(np.sort(clusterer.probabilities_)), plt.ylabel('clusterer.probabilities_')
    plt.xlabel('cells')
    plt.subplot(122), plt.plot(np.unique(clusterer.labels_), cluster_size), plt.ylabel('cluster size')
    plt.xlabel('cluster id')
    plt.suptitle(f'{len(cluster_size)} clusters')
    plt.subplots_adjust(wspace=.4)
    
    if dosavefig:
        nam = f'{what_cre_name}_cluster_prob_size_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{cluster_selection_epsilon}_{gmn}_{now}'
        fign = os.path.join(dir0, dir_now, nam+fmt)     

        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



    #####################################
    #####################################
    # Visualize UMAP embedding and clustered cells
    #####################################
    #####################################


    #%% visualize in 2d using matplotlib

    color_palette = sns.color_palette('Paired', len(rsp['clusterer_labels'].unique()))
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.scatter(rsp['umap_3d_embedding_0'], rsp['umap_3d_embedding_1'], s=12, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.xlabel('umap_3d_embedding_0'), plt.ylabel('umap_3d_embedding_1')
    plt.title(f'{len(cluster_size)} clusters')

    plt.subplot(122)
    plt.scatter(rsp['umap_3d_embedding_0'], rsp['umap_3d_embedding_2'], s=12, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.xlabel('umap_3d_embedding_0'), plt.ylabel('umap_3d_embedding_2')

    if dosavefig:
        nam = f'{what_cre_name}_umapEmbed_clustColored_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{cluster_selection_epsilon}_{gmn}_{now}'
        fign = os.path.join(dir0, dir_now, nam+fmt)     

        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



    #%% visualize in 2d, Doug's version

    fig,ax=plt.subplots(1,2,figsize=(10,6),sharey=True,sharex=True)
    sns.scatterplot(
        x='umap_3d_embedding_1', 
        y='umap_3d_embedding_0',
        hue='clusterer_labels',
        palette=sns.color_palette("hsv", len(rsp['clusterer_labels'].unique())),
        data=rsp,
        legend='full',
        alpha=0.3,
        ax=ax[0],
    )
    sns.scatterplot(
        x='umap_3d_embedding_2', 
        y='umap_3d_embedding_0',
        hue='clusterer_labels',
        palette=sns.color_palette("hsv", len(rsp['clusterer_labels'].unique())),
        data=rsp,
        legend='full',
        alpha=0.3,
        ax=ax[1],
    )

    if dosavefig:
        nam = f'{what_cre_name}_umapEmbed_clustColored_Doug_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{cluster_selection_epsilon}_{gmn}_{now}'
        fign = os.path.join(dir0, dir_now, nam+fmt)     

        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



    ###############################
    #%% visualize in 3d using plotly
    ###############################
    '''
    fig = px.scatter_3d(
        rsp, 
        x='umap_3d_embedding_0', 
        y='umap_3d_embedding_1', 
        z='umap_3d_embedding_2',
        color='clusterer_labels',
        color_continuous_scale='hsv',
    )

    fig.update_traces(marker=dict(size=3))

    # optional view positioning parameters (I don't really understand these, just toying with them to get a view I like!)
    camera = dict(
        up=dict(x=3, y=1.75, z=0.25),
        eye=dict(x=1.5, y=1.5, z=-0.2)
    )

    fig.update_layout(
        scene_camera=camera,
        margin=dict(l=30, r=30, t=10, b=10),
        width=1200,
        height=1000,
    )
    # fig.write_html("/home/dougo/code/dougollerenshaw.github.io/figures_to_share/2020.08.21_k_means_on_umap.html")
    fig.show()
    '''


    #####################################
    #####################################
    # Make heatmaps of GLM dropout coeffs, and do some sorting
    #####################################
    #####################################

    #%% A little bit of prep for plotting
    # identify dominant dropouts for each cluster
    
    gat.identify_dominant_dropouts(rsp, cluster_column_name='clusterer_labels', cols_to_search=cols_for_clustering)
    print(np.unique(rsp['dominant_dropout']))


    #%%
    for column in ['dominant_dropout']:
        make_categorical(rsp, column)



    #%% only get those rsp rows that have non-noise cluster values

    # copy rsp before removing noise points
    rspall = copy.deepcopy(rsp)
    print(rspall.shape)

    # use all rsp rows in clustering
    '''
    rsp = rspall
    rsp.shape
    '''
    rsp['clusterer_labels'].value_counts(), sum(rsp['clusterer_labels'].value_counts())


    # only get those rsp rows that have non-noise cluster values
    rsp = rspall[rspall['clusterer_labels'] != -1]
    print(rsp.shape)
    
    
    ################### 
    # print number of cells with a given dominant dropout
#     print(np.unique(rsp['dominant_dropout']))
    print(rsp['dominant_dropout'].value_counts())

    a = np.unique(rsp['dominant_dropout'])
    ac = [sum(rsp['dominant_dropout']==a[i]) for i in range(len(a))]
    ddtext = [f'{ac[i]}, {a[i]}' for i in range(len(a))]

    
    
    #############################################
    # define heatmap parameters
    # lots here, but there are lots of options!

    cols_fract_change0 = [
        'all-images', 
        'omissions', 
        'pupil', 
        'running',
        'face_motion_energy',
        'single-all-images', 
        'single-omissions', 
        'single-pupil', 
        'single-running',
        'single-face_motion_energy',            
        ]
    
    
    # add to it the dominant dropouts, to make sure the heatmap includes them
    cols_fract_change = np.unique(np.concatenate((np.unique(rsp['dominant_dropout']),
        cols_fract_change0)))

    
    # finalize the list of glm features (fract change in exp var) that will go into heatmap
    cols_to_plot = np.concatenate((
        ['clusterer_labels'],
        cols_fract_change,
        ['cre_line_categorical', 
        'dominant_dropout_categorical',
        'session_id_categorical', 
        'targeted_structure_categorical',
        'imaging_depth',
        'task_dropout_index',
        'equipment_name_categorical',
        ]  
    ))

    
    
    
    ########## make sure there's a heatmap definition for every column, otherwise the column will plot without a heatmap!

    mnf = np.min(np.min(rsp[cols_to_plot[1:len(cols_fract_change)+1]]))
    mxf = np.max(np.max(rsp[cols_to_plot[1:len(cols_fract_change)+1]]))

    heatmap_defs = [
        {
            'columns':cols_to_plot[1:len(cols_fract_change)+1],
            'cbar_label':'adj fraction change\nin var explained',
            'cbar_ticks':[mnf, mxf],
            'vmin':mnf,
            'vmax':mxf,
            'cmap':'viridis', #'bwr',
        },
        {
            'columns':['cre_line_categorical'],
            'cbar_label':'cre_line',
            'cbar_ticks':np.arange(len(np.unique(rsp['cre_line']))),
            'cbar_ticklabels':np.sort(np.unique(rsp['cre_line'])),
            'vmin':-0.5,
            'vmax':len(np.unique(rsp['cre_line']))+0.5,
            'cmap':sns.color_palette("hls", len(np.unique(rsp['cre_line']))),            
        },    
        {
            'columns':['dominant_dropout_categorical'],
            'cbar_label':'dominant_dropout_categorical',
            'cbar_ticks':np.arange(len(rsp['dominant_dropout_categorical'].unique())),
            'cbar_ticklabels':ddtext, #np.sort(np.unique(rsp['dominant_dropout'])),
            'vmin':-0.5,
            'vmax':len(rsp['dominant_dropout_categorical'].unique())-0.5,
            'cmap':sns.color_palette("hls", len(rsp['dominant_dropout_categorical'].unique())),

        },
        {
            'columns':['session_id_categorical'],
            'cbar_label':'session ID',
            'cbar_ticks':[0,1], #[0,1,2,3],
            'cbar_ticklabels':['familiar', 'novel'], #[1,3,4,6],
            'vmin':-0.5,
            'vmax':1.5, #3.5,
            'cmap':sns.color_palette("Dark2", 2), # 4
        },
        {
            'columns':['clusterer_labels'],
            'cbar_label':'clusterer label',
            'cbar_ticks':np.arange(min(rsp['clusterer_labels'].unique()),len(rsp['clusterer_labels'].unique()),2),
            'vmin':min(rsp['clusterer_labels'].unique())-.5,
            'vmax':len(rsp['clusterer_labels'].unique())-.5,
            'cmap':sns.color_palette("hsv", len(rsp['clusterer_labels'].unique())),
        },
        {
            'columns':['equipment_name_categorical'],
            'cbar_label':'equipment name',
            'cbar_ticks':np.arange(len(rsp['equipment_name_categorical'].unique())),
            'cbar_ticklabels':np.sort(np.unique(rsp['equipment_name'])),
            'vmin':-0.5,
            'vmax':len(rsp['equipment_name_categorical'].unique())-0.5,
            'cmap':sns.color_palette("hls", len(rsp['equipment_name_categorical'].unique())),
        },
        {
            'columns':['targeted_structure_categorical'],
            'cbar_label':'targeted structure',
            'cbar_ticks':np.unique(rsp['targeted_structure_categorical']), #np.arange(len(rsp['targeted_structure_categorical'].unique())),
            'cbar_ticklabels':np.unique(rsp['targeted_structure']), #np.sort(np.unique(rsp['targeted_structure'])),
            'vmin':min(rsp['targeted_structure_categorical'].unique())-0.5,
            'vmax':max(rsp['targeted_structure_categorical'].unique())+0.5,
            'cmap':sns.color_palette("hls", len(rsp['targeted_structure'].unique())),
        },
        {
            'columns':['imaging_depth'],
            'cbar_label':'imaging_depth',
            'cbar_ticks':[0,100,200,300,400],
            'vmin':0,
            'vmax':400,
            'cmap':'magma',
        },
        {
            'columns':['task_dropout_index'],
            'cbar_label':'task_dropout_index',
            'cmap':'inferno',
        },    
    ]



    #############################################
    #############################################
    #%% Make heatmaps, sort by whatever we want
    #############################################
    #############################################
    
    # Note that it only makes sense to nest sorting values for categorical data. Any sorting value that follows a continuous variable will not have any effect.

    ### First sort by dominant dropout, the median value of the dominant dropout (for cases with multiple clusters sharing the same dominant dropout), then cre_line, session_id, imaging_depth
   
    sort_order = ['clusterer_labels', 'dominant_dropout','dominant_dropout_median','cre_line_categorical', 'session_id_categorical', 'targeted_structure_categorical', 'imaging_depth'] #'single-all-images', 'single-omissions']
    # sort_order = ['task_dropout_index', 'dominant_dropout','dominant_dropout_median','cre_line_categorical']
    # sort_order = ['dominant_dropout','dominant_dropout_median','cre_line_categorical','session_id_categorical','imaging_depth',]
    # sort_order = ['clusterer_labels', 'dominant_dropout','dominant_dropout_median','cre_line_categorical']
    # sort_order = ['clusterer_labels', 'task_dropout_index']
    # sort_order = ['task_dropout_index']

    sorted_data = gat.sort_data(rsp, sort_order, cluster_column_name='clusterer_labels')

    fig, axes = vbp.make_multi_cmap_heatmap(
        sorted_data[cols_to_plot], 
        heatmap_defs, 
        figsize=(10,12), 
        top_buffer=0, 
        bottom_buffer=0.1, 
        n_cbar_rows=4, 
        heatmap_div=0.7, 
    )

    
    # mark cluster boundaries only if there are <=20 clusters
    if len(rsp['clusterer_labels'].value_counts()) < 21:
        for idx,row in sorted_data.query('cluster_transition').iterrows():
            axes['heatmap'].axhline(idx,color='black')

            
    if dosavefig:
        nam = f'{what_cre_name}_glmHeatmap_clusters_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{cluster_selection_epsilon}_{gmn}_{now}'
        fign = os.path.join(dir0, dir_now, nam+fmt)     

        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


        


    #############################################
    #############################################
    #%% bin a feature (lets call it the primary feature: parameters_primary), and then for each bin look at the distribution of a bunch of other features (parameters_cont and parameters_categ)
    # note: the figure below is not related to umap/clustering (except if we add cluserer_labels as a parameters_categ).
    # it is a general plot that demonstrates how different glm coefficients covary.
    #############################################
    #############################################
    
    norm_by_param_primary = 0 # np.nan # 
    # if nan, don't normalize the counts; 
    # if 1, normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells in that bin of param_primary (eg normalize V1 neurons in bin0 of task_dropout_index by the sum of V1 and LM neurons in bin0 of task_dropout_index ).  
    # If 0, normalize each value of parameters_categ (that is in a given bin of param_primary) by the total number of cells that have that given value of param_categ (acorss all bins of param_primary); (eg normalize V1 neurons in bin0 of task_dropout_index by the total number of V1 neurons (ie in all bins of task_dropout_index)).

    parameter_primary_all = np.concatenate(([['task_dropout_index'], cols_for_clustering]))
    parameters_cont = ['all-images', 'omissions', 'pupil', 'running', 'face_motion_energy', 'pre_licks', 'post_licks', 'single-all-images', 'single-omissions', 'single-pupil', 'single-running', 'single-face_motion_energy', 'single-pre_licks', 'single-post_licks', 'imaging_depth']
    parameters_categ = ['cre_line','session_id','targeted_structure', 'clusterer_labels', 'equipment_name_categorical'] # , 'dominant_dropout' # session_id: 1 # novel

    for parameter_primary in parameter_primary_all:

        if norm_by_param_primary==1:
            nv = 'norm_by_param_prim'
        elif norm_by_param_primary==0:
            nv = 'norm_by_param_categ'
        elif np.isnan(norm_by_param_primary):
            nv = 'no_norm'

        nam = f'{what_cre_name}_bins_{parameter_primary}_{nv}_param_hists_{gmn}_{now}'
        fign = os.path.join(dir0, dir_now, nam+fmt)     

        # reload(plot_hist_parameter_bins)
        plot_hist_parameter_bins.plot_hist_parameter_bins(sorted_data, parameter_primary, parameters_cont, parameters_categ, norm_by_param_primary, [dosavefig, fign])
    
    
    
    
            
        
        
        
        
        

        
        
        
#############################################
#%% Pie charts: for each cluster id, we compute a given parameter's distribution; e.g. for cluster_id 0, we compute the number of cells that took each value of "session_id"

fig,ax = plt.subplots(20,4,figsize=(15,35))

parameters = ['cre_line','session_id','targeted_structure', 'imaging_depth']

for row,cluster_id in enumerate(sorted_data['clusterer_labels'].unique()):
    for col,parameter in enumerate(parameters):
        # make a histogram of imaging depths
        if parameter == 'imaging_depth':
            ax[row,col].hist(
                sorted_data.query('clusterer_labels == @cluster_id')['imaging_depth'],
                bins=np.arange(0,400,50),
                density=True
            )
            
        # make a pie chart for the rest
        else:
            # build a dataframe of value counts for the pie chart (there's probably a better way!)
            df = pd.DataFrame(sorted_data.query('clusterer_labels == @cluster_id')[parameter].value_counts()).sort_index()
            plot = df.plot.pie(
                y = parameter, 
                ax = ax[row,col],
                legend = False
            )
            ax[row,col].set_ylabel('')
        
        # set titles, make first row title different
        if row == 0:
            ax[row, col].set_title('parameter = {}\ncluster ID = {}'.format(parameter, cluster_id))
        else:
            ax[row, col].set_title('cluster ID = {}'.format(cluster_id))
            
            
fig.tight_layout()





