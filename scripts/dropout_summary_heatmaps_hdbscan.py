#!/usr/bin/env python
# coding: utf-8

from importlib import reload
# import hist_param_bins
# reload(hist_param_bins)

dosavefig = 0 #1

if dosavefig:
    import datetime
#     from def_paths import * 

    dir0 = '/home/farzaneh/OneDrive/Analysis'
    now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    dir_now = 'umap_cluster'
    fmt = '.pdf'


# In[ ]:


import visual_behavior_glm.src.GLM_params as glm_params
import visual_behavior_glm.src.GLM_analysis_tools as gat
import visual_behavior_glm.src.GLM_visualization_tools as gvt
from visual_behavior_glm.src.glm import GLM
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


# In[2]:

get_ipython().magic(u'matplotlib inline')   
# get_ipython().run_line_magic('matplotlib', 'notebook')
# get_ipython().run_line_magic('widescreen', '')


# # Gather/organize data

# ## load the results summary for a single GLM version from mongo

# In[3]:


rs = gat.retrieve_results(search_dict = {'glm_version': '4_L2_optimize_by_cell'}, results_type='summary')
rs #.sample(10)


# In[4]:


rs.shape, list(rs['dropout'].unique())


# In[5]:


# rs_now: rs for a single experiment, a single cell
rs_now = rs[np.logical_and(rs['ophys_experiment_id'].values==991852008, rs['cell_specimen_id'].values==1017215561)] #.shape (128, 37) # unique dropout: 32

print(np.shape(rs_now['dropout'].values))
print(list(rs_now['dropout'].values))


# # get a list of columns to use for clustering. Defining this list now makes it easier to avoid accidentally clustering on some identifying feature (e.g., cre_line, session_type, etc.)
# I'm removing the 'visual' dropout, since it's actually a combination of the omission and all-images dropouts.

# In[6]:


cols_for_clustering = list(rs['dropout'].unique())

cols_for_clustering.remove('visual')
'''
cols_for_clustering.remove('model_bias')
cols_for_clustering.remove('model_task0')
cols_for_clustering.remove('model_omissions1')
cols_for_clustering.remove('model_timing1D')
cols_for_clustering.remove('beh_model')

cols_for_clustering.remove('pre_lick_bouts')
cols_for_clustering.remove('post_lick_bouts')
'''
print(np.shape(cols_for_clustering)), cols_for_clustering


# ## build a pivoted version of the results summary, using the `fraction_change_from_full` column as the values.

# In[7]:


rsp = gat.build_pivoted_results_summary(results_summary=rs, cutoff=0.01, value_to_use='fraction_change_from_full')


# In[8]:


# check what fraction of cells are kept after applying the cutoff above.
np.unique(rs['cell_specimen_id']).shape, np.unique(rsp['cell_specimen_id']).shape, np.unique(rsp['cell_specimen_id']).shape[0]/np.unique(rs['cell_specimen_id']).shape[0]


# In[9]:


# rs.columns, rsp.columns, rs.shape, rsp.shape


# ## add a `session_id` column with a numeric value for the session_type (to lump together sessions by order, regardless of image set)

# In[10]:


def map_session_types(session_type):
    session_id = int(session_type[6:7])
    if session_id==4:
        session_id=1 # novel
    else:
        session_id=0 # not novel
    return session_id

rsp['session_id'] = rsp['session_type'].map(lambda st:map_session_types(st))


# In[11]:


sum(rsp['session_id']==0), sum(rsp['session_id']==1)


# ## Turn some categorical columns into numerical columns

# In[12]:


def make_categorical(df, column):
    df['{}_categorical'.format(column)] = pd.Categorical(df[column], ordered=True).codes

for column in ['cre_line','equipment_name','targeted_structure','session_id']:
    make_categorical(rsp, column)


# In[13]:

# keep a copy before merging with "data" that includes behavioral model

rsp00 = copy.deepcopy(rsp)


# ## Merge the behavioral glm df with rsp

# In[14]:


# load the behavioral model for each session
# for now we care about 'task_dropout_index'
# task_weight_index is the average value of the task weight for that session. 
# task_dropout_index is the value of the (task dropout - timing dropout). 
# task_only_dropout_index is just task dropout

data = loading.get_behavior_model_summary_table()
# data['task_dropout_index']
data.shape


# In[15]:


# compare number of sessions for the behavioral and neural glm models
np.unique(data['ophys_session_id'].values).shape, np.unique(rsp00['ophys_session_id'].values).shape



# In[17]:


# merge rsp and data on session_id
# rsp = rsp.merge(data, on=['ophys_session_id'])
rsp = rsp00.merge(data, on=['ophys_session_id', 'cre_line']) #, how='outer')
rsp00.shape, rsp.shape


# In[18]:


np.unique(rsp['ophys_session_id'].values).shape


# In[19]:


list(rsp.columns) #, rsp.shape


#%% take care of column renaming after the merge

rsp = rsp.rename(columns = {'imaging_depth_x': 'imaging_depth'}, inplace = False)


# In[20]:


# rsp.iloc[1298]['ophys_experiment_id_x'], rsp.iloc[1298]['ophys_experiment_id_y']


# In[21]:


# rsp[rsp['ophys_experiment_id_x'].values==795948257][['cell_specimen_id', 'beh_model']]


# ## Redefine cols_for_clustering
# ### add some features from behavioral glm to rsp 

# In[ ]:


# make sure to not add features that get repeated across cells (eg task_dropout_index will be repeated for all cells of the same session)
# cols_for_clustering.append('task_dropout_index')
# np.shape(cols_for_clustering), cols_for_clustering


# # Dimensionality reduction and clustering
# So many more things to try here. This is just a start.

# ## UMAP

# In[24]:


neigh_vals = np.concatenate(([5, 10, 15, 50], np.arange(200, int(rsp.shape[0]/10), 500)))
print(neigh_vals)


# In[26]:


rsp.shape, rsp[cols_for_clustering].shape, cols_for_clustering


# In[101]:

# umap relies on stochastis methods so we need to set the seed for it.
import random
rand_state = 42
np.random.seed(rand_state)
os.environ['PYTHONHASHSEED'] = str(rand_state)
random.seed(rand_state)                                
                                
                                
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


# In[102]:


embedding.shape, neigh, mindist


# ## k-means clustering on 3d umap
# 

# In[29]:


kmeans = KMeans(n_clusters=20)
umap_cols = ['umap_3d_embedding_0','umap_3d_embedding_1','umap_3d_embedding_2']
rsp['clusterer_labels'] = kmeans.fit_predict(rsp[umap_cols])
rsp['clusterer_labels'].value_counts()



# ## hdbscan clustrering

# In[103]:


# Try a range of parameters

min_cluster_size_all = [5, 20, 50, 100, 200] #100 #50 # (default=5)
min_samples_all = [10] #10 #min_cluster_size #18 # default: same as min_cluster_size

clusterer_all = []
clusterer_labels_all = []
clusterer_params = []
val_per_clust_all = []
cluster_ids_all = []

for min_cluster_size in min_cluster_size_all:
    for min_samples in np.unique(np.concatenate((min_samples_all, [min_cluster_size]))):
        
        print('______________________')
        min_cluster_size = int(min_cluster_size)
        min_samples = int(min_samples)
        print([min_cluster_size, min_samples])
        
        ### set cluster_selection_epsilon to .5
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=.5) #(min_cluster_size=9, gen_min_span_tree=True)

        ### don't set cluster_selection_epsilon
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        
        clusterer.fit(embedding) # rsp[umap_cols]
        
        clusterer_labels_ = clusterer.labels_
        val_per_clust = [np.sum(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))] # number of points per cluster
        cluster_ids = np.unique(clusterer.labels_)
        
        print(cluster_ids[[0,-1]])
        print(val_per_clust)
        
        clusterer_labels_all.append(clusterer_labels_) # cluster_labels = clusterer.fit_predict(embedding)
        clusterer_all.append(clusterer)
        clusterer_params.append([min_cluster_size, min_samples])
        val_per_clust_all.append(val_per_clust)
        cluster_ids_all.append(cluster_ids)


# In[80]:


# rsp = rspall


# In[79]:


embedding.shape


# In[104]:
### pick a set of parameters according to the search above

min_cluster_size = 100 #50 # (default=5)
min_samples = 10 #100 #10 #min_cluster_size #18 # default: same as min_cluster_size

# clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=.5)
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples) #(min_cluster_size=9, gen_min_span_tree=True)

clusterer.fit(embedding) # rsp[umap_cols]
# clusterer.labels_ # cluster_labels = clusterer.fit_predict(embedding)
# clusterer.probabilities_
rsp['clusterer_labels'] = clusterer.labels_
# val_per_clust = rsp['clusterer_labels'].value_counts().values # number of points per cluster # this is sorted by the number of cells per cluster
# below is sorted by the cluster id (i prefer below) (ie number of cells per cluster, sorted by cluster id, ie from cluster -1 to end)
val_per_clust = [np.sum(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))] # number of points per cluster

# In[105]:


# fract_per_clust = [np.mean(clusterer.labels_ == np.unique(clusterer.labels_)[i]) for i in range(len(np.unique(clusterer.labels_)))]
print(len(np.unique(clusterer.labels_)))
# noise points
print(sum(clusterer.labels_ == -1), np.mean(clusterer.labels_ == -1))

rsp['clusterer_labels'].value_counts()


# In[84]:


# evaluate clustering results

# print(val_per_clust.shape)
print(np.sort(val_per_clust))

# # number of clusters with <th_neur neurons in them.
# th_n = 20 # we dont want to have less than 20 neurons in a cluster, maybe.
# print(sum(val_per_clust < th_n), np.mean(val_per_clust < th_n))

# # noise points
# print(sum(clusterer.labels_ == -1), np.mean(clusterer.labels_ == -1))


# In[85]:

# plot clusterer probabilities and number of cells in each label.

plt.figure(figsize=(8,3))

plt.subplot(121), plt.plot(np.sort(clusterer.probabilities_)), plt.title('clusterer.probabilities_')
plt.xlabel('cells')
plt.subplot(122), plt.plot(np.unique(clusterer.labels_), val_per_clust), plt.title('cluster size')
plt.xlabel('cluster id')

if dosavefig:
    nam = f'cluster_prob_size_allCre_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{now}'
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    
# ## visualize in 2d using matplotlib

# In[86]:


color_palette = sns.color_palette('Paired', len(rsp['clusterer_labels'].unique()))
cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(rsp['umap_3d_embedding_0'], rsp['umap_3d_embedding_1'], s=10, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.xlabel('umap_3d_embedding_0'), plt.ylabel('umap_3d_embedding_1')
plt.title(f'{len(val_per_clust)} clusters')

plt.subplot(122)
plt.scatter(rsp['umap_3d_embedding_0'], rsp['umap_3d_embedding_2'], s=10, linewidth=0, c=cluster_member_colors, alpha=0.25)
plt.xlabel('umap_3d_embedding_0'), plt.ylabel('umap_3d_embedding_2')

if dosavefig:
    nam = f'umapEmbed_clustColored_allCre_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{now}'
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
    
# In[87]:


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
    nam = f'umapEmbed_clustColored_Doug_allCre_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{now}'
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    

# ## visualize in 3d using plotly

# In[88]:


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




# # A little bit of prep for plotting

# ## first define the dominant dropout in each cluster

# In[106]:


# identify dominant dropouts for each cluster
gat.identify_dominant_dropouts(rsp, cluster_column_name='clusterer_labels', cols_to_search=cols_for_clustering)


# In[107]:


np.unique(rsp['dominant_dropout'])


# In[108]:


for column in ['dominant_dropout']:
    make_categorical(rsp, column)


# ## only get those rsp rows that have non-noise cluster values

# In[109]:


rsp['clusterer_labels'].value_counts(), sum(rsp['clusterer_labels'].value_counts())


# In[96]:


rspall = copy.deepcopy(rsp)
rspall.shape


# In[97]:


# use all rsp rows in clustering
rsp = rspall
rsp.shape


# In[72]:


# only get those rsp rows that have non-noise cluster values
rsp = rspall[rspall['clusterer_labels'] != -1]
rsp.shape


# In[98]:


np.unique(rsp['dominant_dropout'])


#############################################
# ## define heatmap parameters
# lots here, but there are lots of options!

# In[110]:


cols_to_plot = [
    'clusterer_labels',      
    'all-images', 
    'omissions', 
    'pupil', 
    'running',
    'cre_line_categorical', 
    'dominant_dropout_categorical',
    'session_id_categorical', 
#    'equipment_name_categorical',
    'targeted_structure_categorical',
    'imaging_depth',
    'task_dropout_index',  
]

# make sure there's a heatmap definition for every column, otherwise the column will plot without a heatmap!
heatmap_defs = [
    {
        'columns':cols_to_plot[1:5],
        'cbar_label':'fraction change\nin var explained',
        'cbar_ticks':[-1,0,1],
        'vmin':-1,
        'vmax':1,
        'cmap':'bwr',
    },
    {
        'columns':['cre_line_categorical'],
        'cbar_label':'cre_line',
        'cbar_ticks':[0,1,2],
        'cbar_ticklabels':np.sort(np.unique(rsp['cre_line'])),
        'vmin':-0.5,
        'vmax':2.5,
        'cmap':sns.color_palette("hls", 3),
    },    
    {
        'columns':['dominant_dropout_categorical'],
        'cbar_label':'dominant_dropout_categorical',
        'cbar_ticks':np.arange(len(rsp['dominant_dropout_categorical'].unique())),
        'cbar_ticklabels':np.sort(np.unique(rsp['dominant_dropout'])),
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
        'cbar_label':'k-means label',
        'cbar_ticks':np.arange(min(rsp['clusterer_labels'].unique()),len(rsp['clusterer_labels'].unique()),2),
        'vmin':min(rsp['clusterer_labels'].unique()),
        'vmax':len(rsp['clusterer_labels'].unique())-1,
        'cmap':sns.color_palette("hsv", len(rsp['clusterer_labels'].unique())),
    },
#     {
#         'columns':['equipment_name_categorical'],
#         'cbar_label':'equipment name',
#         'cbar_ticks':np.arange(len(rsp['equipment_name_categorical'].unique())),
#         'cbar_ticklabels':np.sort(np.unique(rsp['equipment_name'])),
#         'vmin':-0.5,
#         'vmax':len(rsp['equipment_name_categorical'].unique())-0.5,
#         'cmap':sns.color_palette("hls", len(rsp['equipment_name_categorical'].unique())),
#     },
    {
        'columns':['targeted_structure_categorical'],
        'cbar_label':'targeted structure',
        'cbar_ticks':np.arange(len(rsp['targeted_structure_categorical'].unique())),
        'cbar_ticklabels':np.sort(np.unique(rsp['targeted_structure'])),
        'vmin':-0.5,
        'vmax':len(rsp['targeted_structure'].unique())-0.5,
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


# # Make heatmaps, sort by whatever we want
# Note that it only makes sense to nest sorting values for categorical data. Any sorting value that follows a continuous variable will not have any effect.

# ## First sort by dominant dropout, the median value of the dominant dropout (for cases with multiple clusters sharing the same dominant dropout), then cre_line, session_id, imaging_depth

# In[111]:


# sort_order = ['task_dropout_index', 'dominant_dropout','dominant_dropout_median','cre_line_categorical']
# sort_order = ['dominant_dropout','dominant_dropout_median','cre_line_categorical','session_id_categorical','imaging_depth',]
sort_order = ['clusterer_labels', 'dominant_dropout','dominant_dropout_median','cre_line_categorical']
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

# for idx,row in sorted_data.query('cluster_transition').iterrows():
#     axes['heatmap'].axhline(idx,color='black')

if dosavefig:
    nam = f'glmHeatmap_clusters_allCre_UMAP_{n_components}_{neigh}_{mindist}_hdbscan_{min_cluster_size}_{min_samples}_{now}'
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    




#############################################

parameter_primary = 'task_dropout_index'
parameters_cont = ['all-images', 'omissions', 'pupil', 'running', 'imaging_depth']
parameters_categ = ['cre_line','session_id','targeted_structure', 'clusterer_labels'] # session_id: 1 # novel

nam = f'bins_{parameter_primary}_param_hists_allCre_{now}'
fign = os.path.join(dir0, dir_now, nam+fmt)     


# bin a feature (lets call it the primary feature: parameters_primary), and then for each bin look at the distribution of a bunch of other features (parameters_cont and parameters_categ)

reload(hist_param_bins)

hist_param_bins.hist_param_bins(sorted_data, parameter_primary, parameters_cont, parameters_categ, [dosavefig, fign])
    
    
    
    
            

#############################################

# For each cluster id, we compute a given parameter's distribution; e.g. for cluster_id 0, we compute the number of cells that took each value of "session_id"

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


# ## another sorting, with imaging_depth being the emphasis

# In[ ]:


sort_order = ['dominant_dropout','dominant_dropout_median','imaging_depth','cre_line_categorical','session_id_categorical']
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
for idx,row in sorted_data.query('cluster_transition').iterrows():
    axes['heatmap'].axhline(idx,color='black')


# ## and another sorting, with session_id being the emphasis

# In[ ]:


sort_order = ['dominant_dropout','dominant_dropout_median','session_id_categorical','cre_line_categorical','imaging_depth',]
sorted_data = gat.sort_data(rsp, sort_order, cluster_column_name='clusterer_labels')

fig, axes = vbp.make_multi_cmap_heatmap(
    sorted_data[cols_to_plot], 
    heatmap_defs, 
    figsize=(10,10), 
    top_buffer=0, 
    bottom_buffer=0.1, 
    n_cbar_rows=4, 
    heatmap_div=0.7, 
)
for idx,row in sorted_data.query('cluster_transition').iterrows():
    axes['heatmap'].axhline(idx,color='black')


# ## sort by structure after dropout

# In[ ]:


sort_order = ['dominant_dropout','dominant_dropout_median','targeted_structure_categorical','cre_line_categorical','session_id_categorical','imaging_depth']
sorted_data = gat.sort_data(rsp, sort_order, cluster_column_name='clusterer_labels')

fig, axes = vbp.make_multi_cmap_heatmap(
    sorted_data[cols_to_plot], 
    heatmap_defs, 
    figsize=(10,15), 
    top_buffer=0, 
    bottom_buffer=0.1, 
    n_cbar_rows=4, 
    heatmap_div=0.7, 
)
for idx,row in sorted_data.query('cluster_transition').iterrows():
    axes['heatmap'].axhline(idx,color='black')


# ## Sort by cluster ID and move it to the leftmost column

# In[ ]:


sort_order = ['clusterer_labels','cre_line_categorical','session_id_categorical','imaging_depth']
sorted_data = gat.sort_data(rsp, sort_order, cluster_column_name='clusterer_labels')

cols_to_plot = [
    'clusterer_labels', 
    'all-images', 
    'omissions', 
    'pupil', 
    'running',
    'cre_line_categorical', 
    'session_id_categorical', 
    'equipment_name_categorical',
    'targeted_structure_categorical',
    'imaging_depth'
]

fig, axes = vbp.make_multi_cmap_heatmap(
    sorted_data[cols_to_plot], 
    heatmap_defs, 
    figsize=(10,10), 
    top_buffer=0, 
    bottom_buffer=0.1, 
    n_cbar_rows=4, 
    heatmap_div=0.7, 
)
for idx,row in sorted_data.query('cluster_transition').iterrows():
    axes['heatmap'].axhline(idx,color='black')

