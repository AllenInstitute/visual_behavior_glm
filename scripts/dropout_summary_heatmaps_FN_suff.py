#%%

# fraction_change_from_full = absolute_change_from_full / variance_explained(by the full model)
# confirm the above

# rs_now: rs for a single experiment, a single cell
rs_now = rs[np.logical_and(rs['ophys_experiment_id'].values==991852008, rs['cell_specimen_id'].values==1017215561)] #.shape (128, 37) # unique dropout: 32

variance_explained = rs_now[rs_now['dropout'].values=='Full']['variance_explained'].values
this_feature_abs_change_from_full = rs_now[rs_now['dropout'].values=='intercept']['absolute_change_from_full'].values
this_feature_fraction_change_from_full = rs_now[rs_now['dropout'].values=='intercept']['fraction_change_from_full'].values
variance_explained, this_feature_abs_change_from_full, this_feature_abs_change_from_full/variance_explained, this_feature_fraction_change_from_full



#%%
print(rs_now[rs_now['dropout'].values=='beh_model']['variance_explained'].values, 
rs_now[rs_now['dropout'].values=='model_bias']['variance_explained'].values, 
rs_now[rs_now['dropout'].values=='model_timing1D']['variance_explained'].values, 
rs_now[rs_now['dropout'].values=='model_task0']['variance_explained'].values, 
rs_now[rs_now['dropout'].values=='model_omissions1']['variance_explained'].values)




################# doug's other version's of sorting the heatmap #################

#%% another sorting, with imaging_depth being the emphasis

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



#%% and another sorting, with session_id being the emphasis

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



#%% sort by structure after dropout

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




#%% Sort by cluster ID and move it to the leftmost column

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
