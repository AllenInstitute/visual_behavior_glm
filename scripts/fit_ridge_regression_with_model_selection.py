import pandas as pd
import sys
sys.path.append('/home/nick.ponvert/src/nick-allen/projects/ophys_glm')

import os
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy
import model_utils as m
import numpy as np

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
from visual_behavior.translator.allensdk_sessions import sdk_utils
from visual_behavior.translator.allensdk_sessions import session_attributes
from visual_behavior.ophys.response_analysis import response_processing as rp

#  from allensdk.brain_observatory.behavior import behavior_project_cache as bpc
#  from allensdk.brain_observatory.behavior import response_processing as rp

import argparse

manifest_default = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/behavior_project_cache_20200127/manifest.json'

parser = argparse.ArgumentParser(description='GLM Fitter')
parser.add_argument('--ophys-session-id', type=int, default=877946125,
                    metavar='XXXXXXXXX',
                    help='ophys session ID to process')
parser.add_argument('--manifest', type=str, default=manifest_default,
                    metavar='/data/example.csv',
                    help='path to data manifest (in csv format)')
parser.add_argument('--output-dir', type=str, default='.',
                    metavar='/data/output_dir',
                    help='output data path')
parser.add_argument('--regularization-lambda', type=int, default=70,
                    help='Regularization lambda param')
args = parser.parse_args()


if __name__=="__main__":
    osid = args.ophys_session_id
    output_dir = args.output_dir

    cache = BehaviorProjectCache.from_lims(manifest=args.manifest)
    oeid = sdk_utils.get_oeid_from_osid(osid, cache)

    session = cache.get_session_data(oeid)

    # Add stim response / extended stim information
    session_attributes.filter_invalid_rois_inplace(session)
    sdk_utils.add_stimulus_presentations_analysis(session)
    session.stimulus_response_df = rp.stimulus_response_df(rp.stimulus_response_xr(session))

    dff_trace_timestamps = session.ophys_timestamps

    # clip off the grey screen periods
    timestamps_to_use = m.get_ophys_frames_to_use(session)
    dff_trace_timestamps = dff_trace_timestamps[timestamps_to_use]

    # Get the matrix of dff traces
    dff_trace_arr = m.get_dff_arr(session, timestamps_to_use)

    # Make design matrix
    design = m.DesignMatrix(dff_trace_timestamps[:-1])

    # Get event times
    event_times = session.trials.query('go')['change_time'].values
    event_times = event_times[~np.isnan(event_times)]

    # NOTE: Trying grouping by image index instead of group by name so that things are consistent across sets
    #  flash_time_gb = session.stimulus_presentations.groupby('image_name').apply(lambda group: group['start_time'])
    flash_time_gb = session.stimulus_presentations.groupby('image_index').apply(lambda group: group['start_time'])

    image_names = []
    all_tmats = []
    #  for image_name in flash_time_gb.index.levels[0].values:
    for image_index in flash_time_gb.index.levels[0].values:
        times_this_flash = flash_time_gb[image_index].values
        design_vec_this_image, timestamps = np.histogram(times_this_flash, bins=dff_trace_timestamps)

        image_name = 'image_{}'.format(image_index)

        design.add_kernel(design_vec_this_image, 30, image_name, offset=0)
        image_names.append(image_name)

    # Bin events using the ophys times as bin edges
    events_vec, timestamps = np.histogram(event_times, bins=dff_trace_timestamps)
    design.add_kernel(events_vec, 100, 'change', offset=0)

    # Bin reward times 
    reward_times = session.rewards['timestamps'].values
    rewards_vec, timestamps = np.histogram(reward_times, bins=dff_trace_timestamps)
    design.add_kernel(rewards_vec, 115, 'reward', offset=-15)

    # Bin lick times
    lick_times = session.licks['timestamps'].values
    licks_vec, timestamps = np.histogram(lick_times, bins=dff_trace_timestamps)
    design.add_kernel(licks_vec, 30, 'licks', offset=-10)

    # Get CV splits
    splits = m.split_time(events_vec)
    n_splits = len(splits)

    lam_value = args.regularization_lambda

    # Set up kernels to drop for model selection
    to_drop = []
    drop_labels = []
    to_drop.append([])
    drop_labels.append('full')
    for name in design.labels:
        to_drop.append([name])
        drop_labels.append(name)
    to_drop.append([
        'image_{}'.format(i) for i in range(9)
    ])
    drop_labels.append('all_images')

    # Dictionary to save out the final weights matrix for each reduced model
    all_W = {}

    # Arry to save out cv var explained for each cell, each model
    test_mean_all_models = np.empty((dff_trace_arr.shape[1], len(drop_labels)))
    train_mean_all_models = np.empty((dff_trace_arr.shape[1], len(drop_labels)))

    for ind_model, model_label in enumerate(drop_labels):

        labels_to_use = [label for label in design.labels if label not in to_drop[ind_model]]

        # Design matrix to use for this reduced model fit
        X_this_model = design.get_X(kernels=labels_to_use)
        n_params = X_this_model.shape[0]
        n_neurons = dff_trace_arr.shape[1]

        all_W_this_model = np.empty((n_params, n_neurons, n_splits))
        
        # Do cross-validated fitting for the reduced model
        cv_var_train = np.empty((dff_trace_arr.shape[1], len(splits)))
        cv_var_test = np.empty((dff_trace_arr.shape[1], len(splits)))

        #TODO: Save weights each model
        #  W_this_model = np.empty(dff_trace_arr.shape[1], np.shape(X_this_model))
        for ind_test_split, test_split in tqdm(enumerate(splits)):
            train_split = np.concatenate(
                [split for i, split in enumerate(splits) if i!=ind_test_split]
            )

            X_test = X_this_model[:, test_split].T
            X_train = X_this_model[:, train_split].T

            dff_train = dff_trace_arr[train_split, :]
            dff_test = dff_trace_arr[test_split, :]
            W = m.fit_regularized(dff_train, X_train, lam_value)

            # Save out var explained
            cv_var_test[:, ind_test_split] = m.variance_ratio(
                dff_test,
                W,
                X_test
            )
            cv_var_train[:, ind_test_split] = m.variance_ratio(
                dff_train,
                W,
                X_train
            )
            all_W_this_model[:, :, ind_test_split] = W

        test_mean_all_models[:, ind_model] = cv_var_test.mean(axis=1)
        train_mean_all_models[:, ind_model] = cv_var_train.mean(axis=1)
        all_W.update({model_label:all_W_this_model.mean(axis=2)})

    #  all_cells = session.['cell_specimen_id'].unique()
    all_cell_ids = dff_trace_arr.coords['cell_specimen_id'].values

    df_all_test = pd.DataFrame(test_mean_all_models, columns=drop_labels, index=all_cell_ids)
    df_all_train = pd.DataFrame(train_mean_all_models, columns=drop_labels, index=all_cell_ids)

    df_result = df_all_test.join(df_all_train,  lsuffix='_test', rsuffix='_train')

    #  df_result = pd.DataFrame()
    #  all_w_arrs = []
    #  all_cells = session.stimulus_response_df['cell_specimen_id'].unique()
    #  for ind_cell, csid in enumerate(all_cells):
    #  
    #      pvals = session.stimulus_response_df.query('cell_specimen_id==@csid and pref_stim')['p_value']
    #  
    #      w_this_cell = all_W[:, ind_cell, :] #370x6
    #      avg_w = np.mean(w_this_cell, axis=1)
    #  
    #      cv_test_this_cell = cv_var_test[ind_cell, :] #1x6
    #      cv_train_this_cell = cv_var_train[ind_cell, :] #1x6
    #  
    #      df_result.at[csid, 'weights_pref_image'] = m.pref_image_filter(avg_w, image_names)
    #      df_result.at[csid, 'sdk_pref_image'] = m.pref_stim(session, ind_cell)
    #      df_result.at[csid, 'responsive'] = 1 if np.mean(pvals<0.05) > 0.25 else 0
    #      #  df_result.at[csid, 'cv_var_explained_test'] = np.mean(cv_test_this_cell)
    #      #  df_result.at[csid, 'cv_var_explained_train'] = np.mean(cv_train_this_cell)
    #      dff_this_cell = dff_trace_arr.sel({'cell_specimen_id':csid})
    #      df_result.at[csid, 'dff_max'] = np.max(dff_this_cell)
    #      all_w_arrs.append(w_this_cell.tolist())
    #  
    #  df_result['w'] = all_w_arrs
    
    output_dict = df_result.to_dict()
    
    sparse_X = scipy.sparse.csc_matrix(design.X.T)
    fn_x = 'X_sparse_csc_{}.npz'.format(osid)
    x_full_path = os.path.join(output_dir, fn_x)
    scipy.sparse.save_npz(x_full_path, sparse_X)

    fn_events = 'event_times_{}.h5'.format(osid)
    events_full_path = os.path.join(output_dir, fn_events)
    pd.DataFrame(design.events).to_hdf(events_full_path, key='df')
    
    output_dict.update({'ophys_session_id': osid})
    
    fn = 'osid_{}.json'.format(osid)
    output_full_path = os.path.join(output_dir, fn)
    with open(output_full_path, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)
    print('saved file to: {}'.format(output_full_path))
