import os
import bz2
import scipy
import pickle
import _pickle as cPickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import xarray_mongodb
from tqdm import tqdm
import matplotlib.pyplot as plt

import visual_behavior_glm_strategy.GLM_params as glm_params
import visual_behavior.data_access.loading as loading
import visual_behavior.database as db

from sklearn.decomposition import PCA

def load_fit_pkl(run_params, ophys_experiment_id):
    '''
        Loads the fit dictionary from the pkl file dumped by fit_experiment.
        Attempts to load the compressed pickle file if it exists, otherwise loads the uncompressed file
    
        Inputs:
        run_params, the dictionary of parameters for this version
        ophys_experiment_id, the oeid to load the fit for
    
        Returns:
        the fit dictionary if it exists

    ''' 

    filenamepkl = os.path.join(run_params['experiment_output_dir'],str(ophys_experiment_id)+'.pkl')
    filenamepbz2 = os.path.join(run_params['experiment_output_dir'],str(ophys_experiment_id)+'.pbz2')

    if os.path.isfile(filenamepbz2):
        fit = bz2.BZ2File(filenamepbz2, 'rb')
        fit = cPickle.load(fit)
        return fit
    elif os.path.isfile(filenamepkl):
        with open(filenamepkl,'rb') as f:
            fit = pickle.load(f)
        return fit
    else:
        return None

def log_error(error_dict, keys_to_check = []):
    '''
    logs contents of error_dict to the `error_logs` collection in the `ophys_glm` mongo database
    '''
    conn=db.Database('visual_behavior_data') #establishes connection
    db.update_or_create(
        collection = conn['ophys_glm']['error_logs'],
        document = db.clean_and_timestamp(error_dict),
        keys_to_check = keys_to_check, # keys to check to determine whether an entry already exists. Overwrites if an entry is found with matching keys
    )
    conn.close()

def get_error_log(search_dict = {}):
    '''
    searches the mongo error log for all entries matching the search_dict
    if search dict is an empty dict (default), it will return full contents of the kernel_error_log collection
    '''
    conn=db.Database('visual_behavior_data') #establishes connection
    result = conn['ophys_glm']['error_logs'].find(search_dict)
    conn.close()
    return pd.DataFrame(list(result))

def build_kernel_df(glm, cell_specimen_id):
    '''
    creates a dataframe summarizing each GLM kernel's contribution over timefor a given cell

    '''
    kernel_list = list(glm.design.kernel_dict.keys())
    model_timestamps = glm.fit['fit_trace_arr']['fit_trace_timestamps'].values
    kernel_df = []

    # get all weight names
    all_weight_names = glm.X.weights.values

    # iterate over all kernels
    for ii, kernel_name in enumerate(kernel_list):
        # get the full kernel (dims = n_weights x n_timestamps)
        kernel = glm.design.kernel_dict[kernel_name]['kernel']

        # get the weight matrix for the weights associated with this kernel and cell (dims = 1 x n_weights)
        kernel_weight_names = [w for w in glm.W.weights.values if w.startswith(kernel_name)]
        w_kernel = np.expand_dims(glm.W.loc[dict(
            weights=kernel_weight_names, cell_specimen_id=cell_specimen_id)], axis=0)

        # Version 19 had an edge case where the design matrix gets re-loaded with an extra weight       
        if kernel_name.startswith('image') &(np.shape(kernel)[0] != np.shape(w_kernel)[1]):
            print('Hack, need to fix bug')
            kernel = kernel[0:-1,:]

        # calculate kernel output as w_kernel * kernel (dims = 1 x n_timestamps)
        # add to list of dataframes with cols: timestamps, kernel_outputs, kernel_name
        kernel_df.append(
            pd.DataFrame({
                'timestamps': model_timestamps,
                'timestamp_index':np.arange(len(model_timestamps)),
                'kernel_outputs': (w_kernel @ kernel).squeeze(),
                'kernel_name': [kernel_name]*len(model_timestamps)
            })
        )

    # return the concatenated dataframe (concatenating a list of dataframes makes a single dataframe)
    return pd.concat(kernel_df)

def generate_results_summary(glm):
    nonadj_dropout_summary = generate_results_summary_nonadj(glm)
    adj_dropout_summary = generate_results_summary_adj(glm)

    dropout_summary = pd.merge(
        nonadj_dropout_summary, 
        adj_dropout_summary,
        on=['dropout', 'cell_specimen_id']
    ).reset_index()
    dropout_summary.columns.name = None
    return dropout_summary

def generate_results_summary_adj(glm):
    '''
        Returns a dataframe with summary information from the glm object
    '''
    # Get list of columns to look at, removing the non-adjusted dropouts, and training scores
    test_cols = [col for col in glm.results.columns if ((not col.endswith('train'))&('adj' in col))]
    
    # Set up space
    results_summary_list = []

    # Iterate over cells
    for cell_specimen_id in glm.results.index.values:

        # For each cell, get the relevant columns
        results_summary = pd.DataFrame(glm.results.loc[cell_specimen_id][test_cols]).reset_index().rename(columns={cell_specimen_id:'variance_explained','index':'dropout_name'})

        # For each dropout, separate the name of the dropout from the type of information
        for idx,row in results_summary.iterrows():
            results_summary.at[idx,'dropout'] = row['dropout_name'].split('__')[0]
            results_summary.at[idx,'type'] = row['dropout_name'].split('__')[1]

        # pivot the table on the dropout names
        results_summary = pd.pivot_table(results_summary.drop(columns=['dropout_name']), index=['dropout'],columns=['type'],values =['variance_explained'],dropna=False)
        results_summary.columns = results_summary.columns.droplevel()
        results_summary = results_summary.rename(columns={
            'avg_cv_adjvar_test': 'adj_variance_explained',
            'avg_cv_adjvar_test_full_comparison': 'adj_variance_explained_full',
            'adj_dropout': 'adj_fraction_change_from_full'
        })
 
        # add the cell id info
        results_summary['cell_specimen_id'] = cell_specimen_id
         
        # pack up
        results_summary_list.append(results_summary)

    # Concatenate all cells and return

    return pd.concat(results_summary_list)

def generate_results_summary_nonadj(glm):
    '''
        Returns a dataframe with summary information from the glm object
    '''
    # Get list of columns to look at, removing the non-adjusted dropouts, and training scores
    test_cols = [col for col in glm.results.columns if ((not col.endswith('train'))&('adj' not in col)&('session' not in col)&('cell' not in col))]  
    if 'Full__shuffle_cells' in glm.results.columns:
        test_cols.append('Full__shuffle_cells')
    if 'Full__cell_L2_regularization' in glm.results.columns:
        test_cols.append('Full__cell_L2_regularization')
 
    # Set up space
    results_summary_list = []

    # Iterate over cells
    for cell_specimen_id in glm.results.index.values:

        # For each cell, get the relevant columns
        results_summary = pd.DataFrame(glm.results.loc[cell_specimen_id][test_cols]).reset_index().rename(columns={cell_specimen_id:'variance_explained','index':'dropout_name'})

        # For each dropout, separate the name of the dropout from the type of information
        for idx,row in results_summary.iterrows():
            results_summary.at[idx,'dropout'] = row['dropout_name'].split('__')[0]
            results_summary.at[idx,'type'] = row['dropout_name'].split('__')[1]

        # pivot the table on the dropout names
        results_summary = pd.pivot_table(results_summary.drop(columns=['dropout_name']), index=['dropout'],columns=['type'],values =['variance_explained'],dropna=False)
        results_summary.columns = results_summary.columns.droplevel()
        results_summary = results_summary.rename(columns={
            'avg_cv_var_test':'variance_explained',
            'avg_cv_var_test_full_comparison':'variance_explained_full',
            'dropout':'fraction_change_from_full'})
 
        # add the cell id info
        results_summary['cell_specimen_id'] = cell_specimen_id
         
        # pack up
        results_summary_list.append(results_summary)

    # Concatenate all cells and return

    return pd.concat(results_summary_list)

def generate_results_summary_non_cleaned(glm):
    '''
        Returns a dataframe with summary information from the glm object
    '''
    # Preserving the old functionality for now, but filtering out the adjusted variance columns
    test_cols = [col for col in glm.results.columns if (col.endswith('test') & ('adj' not in col))]
    results_summary_list = []
    for cell_specimen_id in glm.results.index.values:
        results_summary = pd.DataFrame(glm.results.loc[cell_specimen_id][test_cols]).reset_index().rename(columns={cell_specimen_id:'variance_explained','index':'dropout'})
        for idx,row in results_summary.iterrows():
            results_summary.at[idx,'dropout'] = row['dropout'].split('_avg')[0]

        def calculate_fractional_change(row):
            full_model_performance = results_summary[results_summary['dropout']=='Full']['variance_explained'].iloc[0]
            return (row['variance_explained'] - full_model_performance)/full_model_performance

        def calculate_absolute_change(row):
            full_model_performance = results_summary[results_summary['dropout']=='Full']['variance_explained'].iloc[0]
            return row['variance_explained'] - full_model_performance

        results_summary['fraction_change_from_full'] = results_summary.apply(calculate_fractional_change, axis=1)
        results_summary['absolute_change_from_full'] = results_summary.apply(calculate_absolute_change, axis=1)
        results_summary['cell_specimen_id'] = cell_specimen_id
        results_summary_list.append(results_summary)
    return pd.concat(results_summary_list)


def identify_dominant_dropouts(data, cluster_column_name, cols_to_search):
    '''
    for each cluster ID, identifies the dominant dropout value amongst the `cols_to_search`
    adds columns for 'dominant_dropout' and 'dominant_dropout_median'
    operates in place
    inputs:
        data - (pandas dataframe) dataframe to operate on
        cluster_column_name - (string) name of column containing cluster IDs
        cols_to_search - (list) list of columns to search over for dominant column. Should be same set of columns used for clustering
    returns:
        None (operates in place)
    
    '''
    for cluster_id in data[cluster_column_name].unique():
        data_subset = data.query("{} == {}".format(cluster_column_name, cluster_id))

        data_subset_medians = data_subset[cols_to_search].median(axis=0)
        data.loc[data_subset.index, 'dominant_dropout'] = data_subset_medians.idxmin()
        data.loc[data_subset.index, 'dominant_dropout_median'] = data_subset_medians.min()


def sort_data(df_in, sort_order, cluster_column_name):
    '''
    sort dataframe by `sort_order`
    identifies rows where the cluster_id shifts
    '''
    sorted_data = (df_in
            .sort_values(by=sort_order)
            .reset_index(drop=True)
        )

    # identify cluster transitions
    sorted_data['cluster_transition'] = sorted_data[cluster_column_name] != sorted_data[cluster_column_name].shift()
    return sorted_data


def already_fit(oeid, version):
    '''
    check the weight_matrix_lookup_table to see if an oeid/glm_version combination has already been fit
    returns a boolean
    '''
    conn = db.Database('visual_behavior_data')
    coll = conn['ophys_glm']['weight_matrix_lookup_table']
    document_count = coll.count_documents({'ophys_experiment_id':int(oeid), 'glm_version':str(version)})
    conn.close()
    return document_count > 0


def log_results_to_mongo(glm):
    '''
    logs full results and results summary to mongo
    Ensures that there is only one entry per cell/experiment (overwrites if entry already exists)
    '''
    full_results = glm.results.reset_index()
    results_summary = glm.dropout_summary

    full_results['glm_version'] = str(glm.version)
    results_summary['glm_version'] = str(glm.version)

    results_summary['ophys_experiment_id'] = glm.ophys_experiment_id
    results_summary['ophys_session_id'] = glm.ophys_session_id

    full_results['ophys_experiment_id'] = glm.ophys_experiment_id
    full_results['ophys_session_id'] = glm.ophys_session_id

    conn = db.Database('visual_behavior_data')

    keys_to_check = {
        'results_full':['ophys_experiment_id','cell_specimen_id','glm_version'],
        'results_summary':['ophys_experiment_id','cell_specimen_id', 'dropout','glm_version']
    }

    for df,collection in zip([full_results, results_summary], ['results_full','results_summary']):
        coll = conn['ophys_glm'][collection]

        for idx,row in df.iterrows():
            entry = row.to_dict()
            db.update_or_create(
                coll,
                db.clean_and_timestamp(entry),
                keys_to_check = keys_to_check[collection]
            )
    conn.close()

def xarray_to_mongo(xarray):
    '''
    writes xarray to the 'ophys_glm_xarrays' database in mongo
    returns _id of xarray in the 'ophys_glm_xarrays' database
    '''
    conn = db.Database('visual_behavior_data')
    w_matrix_database = conn['ophys_glm_xarrays']
    xdb = xarray_mongodb.XarrayMongoDB(w_matrix_database)
    _id, _ = xdb.put(xarray)
    return _id

def get_weights_matrix_from_mongo(ophys_experiment_id, glm_version):
    '''
    retrieves weights matrix from mongo for a given oeid/glm_version
    throws warning and returns None if no matrix can be found
    '''
    conn = db.Database('visual_behavior_data')
    lookup_table_document = {
        'ophys_experiment_id':ophys_experiment_id,
        'glm_version':glm_version,
    }
    w_matrix_lookup_table = conn['ophys_glm']['weight_matrix_lookup_table']
    w_matrix_database = conn['ophys_glm_xarrays']

    if w_matrix_lookup_table.count_documents(lookup_table_document) == 0:
        warnings.warn('there is no record of a the weights matrix for oeid {}, glm_version {}'.format(ophys_experiment_id, glm_version))
        conn.close()
        return None
    else:
        lookup_result = list(w_matrix_lookup_table.find(lookup_table_document))[0]
        # get the id of the xarray
        w_matrix_id = lookup_result['w_matrix_id']
        xdb = xarray_mongodb.XarrayMongoDB(w_matrix_database)
        W = xdb.get(w_matrix_id)
        conn.close()
        return W


def log_weights_matrix_to_mongo(glm):
    '''
    a method for logging the weights matrix to mongo
    uses the xarray_mongodb library, which automatically distributes the xarray into chunks
    this necessitates building/maintaining a lookup table to link experiments/glm_verisons to the associated xarrays

    input:
        GLM object
    returns:
        None
    '''

    conn = db.Database('visual_behavior_data')
    lookup_table_document = {
        'ophys_experiment_id':int(glm.ophys_experiment_id),
        'glm_version':glm.version,
    }
    w_matrix_lookup_table = conn['ophys_glm']['weight_matrix_lookup_table']
    w_matrix_database = conn['ophys_glm_xarrays']

    if w_matrix_lookup_table.count_documents(lookup_table_document) >= 1:
        # if weights matrix for this experiment/version has already been logged, we need to replace it
        lookup_result = list(w_matrix_lookup_table.find(lookup_table_document))[0]

        # get the id of the xarray
        w_matrix_id = lookup_result['w_matrix_id']

        # delete the existing xarray (both metadata and chunks)
        w_matrix_database['xarray.chunks'].delete_many({'meta_id':w_matrix_id})

        w_matrix_database['xarray.meta'].delete_many({'_id':w_matrix_id})

        # write the new weights matrix to mongo
        new_w_matrix_id = xarray_to_mongo(glm.W)

        # update the lookup table entry
        lookup_result['w_matrix_id'] = new_w_matrix_id
        _id = lookup_result.pop('_id')
        w_matrix_lookup_table.update_one({'_id':_id}, {"$set": db.clean_and_timestamp(lookup_result)})
    else:
        # if the weights matrix had not already been logged

        # write the weights matrix to mongo
        w_matrix_id = xarray_to_mongo(glm.W)
        
        # add the id to the lookup table document
        lookup_table_document.update({'w_matrix_id': w_matrix_id})
        
        # insert the lookup table document into the lookup table
        w_matrix_lookup_table.insert_one(db.clean_and_timestamp(lookup_table_document))

    conn.close()

def get_experiment_table(glm_version, include_4x2_data=False): 
    '''
    gets the experiment table
    appends the following:
        * roi count
        * cluster job summary for each experiment
        * number of existing dropouts
    
    Warning: this takes a couple of minutes to run.
    '''
    experiment_table = loading.get_platform_paper_experiment_table(include_4x2_data=include_4x2_data).reset_index() 
    dropout_summary = retrieve_results({'glm_version':glm_version}, results_type='summary')
    stdout_summary = get_stdout_summary(glm_version)

    # add ROI count to experiment table
    experiment_table['roi_count'] = experiment_table['ophys_experiment_id'].map(lambda oeid: get_roi_count(oeid))

    # get a count of the dropoutsof for each experiment/cell
    dropout_count = pd.DataFrame(
        (dropout_summary
            .groupby(['ophys_experiment_id','cell_specimen_id'])['dropout']
            .count())
            .reset_index()
            .rename(columns={'dropout': 'dropout_count'}
        )
    )

    # merge in stdout summary
    experiment_table_merged = experiment_table.merge(
        stdout_summary,
        left_on = 'ophys_experiment_id',
        right_on = 'ophys_experiment_id',
        how='left'
    )
    # merge in dropout count (average dropout count per experiment - should be same for all cells)
    experiment_table_merged = experiment_table_merged.merge(
        pd.DataFrame(dropout_count.groupby('ophys_experiment_id')['dropout_count'].mean()).reset_index(),
        left_on = 'ophys_experiment_id',
        right_on = 'ophys_experiment_id',
        how='left'
    )

    return experiment_table_merged
    

def get_stdout_summary(glm_version):
    '''
    retrieves statistics about a given model run from mongo
    '''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_glm']['cluster_stdout']
    stdout_summary = pd.DataFrame(list(collection.find({'glm_version':glm_version})))
    conn.close()

    # parse the walltime column
    stdout_summary['required_walltime_seconds'] = stdout_summary['required_walltime'].map(lambda walltime_str: walltime_to_seconds(walltime_str))
    stdout_summary['required_walltime_minutes'] = stdout_summary['required_walltime'].map(lambda walltime_str: walltime_to_seconds(walltime_str)/60)
    stdout_summary['required_walltime_hours'] = stdout_summary['required_walltime'].map(lambda walltime_str: walltime_to_seconds(walltime_str)/3600)

    return stdout_summary

def walltime_to_seconds(walltime_str):
    '''
    converts the walltime string from stdout summary to seconds (int)
    string is assumed to be of format HH:MM:SS
    '''
    h, m, s = walltime_str.split(':')
    return int(h)*60*60 + int(m)*60 + int(s)

def get_roi_count(ophys_experiment_id):
    '''
    a LIMS query to get the valid ROI count for a given experiment
    '''
    query= 'select * from cell_rois where ophys_experiment_id = {}'.format(ophys_experiment_id)
    df = db.lims_query(query)
    return df['valid_roi'].sum()

def retrieve_results(search_dict={}, results_type='full', return_list=None, merge_in_experiment_metadata=True,remove_invalid_rois=True,verbose=False,allow_old_rois=True,invalid_only=False,add_extra_columns=False):
    '''
    gets cached results from mongodb
    input:
        search_dict - dictionary of key/value pairs to use for searching, if empty (default), will return entire database table
        results_type - 'full' or 'summary' (default = 'full')
            * full: 1 row for every unique cell/session (cells that are matched across sessions will have one row for each session.
                Each row contains all of the coefficients of variation (a test and a train value for each dropout)
            * summary: results_summary contains 1 row for every unique cell/session/dropout 
                cells that are matched across sessions will have `N_DROPOUTS` rows for each session.
                Each row contains a `dropout` label describing the particular dropout coefficent(s) that apply to that row. 
                All derived values (`variance_explained`, `fraction_change_from_full`, `absolute_change_from_full`) 
                are calculated only on test data, not train data.
        return_list - a list of columns to return. Returning fewer columns speeds queries
        merge_in_experiment_metadata - boolan which, if True, merges in data from experiment table
        remove_invalid_rois - bool
            if True, removes invalid rois from the returned results
            if False, includes the invalid rois in the returned results
    output:
        dataframe of results
    '''
    assert not (invalid_only & remove_invalid_rois), "Cannot remove invalid rois and only return invalid rois"
    
    if return_list is None:
        return_dict = {'_id': 0}
    else:
        return_dict = {v: 1 for v in return_list}
        if '_id' not in return_list:
            # don't return `_id` unless it was specifically requested
            return_dict.update({'_id': 0})

    if verbose:
        print('Pulling from Mongo')
    conn = db.Database('visual_behavior_data')
    database = 'ophys_glm'
    results = pd.DataFrame(list(conn[database]['results_{}'.format(results_type)].find(search_dict, return_dict)))

    if verbose:
        print('Done Pulling')
    # make 'glm_version' column a string
    if 'glm_version' in results.columns:
        results['glm_version'] = results['glm_version'].astype(str)
    conn.close()
   
    include_4x2_data=False
    if 'glm_version' in search_dict: 
        run_params = glm_params.load_run_json(search_dict['glm_version'])
        include_4x2_data = run_params['include_4x2_data']

    if len(results) > 0 and merge_in_experiment_metadata:
        if verbose:
            print('Merging in experiment metadata')
        # get experiment table, merge in details of each experiment
        experiment_table = loading.get_platform_paper_experiment_table(add_extra_columns=add_extra_columns, include_4x2_data=include_4x2_data).reset_index() 
        results = results.merge(
            experiment_table, 
            left_on='ophys_experiment_id',
            right_on='ophys_experiment_id', 
            how='left',
            suffixes=['', '_duplicated'],
        )
        duplicated_cols = [col for col in results.columns if col.endswith('_duplicated')]
        results = results.drop(columns=duplicated_cols)
    
    if remove_invalid_rois:
        # get list of rois I like
        if verbose:
            print('Loading cell table to remove invalid rois')
        if 'cell_roi_id' in results:
            cell_table = loading.get_cell_table(platform_paper_only=True,add_extra_columns=False,include_4x2_data=include_4x2_data).reset_index() 
            good_cell_roi_ids = cell_table.cell_roi_id.unique()
            results = results.query('cell_roi_id in @good_cell_roi_ids')
        elif allow_old_rois:
            print('WARNING, cell_roi_id not found in database, I cannot filter for old rois. The returned results could be out of date, or QC failed')
        else:
            raise Exception('cell_roi_id not in database, and allow_old_rois=False')
    elif invalid_only:
        if verbose:
            print('Loading cell table to remove valid rois')
        if 'cell_roi_id' in results:
            cell_table = loading.get_cell_table(platform_paper_only=True,add_extra_columns=False,include_4x2_data=include_4x2_data).reset_index() 
            good_cell_roi_ids = cell_table.cell_roi_id.unique()
            results = results.query('cell_roi_id not in @good_cell_roi_ids')
        elif allow_old_rois:
            print('WARNING, cell_roi_id not found in database, I cannot filter for old rois. The returned results could be out of date, or QC failed')
        else:
            raise Exception('cell_roi_id not in database, and allow_old_rois=False') 
    
    if ('variance_explained' in results) and (np.sum(results['variance_explained'].isnull()) > 0):
        print('Warning! Dropout models with NaN variance explained. This shouldn\'t happen')
    elif ('Full__avg_cv_var_test' in results) and (np.sum(results['Full__avg_cv_var_test'].isnull()) > 0):
        print('Warning! Dropout models with NaN variance explained. This shouldn\'t happen')

 
    return results

def make_identifier(row):
    return '{}_{}'.format(row['ophys_experiment_id'],row['cell_specimen_id'])

def get_glm_version_summary(versions_to_compare=None,vrange=[15,20], compact=True,invalid_only=False,remove_invalid_rois=True,save_results=True,additional_columns=[]):
    '''
        Builds a results summary table for comparing variance explained and dropout scores across model versions.
        results_compact = gat.get_glm_version_summary(versions_to_compare)
        gvt.compare_var_explained_by_version(results_compact)
    '''
    if versions_to_compare is None:
        versions_to_compare = glm_params.get_versions(vrange)
        #versions_to_compare = [x[2:] for x in versions_to_compare if 'old' not in x]
        versions_to_compare = [x[2:] for x in versions_to_compare]
    if compact:
        dropouts = ['Full','visual','all-images','omissions','behavioral','task']
        return_list = np.concatenate([[x+'__avg_cv_var_test',x+'__avg_cv_var_train'] for x in dropouts])
        return_list = np.concatenate([return_list, ['ophys_experiment_id','cell_roi_id','cre_line','glm_version']])
        return_list = np.concatenate([return_list, additional_columns])
    else:
        return_list = None
    results_list = []
    print('Iterating model versions')
    for glm_version in versions_to_compare:
        print(glm_version)
        results_list.append(retrieve_results({'glm_version': glm_version},return_list=return_list, results_type='full',invalid_only=invalid_only,remove_invalid_rois=remove_invalid_rois))
    results = pd.concat(results_list, sort=True)

    # Display summary statistics
    summary_table = results.groupby(['cre_line','glm_version'])['Full__avg_cv_var_test'].mean()
    print('')
    print(summary_table)

    # Save Summary Table
    if save_results:
       summary_table.to_csv('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/version_comparisons/summary_table.csv',header=True) 

    return results

def get_glm_version_comparison_table(versions_to_compare, results=None, metric='Full__avg_cv_var_test'):
    '''
    builds a table that allows to glm versions to be directly compared
    input is list of glm versions to compare (list of strings)
    if results dataframe is not passed, it will be queried from Mongo
    
    Returns two items
        comparison_table, where the <metric> for each version is a column, and rows are cells
        results, where each row is a (cell x version)
    '''
    if results is None:
        results_list = []
        for glm_version in versions_to_compare:
            print(glm_version)
            results_list.append(retrieve_results({'glm_version': glm_version}, results_type='full'))
        results = pd.concat(results_list, sort=True)

    results['identifier'] = results.apply(make_identifier, axis=1)
    pivoted_results = results.pivot(index='identifier', columns='glm_version',values=metric)
    cols= [col for col in results.columns if col not in pivoted_results.columns and 'test' not in col and 'train' not in col and '__' not in col and 'dropout' not in col]

    pivoted_results = pivoted_results.merge(
        results[cols].drop_duplicates(subset=['identifier']),
        left_on='identifier',
        right_on='identifier',
        how='left'
    )

    return pivoted_results,pd.concat(results_list, sort=True)

def build_pivoted_results_summary(value_to_use, results_summary=None, glm_version=None, cutoff=None,add_extra_columns=False):
    '''
    pivots the results_summary dataframe to give a dataframe with dropout scores as unique columns
    inputs:
        results_summary: dataframe of results_summary. If none, will be pulled from mongo
        glm_version: glm_version to pull from database (only if results_summary is None)
        cutoff: cutoff for CV score on full model. Cells with CV score less than this value will be excluded from the output dataframe
        value_to_use: which column to use as the value in the pivot table (e.g. 'fraction_change_from_full')
    output:
        wide form results summary
    '''
    
    # some aassertions to make sure the right combination of stuff is input
    assert results_summary is not None or glm_version is not None, 'must pass either a results_summary or a glm_version'
    assert not (results_summary is not None and glm_version is not None), 'cannot pass both a results summary and a glm_version'
    if results_summary is not None:
        assert len(results_summary['glm_version'].unique()) == 1, 'number of glm_versions in the results summary cannot exceed 1'
        
    # get results summary if none was passed
    if results_summary is None:
        results_summary = retrieve_results(search_dict = {'glm_version': glm_version}, results_type='summary',add_extra_columns=add_extra_columns)

    results_summary['identifier'] = results_summary['ophys_experiment_id'].astype(str) + '_' +  results_summary['cell_specimen_id'].astype(str)
 
    # apply cutoff. Set to -inf if not specified
    if cutoff is None:
        cutoff = -np.inf
    cells_to_keep = list(results_summary.query('dropout == "Full" and variance_explained >= @cutoff')['identifier'].unique())
 
    # pivot the results summary so that dropout scores become columns
    results_summary_pivoted = results_summary.query('identifier in @cells_to_keep').pivot(index='identifier',columns='dropout',values=value_to_use).reset_index()

    # merge in other identifying columns, leaving out those that will have more than one unique value per cell
    potential_cols_to_drop = [
        '_id', 
        'index',
        'dropout',
        'absolute_change_from_full',
        'avg_L2_regularization',
        'avg_cv_var_test_full_comparison_raw',
        'avg_cv_var_test_raw',
        'avg_cv_var_train_raw',
        'avg_cv_var_test_sem', 
        'cell_L2_regularization',
        'fraction_change_from_full', 
        'adj_dropout_raw',
        'variance_explained',
        'adj_fraction_change_from_full',
        'avg_cv_adjvar_test_raw',
        'avg_cv_adjvar_test_full_comparison_raw',
        'adj_variance_explained',
        'adj_variance_explained_full',
        'entry_time_utc',
        'driver_line',
        'shuffle_time',
        'shuffle_cells',
    ]
    cols_to_drop = [col for col in potential_cols_to_drop if col in results_summary.columns]
    results_summary_pivoted = results_summary_pivoted.merge(
        results_summary.drop(columns=cols_to_drop).drop_duplicates(),
        left_on='identifier',
        right_on='identifier',
        how='left'
    )
    #if 'avg_cv_var_test_sem' in results_summary.columns:
    #    results_summary_pivoted = results_summary_pivoted.merge(
    #        results_summary.query('dropout == "Full" and variance_explained >=@cutoff')[['identifier','avg_cv_var_test_sem']],
    #        left_on='identifier',
    #        right_on='identifier',
    #        how='left'
    #    )
    #    results_summary_pivoted = results_summary_pivoted.rename(columns={'avg_cv_var_test_sem':'variance_explained_full_sem'})
 
    return results_summary_pivoted


def summarize_variance_explained(results=None):
    '''
    return results summary grouped by version and cre-line
    '''
    if results is None:
        results_dict = retrieve_results()
        results = results_dict['full']
    return results.groupby(['glm_version','cre_line'])['Full_avg_cv_var_test'].describe()


def get_experiment_inventory(results=None):
    '''
    adds a column to the experiments table for every GLM version called 'glm_version_{GLM_VERSION}_exists'
    column is boolean (True if experiment successfully fit for that version, False otherwise)
    '''
    raise Exception('Outdated, do not use')
    def oeid_in_results(oeid, version):
        try:
            res = results['full'].loc[oeid]['glm_version']
            if isinstance(res, str):
                return version == res
            else:
                return version in res.unique()
        except KeyError:
            return False

    if results is None:
        results_dict = retrieve_results()
        results = results_dict['full']
    results = results.set_index(['ophys_experiment_id'])
    
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=False,include_4x2_data=False)

    for glm_version in results['glm_version'].unique():
        for oeid in experiments_table.index.values:
            experiments_table.at[oeid, 'glm_version_{}_exists'.format(glm_version)] = oeid_in_results(oeid, glm_version)

    return experiments_table

def run_pca(dropout_matrix, n_components=40, deal_with_nans='fill_with_zero'):
    '''
    wrapper function for PCA
    inputs:
        dropout_matrix: matrix on which to perform PCA
        n_components: desired PCA components
        deal_with_nans: 'fill_with_zero' fills with zeros. 'drop' drops.
    returns
        pca object with fit performed, pca_result_matrix

    '''
    pca = PCA(n_components=n_components)
    if deal_with_nans == 'fill_with_zero':
        pca_result = pca.fit_transform(dropout_matrix.fillna(0).values)
    elif deal_with_nans == 'drop':
        pca_result = pca.fit_transform(dropout_matrix.dropna().values)
    pca.results = pca_result
    pca.component_names = dropout_matrix.columns
    return pca
    

def process_session_to_df(oeid, run_params):
    '''
        For the ophys_experiment_id, loads the weight matrix, and builds a dataframe
        organized by cell_id and kernel 
    '''
    # Get weights
    W = get_weights_matrix_from_mongo(int(oeid), run_params['version'])
    
    # Make Dataframe with cell and experiment info
    session_df  = pd.DataFrame()
    session_df['cell_specimen_id'] = W.cell_specimen_id.values
    session_df['ophys_experiment_id'] = [int(oeid)]*len(W.cell_specimen_id.values)  
    
    # For each kernel, extract the weights for this kernel
    for k in run_params['kernels']:
        weight_names = [w for w in W.weights.values if w.startswith(k)]
        
        # Check if this kernel was in this model
        if len(weight_names) > 0:
            session_df[k] = W.loc[dict(weights=weight_names)].values.T.tolist()
    return session_df

def build_weights_df(run_params,results_pivoted, cache_results=False,load_cache=False,normalize=False):
    '''
        Builds a dataframe of (cell_specimen_id, ophys_experiment_id) with the weight parameters for each kernel
        Some columns may have NaN if that cell did not have a kernel, for example if a missing datastream  
        
        Takes about 5 minutes to run 
 
        INPUTS:
        run_params, parameter json for the version to analyze
        results_pivoted = build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
        cache_results, if True, save dataframe as csv file
        load_cache, if True, load cached results, if it exists
    
        RETURNS:
        a dataframe
    '''
   
    # Make dataframe for cells and experiments 
    oeids = results_pivoted['ophys_experiment_id'].unique() 
    if len(oeids) == 0:
        return None

    # For each experiment, get the weight matrix from mongo (slow)
    # Then pull the weights from each kernel into a dataframe
    sessions = []
    for index, oeid in enumerate(tqdm(oeids, desc='Iterating Sessions')):
        session_df = process_session_to_df(oeid, run_params)
        sessions.append(session_df)

    # Merge all the session_dfs, and add more session level info
    weights_df = pd.concat(sessions,sort=False)
    weights_df = pd.merge(weights_df,results_pivoted, on = ['cell_specimen_id','ophys_experiment_id'],suffixes=('_weights','')) 
   
    # If we didn't compute dropout scores, then there won't be redundant columns, so the weights won't get appended with _weights
    if not np.any(['weights' in x for x in weights_df.columns.values]):
        rename = {x: x+'_weights' for x in run_params['kernels'].keys()}
        weights_df = weights_df.rename(columns=rename)   
 
    # Interpolate everything onto common time base
    kernels = [x for x in weights_df.columns if 'weights' in x]
    for kernel in tqdm(kernels, desc='Interpolating kernels'):
        weights_df = interpolate_kernels(weights_df, run_params, kernel,normalize=normalize)
 
    print('Computing average kernels') 
    # Compute generic image kernel
    weights_df['all-images_weights'] = weights_df.apply(lambda x: np.mean([
        x['image0_weights'],
        x['image1_weights'],
        x['image2_weights'],
        x['image3_weights'],       
        x['image4_weights'],
        x['image5_weights'],
        x['image6_weights'],
        x['image7_weights']
        ],axis=0),axis=1)

    # Compute preferred image kernel
    weights_df['preferred_image_weights'] = weights_df.apply(lambda x: compute_preferred_kernel([
        x['image0_weights'],
        x['image1_weights'],
        x['image2_weights'],
        x['image3_weights'],       
        x['image4_weights'],
        x['image5_weights'],
        x['image6_weights'],
        x['image7_weights']
        ]),axis=1) 

    # make a combined omissions kernel
    if 'post-omissions_weights' in weights_df:
        weights_df['all-omissions_weights'] = weights_df.apply(lambda x: compute_all_omissions([
        x['omissions_weights'],
        x['post-omissions_weights']
        ]),axis=1)
    
    if 'post-hits_weights' in weights_df:
        weights_df['all-hits_weights'] = weights_df.apply(lambda x: compute_all_kernels([
        x['hits_weights'],
        x['post-hits_weights']
        ]),axis=1)       
        weights_df['all-misses_weights'] = weights_df.apply(lambda x: compute_all_kernels([
        x['misses_weights'],
        x['post-misses_weights']
        ]),axis=1)
        weights_df['all-passive_change_weights'] = weights_df.apply(lambda x: compute_all_kernels([
        x['passive_change_weights'],
        x['post-passive_change_weights']
        ]),axis=1)
        # Make a combined change kernel
        weights_df['task_weights'] = weights_df.apply(lambda x: np.mean([
            x['all-hits_weights'],
            x['all-misses_weights'],
            ],axis=0),axis=1)

    # Make a combined change kernel
    weights_df['task_weights'] = weights_df.apply(lambda x: np.mean([
        x['hits_weights'],
        x['misses_weights'],
        ],axis=0),axis=1)

    # Make a metric of omission excitation/inhibition
    weights_df['omissions_excited'] = weights_df.apply(lambda x: kernel_excitation(x['omissions_weights']),axis=1)
    weights_df['hits_excited']      = weights_df.apply(lambda x: kernel_excitation(x['hits_weights']),axis=1)
    weights_df['misses_excited']    = weights_df.apply(lambda x: kernel_excitation(x['misses_weights']),axis=1)
    weights_df['task_excited']      = weights_df.apply(lambda x: kernel_excitation(x['task_weights']),axis=1)
    weights_df['all-images_excited']= weights_df.apply(lambda x: kernel_excitation(x['all-images_weights']),axis=1)

    # Return weights_df
    return weights_df 

def kernel_excitation(kernel):
    if np.isnan(np.sum(kernel)):
        return np.nan
    else:
        return np.sum(kernel[0:24]) > 0

def compute_all_omissions(omissions):
    if np.isnan(np.sum(omissions[0])) or np.isnan(np.sum(omissions[1])):
        return np.nan
    
    return np.concatenate(omissions)

def compute_all_kernels(kernels):
    if np.isnan(np.sum(kernels[0])) or np.isnan(np.sum(kernels[1])):
        return np.nan
    
    return np.concatenate(kernels)

def compute_preferred_kernel(images):
    
    # If all the weight kernels are nans
    if np.ndim(images) ==1:
        return images[0]
    
    # Find the kernel with the largest magnitude 
    weight_amplitudes = np.sum(np.abs(images),axis=1)
    return images[np.argmax(weight_amplitudes)] 

def interpolate_kernels(weights_df, run_params, kernel_name,normalize=False):
    '''
        Interpolates all kernels onto the scientific time base
        If the kernels are the wrong size for either scientifica or mesoscope, it sets them to NaN
    '''   
 
    # Select the kernel of interest
    df = weights_df[kernel_name].copy()  
    kernel = kernel_name.split('_weights')[0]

    # Normalize each to its max value, if we are doing normalization
    if normalize:
        df = [x/np.max(np.abs(x)) for x in df.values]

    # Interpolate
    time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/31)
    time_vec = np.round(time_vec,2)
    meso_time_vec = np.arange(run_params['kernels'][kernel]['offset'], run_params['kernels'][kernel]['offset'] + run_params['kernels'][kernel]['length'],1/11)#1/10.725)
    time_vecs = {}
    time_vecs['scientifica'] = time_vec
    time_vecs['mesoscope'] = meso_time_vec
    length_mismatch = len(time_vec) != np.max(np.unique([np.size(x) for x in df]))
    if ('image' in kernel_name) & length_mismatch:
        time_vecs['scientifica'] = time_vecs['scientifica'][0:-1]
        time_vecs['mesoscope'] = time_vecs['mesoscope'][0:-1]
    if ('omissions' in kernel_name) & length_mismatch:
        time_vecs['scientifica'] = time_vecs['scientifica'][0:-1]
        time_vecs['mesoscope'] = time_vecs['mesoscope'][0:-1]
    weights_df[kernel_name] = [weight_interpolation(x, time_vecs) for x in df]
    return weights_df

def weight_interpolation(weight_vec, time_vecs={}):
    if np.size(weight_vec) ==1:
        return weight_vec
    elif len(weight_vec) == len(time_vecs['scientifica']):
        return weight_vec
    elif len(weight_vec) == len(time_vecs['mesoscope']):
        return scipy.interpolate.interp1d(time_vecs['mesoscope'], weight_vec, fill_value='extrapolate',bounds_error=False)(time_vecs['scientifica'])
    else:
        return np.nan 

def compute_weight_index(weights_df):
    '''
        Appends columns to the weight_df. One column for each kernel

        The weight index is just the sum(abs(kernel))

        Additionally creates the sume of the individual image kernels
    '''
    kernels = [x for x in weights_df.keys().values if '_weights' in x]
    for k in kernels:
        weights_df[k+'_index'] = [np.sum(np.abs(x)) for x in weights_df[k].values]
    
    weights_df['all-images_weights_index'] = weights_df['image0_weights_index'] + weights_df['image1_weights_index'] + weights_df['image2_weights_index'] + weights_df['image3_weights_index'] + weights_df['image4_weights_index'] +weights_df['image5_weights_index'] +weights_df['image6_weights_index'] +weights_df['image7_weights_index']
    return weights_df

def append_kernel_excitation(weights_df, results_pivoted):
    '''
        Appends labels about kernel weights from weights_df onto results_pivoted
        for some kernels, cells are labeled "excited" or "inhibited" if the average weight over 750ms after
        the aligning event was positive (excited), or negative (inhibited)

        Additionally computes three coding scores for each kernel:
        kernel_positive is the original coding score if the kernel was excited, otherwise 0
        kernel_negative is the original coding score if the kernel was inhibited, otherwise 0
        kernel_signed is kernel_positive - kernel_negative
       
    '''   
 
    results_pivoted = pd.merge(
        results_pivoted,
        weights_df[['identifier','omissions_excited','hits_excited','misses_excited','all-images_excited','task_excited']],
        how = 'inner',
        on = 'identifier',
        validate='one_to_one'
        )
 
    excited_kernels = ['omissions','hits','misses','task','all-images']
    for kernel in excited_kernels:
        results_pivoted[kernel+'_positive'] = results_pivoted[kernel]
        results_pivoted[kernel+'_negative'] = results_pivoted[kernel]
        results_pivoted.loc[results_pivoted[kernel+'_excited'] != True, kernel+'_positive'] = 0
        results_pivoted.loc[results_pivoted[kernel+'_excited'] != False,kernel+'_negative'] = 0   
        results_pivoted[kernel+'_signed'] = results_pivoted[kernel+'_positive'] - results_pivoted[kernel+'_negative']

    return results_pivoted
        

def compute_over_fitting_proportion(results_full,run_params):
    '''
        Computes the over-fitting proportion for each cell on each dropout model:
        (train_ve - test_ve)/train_ve
        1 = completely overfit
        0 = no over-fitting

        Also computes the over-fitting proportion attributable to each dropout:
        1-dropout_over_fit/full_over_fit
        1 = This dropout was responsible for all the overfitting in the full model
        0 = This dropout was responsible for none of the overfitting in the full model

    '''
    dropouts = set(run_params['dropouts'].keys())
    for d in dropouts:
        if d+'__avg_cv_var_train' in results_full.columns:
            results_full[d+'__over_fit'] = (results_full[d+'__avg_cv_var_train']-results_full[d+'__avg_cv_var_test'])/(results_full[d+'__avg_cv_var_train'])
    
    dropouts.remove('Full')
    for d in dropouts:
        if d+'__avg_cv_var_train' in results_full.columns:
            results_full[d+'__dropout_overfit_proportion'] = 1-results_full[d+'__over_fit']/results_full['Full__over_fit']
    return


def find_best_session(results_pivoted, session_number, mouse_id=None, novelty=False):
    '''
        If there are multiple retakes of the same ophys session type, picks one with most 
        registered neurons.
        If novelty is True, picks ophys session with prior exposure to session type = 0
        Returns one ophys session id if there is one, returns None if there is none that meet
        novelty criteria.

        INPUT:
        results_pivoted     glm output with each regressor as a column
        mouse_id            pick one mouse id at a time
        session_number      pick one session type at a time (1,2...6)
        novelty             default = False, if set to True = not a retake

        RETURNS:
        session_number      ophys session number if one is found, None otherwise

    '''
    if mouse_id is not None:  # get glm from one mouse
        df = results_pivoted[(results_pivoted['mouse_id'] == mouse_id) &
                             (results_pivoted['session_number'] == session_number)]
    else:
        df = results_pivoted[results_pivoted['session_number']
                             == session_number]

    sessions = df['ophys_session_id'].unique()
    #print('found {} session(s)...'.format(len(sessions)))


    if len(sessions) == 1 and novelty == False:  # one session
        session_to_use = sessions[0]

    elif not list(sessions):  # no sessions
        session_to_use = None

    elif novelty == True:  # novel session
        try:
            session_to_use = df[df['prior_exposures_to_session_type'] == 0]['ophys_session_id'].unique()[0]
        except:
            print('no novel session, id = {}...'.format(df['ophys_session_id'].unique()))
            session_to_use = None

    else:  # go through sessions and find the one with most registered neurons
        n_csids = 0  # number of cell specimen ids

        for session in sessions:
            n_csid = len(df[df['ophys_session_id'] == session]
                         ['cell_specimen_id'])

            if n_csid > n_csids:
                n_csids = n_csid
                session_to_use = session

    return session_to_use


def get_matched_cell_ids_across_sessions(results_pivoted_sel, session_numbers, novelty=None):
    '''
        Finds cells with the same cell ids across sessions
        INPUT:
        results_pivoted_sel     results_pivoted dataframe without retakes with cell_specimen_id,
                                session_number, mouse_id, and ophys_session_id as columns
        session_numbers         session numbers to compare 
        novelty                 default None, if there are retakes, assumes novelty = True for ophys 4.
                                Set to False if novelty of ophys 4 is not a priority

        RETURNS:
        matched_cell_ids        an array of cell specimen ids matched across sessions
        ophys_session_ids       an array of ophys_session_ids, where the cell ids came from

    '''

    # check for retakes first. You cannot match cells if there are more than one of the same session type.
    ophys_session_ids = []
    tmp = results_pivoted_sel[['mouse_id', 'session_number']].drop_duplicates()
    session_N = tmp.groupby(['mouse_id', 'session_number'])['session_number'].value_counts()

    if session_N.unique() != [1]:

        print('glm output contains retakes; cant match cells')
        matched_cell_ids = None
    else:

        # start with all cell ids
        matched_cell_ids = results_pivoted_sel['cell_specimen_id'].unique()

        for session_number in session_numbers:
            df = results_pivoted_sel[results_pivoted_sel['session_number'] == session_number]
            matched_cell_ids = np.intersect1d(matched_cell_ids, df['cell_specimen_id'].values)
            try:
                ophys_session_ids.append(df['ophys_session_id'].unique()[0])
            except:
                print('no matches')

    return matched_cell_ids, ophys_session_ids


def drop_cells_with_nan(results_pivoted, regressor):
    '''
        Find cells that have NaN dropout scores in either one or more ophys sessions
        and drop them in all ophys sessions. Returns glm df without those cells.

        INPUT:
        results_pivoted    glm output with regressors as columns
        regressor          name of the regressor

        RETURNS:
        results_pivoted_without_nan 
    '''
    cell_with_nan = results_pivoted[results_pivoted[regressor].isnull()]['cell_specimen_id'].values
    results_pivoted_without_nan = results_pivoted[~results_pivoted['cell_specimen_id'].isin(cell_with_nan)]
    return results_pivoted_without_nan


def get_matched_mouse_ids(results_pivoted, session_numbers):
    '''
        Find mouse ids that have matched ophys sessions.

        INPUT:
        results_pivoted     glm output with regressors as columns
        ression_numbers     session numbers to match

        RETURNS:
        mouse_ids           an array with mouse ids that have all listed session numbers
    '''

    mouse_ids = results_pivoted['mouse_id'].unique()
    for session_number in session_numbers:
        mouse_id = results_pivoted[results_pivoted['session_number']
                                   == session_number]['mouse_id'].unique()
        mouse_ids = np.intersect1d(mouse_ids, mouse_id)
    return mouse_ids


def clean_glm_dropout_scores(results_pivoted, run_params, in_session_numbers=None): 
    '''
        Selects only neurons what are explained above threshold var. 
        In_session_numbers allows you specify with sessions to check. 

        INPUT: 
        results_pivoted           glm output witt session_number and variance_explained_full as columns
        in_session_numbers        an array of session number(s) to check. 

        RETURNS:
        results_pivoted_var glm output with cells above threshold of var explained, unmatched cells
    '''
    if 'dropout_threshold' in run_params:
        threshold = run_params['dropout_threshold']
    else:
        threshold = 0.005

    good_cell_ids = results_pivoted[results_pivoted['variance_explained_full']
                       > threshold]['cell_specimen_id'].unique()

    if in_session_numbers is not None:
        for session_number in in_session_numbers:
            cell_ids = results_pivoted[(results_pivoted['session_number'] == session_number) &
                                       (results_pivoted['variance_explained_full'] > threshold)]['cell_specimen_id'].unique()
            good_cell_ids = np.intersect1d(good_cell_ids, cell_ids)
    else:
        good_cell_ids = results_pivoted[results_pivoted['variance_explained_full']
                           > threshold]['cell_specimen_id'].unique()

    results_pivoted_var = results_pivoted[results_pivoted['cell_specimen_id'].isin(
        good_cell_ids)].copy()

    return results_pivoted_var
          
def build_inventory_table(vrange=[18,20],return_inventories=False):
    '''
        Builds a table of all available GLM versions in the supplied range, and reports how many missing/fit experiments/rois in that version
        
        Optionally returns the list of missing experiments and rois
    '''
    versions = glm_params.get_versions(vrange=vrange)
    inventories ={}
    for v in versions:
        inventories[v]=inventory_glm_version(v[2:])
    if return_inventories:
        return inventories_to_table(inventories),inventories    
    else:
        return inventories_to_table(inventories)

def inventories_to_table(inventories):
    '''
        Helper function that takes a dictionary of version inventories and build a summary table
    '''
    summary = inventories.copy()
    for version in summary:
        for value in summary[version]:
            summary[version][value] = len(summary[version][value])
        summary[version]['Complete'] = (summary[version]['missing_experiments'] == 0 ) & (summary[version]['missing_rois'] == 0)
        #summary[version]['Total Experiments'] = summary[version]['fit_experiments'] + summary[version]['extra_experiments']
        #summary[version]['Total ROIs'] = summary[version]['fit_rois'] + summary[version]['extra_rois']
    table = pd.DataFrame.from_dict(summary,orient='index')
    if np.all(table['incomplete_experiments'] == 0):
        table = table.drop(columns=['incomplete_experiments', 'additional_missing_cells'])
    return table

def inventory_glm_version(glm_version, valid_rois_only=True, platform_paper_only=True):
    '''
    checks to see which experiments and cell_roi_ids do not yet exist for a given GLM version
    inputs:
        glm_version: string
        platform_paper_only: bool, if True, only count cells in the platform paper dataset 
    returns: dict
        {
            'missing_experiments': a list of missing experiment IDs
            'missing_rois': a list of missing cell_roi_ids
            'incomplete_experiments': a list of experiments which exist, but for which the cell_roi_id list is incomplete
        }
    '''
    # Get GLM results
    glm_results = retrieve_results(
        search_dict = {'glm_version': glm_version},
        return_list = ['ophys_experiment_id', 'cell_specimen_id', 'cell_roi_id'],
        merge_in_experiment_metadata=False,
        remove_invalid_rois=False
    )
    if len(glm_results) == 0:
        # Check for empty results
        glm_results['ophys_experiment_id'] = [] 
        glm_results['cell_specimen_id'] = [] 
        glm_results['cell_roi_id'] = [] 

    # determine if we need to get 4x2 data for this version
    include_4x2_data=False
    run_params = glm_params.load_run_json(glm_version)
    include_4x2_data = run_params['include_4x2_data']
 
    # Get list of cells in the dataset
    cell_table = loading.get_cell_table(platform_paper_only=platform_paper_only,add_extra_columns=False,include_4x2_data=include_4x2_data).reset_index()

    # get list of rois and experiments we have fit
    total_experiments = glm_results['ophys_experiment_id'].unique()
    total_rois = glm_results['cell_roi_id'].unique()

    # Compute list of rois and experiments that we have fit that are in the dataset
    fit_experiments = list(
        set(cell_table['ophys_experiment_id'].unique()) &
        set(glm_results['ophys_experiment_id'].unique())
    )
    fit_rois = list(
        set(cell_table['cell_roi_id'].unique()) &
        set(glm_results['cell_roi_id'].unique())
    )

    # get list of missing experiments
    missing_experiments = list(
        set(cell_table['ophys_experiment_id'].unique()) - 
        set(glm_results['ophys_experiment_id'].unique())
    )

    # get list of missing rois
    missing_rois = list(
        set(cell_table['cell_roi_id'].unique()) - 
        set(glm_results['cell_roi_id'].unique())
    )

    # Extra experiments, these could be old experiments that have since been failed, or out of scope experiments
    extra_experiments = list(
        set(glm_results['ophys_experiment_id'].unique()) - 
        set(cell_table['ophys_experiment_id'].unique())
    )

    # get list of extra rois
    extra_rois = list(
        set(glm_results['cell_roi_id'].unique()) - 
        set(cell_table['cell_roi_id'].unique())
    )

    # get any experiments for which the ROI count is incomplete. These are 'incomplete_experiments'
    if valid_rois_only==True:
        incomplete_experiments = set()
        additional_missing_cells = list(
            set(cell_table.query('ophys_experiment_id in {}'.format(list(glm_results['ophys_experiment_id'].unique())))['cell_roi_id']) - 
            set(glm_results['cell_roi_id'])
        )
        for missing_cell in additional_missing_cells:
            associated_oeid = cell_table.query('cell_roi_id == @missing_cell').iloc[0]['ophys_experiment_id']
            incomplete_experiments.add(associated_oeid)
        incomplete_experiments = list(incomplete_experiments)
        if len(incomplete_experiments) !=0:
            print('WARNING, incomplete experiments found. This indicates a big data problem, possibly indicating outdated cell segmentation')
    else:
        print('WARNING, ignoring incomplete experiments because valid_rois_only=True')
        incomplete_experiments=[]
        additional_missing_cells=[]

    inventory = {
        'fit_experiments': fit_experiments,
        'fit_rois':fit_rois,
        'missing_experiments': missing_experiments,
        'missing_rois': missing_rois,
        'extra_experiments': extra_experiments,
        'extra_rois': extra_rois,
        'incomplete_experiments': incomplete_experiments,
        'additional_missing_cells':additional_missing_cells,
        'Total Experiments':total_experiments,
        'Total ROIs':total_rois
        }
    
    return inventory
  
  
def select_experiments_for_testing(returns = 'experiment_ids'):
    '''
    This function will return 10 hand-picked experiment IDs to use for testing purposes.
    This will allow multiple versions to test against the same small set of experiments.

    Experiments were chosen as follows:
        2x OPHYS_2_passive
        2x OPHYS_5_passive
        2x active w/ fraction engaged < 0.05 (1 @ 0.00, 1 @ 0.02)
        2x active w/ fraction engaged > 0.99 (1 @ 0.97, 1 @ 0.98)
        2x active w/ fraction engaged in range (0.4, 0.6) (1 @ 0.44, 1 @ 0.59)

    Parameters:
    ----------
    returns : str
        either 'experiment_ids' or 'dataframe'

    Returns:
    --------
    if returns == 'experiment_ids' (default)
        list of 10 pre-chosen experiment IDs
    if returns == 'dataframe':
        experiment table for 10 pre-chosen experiments
    '''

    test_experiments = pd.read_csv('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/experiments_for_testing.csv')

    if returns == 'experiment_ids':
        return test_experiments['ophys_experiment_id'].unique()
    elif returns == 'dataframe':
        return test_experiments


def get_normalized_results_pivoted(glm_version = None, kind = 'max', cutoff = -np.inf):
    '''
    Loads absolute results pivoted, then normalization.
    Currently only normalizes to max

    INPUT:
    glm_version     default is version 15 with events
    kind            type of normalization. Currently only uses max, other options could be session number.

    OUTPUT:
    results_pivoted_normalized
    '''
    if glm_version == None:
        glm_version = '15_events_L2_optimize_by_session'
        print('loading glm version 15 with events by default ...')

    results_pivoted = build_pivoted_results_summary(glm_version=glm_version,
                                                value_to_use='absolute_change_from_full',
                                                results_summary=None,
                                                cutoff=cutoff)

    col_to_exclude = ['identifier', 'Full','variance_explained_full', 'cell_specimen_id', 'cell_roi_id',
       'glm_version', 'ophys_experiment_id', 'ophys_session_id',
       'equipment_name', 'donor_id', 'full_genotype', 'mouse_id',
       'reporter_line', 'driver_line', 'sex', 'age_in_days', 'foraging_id',
       'cre_line', 'indicator', 'session_number',
       'prior_exposures_to_session_type', 'prior_exposures_to_image_set',
       'prior_exposures_to_omissions', 'behavior_session_id',
       'ophys_container_id', 'project_code', 'container_workflow_state',
       'experiment_workflow_state', 'session_name', 'isi_experiment_id',
       'imaging_depth', 'targeted_structure', 'published_at',
       'date_of_acquisition', 'session_type', 'session_tags', 'failure_tags',
       'model_outputs_available', 'location']

    if kind == 'max':
        abs_max_df = results_pivoted[['variance_explained_full', 'cell_specimen_id']].groupby('cell_specimen_id').max()
        results_pivoted_normalized = results_pivoted.join(abs_max_df, on= 'cell_specimen_id', rsuffix='_max').copy()

        for column in results_pivoted.columns:
            if column not in col_to_exclude:
                results_pivoted_normalized.loc[results_pivoted_normalized[column]>0,column]=0
                results_pivoted_normalized[column] = results_pivoted_normalized[column].divide(results_pivoted_normalized['variance_explained_full_max'])

    return results_pivoted_normalized

 
def get_kernel_weights(glm, kernel_name, cell_specimen_id):
    '''
    gets the weights associated with a given kernel for a given cell_specimen_id

    inputs:
        glm : GLM class
        kernel_name : str
            name of desired kernel
        cell_specimen_id : int
            desired cell specimen ID

    returns:
        t_kernel, w_kernel
            t_kernel : array
                timestamps associated with the kernel
            w_kernel : 
                weights associated with the kernel
    '''
    
    # get all of the weight names for the given model
    all_weight_names = glm.X.weights.values
    
    # get the weight names associated with the desired kernel
    kernel_weight_names = [w for w in all_weight_names if w.startswith(kernel_name)]

    # get the weights
    w_kernel = glm.W.loc[dict(weights=kernel_weight_names, cell_specimen_id=cell_specimen_id)]

    # calculate the time array

    # first get the timestep
    timestep = 1/glm.fit['ophys_frame_rate']

    # get the timepoint that is closest to the desired offset
    offset_int = int(round(glm.design.kernel_dict[kernel_name]['offset_seconds']/timestep))

    # calculate t_kernel
    t_kernel = (np.arange(len(w_kernel)) + offset_int) * timestep

    return t_kernel, w_kernel

def get_sem_thresholds(results_pivoted, alpha=0.05,metric='SEM'):
    # Determine thresholds based on either:
    # just overall SEM
    # or whether mean > SEM
    # determine counts of how many cells excluded, etc    
    
    cres = results_pivoted.cre_line.unique()
    thresholds={}
    for cre in cres:
        thresholds[cre] = results_pivoted.query('cre_line ==@cre')['variance_explained_full_sem'].quantile(1-alpha)
    return thresholds

def compare_sem_thresholds(results_pivoted):
    cres = results_pivoted.cre_line.unique()
    
    print('Current, MEAN > 0.005')
    print('Fraction of cells to be set to 0')
    for cre in cres:
        cre_slice = results_pivoted.query('cre_line == @cre')
        frac = (cre_slice['variance_explained_full']< 0.005).astype(int).mean()
        print('{}: {}'.format(cre[0:3], np.round(frac,3)))

    print("\n")
    print('Forcing SEM < MEAN')
    print('Fraction of cells to be set to 0')
    for cre in cres:
        cre_slice = results_pivoted.query('cre_line == @cre')
        frac = (cre_slice['variance_explained_full_sem']>cre_slice['variance_explained_full']).astype(int).mean()
        print('{}: {}'.format(cre[0:3], np.round(frac,3)))

    print("\n")   
    print('Fraction of cells with SEM > 0.005')
    for cre in cres:
        cre_slice = results_pivoted.query('cre_line == @cre')
        frac = (cre_slice['variance_explained_full_sem']> 0.005).astype(int).mean()
        print('{}: {}'.format(cre[0:3], np.round(frac,3)))

def save_targeted_restart_table(run_params, results,save_table=True):
    '''
        Saves a table of experiments to restart. 
        Determines experiments to restart based on the presence of NaN variance explained
    '''    

    # get list of experiments to restart
    nan_oeids = results[results['variance_explained'].isnull()]['ophys_experiment_id'].unique()
    print('{} Experiments with NaN variance explained'.format(len(nan_oeids))) 
    if len(nan_oeids) == 0:
        return
    if save_table:
        restart_table = pd.DataFrame({'ophys_experiment_id':nan_oeids})
        table_path = run_params['output_dir']+'/restart_table.csv'
        restart_table.to_csv(table_path,index=False)
    return nan_oeids 

def make_table_of_nan_cells(run_params, results,save_table=True):
    '''
        Generates a table of cells for which the variance explained is NaN.
        In general, this should not happen
    '''
    nan_cells = results[results['variance_explained'].isnull()].query('dropout=="Full"').copy()
    if save_table:
        table_path = run_params['output_dir']+'/nan_cells_table.csv'
        nan_cells.to_csv(table_path,index=False)
    return nan_cells

def check_nan_cells(fit):
    '''
        Plots the df/f, events, and interpolated events for all cells with exactly 0 activity
    '''
    nan_cells = np.where(np.all(fit['fit_trace_arr'] == 0, axis=0))[0]
    for c in nan_cells:
        plt.figure()
        plt.plot(fit['fit_trace_timestamps'],fit['fit_trace_arr'][:,c],'r',label='Interpolated events')
        plt.plot(fit['stimulus_interpolation']['original_timestamps'], fit['stimulus_interpolation']['original_fit_arr'][:,c],'b--', label='Original Events')
        plt.plot(fit['stimulus_interpolation']['original_timestamps'], fit['stimulus_interpolation']['original_dff_arr'][:,c],'g--', label='Original df/f',alpha=.2)
        plt.legend()
        plt.ylabel('Neural Trace')
        plt.xlabel('Time')
        plt.title(fit['fit_trace_arr'].cell_specimen_id.values[c])

   
def check_cv_nans(fit):
    cv_var_test = fit['dropouts']['Full']['cv_var_test'].copy()
    orig_VE = np.mean(cv_var_test,1)
    orig_VE[orig_VE < 0] = 0

    cv_var_test_zero = fit['dropouts']['Full']['cv_var_test'].copy()
    cv_var_test_zero[np.isinf(cv_var_test_zero)]=0
    zero_VE = np.mean(cv_var_test_zero,1)
    zero_VE[zero_VE < 0] = 0

    cv_var_test_nan = fit['dropouts']['Full']['cv_var_test'].copy()
    cv_var_test_nan[np.isinf(cv_var_test_nan)]=np.nan
    nan_VE = np.nanmean(cv_var_test_nan,1)
    nan_VE[nan_VE < 0] = 0

    max_ve = np.nanmax(np.concatenate([orig_VE,zero_VE, nan_VE]))
    fig, ax = plt.subplots(2,2)
    
    ax[0,0].plot(orig_VE, zero_VE, 'ko')
    ax[0,0].set_ylabel('Set splits to 0')
    ax[0,0].set_xlabel('Original')
    ax[0,0].set_ylim(0,max_ve)
    ax[0,0].set_xlim(0,max_ve)
    ax[0,0].plot([0,max_ve],[0,max_ve],'r--')

    ax[0,1].plot(orig_VE, nan_VE, 'ko')
    ax[0,1].set_ylabel('Set splits to NaN')
    ax[0,1].set_xlabel('Original')
    ax[0,1].set_ylim(0,max_ve)
    ax[0,1].set_xlim(0,max_ve)
    ax[0,1].plot([0,max_ve],[0,max_ve],'r--')

    ax[1,0].plot(nan_VE, zero_VE, 'ko')
    ax[1,0].set_ylabel('Set splits to 0')
    ax[1,0].set_xlabel('Set splits to NaN')
    ax[1,0].set_ylim(0,max_ve)
    ax[1,0].set_xlim(0,max_ve)
    ax[1,0].plot([0,max_ve],[0,max_ve],'r--')

    ax[1,1].hist(zero_VE-orig_VE,alpha=.5,label='ZeroVE - Original',bins=20)
    ax[1,1].hist(nan_VE -orig_VE,alpha=.5,label='NaNVE - Original',bins=20)
    ax[1,1].legend()

    plt.tight_layout()
    return orig_VE, zero_VE, nan_VE


def reshape_rspm_by_experience(results_pivoted = None, model_output_type='adj_fraction_change_from_full',
                 glm_version='24_events_all_L2_optimize_by_session',
                 ophys_experiment_ids_to_use = None,
                 drop_duplicated_cells = True,
                 cutoff=None, features=None, single=False, save_df=False,
                 path=None):

    '''
    Loads and reshapes pivoted GLM results, groups by cell and experience level,
    cleans them (drops NaNs and duplicates), picks chronologically first session if multiple retakes
    for familiar or novel >1 exist. Returns reshaped df = n cells by n features

    Inputes:
        results_pivoted: default = None, you can provide output from build_pivoted_results_summary.
                        Will load it if none provided.
        model_output_type: default = 'adj_fraction_change_from_full'
        glm_version: default = '19_events_all_L2_optimize_by_session'
        cutoff: default = None, cutoff for total variance explained in glm results
        features: an array of regressors to use, default =[allimages, behavioral, omissions, task
        single: default = False, whether to get single dropout scores too
        save_df: default = False, whether to save df
        path: default=None, if save_df = True, must specify path where to save the file

    Output:
        df (n cells by n selected features)
    '''
    if save_df is True:
        assert path is not None, 'must provide path when saving file'

    if path is None and save_df is True:
        raise Warning('Please specify file path if you want to save df '
                      'or set save_df to False. File will not be saved.')
        save_df = False
    elif path is not None and save_df is False:
        raise Warning('Have to set save_df to True if you wish to save the file.')


    if results_pivoted is None:
        results_pivoted = build_pivoted_results_summary(value_to_use=model_output_type, results_summary=None,
                                             glm_version=glm_version, cutoff=cutoff)
        print('loading pivoted results')
    elif cutoff is not None:
        results_pivoted = results_pivoted[results_pivoted['variance_explained_full']>cutoff]
        print('setting a cutoff')

    if ophys_experiment_ids_to_use is not None:
        results_pivoted= results_pivoted[results_pivoted['ophys_experiment_id'].isin(ophys_experiment_ids_to_use)]

    print('loaded glm results')
    if features is None:
        features = get_default_features(single=single)

    # sort by date collected,
    results_pivoted = results_pivoted.sort_values('date_of_acquisition')

    # make cell_specimen_id and experience_level into indices
    df = results_pivoted.groupby(['cell_specimen_id', 'experience_level']).mean()

    # select regressors to look at
    df = df[features]

    # turn experience level into a column level,
    # drop cells that are missing a session,
    # drop duplicated cells (this needs more investigation)
    df = df.unstack()
    print('total N cells = {}'.format(len(df)))
    df = df.dropna()
    print('dropped NaN, now N = {}'.format(len(df)))

    if drop_duplicated_cells is True:
        if len(df) == len(np.unique(df.index.values)):
            print('No duplicated cells found')
        elif len(df) > len(np.unique(df.index.values)):
            print('found {} duplicated cells. But not removed. This needs to be fixed'.format(len(df) - len(np.unique(df.index.values))))
        elif len(df) < len(np.unique(df.index.values)):
            print('something weird happened!!')

    if save_df is True:
        filename = '{}.h5'.format(glm_version)
        full_path = os.path.join(path,filename)
        df.to_hdf(full_path, key='df', mode='w')
        print('saved df file')

    return df

def get_default_features(single=False):
    '''
    Which regressors to select for clustering;
    default = ['all-images','omissions','behavioral','task',]

    Input:
    single: default = False, whether to add 'single' dropout scores, too.

    Output:
    features, an array of glm regressors to use
    '''

    features = ['all-images',
            'omissions',
            'behavioral',
            'task', ]
    if single:
        features = ['single-' + feature for feature in features]

    return features

def check_mesoscope(results,filters=['cre_line','targeted_structure','depth','meso']):
    results['meso'] = ['mesoscope' if x == "MESO.1" else 'scientifica' for x in results['equipment_name']]
    results['depth'] = [50 if x < 100 else 150 if x <200 else 250 if x<300 else 350 for x in results['imaging_depth']]
    summary = pd.DataFrame(results.groupby(filters)['Full__avg_cv_var_test'].mean())
    summary['err']=results.groupby(filters)['Full__avg_cv_var_test'].sem()*2
    summary['count']=results.groupby(filters)['Full__avg_cv_var_test'].count()
    return summary


