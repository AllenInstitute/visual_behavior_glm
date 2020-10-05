import os
import pickle
import warnings
import numpy as np
import pandas as pd
import xarray_mongodb
from tqdm import tqdm

import visual_behavior.data_access.loading as loading
import visual_behavior.database as db

from sklearn.decomposition import PCA

def load_fit_pkl(run_params, ophys_experiment_id):
    '''
        Loads the fit dictionary from the pkl file dumped by fit_experiment
    
        Inputs:
        run_params, the dictionary of parameters for this version
        ophys_experiment_id, the oeid to load the fit for
    
        Returns:
        the fit dictionary if it exists

    '''
    filename = os.path.join(run_params['experiment_output_dir'],str(ophys_experiment_id)+'.pkl')
    with open(filename,'rb') as f:
        fit = pickle.load(f)
    return fit

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
    model_timestamps = glm.fit['dff_trace_arr']['dff_trace_timestamps'].values
    kernel_df = []

    # get all weight names
    all_weight_names = glm.X.weights.values

    # iterate over all kernels
    for ii, kernel_name in enumerate(kernel_list):
        # get the full kernel (dims = n_weights x n_timestamps)
        kernel = glm.design.kernel_dict[kernel_name]['kernel']

        # get the weight matrix for the weights associated with this kernel and cell (dims = 1 x n_weights)
        kernel_weight_names = [w for w in all_weight_names if w.startswith(kernel_name)]
        w_kernel = np.expand_dims(glm.W.loc[dict(
            weights=kernel_weight_names, cell_specimen_id=cell_specimen_id)], axis=0)

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
        results_summary = pd.pivot_table(results_summary.drop(columns=['dropout_name']), index=['dropout'],columns=['type'],values =['variance_explained'])
        results_summary.columns = results_summary.columns.droplevel()
        results_summary = results_summary.rename(columns={'avg_cv_adjvar_test':'adj_variance_explained','avg_cv_adjvar_test_full_comparison':'adj_variance_explained_full','adj_dropout':'adj_fraction_change_from_full'})
 
        # add the cell id info
        results_summary['cell_specimen_id'] = cell_specimen_id
         
        # pack up
        results_summary_list.append(results_summary)

    # Concatenate all cells and return

    return pd.concat(results_summary_list)

def generate_results_summary(glm):
    '''
        Returns a dataframe with summary information from the glm object
    '''
    # Get list of columns to look at, removing the non-adjusted dropouts, and training scores
    test_cols = [col for col in glm.results.columns if ((not col.endswith('train'))&('adj' not in col)&('session' not in col))]  
 
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
        results_summary = pd.pivot_table(results_summary.drop(columns=['dropout_name']), index=['dropout'],columns=['type'],values =['variance_explained'])
        results_summary.columns = results_summary.columns.droplevel()
        results_summary = results_summary.rename(columns={'avg_cv_var_test':'variance_explained','avg_cv_var_test_full_comparison':'variance_explained_full','dropout':'fraction_change_from_full'})
 
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
    # TODO, update to include adjusted dropouts
    # TODO, arent the full_results and results_summary already in the glm object by this point? is it redundant to compute them again?
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
        'ophys_experiment_id':glm.ophys_experiment_id,
        'glm_version':glm.version,
    }
    w_matrix_lookup_table = conn['ophys_glm']['weight_matrix_lookup_table']
    w_matrix_database = conn['ophys_glm_xarrays']

    if w_matrix_lookup_table.count_documents(lookup_table_document) >= 1:
        lookup_result = list(w_matrix_lookup_table.find(lookup_table_document))[0]
        # if weights matrix for this experiment/version has already been logged, we need to replace it

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

def get_experiment_table(glm_version):
    '''
    gets the experiment table
    appends the following:
        * roi count
        * cluster job summary for each experiment
        * number of existing dropouts
    
    Warning: this takes a couple of minutes to run.
    '''
    experiment_table = loading.get_filtered_ophys_experiment_table().reset_index()
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

def retrieve_results(search_dict={}, results_type='full'):
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
    output:
        dataframe of results
    '''
    conn = db.Database('visual_behavior_data')
    database = 'ophys_glm'
    results = pd.DataFrame(list(conn[database]['results_{}'.format(results_type)].find(search_dict)))

    # make 'glm_version' column a string
    results['glm_version'] = results['glm_version'].astype(str)
    conn.close()

    # get experiment table, merge in details of each experiment
    experiment_table = loading.get_filtered_ophys_experiment_table().reset_index()
    results = results.merge(
        experiment_table, 
        left_on='ophys_experiment_id',
        right_on='ophys_experiment_id', 
        how='left',
        suffixes=['', '_duplicated'],
    )
    duplicated_cols = [col for col in results.columns if col.endswith('_duplicated')]
    return results.drop(columns=duplicated_cols)

def make_identifier(row):
    return '{}_{}'.format(row['ophys_experiment_id'],row['cell_specimen_id'])

def get_glm_version_comparison_table(versions_to_compare, metric='Full__avg_cv_var_test'):
    '''
    builds a table that allows to glm versions to be directly compared
    input is list of glm versions to compare (list of strings)
    '''
    results = []
    for glm_version in versions_to_compare:
        results.append(retrieve_results({'glm_version': glm_version}, results_type='full'))
    results = pd.concat(results, sort=True)
    
    results['identifier'] = results.apply(make_identifier, axis=1)
    pivoted_results = results.pivot(index='identifier', columns='glm_version',values=metric)
    cols= [col for col in results.columns if col not in pivoted_results.columns and 'test' not in col and 'train' not in col and '__' not in col and 'dropout' not in col]

    pivoted_results = pivoted_results.merge(
        results[cols],
        left_on='identifier',
        right_on='identifier',
        how='left'
    )

    return pivoted_results

def build_pivoted_results_summary(value_to_use, results_summary=None, glm_version=None, cutoff=None):
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
        assert len(results_summary['glm_version'].unique()) == 1, 'number of glm_versions in the results summary caannot exceed 1'
        
    # get results summary if none was passed
    if results_summary is None:
        results_summary = retrieve_results(search_dict = {'glm_version': glm_version}, results_type='summary')
        
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
        'variance_explained', 
        'fraction_change_from_full', 
        'absolute_change_from_full',
        'adj_fraction_change_from_full',
        'adj_variance_explained',
        'adj_variance_explained_full',
        'entry_time_utc',
    ]
    cols_to_drop = [col for col in potential_cols_to_drop if col in results_summary.columns]
    results_summary_pivoted = results_summary_pivoted.merge(
        results_summary.drop(columns=cols_to_drop).drop_duplicates(),
        left_on='identifier',
        right_on='identifier',
        how='left'
    )
    
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
    
    experiments_table = loading.get_filtered_ophys_experiment_table()

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

def build_weights_df(run_params,results_pivoted, cache_results=False,load_cache=False):
    '''
        Builds a dataframe of (cell_specimen_id, ophys_experiment_id) with the weight parameters for each kernel
        Some columns may have NaN if that cell did not have a kernel, for example if a missing datastream   
 
        INPUTS:
        run_params, parameter json for the version to analyze
        results_pivoted = build_pivoted_results_summary('adj_fraction_change_from_full',results_summary=results)
        cache_results, if True, save dataframe as csv file
        load_cache, if True, load cached results, if it exists
    
        RETURNS:
        a dataframe
    '''
    
    #if load_cache & os.path.exists(run_params['output_dir']+'/weights_df.csv'):
    #    # Need to convert things to np.array
    #    return pd.read_csv(run_params['output_dir']+'/weights_df.csv')
   
    # Make dataframe for cells and experiments 
    oeids = results_pivoted['ophys_experiment_id'].unique() 

    # For each experiment, get the weight matrix from mongo (slow)
    # Then pull the weights from each kernel into a dataframe
    sessions = []
    for index, oeid in enumerate(tqdm(oeids)):
        session_df = process_session_to_df(oeid, run_params)
        sessions.append(session_df)

    # Merge all the session_dfs, and add more session level info
    weights_df = pd.concat(sessions,sort=False)
    weights_df = pd.merge(weights_df,results_pivoted, on = ['cell_specimen_id','ophys_experiment_id'],suffixes=('_weights',''))
    
    ## Cache Results
    #if cache_results:
    #    weights_df.to_csv(run_params['output_dir']+'/weights_df.csv') 

    # Return weights_df
    return weights_df 

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
        results_full[d+'__over_fit'] = (results_full[d+'__avg_cv_var_train']-results_full[d+'__avg_cv_var_test'])/(results_full[d+'__avg_cv_var_train'])
    
    dropouts.remove('Full')
    for d in dropouts:
        results_full[d+'__dropout_overfit_proportion'] = 1-results_full[d+'__over_fit']/results_full['Full__over_fit']
 


# NOTE:
# Everything below this point is carried over from Nick P.'s old repo. Commenting it out to keep it as a resource.
#dirc = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20200102_lambda_70/'
#dirc = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20200102_reward_filter_dev/'
#dff_dirc = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/ophys_glm_dev_dff_traces/'
#global_dir = dirc

# def moving_mean(values, window):
#     weights = np.repeat(1.0, window)/window
#     mm = np.convolve(values, weights, 'valid')
#     return mm


# def compute_full_mean(experiment_ids):
#     x = []
#     for exp_dex, exp_id in enumerate(tqdm(experiment_ids)):
#         try:
#             fit_data = compute_response(exp_id)
#             x = x + compute_mean_error(fit_data)
#         except:
#             pass
#     return x


# def compute_mean_error(fit_data, threshold=0.02):
#     x = []
#     for cell_dex, cell_id in enumerate(fit_data['w'].keys()):
#         if fit_data['cv_var_explained'][cell_id] > threshold:
#             x.append(fit_data['model_err'][cell_id])
#     return x


# def plot_errors(fit_data, threshold=0.02, plot_each=False, smoothing_window=50):
#     plt.figure(figsize=(12, 4))
#     x = []
#     for cell_dex, cell_id in enumerate(fit_data['w'].keys()):
#         if fit_data['cv_var_explained'][cell_id] > threshold:
#             if plot_each:
#                 plt.plot(fit_data['model_err'][cell_id], 'k', alpha=0.2)
#             x.append(fit_data['model_err'][cell_id])
#     plt.plot(moving_mean(np.mean(np.vstack(x), 0), 31*smoothing_window), 'r')
#     plt.axhline(0, color='k', ls='--')


# def plot_cell(fit_data, cell_id):
#     plt.figure(figsize=(12, 4))
#     plt.plot(fit_data['data_dff'][cell_id], 'r', label='Cell')
#     plt.plot(fit_data['model_dff'][cell_id], 'b', label='Model')
#     plt.ylabel('dff')
#     plt.xlabel('time in experiment')


# def get_experiment_design_matrix_temp(oeid, model_dir):
#     return np.load(model_dir+'X_sparse_csc_'+str(oeid)+'.npz')


# def get_experiment_design_matrix(oeid, model_dir):
#     return sparse.load_npz(model_dir+'X_sparse_csc_'+str(oeid)+'.npz').todense()


# def get_experiment_fit(oeid, model_dir):
#     filepath = 'oeid_'+str(oeid)+'.json'
#     with open(model_dir+'/'+filepath) as json_file:
#         data = json.load(json_file)
#     return data


# def get_experiment_dff(oeid):
#     filepath = dff_dirc+str(oeid)+'_dff_array.cd'
#     return xr.open_dataarray(filepath)


# def compute_response(oeid, model_dir):
#     design_matrix = get_experiment_design_matrix(oeid, model_dir)
#     fit_data = get_experiment_fit(oeid)
#     dff_data = get_experiment_dff(oeid)
#     model_dff, model_err, data_dff = compute_response_inner(
#         design_matrix, fit_data, dff_data)
#     fit_data['model_dff'] = model_dff
#     fit_data['model_err'] = model_err
#     fit_data['data_dff'] = data_dff
#     return fit_data


# def compute_response_inner(design_matrix, fit_data, dff_data):
#     model_dff = {}
#     model_err = {}
#     data_dff = {}
#     for cell_dex, cell_id in enumerate(fit_data['w'].keys()):
#         W = np.mean(fit_data['w'][cell_id], 1)
#         model_dff[cell_id] = np.squeeze(np.asarray(design_matrix @ W))
#         model_err[cell_id] = model_dff[cell_id] - \
#             np.array(dff_data.sel(cell_specimen_id=int(cell_id)))
#         data_dff[cell_id] = np.array(
#             dff_data.sel(cell_specimen_id=int(cell_id)))
#     return model_dff, model_err, data_dff

# # Filter for cells tracked on both A1 and A3


# def get_cells_in(df, stage1, stage2):
#     s1 = []
#     s2 = []
#     cell_ids = df['cell_specimen_id'].unique()
#     for cell_dex, cell_id in enumerate(cell_ids):
#         hass1 = len(
#             df.query('cell_specimen_id == @cell_id & stage_name == @stage1')) == 1
#         hass2 = len(
#             df.query('cell_specimen_id == @cell_id & stage_name == @stage2')) == 1
#         if hass1 & hass2:
#             s1.append(df.query('cell_specimen_id == @cell_id & stage_name == @stage1')[
#                       'cv_var_explained'].iloc[0])
#             s2.append(df.query('cell_specimen_id == @cell_id & stage_name == @stage2')[
#                       'cv_var_explained'].iloc[0])
#     return np.array(s1), np.array(s2)


# def plot_session_comparison(s1, s2, label1, label2):
#     plt.figure()
#     plt.plot(s1, s2, 'ko')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel(label1+' Var Explained')
#     plt.ylabel(label2+' Var Explained')
#     plt.xlim(-.1, 1)
#     plt.ylim(-.1, 1)
#     plt.savefig(
#         '/home/alex.piet/codebase/GLM/figs/var_explained_scatter_'+label1+'_'+label2+'.png')
#     plt.figure()
#     plt.hist(s2-s1, 100)
#     plt.axvline(0, color='k', ls='--')
#     mean_val = np.mean(s2-s1)
#     mean_sem = sem(s2-s1)
#     yval = plt.ylim()[1]
#     plt.plot(mean_val, yval, 'rv')
#     plt.plot([mean_val-1.96*mean_sem, mean_val +
#               1.96*mean_sem], [yval, yval], 'r-')
#     plt.xlim(-0.4, 0.4)
#     plt.savefig(
#         '/home/alex.piet/codebase/GLM/figs/var_explained_histogram_'+label1+'_'+label2+'.png')


# def get_ophys_timestamps(session):
#     ophys_frames_to_use = (
#         session.ophys_timestamps > session.stimulus_presentations.iloc[0]['start_time']
#     ) & (
#         session.ophys_timestamps < session.stimulus_presentations.iloc[-1]['stop_time']+0.5
#     )
#     timestamps = session.ophys_timestamps[ophys_frames_to_use]
#     return timestamps[:-1]


# def compute_variance_explained(df):
#     var_expl = [(np.var(x[0]) - np.var(x[1]))/np.var(x[0])
#                 for x in zip(df['data_dff'], df['model_err'])]
#     df['var_expl'] = var_expl


# def process_to_flashes(fit_data, session):
#     ''' 
#         Is now fast
#     '''
#     cells = list(fit_data['w'].keys())
#     timestamps = get_ophys_timestamps(session)
#     df = pd.DataFrame()
#     cell_specimen_id = []
#     stimulus_presentations_id = []
#     model_dff = []
#     model_err = []
#     data_dff = []
#     image_index = []

#     for dex, row in session.stimulus_presentations.iterrows():
#         sdex = np.where(timestamps > row.start_time)[0][0]
#         edex = np.where(timestamps < row.start_time + 0.75)[0][-1]
#         edex = sdex + 21  # due to aliasing, im hard coding this for now
#         for cell_dex, cell_id in enumerate(cells):
#             cell_specimen_id.append(cell_id)
#             stimulus_presentations_id.append(int(dex))
#             model_dff.append(fit_data['model_dff'][cell_id][sdex:edex])
#             model_err.append(fit_data['model_err'][cell_id][sdex:edex])
#             data_dff.append(fit_data['data_dff'][cell_id][sdex:edex])
#             image_index.append(row.image_index)
#     df['cell_specimen_id'] = cell_specimen_id
#     df['stimulus_presentations_id'] = stimulus_presentations_id
#     df['model_dff'] = model_dff
#     df['model_err'] = model_err
#     df['data_dff'] = data_dff
#     df['image_index'] = image_index
#     return df


# def process_to_trials(fit_data, session):
#     ''' 
#         Takes Forever
#     '''
#     cells = list(fit_data['w'].keys())
#     timestamps = get_ophys_timestamps(session)
#     df = pd.DataFrame()
#     for dex, row in session.trials.iterrows():
#         if not np.isnan(row.change_time):
#             sdex = np.where(timestamps > row.change_time-2)[0][0]
#             edex = np.where(timestamps < row.change_time+2)[0][-1]
#             edex = sdex + 124  # due to aliasing, im hard coding this for now
#             for cell_dex, cell_id in enumerate(cells):
#                 d = {'cell_specimen_id': cell_id, 'stimulus_presentations_id': int(dex),
#                      'model_dff': fit_data['model_dff'][cell_id][sdex:edex],
#                      'model_err': fit_data['model_err'][cell_id][sdex:edex],
#                      'data_dff': fit_data['data_dff'][cell_id][sdex:edex]}
#                 df = df.append(d, ignore_index=True)
#     return df


# def compute_shuffle(flash_df):
#     cells = flash_df['cell_specimen_id'].unique()
#     cv_d = {}
#     cv_shuf_d = {}
#     for dex, cellid in enumerate(cells):
#         cv, cv_shuf = compute_shuffle_var_explained(flash_df, cellid)
#         cv_d[cellid] = cv
#         cv_shuf_d[cellid] = cv_shuf
#     return cv_d, cv_shuf_d


# def compute_shuffle_var_explained(flash_df, cell_id):
#     '''
#         Computes the variance explained in a shuffle distribution
#         NOTE: this variance explained is going to be different from the full thing because im being a little hacky. buts I think its ok for the purpose of this analysis 
#     '''
#     cell_df = flash_df.query('cell_specimen_id == @cell_id').reset_index()
#     cell_df['model_dff_shuffle'] = cell_df.sample(
#         frac=1).reset_index()['model_dff']
#     model_dff_shuf = np.hstack(cell_df['model_dff_shuffle'].values)
#     model_dff = np.hstack(cell_df['model_dff'].values)
#     data_dff = np.hstack(cell_df['data_dff'].values)
#     model_err = model_dff - data_dff
#     shuf_err = model_dff_shuf - data_dff
#     var_total = np.var(data_dff)
#     var_resid = np.var(model_err)
#     var_shuf = np.var(shuf_err)
#     cv = (var_total - var_resid) / var_total
#     cv_shuf = (var_total - var_shuf) / var_total
#     return cv, cv_shuf


# def strip_dict(d):
#     value_list = []
#     for dex, key in enumerate(list(d.keys())):
#         value_list.append(d[key])
#     return value_list


# def plot_shuffle_analysis(cv_list, cv_shuf_list, alpha=0.05):
#     plt.figure()
#     nbins = round(len(cv_list)/5)
#     plt.hist(cv_list*100, nbins, alpha=0.5, label='Data')
#     plt.hist(cv_shuf_list*100, nbins, color='k', alpha=0.5, label='Shuffle')
#     plt.axvline(0, ls='--', color='k')
#     plt.legend()
#     threshold = find_threshold(cv_shuf_list, alpha=alpha)
#     plt.axvline(threshold*100, ls='--', color='r')
#     plt.xlabel('Variance Explained')
#     plt.ylabel('count')
#     return threshold


# def find_threshold(cv_shuf_list, alpha=0.05):
#     dex = round(len(cv_shuf_list)*alpha)
#     threshold = np.sort(cv_shuf_list)[-dex]
#     if threshold < 0:
#         return 0
#     else:
#         return threshold


# def shuffle_session(fit_data, session):
#     flash_df = process_to_flashes(fit_data, session)
#     compute_variance_explained(flash_df)
#     cv_df, cv_shuf_df = compute_shuffle(flash_df)
#     threshold = plot_shuffle_analysis(
#         strip_dict(cv_df), strip_dict(cv_shuf_df))
#     return cv_df, cv_shuf_df, threshold

# # Need a function for concatenating cv_df, and cv_shuf_df across sessions


# def shuffle_across_sessions(experiment_list, cache, model_dir=None):
#     if type(model_dir) == type(None):
#         model_dir = global_dir
#     all_cv = []
#     all_shuf = []
#     ophys_experiments = cache.get_experiment_table()
#     for dex, oeid in enumerate(tqdm(experiment_list)):
#         fit_data = compute_response(oeid, model_dir)
#         oeid = ophys_experiments.loc[oeid]['ophys_experiment_id'][0]
#         experiment = cache.get_experiment_data(oeid)
#         flash_df = process_to_flashes(fit_data, experiment)
#         compute_variance_explained(flash_df)
#         cv_df, cv_shuf_df = compute_shuffle(flash_df)
#         all_cv.append(strip_dict(cv_df))
#         all_shuf.append(strip_dict(cv_shuf_df))
#     threshold = plot_shuffle_analysis(np.hstack(all_cv), np.hstack(all_shuf))
#     return all_cv, all_shuf, threshold


# def analyze_threshold(all_cv, all_shuf, threshold):
#     cells_above_threshold = round(
#         np.sum(np.hstack(all_cv) > threshold)/len(np.hstack(all_cv))*100, 2)
#     false_positive_with_zero_threshold = round(
#         np.sum(np.hstack(all_shuf) > 0)/len(np.hstack(all_shuf)), 2)
#     false_positive_with_2_threshold = round(
#         np.sum(np.hstack(all_shuf) > 0.02)/len(np.hstack(all_shuf)), 2)
#     steinmetz_threshold = find_threshold(np.hstack(all_shuf), alpha=0.0033)
#     print("Variance Explained % threshold:       " +
#           str(round(100*threshold, 2)) + " %")
#     print("Percent of cells above threshold:    " +
#           str(cells_above_threshold) + " %")
#     print("False positive if using 0% threshold: " +
#           str(false_positive_with_zero_threshold))
#     print("False positive if using 2% threshold: " +
#           str(false_positive_with_2_threshold))
#     print("Threshold needed for Steinmetz level: " +
#           str(round(100*steinmetz_threshold, 2)) + " %")
