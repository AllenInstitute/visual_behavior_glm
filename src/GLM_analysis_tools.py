import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
import json
import h5py
import xarray as xr
from scipy import sparse
from tqdm import tqdm

import visual_behavior.data_access.loading as loading
import visual_behavior.database as db


dirc = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20200102_lambda_70/'
#dirc = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/20200102_reward_filter_dev/'
dff_dirc = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/ophys_glm_dev_dff_traces/'
global_dir = dirc


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


def generate_results_summary(glm):
    test_cols = [col for col in glm.results.columns if col.endswith('test')]
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


def log_results_to_mongo(glm):
    '''
    logs full results and results summary to mongo
    Ensures that there is only one entry per cell/experiment (overwrites if entry already exists)
    '''
    full_results = glm.results.reset_index()
    results_summary = generate_results_summary(glm)
    experiment_table = loading.get_filtered_ophys_experiment_table().reset_index()
    oeid = glm.oeid
    for key,value in experiment_table.query('ophys_experiment_id == @oeid').iloc[0].items():
        full_results[key] = value
        results_summary[key] = value

    full_results['glm_version'] = glm.version
    results_summary['glm_version'] = glm.version

    conn = db.Database('visual_behavior_data')

    keys_to_check = {
        'results_full':['ophys_experiment_id','cell_specimen_id'],
        'results_summary':['ophys_experiment_id','cell_specimen_id', 'dropout']
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


def retrieve_results(glm_version=None):
    '''
    gets cached results from mongodb
    input:
        GLM version
    output:
        dictionary with keys:
            * full: 1 row for every unique cell/session (cells that are matched across sessions will have one row for each session.
                Each row contains all of the coefficients of variation (a test and a train value for each dropout)
            * summary: results_summary contains 1 row for every unique cell/session/dropout 
                cells that are matched across sessions will have `N_DROPOUTS` rows for each session.
                Each row contains a `dropout` label describing the particular dropout coefficent(s) that apply to that row. 
                All derived values (`variance_explained`, `fraction_change_from_full`, `absolute_change_from_full`) 
                are calculated only on test data, not train data.
    '''
    conn = db.Database('visual_behavior_data')
    database = 'ophys_glm'
    results = {}
    for key in ['full','summary']:
        if glm_version:
            # if version is specified, get results for only specified version
            results[key] = pd.DataFrame(list(conn[database]['results_{}'.format(key)].find({'glm_version':glm_version})))
        else:
            # if no version is specified, get results for all versions
            results[key] = pd.DataFrame(list(conn[database]['results_{}'.format(key)].find({})))
    conn.close()
    return results

def moving_mean(values, window):
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm


def compute_full_mean(experiment_ids):
    x = []
    for exp_dex, exp_id in enumerate(tqdm(experiment_ids)):
        try:
            fit_data = compute_response(exp_id)
            x = x + compute_mean_error(fit_data)
        except:
            pass
    return x


def compute_mean_error(fit_data, threshold=0.02):
    x = []
    for cell_dex, cell_id in enumerate(fit_data['w'].keys()):
        if fit_data['cv_var_explained'][cell_id] > threshold:
            x.append(fit_data['model_err'][cell_id])
    return x


def plot_errors(fit_data, threshold=0.02, plot_each=False, smoothing_window=50):
    plt.figure(figsize=(12, 4))
    x = []
    for cell_dex, cell_id in enumerate(fit_data['w'].keys()):
        if fit_data['cv_var_explained'][cell_id] > threshold:
            if plot_each:
                plt.plot(fit_data['model_err'][cell_id], 'k', alpha=0.2)
            x.append(fit_data['model_err'][cell_id])
    plt.plot(moving_mean(np.mean(np.vstack(x), 0), 31*smoothing_window), 'r')
    plt.axhline(0, color='k', ls='--')


def plot_cell(fit_data, cell_id):
    plt.figure(figsize=(12, 4))
    plt.plot(fit_data['data_dff'][cell_id], 'r', label='Cell')
    plt.plot(fit_data['model_dff'][cell_id], 'b', label='Model')
    plt.ylabel('dff')
    plt.xlabel('time in experiment')


def get_experiment_design_matrix_temp(oeid, model_dir):
    return np.load(model_dir+'X_sparse_csc_'+str(oeid)+'.npz')


def get_experiment_design_matrix(oeid, model_dir):
    return sparse.load_npz(model_dir+'X_sparse_csc_'+str(oeid)+'.npz').todense()


def get_experiment_fit(oeid, model_dir):
    filepath = 'oeid_'+str(oeid)+'.json'
    with open(model_dir+'/'+filepath) as json_file:
        data = json.load(json_file)
    return data


def get_experiment_dff(oeid):
    filepath = dff_dirc+str(oeid)+'_dff_array.cd'
    return xr.open_dataarray(filepath)


def compute_response(oeid, model_dir):
    design_matrix = get_experiment_design_matrix(oeid, model_dir)
    fit_data = get_experiment_fit(oeid)
    dff_data = get_experiment_dff(oeid)
    model_dff, model_err, data_dff = compute_response_inner(
        design_matrix, fit_data, dff_data)
    fit_data['model_dff'] = model_dff
    fit_data['model_err'] = model_err
    fit_data['data_dff'] = data_dff
    return fit_data


def compute_response_inner(design_matrix, fit_data, dff_data):
    model_dff = {}
    model_err = {}
    data_dff = {}
    for cell_dex, cell_id in enumerate(fit_data['w'].keys()):
        W = np.mean(fit_data['w'][cell_id], 1)
        model_dff[cell_id] = np.squeeze(np.asarray(design_matrix @ W))
        model_err[cell_id] = model_dff[cell_id] - \
            np.array(dff_data.sel(cell_specimen_id=int(cell_id)))
        data_dff[cell_id] = np.array(
            dff_data.sel(cell_specimen_id=int(cell_id)))
    return model_dff, model_err, data_dff

# Filter for cells tracked on both A1 and A3


def get_cells_in(df, stage1, stage2):
    s1 = []
    s2 = []
    cell_ids = df['cell_specimen_id'].unique()
    for cell_dex, cell_id in enumerate(cell_ids):
        hass1 = len(
            df.query('cell_specimen_id == @cell_id & stage_name == @stage1')) == 1
        hass2 = len(
            df.query('cell_specimen_id == @cell_id & stage_name == @stage2')) == 1
        if hass1 & hass2:
            s1.append(df.query('cell_specimen_id == @cell_id & stage_name == @stage1')[
                      'cv_var_explained'].iloc[0])
            s2.append(df.query('cell_specimen_id == @cell_id & stage_name == @stage2')[
                      'cv_var_explained'].iloc[0])
    return np.array(s1), np.array(s2)


def plot_session_comparison(s1, s2, label1, label2):
    plt.figure()
    plt.plot(s1, s2, 'ko')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel(label1+' Var Explained')
    plt.ylabel(label2+' Var Explained')
    plt.xlim(-.1, 1)
    plt.ylim(-.1, 1)
    plt.savefig(
        '/home/alex.piet/codebase/GLM/figs/var_explained_scatter_'+label1+'_'+label2+'.png')
    plt.figure()
    plt.hist(s2-s1, 100)
    plt.axvline(0, color='k', ls='--')
    mean_val = np.mean(s2-s1)
    mean_sem = sem(s2-s1)
    yval = plt.ylim()[1]
    plt.plot(mean_val, yval, 'rv')
    plt.plot([mean_val-1.96*mean_sem, mean_val +
              1.96*mean_sem], [yval, yval], 'r-')
    plt.xlim(-0.4, 0.4)
    plt.savefig(
        '/home/alex.piet/codebase/GLM/figs/var_explained_histogram_'+label1+'_'+label2+'.png')


def get_ophys_timestamps(session):
    ophys_frames_to_use = (
        session.ophys_timestamps > session.stimulus_presentations.iloc[0]['start_time']
    ) & (
        session.ophys_timestamps < session.stimulus_presentations.iloc[-1]['stop_time']+0.5
    )
    timestamps = session.ophys_timestamps[ophys_frames_to_use]
    return timestamps[:-1]


def compute_variance_explained(df):
    var_expl = [(np.var(x[0]) - np.var(x[1]))/np.var(x[0])
                for x in zip(df['data_dff'], df['model_err'])]
    df['var_expl'] = var_expl


def process_to_flashes(fit_data, session):
    ''' 
        Is now fast
    '''
    cells = list(fit_data['w'].keys())
    timestamps = get_ophys_timestamps(session)
    df = pd.DataFrame()
    cell_specimen_id = []
    stimulus_presentations_id = []
    model_dff = []
    model_err = []
    data_dff = []
    image_index = []

    for dex, row in session.stimulus_presentations.iterrows():
        sdex = np.where(timestamps > row.start_time)[0][0]
        edex = np.where(timestamps < row.start_time + 0.75)[0][-1]
        edex = sdex + 21  # due to aliasing, im hard coding this for now
        for cell_dex, cell_id in enumerate(cells):
            cell_specimen_id.append(cell_id)
            stimulus_presentations_id.append(int(dex))
            model_dff.append(fit_data['model_dff'][cell_id][sdex:edex])
            model_err.append(fit_data['model_err'][cell_id][sdex:edex])
            data_dff.append(fit_data['data_dff'][cell_id][sdex:edex])
            image_index.append(row.image_index)
    df['cell_specimen_id'] = cell_specimen_id
    df['stimulus_presentations_id'] = stimulus_presentations_id
    df['model_dff'] = model_dff
    df['model_err'] = model_err
    df['data_dff'] = data_dff
    df['image_index'] = image_index
    return df


def process_to_trials(fit_data, session):
    ''' 
        Takes Forever
    '''
    cells = list(fit_data['w'].keys())
    timestamps = get_ophys_timestamps(session)
    df = pd.DataFrame()
    for dex, row in session.trials.iterrows():
        if not np.isnan(row.change_time):
            sdex = np.where(timestamps > row.change_time-2)[0][0]
            edex = np.where(timestamps < row.change_time+2)[0][-1]
            edex = sdex + 124  # due to aliasing, im hard coding this for now
            for cell_dex, cell_id in enumerate(cells):
                d = {'cell_specimen_id': cell_id, 'stimulus_presentations_id': int(dex),
                     'model_dff': fit_data['model_dff'][cell_id][sdex:edex],
                     'model_err': fit_data['model_err'][cell_id][sdex:edex],
                     'data_dff': fit_data['data_dff'][cell_id][sdex:edex]}
                df = df.append(d, ignore_index=True)
    return df


def compute_shuffle(flash_df):
    cells = flash_df['cell_specimen_id'].unique()
    cv_d = {}
    cv_shuf_d = {}
    for dex, cellid in enumerate(cells):
        cv, cv_shuf = compute_shuffle_var_explained(flash_df, cellid)
        cv_d[cellid] = cv
        cv_shuf_d[cellid] = cv_shuf
    return cv_d, cv_shuf_d


def compute_shuffle_var_explained(flash_df, cell_id):
    '''
        Computes the variance explained in a shuffle distribution
        NOTE: this variance explained is going to be different from the full thing because im being a little hacky. buts I think its ok for the purpose of this analysis 
    '''
    cell_df = flash_df.query('cell_specimen_id == @cell_id').reset_index()
    cell_df['model_dff_shuffle'] = cell_df.sample(
        frac=1).reset_index()['model_dff']
    model_dff_shuf = np.hstack(cell_df['model_dff_shuffle'].values)
    model_dff = np.hstack(cell_df['model_dff'].values)
    data_dff = np.hstack(cell_df['data_dff'].values)
    model_err = model_dff - data_dff
    shuf_err = model_dff_shuf - data_dff
    var_total = np.var(data_dff)
    var_resid = np.var(model_err)
    var_shuf = np.var(shuf_err)
    cv = (var_total - var_resid) / var_total
    cv_shuf = (var_total - var_shuf) / var_total
    return cv, cv_shuf


def strip_dict(d):
    value_list = []
    for dex, key in enumerate(list(d.keys())):
        value_list.append(d[key])
    return value_list


def plot_shuffle_analysis(cv_list, cv_shuf_list, alpha=0.05):
    plt.figure()
    nbins = round(len(cv_list)/5)
    plt.hist(cv_list*100, nbins, alpha=0.5, label='Data')
    plt.hist(cv_shuf_list*100, nbins, color='k', alpha=0.5, label='Shuffle')
    plt.axvline(0, ls='--', color='k')
    plt.legend()
    threshold = find_threshold(cv_shuf_list, alpha=alpha)
    plt.axvline(threshold*100, ls='--', color='r')
    plt.xlabel('Variance Explained')
    plt.ylabel('count')
    return threshold


def find_threshold(cv_shuf_list, alpha=0.05):
    dex = round(len(cv_shuf_list)*alpha)
    threshold = np.sort(cv_shuf_list)[-dex]
    if threshold < 0:
        return 0
    else:
        return threshold


def shuffle_session(fit_data, session):
    flash_df = process_to_flashes(fit_data, session)
    compute_variance_explained(flash_df)
    cv_df, cv_shuf_df = compute_shuffle(flash_df)
    threshold = plot_shuffle_analysis(
        strip_dict(cv_df), strip_dict(cv_shuf_df))
    return cv_df, cv_shuf_df, threshold

# Need a function for concatenating cv_df, and cv_shuf_df across sessions


def shuffle_across_sessions(experiment_list, cache, model_dir=None):
    if type(model_dir) == type(None):
        model_dir = global_dir
    all_cv = []
    all_shuf = []
    ophys_experiments = cache.get_experiment_table()
    for dex, oeid in enumerate(tqdm(experiment_list)):
        fit_data = compute_response(oeid, model_dir)
        oeid = ophys_experiments.loc[oeid]['ophys_experiment_id'][0]
        experiment = cache.get_experiment_data(oeid)
        flash_df = process_to_flashes(fit_data, experiment)
        compute_variance_explained(flash_df)
        cv_df, cv_shuf_df = compute_shuffle(flash_df)
        all_cv.append(strip_dict(cv_df))
        all_shuf.append(strip_dict(cv_shuf_df))
    threshold = plot_shuffle_analysis(np.hstack(all_cv), np.hstack(all_shuf))
    return all_cv, all_shuf, threshold


def analyze_threshold(all_cv, all_shuf, threshold):
    cells_above_threshold = round(
        np.sum(np.hstack(all_cv) > threshold)/len(np.hstack(all_cv))*100, 2)
    false_positive_with_zero_threshold = round(
        np.sum(np.hstack(all_shuf) > 0)/len(np.hstack(all_shuf)), 2)
    false_positive_with_2_threshold = round(
        np.sum(np.hstack(all_shuf) > 0.02)/len(np.hstack(all_shuf)), 2)
    steinmetz_threshold = find_threshold(np.hstack(all_shuf), alpha=0.0033)
    print("Variance Explained % threshold:       " +
          str(round(100*threshold, 2)) + " %")
    print("Percent of cells above threshold:    " +
          str(cells_above_threshold) + " %")
    print("False positive if using 0% threshold: " +
          str(false_positive_with_zero_threshold))
    print("False positive if using 2% threshold: " +
          str(false_positive_with_2_threshold))
    print("Threshold needed for Steinmetz level: " +
          str(round(100*steinmetz_threshold, 2)) + " %")
