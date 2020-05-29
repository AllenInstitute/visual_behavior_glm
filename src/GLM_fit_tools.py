import os
import pwd
import json
import shutil
import pickle
import xarray as xr
import numpy as np
import pandas as pd
import datetime

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
from visual_behavior.translator.allensdk_sessions import sdk_utils
from visual_behavior.translator.allensdk_sessions import session_attributes
from visual_behavior.ophys.response_analysis import response_processing as rp



CODEBASE = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/GLM/visual_behavior_glm/' 
OUTPUT_DIR_BASE = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
DEFAULT_OPHYS_SESSION_IDS = [881236651, 880753403] 
MANIFEST = ''# TODO

def load_run_json(VERSION):
    '''
        Loads the run parameters for model v_<VERSION>
        Assumes verion is saved with root directory global OUTPUT_DIR_BASE       
        returns a dictionary of run parameters
    '''
    json_path           = OUTPUT_DIR_BASE + 'v_'+str(VERSION) +'/' +'run_params.json'
    with open(json_path,'r') as json_file:
        run_params = json.load(json_file)
    return run_params

def make_run_json(VERSION,label='',username=None,src_path=None):
    '''
        Freezes model files, parameters, and ophys experiment ids
        If the model iteration already exists, throws an error
        root directory is global OUTPUT_DIR_BASE

        v_<VERSION>             contains the model iteration
        ./frozen_model_files/   contains the model files
        ./session_model_files/  contains the output for each session
        ./README.txt            contains information about the model creation
        ./run_params.json       contains a dictionary of the model parameters

        <username>  include a string to README.txt who created each model iteration. If none is provided
                    attempts to load linux username. Will default to "unknown" on error
        <label>     include a string to README.txt with a brief summary of the model iteration
        <src_path>  path to repo home. Will default to alex's path
    '''

    # Make directory, will throw an error if already exists
    output_dir              = OUTPUT_DIR_BASE + 'v_'+str(VERSION) +'/'
    model_freeze_dir        = output_dir +'frozen_model_files/'
    experiment_output_dir   = output_dir +'experiment_model_files/'
    job_dir                 = output_dir +'log_files/'
    json_path               = output_dir +'run_params.json'
    manifest_path           = output_dir +'manifest_v_'+str(VERSION)+'.csv'
    os.mkdir(output_dir)
    os.mkdir(model_freeze_dir)
    os.mkdir(experiment_output_dir)
    os.mkdir(job_dir)
    
    # Add a readme file with information about when the model was created
    if username is None:
        try:
            username = pwd.getpwuid(os.getuid())[0]
        except:
            username = 'unknown'
    readme = open(output_dir+'README.txt','w')
    readme.writelines([ 'OPHYS GLM  v',str(VERSION),
                        '\nCreated on ',str(datetime.datetime.now()), 
                        '\nCreated by ',username,
                        '\nComment    ',label,'\n\n'])
    readme.close()

    # Copy model files to frozen directory
    python_file_full_path = model_freeze_dir+'GLM_fit_tools.py'
    python_fit_script = model_freeze_dir +'fit_glm_v_'+str(VERSION)+'.py'
    if src_path is None:
        print('WARNING, no path provided, defaulting to: '+CODEBASE)
        src_path = CODEBASE
    shutil.copyfile(src_path+'src/GLM_fit_tools.py',   python_file_full_path)
    shutil.copyfile(src_path+'scripts/fit_glm.py',     python_fit_script)
    
    # Define list of experiments to fit
    manifest = get_manifest()
    manifest.to_csv(manifest_path)
    
    # Define job settings
    job_settings = {'queue': 'braintv',
                    'mem': '15g',
                    'walltime': '2:00:00',
                    'ppn':4,
                    }

    # Make JSON file with parameters
    run_params = {
        'output_dir':output_dir,
        'model_freeze_dir':model_freeze_dir,
        'experiment_output_dir':experiment_output_dir,
        'job_dir':job_dir,
        'json_path':json_path,
        'version':VERSION,
        'creation_time':str(datetime.datetime.now()),
        'user':username,
        'label':label,
        'manifest_path':manifest_path,
        'src_file':python_file_full_path,
        'fit_script':python_fit_script,
        'regularization_lambda':0,  # TODO need to define the regularization strength
        'ophys_experiment_ids':manifest.index.values.tolist(),
        'job_settings':job_settings
    }
    with open(json_path, 'w') as json_file:
        json.dump(run_params, json_file, indent=4)

    # Print Success
    print('Model Successfully Saved, version '+str(VERSION))

def get_manifest():     # TODO need to define manifest
    # Should include ophys_experiment_ids as index, and include ophys_session_ids for each experiment
    return pd.DataFrame(index =DEFAULT_OPHYS_SESSION_IDS)

def fit_experiment(oeid, run_params):
    print(oeid) 
    # Load Data
        # load SDK session object
        # Filter invalid ROIs
        # Add stimulus_presentations_analysis
        # add stimulus_response_df
        # clip gray screen periods off dff_timestamps
    #session = load_data(session,run_params)
    #dff_trace_arry, dff_trace_timestamps = process_data(session)
    
    # Make Design Matrix
    #design = DesignMatrix(dff_trace_timestamps[:-1])
        # Add kernels
    
    # Set up CV splits
    
    # Set up kernels to drop for model selection

    # Iterate over model selections
        # Iterate CV

    # Save Results
    fit = dict()
    filepath = run_params['experiment_output_dir']+str(oeid)+'.pkl' 
    file_temp = open(filepath, 'wb')
    pickle.dump(fit, file_temp)
    file_temp.close()  

def load_data(oeid,run_params):
    '''
        Returns SDK session object
    '''
    cache = BehaviorProjectCache.from_lims(manifest=MANIFEST)
    session = cache.get_session_data(oeid)
    # TODO, update this block to use the new VBA things?
    session_attributes.filter_invalid_rois_inplace(session)
    sdk_utils.add_stimulus_presentations_analysis(session)
    session.stimulus_response_df = rp.stimulus_response_df(rp.stimulus_response_xr(session))
    return session

def process_data(session):
    '''
        Adds stimulus response and extended stimulus information
        clips off gray screen periods
        returns the proper timestamps and dff
    '''
    dff_trace_timestamps = session.ophys_timestamps

    # clip off the grey screen periods
    timestamps_to_use = get_ophys_frames_to_use(session)
    dff_trace_timestamps = dff_trace_timestamps[timestamps_to_use]

    # Get the matrix of dff traces
    dff_trace_arr = get_dff_arr(session, timestamps_to_use)

    return dff_trace_arr, dff_trace_timestamps



######## DEV AFTER HERE

# TODO What are these values for?
#test_timebase = np.arange(1000)
#test_values = np.random.random(1000)


# TODO What does this function do?
def split_time(timebase, subsplits_per_split=10, output_splits=6):
    num_timepoints = len(timebase)
    total_splits = output_splits * subsplits_per_split

    # all the split inds broken into the small subsplits
    split_inds = np.array_split(np.arange(num_timepoints), total_splits)

    # make output splits by combining random subsplits
    random_subsplit_inds = np.random.choice(np.arange(total_splits), size=total_splits, replace=False)
    subsplit_inds_per_split = np.array_split(random_subsplit_inds, output_splits)

    output_split_inds = []
    for ind_output in range(output_splits):
        subsplits_this_split = subsplit_inds_per_split[ind_output]
        inds_this_split = np.concatenate([split_inds[sub_ind] for sub_ind in subsplits_this_split])
        output_split_inds.append(inds_this_split)
    return output_split_inds

class DesignMatrix(object):
    def __init__(self, event_timestamps, intercept=True):
        '''
        A toeplitz-matrix builder for running regression with multiple temporal kernels. 

        Args
            event_timestamps: The actual timestamps for each time bin that will be used in the regression model. 
            intercept: Whether to fit an intercept term.
        '''

        # Add some kernels
        self.X = None
        self.kernel_list = []
        self.labels = []
        self.ind_start = []
        self.ind_stop = []
        self.running_stop = 0
        self.events = {'timestamps':event_timestamps}
        self.include_intercept=intercept
        if self.include_intercept:
            # Add an intercept column by adding a 'kernel' of length 1 starting at every time point.
            # This allows us to do model selection like usual if we want and registers start/stop inds.
            self.add_kernel(np.ones(len(event_timestamps)), 1, 'intercept', 0)

    def kernel_dict(self):
        return {label:kernel for label, kernel in zip(self.labels, self.kernel_list)}

    def get_X(self, kernels=None):
        '''
        Get the design matrix. 

        Args:
            kernels (optional list of kernel string names): which kernels to include (for model selection)
        Returns:
            X (np.array): The design matrix
        '''
        if kernels is None:
            return np.vstack(self.kernel_list)
        else:
            kernel_dict = self.kernel_dict()
            kernels_to_use = []
            for kernel_name in kernels:
                kernels_to_use.append(kernel_dict[kernel_name])
            return np.vstack(kernels_to_use)

    #TODO: Allow kernel length/offset to be specified in actual time if you want
    def add_kernel(self, events, kernel_length, label, offset=0):
        '''
        Add a temporal kernel. 

        Args:
            events (np.array): The timestamps of each event that the kernel will align to. 
            kernel_length (int): NUMBER OF SAMPLES length of the kernel. 
            label (string): Name of the kernel. 
            offset (int) :NUMBER OF SAMPLES offset relative to the events. Negative offsets cause the kernel
                          to overhang before the event
        '''
        #Enforce unique labels
        if label in self.labels:
            raise ValueError('Labels must be unique')

        self.events[label] = events

        this_kernel = toeplitz(events, kernel_length)

        #Pad with zeros, roll offset, and truncate to length
        if offset < 0:
            this_kernel = np.concatenate([np.zeros((this_kernel.shape[0], np.abs(offset))), this_kernel], axis=1)
            this_kernel = np.roll(this_kernel, offset)[:, np.abs(offset):]
        elif offset > 0:
            this_kernel = np.concatenate([this_kernel, np.zeros((this_kernel.shape[0], offset))], axis=1)
            this_kernel = np.roll(this_kernel, offset)[:, :-offset]

        self.kernel_list.append(this_kernel)

        #Keep track of start and stop inds
        self.ind_start.append(self.running_stop)
        self.ind_stop.append(self.running_stop + kernel_length)
        self.running_stop += kernel_length

        #Keep track of labels
        self.labels.append(label)

    def test_get_kernel(self, label):
        kernel_ind = self.labels.index(label)
        kernel_start = self.ind_start[kernel_ind]
        kernel_stop = self.ind_stop[kernel_ind]
        assert np.all(self.get_X[kernel_start:kernel_stop, :] == self.kernel_list[kernel_ind])

    #TODO: I need to define the schema somehow. Perhaps just have a complementary `write_json` method.
    @classmethod
    def from_json(cls, json_path):
        '''
        Make a design matrix from a json file. 
        The json schema has individual timestamps, so need to histogram them. 
        '''
        with open(json_path, 'r') as json_data:
            json_data = json.load(json_data)

        this_design_mat = cls(np.array(json_data['timestamps'])[:-1])
        for kernel_dict in json_data['kernels']:
            events_vec, _ = np.histogram(kernel_dict['times'], bins=json_data['timestamps'])
            this_design_mat.add_kernel(events=events_vec,
                                       label=kernel_dict['label'],
                                       kernel_length=kernel_dict['kernel_length'],
                                       offset=kernel_dict['kernel_offset']
                                       )
        return this_design_mat

def toeplitz(events, kernel_length):
    '''
    Build a toeplitz matrix aligned to events.

    Args:
        events (np.array of 1/0): Array with 1 if the event happened at that time, and 0 otherwise.
        kernel_length (int): How many kernel parameters
    Returns
        np.array, size(len(events), kernel_length) of 1/0
    '''

    total_len = len(events)
    events = np.concatenate([events, np.zeros(kernel_length)])
    arrays_list = [events]
    for i in range(kernel_length-1):
        arrays_list.append(np.roll(events, i+1))
    return np.vstack(arrays_list)[:,:total_len]

def get_ophys_frames_to_use(session):
    '''
    Trims out the grey period at start, end, and the fingerprint.
    Args:
        session (allensdk.brain_observatory.behavior.behavior_ophys_session.BehaviorOphysSession)
    Returns:
        ophys_frames_to_use (np.array of bool): Boolean mask with which ophys frames to use
    '''
    #TODO what does this do when there is a random early omission?
    ophys_frames_to_use = (
        session.ophys_timestamps > session.stimulus_presentations.iloc[0]['start_time']
    ) & (
        session.ophys_timestamps < session.stimulus_presentations.iloc[-1]['stop_time']+0.5
    )
    return ophys_frames_to_use

def get_dff_arr(session, timestamps_to_use):
    '''
    Get the dff traces from a session in xarray format (preserves cell ids and timestamps)
    '''
    all_dff = np.stack(session.dff_traces['dff'].values)
    all_dff_to_use = all_dff[:, timestamps_to_use]

    #Predictors get binned against dff timestamps, so throw the last bin edge
    all_dff_to_use = all_dff_to_use[:, :-1]

    # Return a (n_timepoints, n_cells) array

    dff_trace_timestamps = session.ophys_timestamps
    dff_trace_timestamps = dff_trace_timestamps[timestamps_to_use]

    dff_trace_xr = xr.DataArray(
            data = all_dff_to_use.T,
            dims = ("dff_trace_timestamps", "cell_specimen_id"),
            coords = {
                "dff_trace_timestamps": dff_trace_timestamps[:-1],
                "cell_specimen_id": session.cell_specimen_table.index.values
            }
        )
    return dff_trace_xr


def fit(dff_trace_arr, X):
    '''
    Analytical OLS solution to linear regression. 

    dff_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    '''
    W = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, dff_trace_arr))
    return W

def fit_regularized(dff_trace_arr, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    dff_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization (hyperparameter to tune)
    '''
    W = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, dff_trace_arr))
    return W

def variance_ratio(dff_trace_arr, W, X):
    '''
    dff_trace_arr: (n_timepoints, n_cells)
    W: (n_kernel_params, n_cells)
    X: (n_timepoints, n_kernel_params)
    '''
    Y = X @ W
    var_total = np.var(dff_trace_arr, axis=0) #Total variance in the dff trace for each cell
    var_resid = np.var(dff_trace_arr-Y, axis=0) #Residual variance
    return (var_total - var_resid) / var_total

##### Evaluation/visualization functions (potentially not up to date) below here #######

def all_cells_psth(dff_traces_arr, ophys_timestamps, flash_time_gb, change_events):

    # Get psth for each flash
    all_psth = {}
    for image_name in flash_time_gb.index.levels[0].values:
        times_this_image = flash_time_gb[image_name].values
        dff_frames_this_image = rp.index_of_nearest_value(ophys_timestamps, times_this_image)
        dff_frames_this_image = dff_frames_this_image[
            (len(ophys_timestamps) - dff_frames_this_image > 31)
        ]
        data_this_image = rp.eventlocked_traces(dff_traces_arr, dff_frames_this_image,
                                                0, 30)
        psth_this_image = data_this_image.mean(axis=1)
        all_psth.update({image_name:psth_this_image})

    # Get psth for average change effect
    dff_frames_changes = rp.index_of_nearest_value(ophys_timestamps, change_events)
    dff_frames_changes = dff_frames_changes[
        (len(ophys_timestamps) - dff_frames_changes > 101)
    ]
    data_changes = rp.eventlocked_traces(dff_traces_arr, dff_frames_changes,
                                            0, 100)
    psth_changes = data_changes.mean(axis=1)
    all_psth.update({'change':psth_changes})
    
    return all_psth

def split_filters(W, image_names):
    '''
    W: n_params, n_cells
    image_names: list of strings
    '''

    start=0
    all_filters = {}
    for ind_image, image_name in enumerate(image_names):
        all_filters.update({image_name:W[start:start+30, :]})
        start += 30
    all_filters.update({'change':W[start:]})
    return all_filters

# TODO what does this function do?
def compare_filter_and_psth(ind_cell, dff_traces_arr, ophys_timestamps, flash_time_gb, change_events, all_W):
    all_psths = all_cells_psth(dff_traces_arr.T, ophys_timestamps, flash_time_gb, change_events)
    image_names = flash_time_gb.index.levels[0].values
    all_filters_mean = split_filters(all_W.mean(axis=2), image_names)
    all_filters_std = split_filters(all_W.std(axis=2), image_names)
    plt.figure()
    num_images = len(image_names)
    for ind_image, image_name in enumerate(image_names):
        plt.subplot(num_images+1, 1, ind_image+1)
        this_psth = all_psths[image_name][:,ind_cell]
        this_filter_mean = all_filters_mean[image_name][:,ind_cell]
        this_filter_std = all_filters_std[image_name][:,ind_cell]
        plt.plot(this_psth, 'k-')
        plt.plot(this_filter_mean, 'r--')
        plt.plot(this_filter_mean+this_filter_std, 'r--', alpha=0.5)
        plt.plot(this_filter_mean-this_filter_std, 'r--', alpha=0.5)
        plt.ylabel(image_name)

    # Average change
    plt.subplot(num_images+1, 1, ind_image+2)
    this_psth = all_psths['change'][:,ind_cell]
    this_filter_mean = all_filters_mean['change'][:,ind_cell]
    this_filter_std = all_filters_std['change'][:,ind_cell]
    plt.plot(this_psth, 'k-')
    plt.plot(this_filter_mean, 'r--')
    plt.plot(this_filter_mean+this_filter_std, 'r--', alpha=0.5)
    plt.plot(this_filter_mean-this_filter_std, 'r--', alpha=0.5)
    plt.ylabel('change')

def compare_all_filters_and_psth(ind_cell, dff_traces_arr, ophys_timestamps, flash_time_gb, change_events, all_W):
    all_psths = all_cells_psth(dff_traces_arr.T, ophys_timestamps, flash_time_gb, change_events)
    image_names = flash_time_gb.index.levels[0].values
    #  all_filters_mean = split_filters(all_W.mean(axis=2), image_names)
    #  all_filters_std = split_filters(all_W.std(axis=2), image_names)
    plt.figure()
    num_images = len(image_names)
    for ind_split in range(6):
        filters_this_split = split_filters(all_W[:, :, ind_split], image_names)
        for ind_image, image_name in enumerate(image_names):
            plt.subplot(num_images+1, 1, ind_image+1)
            this_psth = all_psths[image_name][:,ind_cell]

            this_filter = filters_this_split[image_name][:,ind_cell]
            plt.plot(this_psth, 'k-')
            plt.plot(this_filter, 'r--')
            plt.ylabel(image_name)

def plot_filters(w, image_names):
    num_subplots = len(image_names)+2
    fig, axes = plt.subplots(3, 1)
    start=0
    for ind_image, image_name in enumerate(image_names):
        #  plt.subplot(num_subplots, 1, ind_image+1)
        axes[0].plot(np.linspace(0, 1-1/31, 30), w[start:start+30], label=image_name)
        #  plt.title(image_name)
        #  plt.plot(w[start:start+30])
        start += 30
    axes[0].legend()

    #  plt.subplot(num_subplots, 1, num_subplots)
    axes[1].plot(np.arange(0, 100/31, 1/31), w[start:start+100], label='change')
    start += 100
    axes[1].legend()
    
    axes[2].plot(np.arange(0, 100/31, 1/31), w[start:], label='reward')
    axes[2].legend()

def pref_stim(session, cell_ind):
    csid = session.dff_traces.iloc[cell_ind].name
    stim_id = session.stimulus_response_df.query('cell_specimen_id==@csid and pref_stim').iloc[0]['stimulus_presentations_id']
    return session.stimulus_presentations.loc[stim_id]['image_name']

def fit_and_evaluate(session, ind_cell, X):
    dff_trace, w = fit_cell(session, ind_cell, X)
    plot_filters(w, image_names)
    var_explained = variance_ratio(dff_trace, w, X)
    actual_pref = pref_stim(session, ind_cell)
    plt.suptitle('var explained: {}\npref stim: {}'.format(var_explained, actual_pref))
    plt.tight_layout()

def pref_image_filter(w, image_names):
    start=0
    filter_sums = np.empty(len(image_names))
    for ind_image, image_name in enumerate(image_names):
        filter_sums[ind_image] = np.sum(w[start:start+30])
        start += 30
    return image_names[np.argmax(filter_sums)]

def plot_cv_train_test(cv_train, cv_test, lambda_vals):
    train_mean = cv_train.mean((0, 1))
    test_mean = cv_test.mean((0, 1))
    fig, ax1 = plt.subplots()
    ax1.plot(test_mean, color='r')
    ax1.set_ylabel('cv var explained, test')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(train_mean, 'k')
    ax2.set_ylabel('cv var explained, train')
    ax2.tick_params(axis='y', labelcolor='k')

    ax1.set_xticks(range(len(lambda_vals)))
    ax1.set_xticklabels(np.round(lambda_vals, 2), rotation='vertical')
    ax1.set_xlabel('lambdas')
