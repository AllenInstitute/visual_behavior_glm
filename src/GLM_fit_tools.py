import os
import pwd
import json
import shutil
import pickle
import xarray as xr
import numpy as np
import pandas as pd
import scipy 
import datetime
from tqdm import tqdm
from copy import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache
from visual_behavior.translator.allensdk_sessions import sdk_utils
from visual_behavior.translator.allensdk_sessions import session_attributes
from visual_behavior.ophys.response_analysis import response_processing as rp
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.data_access.loading as loading

OUTPUT_DIR_BASE = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'

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

def check_run_fits(VERSION):
    '''
        Returns the experiment table for this model version with a column 'GLM_fit' 
        appended that is a bool of whether the output pkl file exists for that 
        experiment. It does not try to load the file, or determine if the fit is good.
    '''
    run_params = load_run_json(VERSION)
    experiment_table = pd.read_csv(run_params['experiment_table_path']).reset_index(drop=True).set_index('ophys_experiment_id')
    experiment_table['GLM_fit'] = False
    for index, oeid in enumerate(experiment_table.index.values):
        filename = run_params['experiment_output_dir']+str(oeid)+".pkl" 
        experiment_table.at[oeid, 'GLM_fit'] = os.path.isfile(filename) 
    return experiment_table

def make_run_json(VERSION,label='',username=None,src_path=None, TESTING=False):
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
        <src_path>  path to repo home. Will throw an error if not passed in 
        <TESTING>   if true, will only include 5 sessions in the experiment list
    '''

    # Make directory, will throw an error if already exists
    output_dir              = OUTPUT_DIR_BASE + 'v_'+str(VERSION) +'/'
    model_freeze_dir        = output_dir +'frozen_model_files/'
    experiment_output_dir   = output_dir +'experiment_model_files/'
    manifest_dir            = output_dir +'manifest/'
    manifest                = output_dir +'manifest/manifest.json'
    job_dir                 = output_dir +'log_files/'
    json_path               = output_dir +'run_params.json'
    experiment_table_path   = output_dir +'experiment_table_v_'+str(VERSION)+'.csv'
    beh_model_dir           = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/model_output/'
    os.mkdir(output_dir)
    os.mkdir(model_freeze_dir)
    os.mkdir(experiment_output_dir)
    os.mkdir(job_dir)
    os.mkdir(manifest_dir)
    
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
        raise Exception('You need to provide a path to the model source code')

    shutil.copyfile(src_path+'src/GLM_fit_tools.py',   python_file_full_path)
    shutil.copyfile(src_path+'scripts/fit_glm.py',     python_fit_script)
    
    # Define list of experiments to fit
    experiment_table = get_experiment_table()
    if TESTING:
        experiment_table = experiment_table.query('project_code == "VisualBehavior"').tail(5)
    experiment_table.to_csv(experiment_table_path)
    
    # Define job settings
    job_settings = {'queue': 'braintv',
                    'mem': '15g',
                    'walltime': '2:00:00',
                    'ppn':4,
                    }

    # Define Kernels
    kernels_orig = {
        'intercept':    {'event':'intercept',   'type':'continuous',    'length':0,     'offset':0},
        'time':         {'event':'time',        'type':'continuous',    'length':0,     'offset':0},
        #'licks':        {'event':'licks',       'type':'discrete',      'length':1.5,   'offset':-0.5},
        'pre_licks':    {'event':'licks',       'type':'discrete',      'length':0.5,   'offset':-0.5},
        'post_licks':   {'event':'licks',       'type':'discrete',      'length':1,     'offset':0},
        'rewards':      {'event':'rewards',     'type':'discrete',      'length':4,     'offset':-0.5},
        'change':       {'event':'change',      'type':'discrete',      'length':4,     'offset':0},
        'omissions':    {'event':'omissions',   'type':'discrete',      'length':2,     'offset':0},
        'each-image':   {'event':'each-image',  'type':'discrete',      'length':1.5,  'offset':0},
        'running':      {'event':'running',     'type':'continuous',    'length':2,     'offset':-1},
        #'population_mean':{'event':'population_mean','type':'continuous','length':.5,   'offset':-.25},
        #'Population_Activity_PC1':        {'event':'Population_Activity_PC1',       'type':'continuous',    'length':.5,    'offset':-.25},
        'beh_model':    {'event':'beh_model',   'type':'continuous',    'length':.5,    'offset':-.25},
        'pupil':        {'event':'pupil',       'type':'continuous',    'length':2,     'offset':-1}
    }
    kernels = process_kernels(copy(kernels_orig))
    dropouts = define_dropouts(kernels,kernels_orig)

    # Make JSON file with parameters
    run_params = {
        'output_dir':output_dir,                        
        'model_freeze_dir':model_freeze_dir,            
        'experiment_output_dir':experiment_output_dir,
        'job_dir':job_dir,
        'manifest':manifest,
        'json_path':json_path,
        'beh_model_dir':beh_model_dir,
        'version':VERSION,
        'creation_time':str(datetime.datetime.now()),
        'user':username,
        'label':label,
        'experiment_table_path':experiment_table_path,
        'src_file':python_file_full_path,
        'fit_script':python_fit_script,
        'L2_fixed_lambda':70,       # This value is used if L2_use_fixed_value
        'L2_use_fixed_value':False, # If False, find L2 values over grid
        'L2_use_avg_value':True,    # If True, uses the average value over grid
        'L2_grid_range':[.1, 500],
        'L2_grid_num': 20,
        'ophys_experiment_ids':experiment_table.index.values.tolist(),
        'job_settings':job_settings,
        'kernels':kernels,
        'dropouts':dropouts,
        'CV_splits':5,
        'CV_subsplits':10
    }
    with open(json_path, 'w') as json_file:
        json.dump(run_params, json_file, indent=4)

    # Print Success
    print('Model Successfully Saved, version '+str(VERSION))

def get_experiment_table(require_model_outputs = True):
    """
    get a list of filtered experiments and associated attributes
    returns only experiments that have relevant project codes and have passed QC

    Keyword arguments:
    require_model_outputs (bool) -- if True, limits returned experiments to those that have been fit with behavior model
    """
    experiments_table = loading.get_filtered_ophys_experiment_table()
    if require_model_outputs:
        return experiments_table.query('model_outputs_available == True')
    else:
        return experiments_table

def fit_experiment(oeid, run_params,NO_DROPOUTS=False):
    print("Fitting ophys_experiment_id: "+str(oeid)) 

    # Load Data
    print('Loading data')
    session = load_data(oeid)

    # Processing df/f data
    print('Processing df/f data')
    fit= dict()
    fit['dff_trace_arr'] = process_data(session)
    fit = annotate_dff(fit)
    fit['ophys_frame_rate'] = session.dataset.metadata['ophys_frame_rate'] 

    # Make Design Matrix
    print('Build Design Matrix')
    design = DesignMatrix(fit) 

    # Add kernels
    design = add_kernels(design, run_params, session, fit) 

    # Set up CV splits
    print('Setting up CV')
    fit['splits'] = split_time(fit['dff_trace_timestamps'], output_splits=run_params['CV_splits'], subsplits_per_split=run_params['CV_subsplits'])
    fit['ridge_splits'] = split_time(fit['dff_trace_timestamps'], output_splits=run_params['CV_splits'], subsplits_per_split=run_params['CV_subsplits'])

    # Determine Regularization Strength
    print('Evaluating Regularization values')
    fit = evaluate_ridge(fit, design, run_params)

    # Set up kernels to drop for model selection
    print('Setting up model selection dropout')
    fit['dropouts'] = copy(run_params['dropouts'])
    if NO_DROPOUTS:
        fit['dropouts'] = {'Full':copy(fit['dropouts']['Full'])}

    # Iterate over model selections
    print('Iterating over model selection')
    fit = evaluate_models(fit, design, run_params)

    # Save Results
    print('Saving results')
    filepath = run_params['experiment_output_dir']+str(oeid)+'.pkl' 
    file_temp = open(filepath, 'wb')
    pickle.dump(fit, file_temp)
    file_temp.close()  
    
    # Save Design Matrix
    print('Saving Design Matrix')  
    sparse_X = scipy.sparse.csc_matrix(design.get_X().values)
    filepath = run_params['experiment_output_dir']+'X_sparse_csc_'+str(oeid)+'.npz'
    scipy.sparse.save_npz(filepath, sparse_X)

    # Save Event Table
    print('Saving Events Table')
    filepath = run_params['experiment_output_dir']+'event_times_'+str(oeid)+'.h5'
    pd.DataFrame(design.events).to_hdf(filepath,key='df')

    print('Finished') 
    return session, fit, design

def process_kernels(kernels):
    '''
        Replaces the 'each-image' kernel with each individual image (not omissions), with the same parameters
    '''
    if ('each-image' in kernels) & ('any-image' in kernels):
        raise Exception('Including both each-image and any-image kernels makes the model unstable')
    if 'each-image' in kernels:
        specs = kernels.pop('each-image')
        for index, val in enumerate(range(0,8)):
            kernels['image'+str(val)] = copy(specs)
            kernels['image'+str(val)]['event'] = 'image'+str(val)
    if 'beh_model' in kernels:
        specs = kernels.pop('beh_model')
        weight_names = ['bias','task0','omissions1','timing1D']
        for index, val in enumerate(weight_names):
            kernels['model_'+str(val)] = copy(specs)
            kernels['model_'+str(val)]['event'] = 'model_'+str(val)
    return kernels
 
def define_dropouts(kernels,kernel_definitions):
    '''
        Creates a dropout dictionary. Each key is the label for the dropout, and the value is a list of kernels to include
        Creates a dropout for each kernel by removing just that kernel.
        In addition creates a 'visual' dropout by removing 'any-image' and 'each-image' and 'omissions'
        If 'each-image' is in the kernel_definitions, then creates a dropout 'each-image' with all 8 images removed
    '''
    # Remove each kernel one-by-one
    dropouts = {'Full': {'kernels':list(kernels.keys())}}
    for kernel in kernels.keys():
        dropouts[kernel]={'kernels':list(kernels.keys())}
        dropouts[kernel]['kernels'].remove(kernel)

    # Removes all individual image kernels
    if 'each-image' in kernel_definitions:
        dropouts['all-images'] = {'kernels':list(kernels.keys())}
        for i in range(0,8):
            dropouts['all-images']['kernels'].remove('image'+str(i))

    # Removes all Stimulus Kernels
    if ('each-image' in kernel_definitions) or ('any-image' in kernel_definitions) or ('omissions' in kernel_definitions):
        dropouts['visual'] = {'kernels':list(kernels.keys())}
        if 'each-image' in kernel_definitions:
            for i in range(0,8):
                dropouts['visual']['kernels'].remove('image'+str(i))
        if 'omissions' in kernel_definitions:
            dropouts['visual']['kernels'].remove('omissions')
        if 'any-image' in kernel_definitions:
            dropouts['visual']['kernels'].remove('any-image')

    # Remove all behavior model kernels
    if 'beh_model' in kernel_definitions:
        dropouts['beh_model'] = {'kernels':list(kernels.keys())}
        dropouts['beh_model']['kernels'].remove('model_bias')
        dropouts['beh_model']['kernels'].remove('model_task0')
        dropouts['beh_model']['kernels'].remove('model_timing1D')
        dropouts['beh_model']['kernels'].remove('model_omissions1')

    return dropouts

def evaluate_ridge(fit, design,run_params):
    '''
        Finds the best L2 value by fitting the model on a grid of L2 values and reporting training/test error
    
        fit, model dictionary
        design, design matrix
        run_params, dictionary of parameters, which needs to include:
            L2_use_fixed_value, if True, skips this step and uses a hard coded value given by L2_fixed_lambda
            L2_grid_range, a min/max L2 value to use
            L2_grid_num, the number of log-spaced points to use in the grid range

        returns fit, with the values added:
            L2_grid,    the L2 grid evaluated
            avg_regularization, the average optimal L2 value, or the fixed value
            cell_regularization, the optimal L2 value for each cell
    '''
    if run_params['L2_use_fixed_value']:
        print('Using a hard-coded regularization value')
        fit['avg_regularization'] = run_params['L2_fixed_lambda']
    else:
        print('Evaluating a grid of regularization values')
        fit['L2_grid'] = np.concatenate([[0],np.geomspace(run_params['L2_grid_range'][0], run_params['L2_grid_range'][1],num = run_params['L2_grid_num'])])
        train_cv = np.empty((fit['dff_trace_arr'].shape[1], len(fit['L2_grid']))) 
        test_cv  = np.empty((fit['dff_trace_arr'].shape[1], len(fit['L2_grid']))) 
        X = design.get_X()
      
        # Iterate over L2 Values 
        for L2_index, L2_value in enumerate(fit['L2_grid']):
            cv_var_train = np.empty((fit['dff_trace_arr'].shape[1], len(fit['ridge_splits'])))
            cv_var_test = np.empty((fit['dff_trace_arr'].shape[1], len(fit['ridge_splits'])))

            # Iterate over CV splits
            for split_index, test_split in tqdm(enumerate(fit['ridge_splits']), total=len(fit['ridge_splits']), desc='    Fitting L2, {}'.format(L2_value)):
                train_split = np.concatenate([split for i, split in enumerate(fit['ridge_splits']) if i!=split_index])
                X_test  = X[test_split,:]
                X_train = X[train_split,:]
                dff_train = fit['dff_trace_arr'][train_split,:]
                dff_test  = fit['dff_trace_arr'][test_split,:]
                W = fit_regularized(dff_train, X_train, L2_value)
                cv_var_train[:,split_index] = variance_ratio(dff_train, W, X_train)
                cv_var_test[:,split_index]  = variance_ratio(dff_test, W, X_test)

            train_cv[:,L2_index] = np.mean(cv_var_train,1)
            test_cv[:,L2_index]  = np.mean(cv_var_test,1)

        fit['avg_regularization'] = np.mean([fit['L2_grid'][x] for x in np.argmax(test_cv,1)])      
        fit['cell_regularization'] = [fit['L2_grid'][x] for x in np.argmax(test_cv,1)]     
        fit['L2_test_cv'] = test_cv
        fit['L2_train_cv'] = train_cv
    return fit

def evaluate_models(fit, design, run_params):
    '''
        Evaluates the model selections across all dropouts using either the single L2 value, or each cell's optimal value

    '''
    if run_params['L2_use_avg_value'] or run_params['L2_use_fixed_value']:
        print('Using a constant regularization value across all cells')
        return evaluate_models_same_ridge(fit,design, run_params)
    else:
        print('Using an optimized regularization value for each cell')
        return evaluate_models_different_ridge(fit,design,run_params)

def evaluate_models_different_ridge(fit,design,run_params):
    '''
        Fits and evaluates each model defined in fit['dropouts']
    
        For each model, it creates the design matrix, finds the optimal weights, and saves the variance explained. 
            It does this for the entire dataset as test and train. As well as CV, saving each test/train split
    '''
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])

        # Iterate CV
        cv_var_train    = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_var_test     = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_weights      = np.empty((np.shape(X)[1], fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        all_weights     = np.empty((np.shape(X)[1], fit['dff_trace_arr'].shape[1]))
        all_var_explain = np.empty((fit['dff_trace_arr'].shape[1]))
        all_prediction  = np.empty(fit['dff_trace_arr'].shape)

        for cell_index, cell_value in tqdm(enumerate(fit['dff_trace_arr']['cell_specimen_id'].values),total=len(fit['dff_trace_arr']['cell_specimen_id'].values),desc='   Fitting Cells'):

            dff = fit['dff_trace_arr'][:,cell_index]
            Wall = fit_cell_regularized(dff, X,fit['cell_regularization'][cell_index])     
            var_explain = variance_ratio(dff, Wall,X)
            all_weights[:,cell_index] = Wall
            all_var_explain[cell_index] = var_explain
            all_prediction[:,cell_index] = X.values @ Wall.values

            for index, test_split in enumerate(fit['splits']):
                train_split = np.concatenate([split for i, split in enumerate(fit['splits']) if i!=index])
                X_test = X[test_split,:]
                X_train = X[train_split,:]
                dff_train = fit['dff_trace_arr'][train_split,cell_index]
                dff_test = fit['dff_trace_arr'][test_split,cell_index]
                W = fit_cell_regularized(dff_train, X_train, fit['cell_regularization'][cell_index])
                cv_var_train[cell_index,index] = variance_ratio(dff_train, W, X_train)
                cv_var_test[cell_index,index] = variance_ratio(dff_test, W, X_test)
                cv_weights[:,cell_index,index] = W 

        fit['dropouts'][model_label]['train_weights'] = all_weights
        fit['dropouts'][model_label]['train_variance_explained']=all_var_explain
        fit['dropouts'][model_label]['full_model_train_prediction'] =  all_prediction
        fit['dropouts'][model_label]['cv_weights'] = cv_weights
        fit['dropouts'][model_label]['cv_var_train'] = cv_var_train
        fit['dropouts'][model_label]['cv_var_test'] = cv_var_test

    return fit 


def evaluate_models_same_ridge(fit, design, run_params):
    '''
        Fits and evaluates each model defined in fit['dropouts']
    
        For each model, it creates the design matrix, finds the optimal weights, and saves the variance explained. 
            It does this for the entire dataset as test and train. As well as CV, saving each test/train split
    '''
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])

        dff = fit['dff_trace_arr']
        Wall = fit_regularized(dff, X,fit['avg_regularization'])     
        var_explain = variance_ratio(dff, Wall,X)
        fit['dropouts'][model_label]['weights'] = Wall
        fit['dropouts'][model_label]['variance_explained']=var_explain
        fit['dropouts'][model_label]['full_model_train_prediction'] =  X.values @ Wall.values

        # Iterate CV
        cv_var_train = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_var_test = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_weights = np.empty((np.shape(Wall)[0], np.shape(Wall)[1], len(fit['splits'])))

        for index, test_split in tqdm(enumerate(fit['splits']), total=len(fit['splits']), desc='    Fitting model, {}'.format(model_label)):
            train_split = np.concatenate([split for i, split in enumerate(fit['splits']) if i!=index])
            X_test = X[test_split,:]
            X_train = X[train_split,:]
            dff_train = fit['dff_trace_arr'][train_split,:]
            dff_test = fit['dff_trace_arr'][test_split,:]
            W = fit_regularized(dff_train, X_train, fit['avg_regularization'])
            cv_var_train[:,index] = variance_ratio(dff_train, W, X_train)
            cv_var_test[:,index] = variance_ratio(dff_test, W, X_test)
            cv_weights[:,:,index] = W 

        fit['dropouts'][model_label]['cv_weights'] = cv_weights
        fit['dropouts'][model_label]['cv_var_train'] = cv_var_train
        fit['dropouts'][model_label]['cv_var_test'] = cv_var_test

    return fit 

def build_dataframe_from_dropouts(fit):
    '''
        Returns a dataframe with 
        Index: Cell specimen id
        Columns: Average (across CV folds) variance explained on the test and training sets for each model defined in fit['dropouts']
    '''
    cellids = fit['dff_trace_arr']['cell_specimen_id'].values
    results = pd.DataFrame(index=pd.Index(cellids, name='cell_specimen_id'))
    for model_label in fit['dropouts'].keys():
        results[model_label+"_avg_cv_var_train"] = np.mean(fit['dropouts'][model_label]['cv_var_train'],1)
        results[model_label+"_avg_cv_var_test"]  = np.mean(fit['dropouts'][model_label]['cv_var_test'],1)
    return results

def L2_report(fit):
    '''
        Evaluates how well the L2 grid worked. Plots the train/test error across L2 Values to visually see the best value
        Plots the CV_test for each L2 value
    
    '''
    plt.figure()
    plt.plot(fit['L2_grid'], np.mean(fit['L2_train_cv'],0), 'b-')
    plt.plot(fit['L2_grid'], np.mean(fit['L2_test_cv'],0), 'r-')
    plt.gca().set_xscale('log')
    plt.ylabel('Session avg test CV')
    plt.xlabel('L2 Strength')
    plt.axvline(fit['avg_regularization'], color='k', linestyle='--', alpha = 0.5)
    plt.ylim(0,.15) 

    cellids = fit['dff_trace_arr']['cell_specimen_id'].values
    results = pd.DataFrame(index=cellids)
    for index, value in enumerate(fit['L2_grid']):
        results["cv_train_"+str(index)] = fit['L2_train_cv'][:,index]
        results["cv_test_"+str(index)]  = fit['L2_test_cv'][:,index]
    results.plot.scatter('cv_test_1','cv_test_'+str(index))
    plt.plot([0,1],[0,1],'k--')
    return results
 
def load_data(oeid, dataframe_format='wide'):
    '''
        Returns Visual Behavior ResponseAnalysis object
        Allen SDK dataset is an attribute of this object (session.dataset)
        Keyword arguments:
            oeid (int) -- ophys_experiment_id
            dataframe_format (str) -- whether the response dataframes should be in 'wide' or 'tidy'/'long' formats (default = 'wide')
                                      'wide' format is one row per stimulus/cell, with all timestamps and dff traces in a single cell
                                      'tidy' or 'long' format has one row per timepoint (this format conforms to seaborn standards)
    '''
    dataset = loading.get_ophys_dataset(oeid, include_invalid_rois=False)
    session = ResponseAnalysis(
        dataset, 
        overwrite_analysis_files=False, 
        use_extended_stimulus_presentations=True, 
        dataframe_format = dataframe_format
    ) 
    return session

def process_data(session,ignore_errors=False):
    '''
    Processes dff traces by trimming off portions of recording session outside of the task period. These include:
        * a ~5 minute gray screen period before the task begins
        * a ~5 minute gray screen period after the task ends
        * a 5-10 minute movie following the second gray screen period
    
    input -- session object

    returns -- an xarray of of deltaF/F traces with dimensions [timestamps, cell_specimen_ids]
    '''

    # clip off the grey screen periods
    dff_trace_timestamps = session.ophys_timestamps
    timestamps_to_use = get_ophys_frames_to_use(session)

    # Get the matrix of dff traces
    dff_trace_arr = get_dff_arr(session, timestamps_to_use)

    # some assert statements to ensure that dimensions are correct
    assert np.sum(timestamps_to_use) == len(dff_trace_arr['dff_trace_timestamps'].values), 'length of `timestamps_to_use` must match length of `dff_trace_timestamps` in `dff_trace_arr`'
    assert np.sum(timestamps_to_use) == dff_trace_arr.values.shape[0], 'length of `timestamps_to_use` must match 0th dimension of `dff_trace_arr`'
    assert len(session.cell_specimen_table.query('valid_roi == True')) == dff_trace_arr.values.shape[1], 'number of valid ROIs must match 1st dimension of `dff_trace_arr`'

    return dff_trace_arr

def annotate_dff(fit):
    fit['dff_trace_timestamps'] = fit['dff_trace_arr']['dff_trace_timestamps'].values
    fit['dff_trace_bins'] = np.concatenate([fit['dff_trace_timestamps'],[fit['dff_trace_timestamps'][-1]+np.mean(np.diff(fit['dff_trace_timestamps']))]])  
    return fit

def add_kernels(design, run_params,session, fit):
    '''
        Iterates through the kernels in run_params['kernels'] and adds
        each to the design matrix
        Each kernel must have fields:
            offset:
            length:
    
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model
    '''
    for kernel_name in run_params['kernels']:
        if run_params['kernels'][kernel_name]['type'] == 'discrete':
            design = add_discrete_kernel_by_label(kernel_name, design, run_params, session, fit)
        else:
            design = add_continuous_kernel_by_label(kernel_name, design, run_params, session, fit)   
    return design

def add_continuous_kernel_by_label(kernel_name, design, run_params, session,fit):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name          <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model       
    ''' 
    print('    Adding kernel: '+kernel_name)
    event = run_params['kernels'][kernel_name]['event']
    if event == 'intercept':
        timeseries = np.ones(len(fit['dff_trace_timestamps']))
    elif event == 'time':
        timeseries = np.array(range(1,len(fit['dff_trace_timestamps'])+1))
        timeseries = timeseries/len(timeseries)
    elif event == 'running':
        running_df = session.dataset.running_speed
        running_df = running_df.rename(columns={'speed':'values'})
        timeseries = interpolate_to_dff_timestamps(fit,running_df)['values'].values
    elif event == 'population_mean':
        timeseries = np.mean(fit['dff_trace_arr'],1).values
    elif event == 'Population_Activity_PC1':
        pca = PCA()
        pca.fit(fit['dff_trace_arr'].values)
        dff_pca = pca.transform(fit['dff_trace_arr'].values)
        timeseries = dff_pca[:,0]
    elif (len(event) > 6) & ( event[0:6] == 'model_'):
        bsid = session.dataset.metadata['behavior_session_id']
        weight_name = event[6:]
        weight = get_model_weight(bsid, weight_name, run_params)
        weight_df = pd.DataFrame()
        weight_df['timestamps'] = session.dataset.stimulus_presentations.start_time.values
        weight_df['values'] = weight.values
        timeseries = interpolate_to_dff_timestamps(fit, weight_df)
        timeseries['values'].fillna(method='ffill',inplace=True) # TODO investigate where these NaNs come from
        timeseries = timeseries['values'].values
    elif event == 'pupil':
        pupil_df = session.dataset.eye_tracking
        pupil_df = pupil_df.rename(columns={'time':'timestamps','pupil_area':'values'})
        timeseries = interpolate_to_dff_timestamps(fit, pupil_df)
        timeseries['values'].fillna(method='ffill',inplace=True)
        timeseries['values'].fillna(method='bfill',inplace=True)
        timeseries = timeseries['values'].values
    else:
        raise Exception('Could not resolve kernel label')

    #assert length of values is same as length of timestamps
    assert len(timeseries) == fit['dff_trace_arr'].values.shape[0], 'Length of continuous regressor must match length of dff_trace_timestamps'

    design.add_kernel(timeseries, run_params['kernels'][kernel_name]['length'], kernel_name, offset=run_params['kernels'][kernel_name]['offset'])   
    return design

def add_discrete_kernel_by_label(kernel_name,design, run_params,session,fit):
    '''
        Adds the kernel specified by <kernel_name> to the design matrix
        kernel_name     <str> the label for this kernel, will raise an error if not implemented
        design          the design matrix for this model
        run_params      the run_json for this model
        session         the SDK session object for this experiment
        fit             the fit object for this model       
    ''' 
    print('    Adding kernel: '+kernel_name)
    event = run_params['kernels'][kernel_name]['event']
    if event == 'licks':
        event_times = session.dataset.licks['timestamps'].values
    elif event == 'rewards':
        event_times = session.dataset.rewards['timestamps'].values
    elif event == 'change':
        event_times = session.dataset.trials.query('go')['change_time'].values
        event_times = event_times[~np.isnan(event_times)]
    elif event == 'any-image':
        event_times = session.dataset.stimulus_presentations.query('not omitted')['start_time'].values
    elif event == 'omissions':
        event_times = session.dataset.stimulus_presentations.query('omitted')['start_time'].values
    elif (len(event)>5) & (event[0:5] == 'image'):
        event_times = session.dataset.stimulus_presentations.query('image_index == @event[-1]')['start_time'].values
    else:
        raise Exception('Could not resolve kernel label')
    events_vec, timestamps = np.histogram(event_times, bins=fit['dff_trace_bins'])
    design.add_kernel(events_vec, run_params['kernels'][kernel_name]['length'], kernel_name, offset=run_params['kernels'][kernel_name]['offset'])   
    return design

class DesignMatrix(object):
    def __init__(self, fit_dict):
        '''
        A toeplitz-matrix builder for running regression with multiple temporal kernels. 

        Args
            fit_dict, a dictionary with:
                event_timestamps: The actual timestamps for each time bin that will be used in the regression model. 
                ophys_frame_rate: the number of ophys timestamps per second
        '''

        # Add some kernels
        self.X = None
        self.kernel_dict = {}
        self.running_stop = 0
        self.events = {'timestamps':fit_dict['dff_trace_timestamps']}
        self.ophys_frame_rate = fit_dict['ophys_frame_rate']

    def make_labels(self, label, num_weights,offset, length): 
        base = [label] * num_weights 
        numbers = [str(x) for x in np.array(range(0,length+1))+offset]
        return [x[0] + '_'+ x[1] for x in zip(base, numbers)]

    def get_X(self, kernels=None):
        '''
        Get the design matrix. 

        Args:
            kernels (optional list of kernel string names): which kernels to include (for model selection)
        Returns:
            X (np.array): The design matrix
        '''
        if kernels is None:
            kernels = self.kernel_dict.keys()

        kernels_to_use = []
        param_labels = []
        for kernel_name in kernels:
            kernels_to_use.append(self.kernel_dict[kernel_name]['kernel'])
            param_labels.append(self.make_labels(   kernel_name, 
                                                    np.shape(self.kernel_dict[kernel_name]['kernel'])[0], 
                                                    self.kernel_dict[kernel_name]['offset_samples'],
                                                    self.kernel_dict[kernel_name]['kernel_length_samples'] ))

        X = np.vstack(kernels_to_use) 
        x_labels = np.hstack(param_labels)

        assert np.shape(X)[0] == np.shape(x_labels)[0], 'Weight Matrix must have the same length as the weight labels'

        X_array = xr.DataArray(
            X, 
            dims =('weights','timestamps'), 
            coords = {  'weights':x_labels, 
                        'timestamps':self.events['timestamps']}
            )
        return X_array.T

    def add_kernel(self, events, kernel_length, label, offset=0):
        '''
        Add a temporal kernel. 

        Args:
            events (np.array): The timestamps of each event that the kernel will align to. 
            kernel_length (int): length of the kernel (in SECONDS). 
            label (string): Name of the kernel. 
            offset (int) :offset relative to the events. Negative offsets cause the kernel
                          to overhang before the event (in SECONDS)
        '''
        #Enforce unique labels
        if label in self.kernel_dict.keys():
            raise ValueError('Labels must be unique')

        self.events[label] = events

        # CONVERT kernel_length to kernel_length_samples
        if kernel_length == 0:
            kernel_length_samples = 1
        else:
            kernel_length_samples = int(np.ceil(self.ophys_frame_rate*kernel_length)) 

        # CONVERT offset to offset_samples
        offset_samples = int(np.floor(self.ophys_frame_rate*offset))

        this_kernel = toeplitz(events, kernel_length_samples)

        #Pad with zeros, roll offset_samples, and truncate to length
        if offset_samples < 0:
            this_kernel = np.concatenate([np.zeros((this_kernel.shape[0], np.abs(offset_samples))), this_kernel], axis=1)
            this_kernel = np.roll(this_kernel, offset_samples)[:, np.abs(offset_samples):]
        elif offset_samples > 0:
            this_kernel = np.concatenate([this_kernel, np.zeros((this_kernel.shape[0], offset_samples))], axis=1)
            this_kernel = np.roll(this_kernel, offset_samples)[:, :-offset_samples]

        self.kernel_dict[label] = {
            'kernel':this_kernel,
            'kernel_length_samples':kernel_length_samples,
            'offset_samples':offset_samples,
            'kernel_length_seconds':kernel_length,
            'offset_seconds':offset,
            'ind_start':self.running_stop,
            'ind_stop':self.running_stop+kernel_length_samples
            }
        self.running_stop += kernel_length_samples


def split_time(timebase, subsplits_per_split=10, output_splits=6):
    '''
        Defines the timepoints for each cross validation split
        timebase        vector for timestamps
        output_splits   number of cross validation splits
        subsplits_per_split     each cv split will be composed of this many continuous blocks

        We have to compose each CV split of a bunch of subsplits to prevent issues  
        time in session from distorting each CV split
    
        returns:
        a list of CV splits. For each list element, a list of the timestamps in that CV split
    '''
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

def toeplitz(events, kernel_length_samples):
    '''
    Build a toeplitz matrix aligned to events.

    Args:
        events (np.array of 1/0): Array with 1 if the event happened at that time, and 0 otherwise.
        kernel_length_samples (int): How many kernel parameters
    Returns
        np.array, size(len(events), kernel_length_samples) of 1/0
    '''

    total_len = len(events)
    events = np.concatenate([events, np.zeros(kernel_length_samples)])
    arrays_list = [events]
    for i in range(kernel_length_samples-1):
        arrays_list.append(np.roll(events, i+1))
    return np.vstack(arrays_list)[:,:total_len]

def get_ophys_frames_to_use(session, end_buffer=0.5,stim_dur = 0.25):
    '''
    Trims out the grey period at start, end, and the fingerprint.
    Args:
        session (allensdk.brain_observatory.behavior.behavior_ophys_session.BehaviorOphysSession)
        end_buffer (float): duration in seconds to extend beyond end of last stimulus presentation (default = 0.5)
        stim_dur (float): duration in seconds of stimulus presentations
    Returns:
        ophys_frames_to_use (np.array of bool): Boolean mask with which ophys frames to use
    '''
    # filter out omitted flashes to avoid omitted flashes at the start of the session from affecting analysis range
    filtered_stimulus_presentations = session.stimulus_presentations
    while filtered_stimulus_presentations.iloc[0]['omitted'] == True:
        filtered_stimulus_presentations = filtered_stimulus_presentations.iloc[1:]
    
    ophys_frames_to_use = (
        (session.ophys_timestamps > filtered_stimulus_presentations.iloc[0]['start_time']) 
        & (session.ophys_timestamps < filtered_stimulus_presentations.iloc[-1]['start_time'] +stim_dur+ end_buffer)
    )
    return ophys_frames_to_use

def get_dff_arr(session, timestamps_to_use):
    '''
    Get the dff traces from a session in xarray format (preserves cell ids and timestamps)

    timestamps_to_use is a boolean vector that contains which timestamps to use in the analysis
    '''
    # Get dff and trim off ends
    all_dff = np.stack(session.dff_traces['dff'].values)
    all_dff_to_use = all_dff[:, timestamps_to_use]

    # Get the timestamps
    dff_trace_timestamps = session.ophys_timestamps
    dff_trace_timestamps_to_use = dff_trace_timestamps[timestamps_to_use]

    # Note: it may be more efficient to get the xarrays directly, rather than extracting/building them from session.dff_traces
    #       The dataframes are built from xarrays to start with, so we are effectively converting them twice by doing this
    #       But if there's no big time penalty to doing it this way, then maybe just leave it be.
    dff_trace_xr = xr.DataArray(
            data = all_dff_to_use.T,
            dims = ("dff_trace_timestamps", "cell_specimen_id"),
            coords = {
                "dff_trace_timestamps": dff_trace_timestamps_to_use,
                "cell_specimen_id": session.cell_specimen_table.index.values
            }
        )
    return dff_trace_xr

def interpolate_to_dff_timestamps(fit,df):
    '''
    interpolate timeseries onto ophys timestamps

    input:  fit, dictionary containing 'dff_trace_timestamps':<array of timestamps>
            df, dataframe with columns:
                timestamps (timestamps of signal)
                values  (signal of interest)

    returns: dataframe with columns:
                timestamps (dff_trace_timestamps)
                values (values interpolated onto dff_trace_timestamps)
   
    '''
    f = scipy.interpolate.interp1d(
        df['timestamps'],
        df['values'],
        bounds_error=False
    )

    interpolated = pd.DataFrame({
        'timestamps':fit['dff_trace_timestamps'],
        'values':f(fit['dff_trace_timestamps'])
    })

    return interpolated

def get_model_weight(bsid, weight_name, run_params):
    beh_model = pd.read_csv(run_params['beh_model_dir']+str(bsid)+'.csv')
    return beh_model[weight_name].copy()

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

    Returns: XArray
    '''
    # Compute the weights
    if lam == 0:
        W = fit(dff_trace_arr,X)
    else:
        W = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, dff_trace_arr))

    # Make xarray
    cellids = dff_trace_arr['cell_specimen_id'].values
    W_xarray= xr.DataArray(
            W, 
            dims =('weights','cell_specimen_id'), 
            coords = {  'weights':X.weights.values, 
                        'cell_specimen_id':cellids}
            )
    return W_xarray

def fit_cell_regularized(dff_trace_arr, X, lam):
    '''
    Analytical OLS solution with added L2 regularization penalty. 

    dff_trace_arr: shape (n_timestamps * n_cells)
    X: shape (n_timestamps * n_kernel_params)
    lam (float): Strength of L2 regularization (hyperparameter to tune)

    Returns: XArray
    '''
    # Compute the weights
    if lam == 0:
        W = fit(dff_trace_arr,X)
    else:
        W = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, dff_trace_arr))

    # Make xarray 
    W_xarray= xr.DataArray(
            W, 
            dims =('weights'), 
            coords = {  'weights':X.weights.values}
            )
    return W_xarray

def variance_ratio(dff_trace_arr, W, X): # TODO Double check this function
    '''
    dff_trace_arr: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    '''
    Y = X.values @ W.values
    var_total = np.var(dff_trace_arr, axis=0) #Total variance in the dff trace for each cell
    var_resid = np.var(dff_trace_arr-Y, axis=0) #Residual variance
    return (var_total - var_resid) / var_total



