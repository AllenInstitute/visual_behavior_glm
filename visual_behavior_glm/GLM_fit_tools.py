import os
import pickle
import xarray as xr
import numpy as np
import pandas as pd
import scipy 
from tqdm import tqdm
from copy import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.data_access.loading as loading
from visual_behavior.encoder_processing.running_data_smoothing import process_encoder_data
import visual_behavior_glm.GLM_analysis_tools as gat

def load_fit_experiment(ophys_experiment_id, run_params):
    '''
        Loads the session data, the fit dictionary and the design matrix for this oeid/run_params
    
        Will raise a FileNotFound Exception if the fit did not happen    
    
        INPUTS:
        ophys_experiment_id,    oeid for this experiment
        run_params,             dictionary of parameters for the fit version
        
        RETURNS:
        session     SDK session object
        fit         fit dictionary with model results
        design      DesignMatrix object for this experiment
    '''
    fit = gat.load_fit_pkl(run_params, ophys_experiment_id)
    session = load_data(ophys_experiment_id)
    design = DesignMatrix(fit)
    design = add_kernels(design, run_params, session,fit)
    return session, fit, design

def check_run_fits(VERSION):
    '''
        Returns the experiment table for this model version with a column 'GLM_fit' 
        appended that is a bool of whether the output pkl file exists for that 
        experiment. It does not try to load the file, or determine if the fit is good.
    
        INPUTS:
        VERSION,    a string of which model version to check
        
        RETURNS
        experiment_table,   a dataframe with a boolean column 'GLM_fit' that says whether VERSION was fit for that experiment id
    '''
    run_params = load_run_json(VERSION)
    experiment_table = pd.read_csv(run_params['experiment_table_path']).reset_index(drop=True).set_index('ophys_experiment_id')
    experiment_table['GLM_fit'] = False
    for index, oeid in enumerate(experiment_table.index.values):
        filename = run_params['experiment_output_dir']+str(oeid)+".pkl" 
        experiment_table.at[oeid, 'GLM_fit'] = os.path.isfile(filename) 
    return experiment_table

def fit_experiment(oeid, run_params,NO_DROPOUTS=False,TESTING=False):
    '''
        Fits the GLM to the ophys_experiment_id
        
        Inputs:
        oeid            experiment to fit
        run_params      dictionary of parameters for this fit
        NO_DROPOUTS     if True, does not perform dropout analysis
        TESTING         if True, fits only the first 6 cells in the experiment
    
        Returns:
        session         the VBA session object for this experiment
        fit             a dictionary containing the results of the fit
        design          the design matrix for this fit
    '''
    
    # Log oeid
    print("Fitting ophys_experiment_id: "+str(oeid)) 

    # Warn user if debugging tools are active
    if NO_DROPOUTS:
        print('WARNING! NO_DROPOUTS=True in fit_experiment(), dropout analysis will NOT run')

    if TESTING:
        print('WARNING! TESTING=True in fit_experiment(), will only fit the first 6 cells of this experiment')

    # Load Data
    print('Loading data')
    session = load_data(oeid)

    # Processing df/f data
    print('Processing df/f data')
    fit= dict()
    fit['dff_trace_arr'] = process_data(session,TESTING=TESTING)
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
        # Cancel dropouts if we are in debugging mode
        fit['dropouts'] = {'Full':copy(fit['dropouts']['Full'])}

    # Iterate over model selections
    print('Iterating over model selection')
    fit = evaluate_models(fit, design, run_params)

    # Start Diagnostic analyses
    print('Starting diagnostics')
    print('Bootstrapping synthetic data')
    fit = bootstrap_model(fit, design, run_params)

    # Perform shuffle analysis, with two shuffle methods
    print('Evaluating shuffle fits')
    fit = evaluate_shuffle(fit, design, method='cells')
    fit = evaluate_shuffle(fit, design, method='time')

    # Save fit dictionary 
    print('Saving results')
    fit['failed_kernels'] = run_params['failed_kernels']
    fit['failed_dropouts'] = run_params['failed_dropouts']
    filepath = os.path.join(run_params['experiment_output_dir'],str(oeid)+'.pkl')
    file_temp = open(filepath, 'wb')
    pickle.dump(fit, file_temp)
    file_temp.close()  
    
    # Save Design Matrix
    print('Saving Design Matrix')  
    sparse_X = scipy.sparse.csc_matrix(design.get_X().values)
    filepath = os.path.join(run_params['experiment_output_dir'],'X_sparse_csc_'+str(oeid)+'.npz')
    scipy.sparse.save_npz(filepath, sparse_X)

    # Save Event Table
    print('Saving Events Table')
    filepath = os.path.join(run_params['experiment_output_dir'],'event_times_'+str(oeid)+'.h5')
    pd.DataFrame(design.events).to_hdf(filepath,key='df')

    # Pack up
    print('Finished') 
    return session, fit, design

def bootstrap_model(fit, design, run_params, regularization=50, norm_preserve=False, check_every=100, PLOT=False):
    '''
        Generates synthetic df/f traces using normally distributed random parameters. 
        Then tries to recover those parameters using the CV procedure in fit_experiment()

        In addition creates a series of synthetic df/f traces where a subset of weights are set to 0 in generating the data, but
        are still included in the fit recovery. These junk regressors let us evaluate the regularization process. 

        INPUTS:
        fit             fit dictionary
        design          design matrix
        run_params      run_params json
        regularization  fixed L2 value used for fitting
        norm_preserve   if true, when zeroing weights preserves the norm of the weight vector (sanity check for weird regularization effects)
        check_every     Zeros out weights in steps of this many parameters
        PLOT            if true, makes a summary plot of the bootstrap analysis

        RETURNS:
        fit             fit dictionary with 'bootstrap' key added which maps to a dictionary of results
    '''
    # Set up storage, and how many datasets to make
    zero_dexes = range(100, len(fit['dropouts']['Full']['cv_weights'][:,:,0]),check_every)
    cv_var_train    = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']), len(zero_dexes)))
    cv_var_test     = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']), len(zero_dexes)))
    cv_var_def      = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']), len(zero_dexes)))
    cv_weight_shift = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']), len(zero_dexes)))

    # Iterate over datasets, making synthetic data, and doing recovery
    for index, zero_dex in enumerate(zero_dexes):

        # Set up design matrix and Y variable (will get over-ridden)p
        X = design.get_X()
        Y_boot = copy(fit['dff_trace_arr'])

        # iterate over cross validation
        for split_index, test_split in tqdm(enumerate(fit['ridge_splits']), total=len(fit['ridge_splits']), desc='    Bootstrapping with {} regressors'.format(zero_dex)):
            # Set up training/test splits
            train_split = np.concatenate([split for i, split in enumerate(fit['ridge_splits']) if i!=split_index])
            test_split = fit['ridge_splits'][split_index]

            # Make weights 
            W = fit['dropouts']['Full']['cv_weights'][:,:,0].copy()
            W = np.random.randn(np.shape(W)[0], np.shape(W)[1])
            if norm_preserve:
                orig_norm = np.linalg.norm(W,axis=0)
            idx = np.random.permutation(np.arange(np.shape(W)[0]-1)+1)[zero_dex:]
            W[idx,:] = 0
            if norm_preserve:
                new_norm = np.linalg.norm(W,axis=0)
                ratio_norm = orig_norm/new_norm
                W = W @ np.diag(ratio_norm)

            # Generate synthetic data, and get best W estimate
            Y_boot.values = X.values @ W
            W_boot = fit_regularized(Y_boot[train_split,:], X[train_split,:],regularization)     

            # Evaluate and save results
            W_orig = copy(W_boot)
            W_orig.values = W
            cv_var_train[:,split_index,index]     = variance_ratio(Y_boot[train_split,:], W_boot, X[train_split,:]) 
            cv_var_test[:,split_index,index]      = variance_ratio(Y_boot[test_split,:],  W_boot, X[test_split,:])
            cv_var_def[:,split_index,index]       = variance_ratio(Y_boot[test_split,:],  W_orig, X[test_split,:])
            cv_weight_shift[:,split_index,index]  = np.mean(np.abs(W_boot.values - W_orig.values))

    # Pack up 
    keyword = 'bootstrap'
    if not norm_preserve:
        keyword = keyword+"_no_norm"
    fit['bootstrap']={}
    fit['bootstrap']['W_boot'] = W_boot
    fit['bootstrap']['W_orig'] = W_orig
    fit['bootstrap']['var_explained_train'] = cv_var_train.mean(axis=1) 
    fit['bootstrap']['var_explained_test']  = cv_var_test.mean(axis=1)  
    fit['bootstrap']['var_explained_def']   = cv_var_def.mean(axis=1)      
    fit['bootstrap']['mean_weight_shift']   = cv_weight_shift.mean(axis=1)
    fit['bootstrap']['zero_dexes'] = zero_dexes
        
    # If plotting is requested
    if PLOT:
        plot_bootstrap(fit, keyword, zero_dexes)
    
    # return fit dictionary with added key/value
    return fit

def plot_bootstrap(fit, keyword,zero_dexes):
    '''
        Plots the bootstrapping analysis
        
        INPUTS:
        fit         fit dictionary
        keywowrd    name of file to save figure as <keyword>.png
        zero_dexes  index of how many weights were zeroed out for each dataset
    '''
    plt.figure()
    plt.plot(zero_dexes, fit['bootstrap']['var_explained_train'].mean(axis=0), 'bo-', label='Train')
    plt.plot(zero_dexes, fit['bootstrap']['var_explained_test'].mean(axis=0), 'ro-', label='Test')
    plt.ylabel('Var Explained')
    plt.xlabel('Number of non-zero values in generative weights')
    plt.legend()
    plt.tight_layout()
    plt.savefig(keyword+'.png')

def evaluate_shuffle(fit, design, method='cells', num_shuffles=50):
    '''
        Evaluates the model on shuffled df/f data to determine a noise floor for the model
    
        Inputs:
        fit, the fit dictionary
        design, the design matrix
        method, either 'cells' or 'time'
            cells, shuffle cell labels, but preserves time. 
            time, circularly permutes each cell's df/f trace
        num_shuffles, how many times to shuffle.
    
        returns:
        fit, with the added keys:
            var_shuffle_<method>    a cells x num_shuffles arrays of shuffled variance explained
            var_shuffle_<method>_threshold  the threshold for a 5% false positive rate
    '''
    # Set up space
    W = fit['dropouts']['Full']['train_weights']
    X = design.get_X()
    var_shuffle = np.empty((fit['dff_trace_arr'].shape[1], num_shuffles)) 
    dff_shuffle = np.copy(fit['dff_trace_arr'].values)
    max_shuffle = np.shape(dff_shuffle)[0]
    
    # Iterate over shuffles
    for count in tqdm(range(0, num_shuffles), total=num_shuffles, desc='    Shuffling by {}'.format(method)):
        if method == 'time':
            for dex in range(0, np.shape(dff_shuffle)[1]):
                shuffle_count = np.random.randint(1, max_shuffle)
                dff_shuffle[:,dex] = np.roll(dff_shuffle[:,dex], shuffle_count, axis=0) 
        elif method == 'cells':
            idx = np.random.permutation(np.shape(dff_shuffle)[1])
            while np.any(idx == np.array(range(0, np.shape(dff_shuffle)[1]))):
                idx = np.random.permutation(np.shape(dff_shuffle)[1])
            dff_shuffle = np.copy(fit['dff_trace_arr'].values)[:,idx]
        var_shuffle[:,count]  = variance_ratio(dff_shuffle, W, X)

    # Make summary evaluation of shuffle threshold
    fit['var_shuffle_'+method] = var_shuffle
    x = np.sort(var_shuffle.flatten())
    dex = np.floor(len(x)*0.95).astype(int)
    fit['var_shuffle_'+method+'_threshold'] = x[dex]
    return fit

def evaluate_ridge(fit, design,run_params):
    '''
        Finds the best L2 value by fitting the model on a grid of L2 values and reporting training/test error
    
        fit, model dictionary
        design, design matrix
        run_params, dictionary of parameters, which needs to include:
            L2_optimize_by_cell     # If True, uses the best L2 value for each cell
            L2_optimize_by_session  # If True, uses the best L2 value for this session
            L2_use_fixed_value      # If True, uses the hard coded L2_fixed_lambda
            L2_fixed_lambda         # This value is used if L2_use_fixed_value
            L2_grid_range           # Min/Max L2 values for optimization
            L2_grid_num             # Number of L2 values for optimization

        returns fit, with the values added:
            L2_grid                 # the L2 grid evaluated (if L2_optimize_by_cell, or L2_optimize_by_session)
            avg_regularization      # the average optimal L2 value, or the fixed value
            cell_regularization     # the optimal L2 value for each cell (if L2_optimize_by_cell)
    '''
    if run_params['L2_use_fixed_value']:
        print('Using a hard-coded regularization value')
        fit['avg_regularization'] = run_params['L2_fixed_lambda']
    else:
        print('Evaluating a grid of regularization values')
        if run_params['L2_grid_type'] == 'log':
            fit['L2_grid'] = np.geomspace(run_params['L2_grid_range'][0], run_params['L2_grid_range'][1],num = run_params['L2_grid_num'])
        else:
            fit['L2_grid'] = np.linspace(run_params['L2_grid_range'][0], run_params['L2_grid_range'][1],num = run_params['L2_grid_num'])
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
        fit['L2_at_grid_min'] = [x==0 for x in np.argmax(test_cv,1)]
        fit['L2_at_grid_max'] = [x==(len(fit['L2_grid'])-1) for x in np.argmax(test_cv,1)]
    return fit

def evaluate_models(fit, design, run_params):
    '''
        Evaluates the model selections across all dropouts using either the single L2 value, or each cell's optimal value

    '''
    if run_params['L2_use_fixed_value'] or run_params['L2_optimize_by_session']:
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

        Each cell uses a different L2 value defined in fit['cell_regularization']
    '''
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])
        X_inner = np.dot(X.T, X)
        mask = get_mask(fit['dropouts'][model_label],design)
        Full_X = design.get_X(kernels=fit['dropouts']['Full']['kernels'])

        # Iterate CV
        cv_var_train    = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_var_test     = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_adjvar_train = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test  = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_train_fc = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test_fc= np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))  
        cv_weights      = np.empty((np.shape(X)[1], fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        all_weights     = np.empty((np.shape(X)[1], fit['dff_trace_arr'].shape[1]))
        all_var_explain = np.empty((fit['dff_trace_arr'].shape[1]))
        all_adjvar_explain = np.empty((fit['dff_trace_arr'].shape[1]))
        all_prediction  = np.empty(fit['dff_trace_arr'].shape)
        X_test_array = []   # Cache the intermediate steps for each cell
        X_train_array = []
        X_cov_array = []

        for cell_index, cell_value in tqdm(enumerate(fit['dff_trace_arr']['cell_specimen_id'].values),total=len(fit['dff_trace_arr']['cell_specimen_id'].values),desc='   Fitting Cells'):

            dff = fit['dff_trace_arr'][:,cell_index]
            Wall = fit_cell_regularized(X_inner,dff, X,fit['cell_regularization'][cell_index])     
            var_explain = variance_ratio(dff, Wall,X)
            adjvar_explain = masked_variance_ratio(dff, Wall,X, mask) 
            all_weights[:,cell_index] = Wall
            all_var_explain[cell_index] = var_explain
            all_adjvar_explain[cell_index] = adjvar_explain
            all_prediction[:,cell_index] = X.values @ Wall.values

            for index, test_split in enumerate(fit['splits']):
                train_split = np.concatenate([split for i, split in enumerate(fit['splits']) if i!=index])
        
                # If this is the first cell, stash the design matrix and covariance result
                if cell_index == 0:
                    X_test_array.append(X[test_split,:])
                    X_train_array.append(X[train_split,:])
                    X_cov_array.append(np.dot(X[train_split,:].T,X[train_split,:]))
                # Grab the stashed result
                X_test  = X_test_array[index]
                X_train = X_train_array[index]
                X_cov   = X_cov_array[index]

                dff_train = fit['dff_trace_arr'][train_split,cell_index]
                dff_test = fit['dff_trace_arr'][test_split,cell_index]
                W = fit_cell_regularized(X_cov,dff_train, X_train, fit['cell_regularization'][cell_index])
                cv_var_train[cell_index,index] = variance_ratio(dff_train, W, X_train)
                cv_var_test[cell_index,index] = variance_ratio(dff_test, W, X_test)
                cv_adjvar_train[cell_index,index]= masked_variance_ratio(dff_train, W, X_train, mask[train_split]) 
                cv_adjvar_test[cell_index,index] = masked_variance_ratio(dff_test, W, X_test, mask[test_split])
                cv_weights[:,cell_index,index] = W 
                if model_label == 'Full':
                    # If this is the Full model, the value is the same
                    cv_adjvar_train_fc[cell_index,index]= masked_variance_ratio(dff_train, W, X_train, mask[train_split])  
                    cv_adjvar_test_fc[cell_index,index] = masked_variance_ratio(dff_test, W, X_test, mask[test_split])  
                else:
                    # Otherwise, get weights and design matrix for this cell/cv_split and compute the variance explained on this mask
                    Full_W = xr.DataArray(fit['dropouts']['Full']['cv_weights'][:,cell_index,index])
                    Full_X_test = Full_X[test_split,:]
                    Full_X_train = Full_X[train_split,:]
                    cv_adjvar_train_fc[cell_index,index]= masked_variance_ratio(dff_train, Full_W, Full_X_train, mask[train_split])  
                    cv_adjvar_test_fc[cell_index,index] = masked_variance_ratio(dff_test, Full_W, Full_X_test, mask[test_split])    

        all_weights_xarray = xr.DataArray(
            data = all_weights,
            dims = ("weights", "cell_specimen_id"),
            coords = {
                "weights": X.weights.values,
                "cell_specimen_id": fit['dff_trace_arr'].cell_specimen_id.values
            }
        )

        fit['dropouts'][model_label]['train_weights']   = all_weights_xarray
        fit['dropouts'][model_label]['train_variance_explained']    = all_var_explain
        fit['dropouts'][model_label]['train_adjvariance_explained'] = all_adjvar_explain
        fit['dropouts'][model_label]['full_model_train_prediction'] = all_prediction
        fit['dropouts'][model_label]['cv_weights']      = cv_weights
        fit['dropouts'][model_label]['cv_var_train']    = cv_var_train
        fit['dropouts'][model_label]['cv_var_test']     = cv_var_test
        fit['dropouts'][model_label]['cv_adjvar_train'] = cv_adjvar_train
        fit['dropouts'][model_label]['cv_adjvar_test']  = cv_adjvar_test
        fit['dropouts'][model_label]['cv_adjvar_train_full_comparison'] = cv_adjvar_train_fc
        fit['dropouts'][model_label]['cv_adjvar_test_full_comparison']  = cv_adjvar_test_fc

    return fit 


def evaluate_models_same_ridge(fit, design, run_params):
    '''
        Fits and evaluates each model defined in fit['dropouts']
    
        For each model, it creates the design matrix, finds the optimal weights, and saves the variance explained. 
            It does this for the entire dataset as test and train. As well as CV, saving each test/train split
    
        All cells use the same regularization value defined in fit['avg_regularization']  
        
    '''
    for model_label in fit['dropouts'].keys():

        # Set up design matrix for this dropout
        X = design.get_X(kernels=fit['dropouts'][model_label]['kernels'])
        mask = get_mask(fit['dropouts'][model_label],design)
        Full_X = design.get_X(kernels=fit['dropouts']['Full']['kernels'])

        # Fit on full dataset for references as training fit
        dff = fit['dff_trace_arr']
        Wall = fit_regularized(dff, X,fit['avg_regularization'])     
        var_explain = variance_ratio(dff, Wall,X)
        adjvar_explain = masked_variance_ratio(dff, Wall,X, mask) 
        fit['dropouts'][model_label]['train_weights'] = Wall
        fit['dropouts'][model_label]['train_variance_explained']    = var_explain
        fit['dropouts'][model_label]['train_adjvariance_explained'] = adjvar_explain
        fit['dropouts'][model_label]['full_model_train_prediction'] = X.values @ Wall.values

        # Iterate CV
        cv_var_train = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_var_test = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))
        cv_adjvar_train = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))  
        cv_adjvar_train_fc = np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits']))) 
        cv_adjvar_test_fc= np.empty((fit['dff_trace_arr'].shape[1], len(fit['splits'])))  
        cv_weights = np.empty((np.shape(Wall)[0], np.shape(Wall)[1], len(fit['splits'])))

        for index, test_split in tqdm(enumerate(fit['splits']), total=len(fit['splits']), desc='    Fitting model, {}'.format(model_label)):
            train_split = np.concatenate([split for i, split in enumerate(fit['splits']) if i!=index])
            X_test = X[test_split,:]
            X_train = X[train_split,:]
            mask_test = mask[test_split]
            mask_train = mask[train_split]
            dff_train = fit['dff_trace_arr'][train_split,:]
            dff_test = fit['dff_trace_arr'][test_split,:]
            W = fit_regularized(dff_train, X_train, fit['avg_regularization'])
            cv_var_train[:,index]   = variance_ratio(dff_train, W, X_train)
            cv_var_test[:,index]    = variance_ratio(dff_test, W, X_test)
            cv_adjvar_train[:,index]= masked_variance_ratio(dff_train, W, X_train, mask_train) 
            cv_adjvar_test[:,index] = masked_variance_ratio(dff_test, W, X_test, mask_test)
            cv_weights[:,:,index]   = W 
            if model_label == 'Full':
                # If this model is Full, then the masked variance ratio is the same
                cv_adjvar_train_fc[:,index]= masked_variance_ratio(dff_train, W, X_train, mask_train)  
                cv_adjvar_test_fc[:,index] = masked_variance_ratio(dff_test, W, X_test, mask_test)  
            else:
                # Otherwise load the weights and design matrix for this cv_split, and compute VE with this support mask
                Full_W = xr.DataArray(fit['dropouts']['Full']['cv_weights'][:,:,index])
                Full_X_test = Full_X[test_split,:]
                Full_X_train = Full_X[train_split,:]
                cv_adjvar_train_fc[:,index]= masked_variance_ratio(dff_train, Full_W, Full_X_train, mask_train)  
                cv_adjvar_test_fc[:,index] = masked_variance_ratio(dff_test, Full_W, Full_X_test, mask_test)    

        fit['dropouts'][model_label]['cv_weights']      = cv_weights
        fit['dropouts'][model_label]['cv_var_train']    = cv_var_train
        fit['dropouts'][model_label]['cv_var_test']     = cv_var_test
        fit['dropouts'][model_label]['cv_adjvar_train'] = cv_adjvar_train
        fit['dropouts'][model_label]['cv_adjvar_test']  = cv_adjvar_test
        fit['dropouts'][model_label]['cv_adjvar_train_full_comparison'] = cv_adjvar_train_fc
        fit['dropouts'][model_label]['cv_adjvar_test_full_comparison']  = cv_adjvar_test_fc

    return fit 

def get_mask(dropout,design):
    '''
        For the dropout dictionary returns the mask of where the kernels have support in the design matrix.
        Ignores the support of the intercept regressor
    
        INPUTS:
        dropout     a dictionary with keys:
            'is_single' (bool)
            'dropped_kernels' (list)
            'kernels' (list)
        design      DesignMatrix object
    
        RETURNS:
        mask, a boolean vector for the indicies with support

        if the dropout is_single, then support is defined by the included kernels
        if the dropout is not is_single, then support is defined by the dropped kernels
    '''
    if dropout['is_single']:
        # Support is defined by the included kernels
        kernels=dropout['kernels']
    else:
        # Support is defined by the dropped kernels
        kernels=dropout['dropped_kernels']
    
    # Need to remove 'intercept'
    if 'intercept' in kernels:
        kernels.remove('intercept')    

    # Get mask from design matrix object 
    return design.get_mask(kernels=kernels)

def build_dataframe_from_dropouts(fit,threshold=0.005):
    '''
        INPUTS:
        threshold (0.005 default) is the minimum amount of variance explained by the full model. The minimum amount of variance explained by a dropout model        

        Returns a dataframe with 
        Index: Cell specimen id
        Columns: Average (across CV folds) variance explained on the test and training sets for each model defined in fit['dropouts']
    '''
        
    cellids = fit['dff_trace_arr']['cell_specimen_id'].values
    results = pd.DataFrame(index=pd.Index(cellids, name='cell_specimen_id'))

    # Iterate over models
    for model_label in fit['dropouts'].keys():
        # For each model, average over CV splits for variance explained on train/test
        results[model_label+"__avg_cv_var_train"] = np.mean(fit['dropouts'][model_label]['cv_var_train'],1) 
        results[model_label+"__avg_cv_var_test"]  = np.mean(fit['dropouts'][model_label]['cv_var_test'],1) 
        results[model_label+"__avg_cv_var_test_full_comparison"] = np.mean(fit['dropouts']['Full']['cv_var_test'],1)

        # For each model, average over CV splits for adjusted variance explained on train/test, and the full model comparison
        # If a CV split did not have an event in a test split, so the kernel has no support, the CV is NAN. Here we use nanmean to
        # ignore those CV splits without information
        results[model_label+"__avg_cv_adjvar_train"] = np.nanmean(fit['dropouts'][model_label]['cv_adjvar_train'],1) 
        results[model_label+"__avg_cv_adjvar_test"]  = np.nanmean(fit['dropouts'][model_label]['cv_adjvar_test'],1) 
        results[model_label+"__avg_cv_adjvar_test_full_comparison"]  = np.nanmean(fit['dropouts'][model_label]['cv_adjvar_test_full_comparison'],1) 
    
        # Clip the variance explained values to >= 0
        results.loc[results[model_label+"__avg_cv_var_test"] < 0,model_label+"__avg_cv_var_test"] = 0
        results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < 0,model_label+"__avg_cv_var_test_full_comparison"] = 0
        results.loc[results[model_label+"__avg_cv_adjvar_test"] < 0,model_label+"__avg_cv_adjvar_test"] = 0
        results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < 0,model_label+"__avg_cv_adjvar_test_full_comparison"] = 0
        
        # Compute the absolute change in variance
        results[model_label+"__absolute_change_from_full"] = results[model_label+"__avg_cv_var_test"] - results[model_label+"__avg_cv_var_test_full_comparison"] 
 
        # Compute the dropout scores, which is dependent on whether this was a single-dropout or not
        if fit['dropouts'][model_label]['is_single']:  
            # Compute the dropout
            results[model_label+"__dropout"] = -results[model_label+"__avg_cv_var_test"]/results[model_label+"__avg_cv_var_test_full_comparison"]
            results[model_label+"__adj_dropout"] = -results[model_label+"__avg_cv_adjvar_test"]/results[model_label+"__avg_cv_adjvar_test_full_comparison"]

            # Cleaning Steps, careful eye here! TODO
            # If the single-dropout explained more variance than the full_comparison, clip dropout to -1
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < results[model_label+"__avg_cv_adjvar_test"], model_label+"__adj_dropout"] = -1 
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < results[model_label+"__avg_cv_var_test"], model_label+"__dropout"] = -1

            # If the single-dropout explained less than THRESHOLD variance, clip dropout to 0            
            results.loc[results[model_label+"__avg_cv_adjvar_test"] < threshold, model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test"] < threshold, model_label+"__dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test"] < threshold, model_label+"__dropout"] = 0
    
            # If the full_comparison model explained less than THRESHOLD variance, clip the dropout to 0.
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < threshold, model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < threshold, model_label+"__dropout"] = 0 
            #results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < threshold, model_label+"__absolute_change_from_full"] = 0 
        else:
            # Compute the dropout
            results[model_label+"__adj_dropout"] = -(1-results[model_label+"__avg_cv_adjvar_test"]/results[model_label+"__avg_cv_adjvar_test_full_comparison"]) 
            results[model_label+"__dropout"] = -(1-results[model_label+"__avg_cv_var_test"]/results[model_label+"__avg_cv_var_test_full_comparison"]) 
   
            # Cleaning Steps, careful eye here! TODO            
            # If the dropout explained more variance than the full_comparison, clip the dropout to 0
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < results[model_label+"__avg_cv_adjvar_test"], model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < results[model_label+"__avg_cv_var_test"], model_label+"__dropout"] = 0

            # If the full_comparison model explained less than THRESHOLD variance, clip the dropout to 0
            results.loc[results[model_label+"__avg_cv_adjvar_test_full_comparison"] < threshold, model_label+"__adj_dropout"] = 0
            results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < threshold, model_label+"__dropout"] = 0
            #results.loc[results[model_label+"__avg_cv_var_test_full_comparison"] < threshold, model_label+"__absolute_change_from_full"] = 0

            # OLD STEPS TO BE REMOVED TODO
            # Removing the requirement that there is a minimum amount of difference between the dropout and the full
            # Compute the difference between the dropout and the full model comparison
            #results[model_label+"__absolute_change_from_full"] = results[model_label+"__avg_cv_adjvar_test"] - results[model_label+"__avg_cv_adjvar_test_full_comparison"]
        
            # If the dropout didnt decrease the variance explained by at least THRESHOLD amount, clip dropout to 0
            #results.loc[results[model_label+"__absolute_change_from_full"] > -threshold, model_label+"__adj_dropout"] = 0


        # Not removing the code because I want to document things first TODO
        #d = copy(fit['dropouts'][model_label]['cv_adjvar_test'])
        #F = copy(fit['dropouts'][model_label]['cv_adjvar_test_full_comparison'])
        #if fit['dropouts'][model_label]['is_single']:
        #    #results[model_label+"__adj_dropout"] = np.mean(-d/F,axis=1) # Average over cross validations before or after computing dropout?
        #    # This way is way more noisy, so averaging first
        #    
        #    results[model_label+"__adj_dropout"] = -np.mean(d,axis=1)/np.mean(F,axis=1) 
        #else:
        #    #results[model_label+"__adj_dropout"] = np.mean(-(1-d/F),axis=1) # Average over cross validations before or after computing dropout?
        #    results[model_label+"__adj_dropout"] = -(1-np.mean(d,axis=1)/np.mean(F,axis=1)) 
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
 
def load_data(oeid, dataframe_format='wide', smooth_running_data=True):
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
    if smooth_running_data:
        session.dataset.running_data_df = process_encoder_data(
            session.dataset.running_data_df.reset_index(), 
            v_max='v_sig_max'
        )
    return session

def process_eye_data(session,run_params,ophys_timestamps=None):
    '''
        Returns a dataframe of eye tracking data with several processing steps
        1. All columns are interpolated onto ophys timestamps
        2. Likely blinks are removed with a threshold set by run_params['eye_blink_z']
        3. After blink removal, a second transient step removes outliers with threshold run_params['eye_tranisent_threshold']
        4. After interpolating onto the ophys timestamps, Z-scores the eye_width and pupil_radius
        
        Does not modifiy the original eye_tracking dataframe
    '''    

    # Set parameters for blink detection, and load data
    session.dataset.set_params(eye_tracking_z_threshold=run_params['eye_blink_z'])
    eye = session.dataset.eye_tracking.copy(deep=True)

    # Compute pupil radius
    eye['pupil_radius'] = np.sqrt(eye['pupil_area']*(1/np.pi))
    
    # Remove likely blinks and interpolate
    eye.loc[eye['likely_blink'],:] = np.nan
    eye = eye.interpolate()   

    # Do a second transient removal step
    x = scipy.stats.zscore(eye['pupil_radius'],nan_policy='omit')
    d_mask = np.abs(np.diff(x,append=x[-1])) > run_params['eye_transient_threshold']
    eye.loc[d_mask,:]=np.nan
    eye = eye.interpolate()

    # Interpolate everything onto ophys_timestamps
    ophys_eye = pd.DataFrame({'timestamps':ophys_timestamps})
    z_score = ['eye_width','pupil_radius']
    for column in eye.keys():
        if column != 'time':
            f = scipy.interpolate.interp1d(eye['time'], eye[column], bounds_error=False)
            ophys_eye[column] = f(ophys_eye['timestamps'])
            ophys_eye[column].fillna(method='ffill',inplace=True)
            if column in z_score:
                ophys_eye[column+'_zscore'] = scipy.stats.zscore(ophys_eye[column],nan_policy='omit')
    print('                 : '+'Mean Centering')
    print('                 : '+'Standardized to unit variance')
    return ophys_eye 


def process_data(session,TESTING=False):
    '''
    Processes dff traces by trimming off portions of recording session outside of the task period. These include:
        * a ~5 minute gray screen period before the task begins
        * a ~5 minute gray screen period after the task ends
        * a 5-10 minute movie following the second gray screen period
    
    input -- session object 
    TESTING,        if True, only includes the first 6 cells of the experiment

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

    # Clip the array to just the first 6 cells
    if TESTING:
        dff_trace_arr = dff_trace_arr[:,0:6]

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
    run_params['failed_kernels']=set()
    run_params['failed_dropouts']=set()
    run_params['kernel_error_dict'] = dict()
    for kernel_name in run_params['kernels']:
        if run_params['kernels'][kernel_name]['type'] == 'discrete':
            design = add_discrete_kernel_by_label(kernel_name, design, run_params, session, fit)
        else:
            design = add_continuous_kernel_by_label(kernel_name, design, run_params, session, fit)   

    clean_failed_kernels(run_params)
    return design

def clean_failed_kernels(run_params):
    '''
        Modifies the model definition to handle any kernels that failed to fit during the add_kernel process
        Removes the failed kernels from run_params['kernels'], and run_params['dropouts']
    '''
    if run_params['failed_kernels']:
        print('The following kernels failed to be added to the model: ')
        print(run_params['failed_kernels'])
    
    # Iterate failed kernels
    for kernel in run_params['failed_kernels']:     
        # Remove the failed kernel from the full list of kernels
        if kernel in run_params['kernels'].keys():
            run_params['kernels'].pop(kernel)

        # Remove the dropout associated with this kernel
        if kernel in run_params['dropouts'].keys():
            run_params['dropouts'].pop(kernel)        
        
        # Remove the failed kernel from each dropout list of kernels
        for dropout in run_params['dropouts'].keys(): 
            # If the failed kernel is in this dropout, remove the kernel from the kernel list
            if kernel in run_params['dropouts'][dropout]['kernels']:
                run_params['dropouts'][dropout]['kernels'].remove(kernel) 
            # If the failed kernel is in the dropped kernel list, remove from dropped kernel list
            if kernel in run_params['dropouts'][dropout]['dropped_kernels']:
                run_params['dropouts'][dropout]['dropped_kernels'].remove(kernel) 

    # Iterate Dropouts, checking for empty dropouts
    drop_list = list(run_params['dropouts'].keys())
    for dropout in drop_list:
        if not (dropout == 'Full'):
            if len(run_params['dropouts'][dropout]['dropped_kernels']) == 0:
                run_params['dropouts'].pop(dropout)
                run_params['failed_dropouts'].add(dropout)
            elif len(run_params['dropouts'][dropout]['kernels']) == 1:
                run_params['dropouts'].pop(dropout)
                run_params['failed_dropouts'].add(dropout)

    if run_params['failed_dropouts']:
        print('The following dropouts failed to be added to the model: ')
        print(run_params['failed_dropouts'])


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
    try:
        event = run_params['kernels'][kernel_name]['event']
        if event == 'intercept':
            timeseries = np.ones(len(fit['dff_trace_timestamps']))
        elif event == 'time':
            timeseries = np.array(range(1,len(fit['dff_trace_timestamps'])+1))
            timeseries = timeseries/len(timeseries)
        elif event == 'running':
            running_df = session.dataset.running_data_df
            running_df = running_df.rename(columns={'speed':'values'})
            timeseries = interpolate_to_dff_timestamps(fit, running_df)['values'].values
            timeseries = standardize_inputs(timeseries, mean_center=False,unit_variance=False, max_value=run_params['max_run_speed'])
        elif event.startswith('face_motion'):
            PC_number = int(event.split('_')[-1])
            face_motion_df =  pd.DataFrame({
                'timestamps': session.dataset.behavior_movie_timestamps,
                'values': session.dataset.behavior_movie_pc_activations[:,PC_number]
            })
            timeseries = interpolate_to_dff_timestamps(fit, face_motion_df)['values'].values
            timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'],unit_variance=run_params['unit_variance_inputs'])
        elif event == 'population_mean':
            timeseries = np.mean(fit['dff_trace_arr'],1).values
            timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'],unit_variance=run_params['unit_variance_inputs'])
        elif event == 'Population_Activity_PC1':
            pca = PCA()
            pca.fit(fit['dff_trace_arr'].values)
            dff_pca = pca.transform(fit['dff_trace_arr'].values)
            timeseries = dff_pca[:,0]
            timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'],unit_variance=run_params['unit_variance_inputs'])
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
            timeseries = standardize_inputs(timeseries, mean_center=run_params['mean_center_inputs'],unit_variance=run_params['unit_variance_inputs'])
        elif event == 'pupil':
            session.ophys_eye = process_eye_data(session,run_params,ophys_timestamps =fit['dff_trace_timestamps'] )
            timeseries = session.ophys_eye['pupil_radius'].values
        else:
            raise Exception('Could not resolve kernel label')
    except Exception as e:
        print('Error encountered while adding kernel for '+kernel_name+'. Attemping to continue without this kernel. ' )
        print(e)
        # Need to remove from relevant lists
        run_params['failed_kernels'].add(kernel_name)      
        run_params['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel', 
            'kernel_name': kernel_name, 
            'exception':e.args[0], 
            'oeid':session.dataset.metadata['ophys_experiment_id'], 
            'glm_version':run_params['version']
        }
        # log error to mongo
        gat.log_error(
            run_params['kernel_error_dict'][kernel_name], 
            keys_to_check = ['oeid', 'glm_version', 'kernel_name']
        )
        return design
    else:
        #assert length of values is same as length of timestamps
        assert len(timeseries) == fit['dff_trace_arr'].values.shape[0], 'Length of continuous regressor must match length of dff_trace_timestamps'

        # Add to design matrix
        design.add_kernel(timeseries, run_params['kernels'][kernel_name]['length'], kernel_name, offset=run_params['kernels'][kernel_name]['offset'])   
        return design


def standardize_inputs(timeseries, mean_center=True, unit_variance=True,max_value=None):
    '''
        Performs three different input standarizations to the timeseries
    
        if mean_center, the timeseries is adjusted to have 0-mean. This can be performed with unit_variance. 

        if unit_variance, the timeseries is adjusted to have unit variance. This can be performed with mean_center.
    
        if max_value is given, then the timeseries is normalized by max_value. This cannot be performed with mean_center and unit_variance.

    '''
    if (max_value is not None ) & (mean_center or unit_variance):
        raise Exception('Cannot perform max_value standardization and mean_center or unit_variance standardizations together.')

    if mean_center:
        print('                 : '+'Mean Centering')
        timeseries = timeseries -np.mean(timeseries) # mean center
    if unit_variance:
        print('                 : '+'Standardized to unit variance')
        timeseries = timeseries/np.std(timeseries)
    if max_value is not None:
        print('                 : '+'Normalized by max value: '+str(max_value))
        timeseries = timeseries/max_value

    return timeseries

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
    try:
        event = run_params['kernels'][kernel_name]['event']
        if event == 'licks':
            event_times = session.dataset.licks['timestamps'].values
        elif event == 'lick_bouts':
            licks = session.dataset.licks
            licks['pre_ILI'] = licks['timestamps'] - licks['timestamps'].shift(fill_value=-10)
            licks['bout_start'] = licks['pre_ILI'] > run_params['lick_bout_ILI']
            event_times = session.dataset.licks.query('bout_start')['timestamps'].values
        elif event == 'rewards':
            event_times = session.dataset.rewards['timestamps'].values
        elif event == 'change':
            event_times = session.dataset.trials.query('go')['change_time'].values
            event_times = event_times[~np.isnan(event_times)]
        elif event in ['hit', 'miss', 'false_alarm', 'correct_reject']:
            event_times = session.dataset.trials.query(event)['change_time'].values
            event_times = event_times[~np.isnan(event_times)]
        elif event == 'any-image':
            event_times = session.dataset.stimulus_presentations.query('not omitted')['start_time'].values
        elif event == 'omissions':
            event_times = session.dataset.stimulus_presentations.query('omitted')['start_time'].values
        elif (len(event)>5) & (event[0:5] == 'image'):
            event_times = session.dataset.stimulus_presentations.query('image_index == @event[-1]')['start_time'].values
        else:
            raise Exception('Could not resolve kernel label')
    except Exception as e:
        print('Error encountered while adding kernel for '+kernel_name+'. Attemping to continue without this kernel. ' )
        print(e)
        # Need to remove from relevant lists
        run_params['failed_kernels'].add(kernel_name)      
        run_params['kernel_error_dict'][kernel_name] = {
            'error_type': 'kernel', 
            'kernel_name': kernel_name, 
            'exception':e.args[0], 
            'oeid':session.dataset.metadata['ophys_experiment_id'], 
            'glm_version':run_params['version']
        }
        # log error to mongo:
        gat.log_error(
            run_params['kernel_error_dict'][kernel_name], 
            keys_to_check = ['oeid', 'glm_version', 'kernel_name']
        )        
        return design       
    else:
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

    def get_mask(self, kernels=None):
        ''' 
            Args:
            kernels, a list of kernel string names
            Returns:
            mask ( a boolean vector), where these kernels have support
        '''
        if len(kernels) == 0:
            X = self.get_X() 
        else:
            X = self.get_X(kernels=kernels) 
        mask = np.any(~(X==0), axis=1)
        return mask.values
 
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
    '''
        Loads the model weights for <bsid> behavior_ophys_session_id
        Loads only the <weight_name> weight
        run_params gives the directory to the fit location
    '''
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

def fit_cell_regularized(X_cov,dff_trace_arr, X, lam):
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
        W = np.dot(np.linalg.inv(X_cov + lam * np.eye(X.shape[-1])),
               np.dot(X.T, dff_trace_arr))

    # Make xarray 
    W_xarray= xr.DataArray(
            W, 
            dims =('weights'), 
            coords = {  'weights':X.weights.values}
            )
    return W_xarray

def variance_ratio(dff_trace_arr, W, X): 
    '''
    Computes the fraction of variance in dff_trace_arr explained by the linear model Y = X*W
    
    dff_trace_arr: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    '''
    Y = X.values @ W.values
    var_total = np.var(dff_trace_arr, axis=0)   # Total variance in the dff trace for each cell
    var_resid = np.var(dff_trace_arr-Y, axis=0) # Residual variance in the difference between the model and data
    return (var_total - var_resid) / var_total  # Fraction of variance explained by linear model

def masked_variance_ratio(dff_trace_arr, W, X, mask): 
    '''
    Computes the fraction of variance in dff_trace_arr explained by the linear model Y = X*W
    but only looks at the timepoints in mask
    
    dff_trace_arr: (n_timepoints, n_cells)
    W: Xarray (n_kernel_params, n_cells)
    X: Xarray (n_timepoints, n_kernel_params)
    mask: bool vector (n_timepoints,)
    '''

    Y = X.values @ W.values

    # Define variance function that lets us isolate the mask timepoints
    def my_var(dff, support_mask):
        if len(np.shape(dff)) ==1:
            dff = dff.values[:,np.newaxis]
        mu = np.mean(dff,axis=0)
        return np.mean((dff[support_mask,:]-mu)**2,axis=0)

    var_total = my_var(dff_trace_arr, mask)#Total variance in the dff trace for each cell
    var_resid = my_var(dff_trace_arr-Y, mask)#Residual variance in the difference between the model and data
    return (var_total - var_resid) / var_total  # Fraction of variance explained by linear model

def error_by_time(fit, design):
    '''
        Plots the model error over the course of the session
    '''
    plt.figure()
    Y = design.get_X().values @ fit['dropouts']['Full']['cv_var_weights'][:,:,0]
    diff = fit['dff_trace_arr'] - Y
    plt.figure()
    plt.plot(np.abs(diff.mean(axis=1)), 'k-')
    plt.ylabel('Model Error (df/f)')
    

