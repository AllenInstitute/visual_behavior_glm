import os 
import json
from copy import copy
import datetime
import shutil

import visual_behavior.data_access.loading as loading

OUTPUT_DIR_BASE = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm'

def define_kernels():
    kernels = {
        'intercept':    {'event':'intercept',   'type':'continuous',    'length':0,     'offset':0, 'dropout':True},
        'time':         {'event':'time',        'type':'continuous',    'length':0,     'offset':0, 'dropout':True},
        'pre_licks':    {'event':'licks',       'type':'discrete',      'length':5,   'offset':-5, 'dropout':True},
        'post_licks':   {'event':'licks',       'type':'discrete',      'length':1,     'offset':0, 'dropout':True},
        'pre_lick_bouts':    {'event':'lick_bouts',       'type':'discrete',      'length':5,   'offset':-5, 'dropout':True},
        'post_lick_bouts':   {'event':'lick_bouts',       'type':'discrete',      'length':1,     'offset':0, 'dropout':True},
        'rewards':      {'event':'rewards',     'type':'discrete',      'length':4,     'offset':-0.5, 'dropout':True},
        'change':       {'event':'change',      'type':'discrete',      'length':2,     'offset':0, 'dropout':True},
        'hits':       {'event':'hit',      'type':'discrete',      'length':3,     'offset':-1, 'dropout':True},
        'misses':       {'event':'miss',      'type':'discrete',      'length':3,     'offset':-1, 'dropout':True},
        'false_alarms':       {'event':'false_alarm',      'type':'discrete',      'length':3,     'offset':-1, 'dropout':True},
        'correct_rejects':       {'event':'correct_reject',      'type':'discrete',      'length':3,     'offset':-1, 'dropout':True},
        'omissions':    {'event':'omissions',   'type':'discrete',      'length':6,     'offset':-1, 'dropout':True},
        'each-image':   {'event':'each-image',  'type':'discrete',      'length':0.8,  'offset':0, 'dropout':True},
        'image_expectation':   {'event':'any-image',  'type':'discrete','length':0.8,  'offset':-0.767, 'dropout':True},
        'running':      {'event':'running',     'type':'continuous',    'length':2,     'offset':-1, 'dropout':True},
        'beh_model':    {'event':'beh_model',   'type':'continuous',    'length':.5,    'offset':-.25, 'dropout':True},
        'pupil':        {'event':'pupil',       'type':'continuous',    'length':2,     'offset':-1, 'dropout':True},
    }
    ## add face motion energy PCs
    for PC in range(10):
        kernels['face_motion_PC_{}'.format(PC)] = {'event':'face_motion_PC_{}'.format(PC), 'type':'continuous', 'length':2, 'offset':-1, 'dropout':False}
    return kernels


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

def make_run_json(VERSION,label='',username=None, src_path=None, TESTING=False):
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
    output_dir              = os.path.join(OUTPUT_DIR_BASE, 'v_'+str(VERSION))
    model_freeze_dir        = os.path.join(output_dir, 'frozen_model_files')
    experiment_output_dir   = os.path.join(output_dir, 'experiment_model_files')
    manifest_dir            = os.path.join(output_dir, 'manifest')
    manifest                = os.path.join(output_dir, 'manifest', 'manifest.json')
    job_dir                 = os.path.join(output_dir, 'log_files')
    json_path               = os.path.join(output_dir, 'run_params.json')
    experiment_table_path   = os.path.join(output_dir, 'experiment_table_v_'+str(VERSION)+'.csv')
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
    python_file_full_path = os.path.join(model_freeze_dir, 'GLM_fit_tools.py')
    python_fit_script = os.path.join(model_freeze_dir, 'fit_glm_v_'+str(VERSION)+'.py')
    if src_path is None:
        raise Exception('You need to provide a path to the model source code')

    shutil.copyfile(os.path.join(src_path, 'src/GLM_fit_tools.py'),   python_file_full_path)
    shutil.copyfile(os.path.join(src_path, 'scripts/fit_glm.py'),     python_fit_script)
    
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
    kernels_orig = define_kernels()
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
        'L2_optimize_by_cell': False,    # If True, uses the best L2 value for each cell
        'L2_optimize_by_session': False, # If True, uses the best L2 value for this session
        'L2_use_fixed_value': True,    # If True, uses the hard coded L2_fixed_lambda
        'L2_fixed_lambda':1,         # This value is used if L2_use_fixed_value
        'L2_grid_range':[.1, 500],      # Min/Max L2 values for L2_optimize_by_cell, or L2_optimize_by_session
        'L2_grid_num': 40,              # Number of L2 values for L2_optimize_by_cell, or L2_optimize_by_session
        'L2_grid_type':'linear',        # how to space L2 options, must be: 'log' or 'linear'
        'ophys_experiment_ids':experiment_table.index.values.tolist(),
        'job_settings':job_settings,
        'kernels':kernels,
        'dropouts':dropouts,
        'lick_bout_ILI': 0.7,           # The minimum duration of time between two licks to segment them into separate lick bouts
        'CV_splits':5,
        'CV_subsplits':10,
        'eye_blink_z': 5.0,             # Threshold for excluding likely blinks
        'eye_transient_threshold': 0.5, # Threshold for excluding eye transients after blink removal
        'mean_center_inputs': True,     # If True, mean centers continuous inputs
        'unit_variance_inputs': True,   # If True, continuous inputs have unit variance
        'max_run_speed': 100              # If 1, has no effect. Scales running speed to be O(1). 
    }

    # Regularization parameter checks
    a = run_params['L2_optimize_by_cell']
    b = run_params['L2_optimize_by_session']
    c = run_params['L2_use_fixed_value']
    assert (a or b or c) and not ((a and b) or (b and c) or (a and c)), "Must select one and only on L2 option: L2_optimize_by_cell, L2_optimize_by_session, or L2_use_fixed_value"

    # Check L2 Fixed value parameters
    if run_params['L2_use_fixed_value'] and (run_params['L2_fixed_lambda'] is None):
        raise Exception('L2_use_fixed_value is True, but have None for L2_fixed_lambda')
    if (not run_params['L2_use_fixed_value']) and (run_params['L2_fixed_lambda'] is not None):
        raise Exception('L2_use_fixed_value is False, but L2_fixed_lambda has been set')      
    if run_params['L2_use_fixed_value']:
        assert run_params['L2_fixed_lambda'] > 0, "Must have some positive regularization value to prevent singular matrix"

    # Check L2 Optimization parameters
    if (a or b):
        assert run_params['L2_grid_num'] > 0, "Must have at least one grid option for L2 optimization"
        assert len(run_params['L2_grid_range']) ==2, "Must have a minimum and maximum L2 grid option"
        assert run_params['L2_grid_type'] in ['log','linear'], "L2_grid_type must be log or linear"
        assert run_params['L2_grid_range'][0] > 0, "Must have a positive regularization minimum value."

    with open(json_path, 'w') as json_file:
        json.dump(run_params, json_file, indent=4)

    # Print Success
    print('Model Successfully Saved, version '+str(VERSION))

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
    for kernel in [kernel for kernel in kernels.keys() if kernels[kernel]['dropout']]:
        dropouts[kernel]={'kernels':list(kernels.keys())}
        dropouts[kernel]['kernels'].remove(kernel)

    # Removes all face motion PC kernels as a group
    if 'face_motion_PC_0' in kernel_definitions:
        dropouts['face_motion_energy'] = {'kernels':list(kernels.keys())}
        kernels_to_drop = [kernel for kernel in dropouts['face_motion_energy']['kernels'] if kernel.startswith('face_motion')] 
        for kernel in kernels_to_drop:
            dropouts['face_motion_energy']['kernels'].remove(kernel)

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
    

def load_run_json(version):
    '''
        Loads the run parameters for model v_<version>
        Assumes verion is saved with root directory global OUTPUT_DIR_BASE       
        returns a dictionary of run parameters
    '''
    json_path = os.path.join(OUTPUT_DIR_BASE, 'v_'+str(version), 'run_params.json')
    with open(json_path,'r') as json_file:
        run_params = json.load(json_file)
    return run_params
