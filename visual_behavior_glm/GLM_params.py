import os 
import json
import numpy as np
from copy import copy
import datetime
import shutil

import visual_behavior.data_access.loading as loading

OUTPUT_DIR_BASE = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm'

def define_kernels():
    kernels = {
        'intercept':    {'event':'intercept',   'type':'continuous',    'length':0,     'offset':0,     'dropout':True, 'text': 'constant value'},
        'time':         {'event':'time',        'type':'continuous',    'length':0,     'offset':0,     'dropout':True, 'text': 'linear ramp from 0 to 1'},
        'pre_licks':    {'event':'licks',       'type':'discrete',      'length':5,     'offset':-5,    'dropout':True, 'text': 'mouse lick'},
        'post_licks':   {'event':'licks',       'type':'discrete',      'length':1,     'offset':0,     'dropout':True, 'text': 'mouse lick'},
        'pre_lick_bouts':   {'event':'lick_bouts','type':'discrete',    'length':5,     'offset':-5,    'dropout':True, 'text': 'lick bout'},
        'post_lick_bouts':  {'event':'lick_bouts','type':'discrete',    'length':1,     'offset':0,     'dropout':True, 'text': 'lick bout'},
        'rewards':      {'event':'rewards',     'type':'discrete',      'length':4,     'offset':-0.5,  'dropout':True, 'text': 'water reward'},
        'change':       {'event':'change',      'type':'discrete',      'length':2,     'offset':0,     'dropout':True, 'text': 'image change'},
        'hits':         {'event':'hit',         'type':'discrete',      'length':3,     'offset':-1,    'dropout':True, 'text': 'lick to image change'},
        'misses':       {'event':'miss',        'type':'discrete',      'length':3,     'offset':-1,    'dropout':True, 'text': 'no lick to image change'},
        'false_alarms':     {'event':'false_alarm',   'type':'discrete','length':3,     'offset':-1,    'dropout':True, 'text': 'lick on catch trials'},
        'correct_rejects':  {'event':'correct_reject','type':'discrete','length':3,     'offset':-1,    'dropout':True, 'text': 'no lick on catch trials'},
        'omissions':    {'event':'omissions',   'type':'discrete',      'length':2.5,   'offset':0,     'dropout':True, 'text': 'image was omitted'},
        'each-image':   {'event':'each-image',  'type':'discrete',      'length':0.8,   'offset':0,     'dropout':True, 'text': 'image presentation'},
        'image_expectation':   {'event':'any-image',  'type':'discrete','length':0.8,   'offset':-0.767,'dropout':True, 'text': '750ms from last image'},
        'running':      {'event':'running',     'type':'continuous',    'length':2,     'offset':-1,    'dropout':True, 'text': 'normalized running speed'},
        'beh_model':    {'event':'beh_model',   'type':'continuous',    'length':.5,    'offset':-.25,  'dropout':True, 'text': 'behavioral model weights'},
        'pupil':        {'event':'pupil',       'type':'continuous',    'length':2,     'offset':-1,    'dropout':True, 'text': 'Z-scored pupil diameter'},
    }
    ## add face motion energy PCs
    for PC in range(5):
        kernels['face_motion_PC_{}'.format(PC)] = {'event':'face_motion_PC_{}'.format(PC), 'type':'continuous', 'length':2, 'offset':-1, 'dropout':True, 'text':'PCA from face motion videos'}
    return kernels


def get_experiment_table(require_model_outputs = False):
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
    readme_file = os.path.join(output_dir, 'README.txt')
    readme = open(readme_file,'w')
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
        'L2_fixed_lambda':50,         # This value is used if L2_use_fixed_value
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
    dropouts = {'Full': {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}}
    for kernel in [kernel for kernel in kernels.keys() if kernels[kernel]['dropout']]:
        dropouts[kernel]={'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
        dropouts[kernel]['kernels'].remove(kernel)
        dropouts[kernel]['dropped_kernels'].append(kernel)

    # Removes all face motion PC kernels as a group
    if 'face_motion_PC_0' in kernel_definitions:
        dropouts['face_motion_energy'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
        kernels_to_drop = [kernel for kernel in dropouts['face_motion_energy']['kernels'] if kernel.startswith('face_motion')] 
        for kernel in kernels_to_drop:
            dropouts['face_motion_energy']['kernels'].remove(kernel)
            dropouts['face_motion_energy']['dropped_kernels'].append(kernel)

    # Removes all individual image kernels
    if 'each-image' in kernel_definitions:
        dropouts['all-images'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
        for i in range(0,8):
            dropouts['all-images']['kernels'].remove('image'+str(i))
            dropouts['all-images']['dropped_kernels'].append('image'+str(i))

    # Removes all Stimulus Kernels, creating the visual dropout
    if ('each-image' in kernel_definitions) or ('any-image' in kernel_definitions) or ('omissions' in kernel_definitions):
        dropouts['visual'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
        if 'each-image' in kernel_definitions:
            for i in range(0,8):
                dropouts['visual']['kernels'].remove('image'+str(i))
                dropouts['visual']['dropped_kernels'].append('image'+str(i))
        if 'omissions' in kernel_definitions:
            dropouts['visual']['kernels'].remove('omissions')
            dropouts['visual']['dropped_kernels'].append('omissions')
        if 'image_expectation' in kernel_definitions:
            dropouts['visual']['kernels'].remove('image_expectation')
            dropouts['visual']['dropped_kernels'].append('image_expectation')
        if 'any-image' in kernel_definitions:
            dropouts['visual']['kernels'].remove('any-image')
            dropouts['visual']['dropped_kernels'].append('any-image')

    # Create behavioral dropout:
    behavioral = ['running','pupil','pre_licks','post_licks','pre_lick_bouts','post_lick_bouts','model_bias','model_task0','model_timing1D','model_omission1']
    if 'face_motion_energy' in dropouts:
        behavioral=behavioral+dropouts['face_motion_energy']['dropped_kernels']
    dropouts['behavioral'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in behavioral:
        if k in kernel_definitions:
            dropouts['behavioral']['kernels'].remove(k)
            dropouts['behavioral']['dropped_kernels'].append(k)

    # Create licking dropout
    licking = ['pre_licks','post_licks','pre_lick_bouts','post_lick_bouts']
    dropouts['licking'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in licking:
        if k in kernel_definitions:
            dropouts['licking']['kernels'].remove(k)
            dropouts['licking']['dropped_kernels'].append(k)

    # Create licking bouts dropout
    licking = ['pre_lick_bouts','post_lick_bouts']
    dropouts['licking_bouts'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in licking:
        if k in kernel_definitions:
            dropouts['licking_bouts']['kernels'].remove(k)
            dropouts['licking_bouts']['dropped_kernels'].append(k)
 
    # Create licking_each_lick dropout
    licking = ['pre_licks','post_licks']
    dropouts['licking_each_lick'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in licking:
        if k in kernel_definitions:
            dropouts['licking_each_lick']['kernels'].remove(k)
            dropouts['licking_each_lick']['dropped_kernels'].append(k)
 
 
    # Create pupil/running 
    pupil_and_running = ['pupil','running']
    dropouts['pupil_and_running'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in pupil_and_running:
        if k in kernel_definitions:
            dropouts['pupil_and_running']['kernels'].remove(k)
            dropouts['pupil_and_running']['dropped_kernels'].append(k)

    # Omissions vs pupil
    pupil_and_omissions = ['pupil','omissions']
    dropouts['pupil_and_omissions'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in pupil_and_omissions:
        if k in kernel_definitions:
            dropouts['pupil_and_omissions']['kernels'].remove(k)
            dropouts['pupil_and_omissions']['dropped_kernels'].append(k)

    # Omissions vs running
    running_and_omissions = ['running','omissions']
    dropouts['running_and_omissions'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in running_and_omissions:
        if k in kernel_definitions:
            dropouts['running_and_omissions']['kernels'].remove(k)
            dropouts['running_and_omissions']['dropped_kernels'].append(k)

    # Create task 
    task = ['hits','misses','false_alarms','correct_rejects','change','rewards']
    dropouts['task'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in task:
        if k in kernel_definitions:
            dropouts['task']['kernels'].remove(k)
            dropouts['task']['dropped_kernels'].append(k)

    # Create trial type 
    trial_type = ['hits','misses','false_alarms','correct_rejects']
    dropouts['trial_type'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in trial_type:
        if k in kernel_definitions:
            dropouts['trial_type']['kernels'].remove(k)
            dropouts['trial_type']['dropped_kernels'].append(k)

    # Create change_and_rewards
    change_and_rewards = ['change','rewards']
    dropouts['change_and_rewards'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in change_and_rewards:
        if k in kernel_definitions:
            dropouts['change_and_rewards']['kernels'].remove(k)
            dropouts['change_and_rewards']['dropped_kernels'].append(k)

    # Create hits_and_rewards
    hits_and_rewards = ['hits','rewards']
    dropouts['hits_and_rewards'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in hits_and_rewards:
        if k in kernel_definitions:
            dropouts['hits_and_rewards']['kernels'].remove(k)
            dropouts['hits_and_rewards']['dropped_kernels'].append(k)

    # Expectation Dropout 
    expectation = ['image_expectation','omissions']
    dropouts['expectation'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in expectation:
        if k in kernel_definitions:
            dropouts['expectation']['kernels'].remove(k)
            dropouts['expectation']['dropped_kernels'].append(k)
 
    # Create cognitive 
    cognitive = ['hits','misses','false_alarms','correct_rejects','change','rewards']
    dropouts['cognitive'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
    for k in cognitive:
        if k in kernel_definitions:
            dropouts['cognitive']['kernels'].remove(k)
            dropouts['cognitive']['dropped_kernels'].append(k)


    # Remove all behavior model kernels
    if 'beh_model' in kernel_definitions:
        dropouts['beh_model'] = {'kernels':list(kernels.keys()),'dropped_kernels':[],'is_single':False}
        dropouts['beh_model']['kernels'].remove('model_bias')
        dropouts['beh_model']['kernels'].remove('model_task0')
        dropouts['beh_model']['kernels'].remove('model_timing1D')
        dropouts['beh_model']['kernels'].remove('model_omissions1')
        dropouts['beh_model']['dropped_kernels'].append('model_bias')
        dropouts['beh_model']['dropped_kernels'].append('model_task0')
        dropouts['beh_model']['dropped_kernels'].append('model_timing1D')
        dropouts['beh_model']['dropped_kernels'].append('model_omissions1')
        dropouts['cognitive']['kernels'].remove('model_bias')
        dropouts['cognitive']['kernels'].remove('model_task0')
        dropouts['cognitive']['kernels'].remove('model_timing1D')
        dropouts['cognitive']['kernels'].remove('model_omissions1')
        dropouts['cognitive']['dropped_kernels'].append('model_bias')
        dropouts['cognitive']['dropped_kernels'].append('model_task0')
        dropouts['cognitive']['dropped_kernels'].append('model_timing1D')
        dropouts['cognitive']['dropped_kernels'].append('model_omissions1')

    
    # Adds single kernel dropouts:
    for drop in [drop for drop in dropouts.keys()]:
        if (drop is not 'Full') & (drop is not 'intercept'):
            # Make a list of kernels by taking the difference between the kernels in 
            # the full model, and those in the dropout specified by this kernel.
            # This formulation lets us do single kernel dropouts for things like beh_model,
            # or all-images
            kernels = set(dropouts['Full']['kernels'])-set(dropouts[drop]['kernels'])
            kernels.add('intercept') # We always include the intercept
            dropped_kernels = set(dropouts['Full']['kernels']) - kernels
            dropouts['single-'+drop] = {'kernels':list(kernels),'dropped_kernels':list(dropped_kernels),'is_single':True} 
    
    for drop in dropouts.keys():
        assert len(dropouts[drop]['kernels']) + len(dropouts[drop]['dropped_kernels']) == len(dropouts['Full']['kernels']), 'bad length'
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

def describe_model_version(version):
    '''
        Prints a text description of the model kernels and dropouts. Tries to load the information from v_alex_test
    '''
    run_params = load_run_json(version)
    just_for_text = load_run_json('alex_test')   

    print('\nThe model contains the following kernels:') 
    for kernel in run_params['kernels']:
        if 'text' in run_params['kernels'][kernel]:
            text = run_params['kernels'][kernel]['text']       
        else:
            if kernel not in just_for_text['kernels']:
                text = 'no description available'
            else:
                text = just_for_text['kernels'][kernel]['text']              
        if run_params['kernels'][kernel]['type'] == 'discrete':
            start = np.round(run_params['kernels'][kernel]['offset'],2)
            end  = np.round(run_params['kernels'][kernel]['length'] + start,1)
            print(kernel.ljust(18) + " is aligned from ("+str(start)+", "+str(end)+") seconds around each "+text)
        else:
            print(kernel.ljust(18) + " runs the full length of the session, and is "+ text)
    
    print('\nThe model contains the following dropout, or reduced models:') 
    for d in run_params['dropouts']:
        if 'is_single' in run_params['dropouts'][d]:
            is_single=run_params['dropouts'][d]['is_single']
        elif d in just_for_text['dropouts'][d]:
            is_single=just_for_text['dropouts'][d]['is_single']
        else:
            is_single=False
        if is_single:
            k = run_params['dropouts'][d]['kernels']
            if 'intercept' in k:
                k.remove('intercept')
            if len(k) > 1:
                print(d.ljust(25) +" contains just the kernels "+ ', '.join(k))
            else:
                print(d.ljust(25) +" contains just the kernel "+ ', '.join(k))   
        else:
            if 'dropped_kernels' in run_params['dropouts'][d]:
                drops = run_params['dropouts'][d]['dropped_kernels']
            elif d in just_for_text['dropouts']:
                drops = just_for_text['dropouts'][d]['dropped_kernels']
            else:
                drops = ['?']
            if len(drops) == 0:
                print(d.ljust(25) +" contains all kernels")     
            else:
                print(d.ljust(25) +" contains all kernels except "+', '.join(drops))    
 

