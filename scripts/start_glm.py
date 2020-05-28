import os
import json
import sys

# Path to pbstools updated to work for python 3
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob

# Path to python executable
python_executable = '/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python'

# Path to where the json file will be located
OUTPUT_DIR_BASE = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'

# get what version number to run
VERSION = sys.argv[1]

# Load json file that contains the parameters for this model version 
run_param_json = OUTPUT_DIR_BASE +'v_'+str(VERSION)+'/run_params.json'
with open(run_param_json, 'r') as json_file:
    run_params = json.load(json_file)

# Extract settings
ophys_experiment_ids    = run_params['ophys_experiment_ids']   
job_settings            = run_params['job_settings']

# Start a job for each experiment
if __name__=="__main__":
    for oeid in ophys_experiment_ids:
        filename = run_params['experiment_output_dir']+str(oeid)+".pkl" 
        if os.path.isfile(filename):
            # If the output file already exists, it will not over-ride
            print(str(oeid) + " Already done!")
        else:
            # No output file found, starting python job for this experiment
            args = [str(oeid),str(run_param_json)]
            args_string = ' '.join(args)
            job_title = 'oeid_{}'.format(oeid)
            PythonJob(
                run_params['fit_script'],
                python_executable,
                python_args = args_string,
                jobname = job_title,
                jobdir = run_params['job_dir'],
                **job_settings
            ).run(dryrun=False)

