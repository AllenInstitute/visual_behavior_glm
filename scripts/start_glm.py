import os
import json
import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob
python_executable = '/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python'

# Load json file that contains the parameters for this model iteration
VERSION = sys.argv[1]
OUTPUT_DIR_BASE = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/'
run_param_json = OUTPUT_DIR_BASE +'v_'+str(VERSION)+'/run_params.json'
with open(run_param_json, 'r') as json_file:
    run_params = json.load(json_file)
ophys_experiment_ids = run_params['ophys_experiment_ids']   
job_settings = run_params['job_settings']

# Start a job for each experiment
if __name__=="__main__":
    for oeid in ophys_experiment_ids:

        args = ['--ophys-experiment-id {}'.format(oeid),
                '--json {}'.format(run_param_json),
                ]
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

