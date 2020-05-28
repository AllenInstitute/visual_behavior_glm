import os
import numpy as np
import json
import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/') # TODO, update?
from pbstools import pbstools
python_executable = r"/home/nick.ponvert/anaconda3/envs/allen/bin/python" # TODO, update?

# Load json file that contains the parameters for this model iteration
#   as well as a list of sessions to fit
run_param_json = sys.argv[1]
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
        pbstools.PythonJob(
            run_params['python_fit_script'],
            python_executable,
            python_args = args_string,
            jobname = job_title,
            jobdir = run_params['job_dir'],
            **job_settings
        ).run(dryrun=False)

