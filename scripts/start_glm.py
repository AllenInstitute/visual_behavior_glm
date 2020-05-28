import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/')
from pbstools import pbstools
import os
import numpy as np
import json
python_executable = r"/home/nick.ponvert/anaconda3/envs/allen/bin/python"

# Load json file that contains the parameters for this model iteration
#   as well as a list of sessions to fit
run_param_json = sys.argv[1]
with open(run_param_json, 'r') as json_file:
    run_params = json.load(json_file)
ophys_session_ids = run_params['ophys_sessions']   

# Settings for HPC Job
# TODO these settings should be in run_params
job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '2:00:00',
                'ppn':4,
                }

# Start a job for each session
if __name__=="__main__":
    for osid in ophys_session_ids:

        args = ['--ophys-session-id {}'.format(osid),
                '--output-dir {}'.format(run_params['output_dir']),
                '--regularization-lambda {}'.format(run_params['regularization_lambda'])
                ]
        args_string = ' '.join(args)

        job_title = 'session_{}'.format(osid)
        pbstools.PythonJob(
            run_params['python_file'],
            python_executable,
            python_args = args_string,
            jobname = job_title,
            jobdir = run_params['job_dir'],
            **job_settings
        ).run(dryrun=False)

