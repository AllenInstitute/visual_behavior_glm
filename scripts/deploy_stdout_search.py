import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

import visual_behavior.data_access.loading as loading

sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='find and log STDOUT values')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')
parser.add_argument('--glm-version', type=str, default='7_L2_optimize_by_session', metavar='glm version')

job_dir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/job_records"

job_settings = {'queue': 'braintv',
                'mem': '2g',
                'walltime': '00:05:00',
                'ppn': 1,
                }

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = os.path.join(os.getcwd(), "get_stdout_contents.py")
    print('python file = {}'.format(python_file))

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiment_ids = experiments_table.index.values
    job_count = 0

    job_string = "--oeid {} --glm-version {}"

    for experiment_id in experiment_ids:

        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(experiment_id, job_count))
        job_title = 'log_STDOUT_for_oeid_{}_fit_glm_v_{}'.format(experiment_id, args.glm_version)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args=job_string.format(experiment_id, args.glm_version),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)