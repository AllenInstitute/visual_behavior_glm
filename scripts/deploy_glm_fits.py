import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

import visual_behavior.data_access.loading as loading
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')
parser.add_argument('--glm-version', type=str, default='0', metavar='glm version')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/ophys_glm"

job_settings = {'queue': 'braintv',
                'mem': '16g',
                'walltime': '4:00:00',
                'ppn': 1,
                }

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/visual_behavior_glm/scripts/fit_glm.py".format(os.path.expanduser('~'))

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiment_ids = experiments_table.query('model_outputs_available == True').index.values

    for experiment_id in experiment_ids:

        print('starting cluster job for {}'.format(experiment_id))
        job_title = 'oeid_{}_fit_glm'.format(experiment_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--oeid {} --version {}".format(experiment_id, args.glm_version),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)