import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

import visual_behavior.data_access.loading as loading
import visual_behavior_glm.GLM_analysis_tools as gat

sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')
parser.add_argument('--version', type=str, default='0', metavar='glm version')
parser.add_argument(
    '--src-path', 
    type=str, 
    default='',
    metavar='src_path',
    help='folder where code lives'
)
parser.add_argument(
    '--force-overwrite', 
    action='store_true',
    default=False,
    dest='force_overwrite', 
    help='Overwrites existing fits for this version if enabled. Otherwise only experiments without existing results are fit'
)
parser.add_argument(
    '--use-previous-fit', 
    action='store_true',
    default=False,
    dest='use_previous_fit', 
    help='use previous fit if it exists (boolean, default = False)'
)

job_dir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/ophys_glm"

job_settings = {'queue': 'braintv',
                'mem': '64g',
                'walltime': '12:00:00',
                'ppn': 16,
                }

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/scripts/fit_glm.py".format(args.src_path)

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiment_ids = experiments_table.index.values
    job_count = 0

    if args.use_previous_fit:
        job_string = "--oeid {} --version {} --use-previous-fit"
    else:
        job_string = "--oeid {} --version {}"

    for experiment_id in experiment_ids:

        if args.force_overwrite or not gat.already_fit(experiment_id, args.version):
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(experiment_id, job_count))
            job_title = 'oeid_{}_fit_glm_v_{}'.format(experiment_id, args.version)
            pbstools.PythonJob(
                python_file,
                python_executable,
                python_args=job_string.format(experiment_id, args.version),
                jobname=job_title,
                jobdir=job_dir,
                **job_settings
            ).run(dryrun=False)
            time.sleep(0.001)