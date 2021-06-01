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
parser.add_argument('--env-path', type=str, default='visual_behavior', metavar='path to conda environment to use')
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
    '--testing', 
    action='store_true',
    default=False,
    dest='testing', 
    help='If this flag is called, only 10 test sessions will be deployed'
)
parser.add_argument(
    '--use-previous-fit', 
    action='store_true',
    default=False,
    dest='use_previous_fit', 
    help='use previous fit if it exists (boolean, default = False)'
)
parser.add_argument(
    '--job-start-fraction', 
    type=float, 
    default=0.0,
    metavar='start_fraction',
    help='which fraction of all jobs to start on. useful if splitting jobs amongst users. Default = 0.0'
)
parser.add_argument(
    '--job-end-fraction', 
    type=float, 
    default=1.0,
    metavar='end_fraction',
    help='which fraction of all jobs to end on. useful if splitting jobs amongst users. Default = 1.0'
)

job_dir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm/cluster_records_v13"

walltime = '{}:00:00'
mem = '{}g'
job_settings = {'queue': 'braintv',
                'mem': '16g',
                'walltime': '2:00:00',
                'ppn': 16,
                }

def calculate_required_mem(roi_count):
    '''calculate required memory in GB'''
    return 12 + 0.25*roi_count

def calculate_required_walltime(roi_count):
    '''calculate required walltime in hours'''
    return 10 + 0.125*roi_count

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/scripts/fit_glm.py".format(args.src_path)

    if args.testing:
        experiments_table = gat.select_experiments_for_testing(returns = 'dataframe')
    else:
        experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True).reset_index()

    job_count = 0

    # get ROI count for each experiment
    experiments_table['roi_count'] = experiments_table['ophys_experiment_id'].map(lambda oeid: gat.get_roi_count(oeid))

    experiment_ids = experiments_table['ophys_experiment_id'].values

    if args.use_previous_fit:
        job_string = "--oeid {} --version {} --use-previous-fit"
    else:
        job_string = "--oeid {} --version {}"

    n_experiment_ids = len(experiment_ids)
    for experiment_id in experiment_ids[int(n_experiment_ids * args.job_start_fraction): int(n_experiment_ids * args.job_end_fraction)]:

        # calculate resource needs based on ROI count
        roi_count = experiments_table.query('ophys_experiment_id == @experiment_id').iloc[0]['roi_count']
        job_settings['walltime'] = walltime.format(int(np.ceil((calculate_required_walltime(roi_count)))))
        job_settings['mem'] = mem.format(int(np.ceil((calculate_required_mem(roi_count)))))

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
