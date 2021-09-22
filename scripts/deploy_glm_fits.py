import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import visual_behavior_glm.GLM_analysis_tools as gat

from simple_slurm import Slurm

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

    stdout_basedir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm"
    stdout_location = os.path.join(stdout_basedir, 'job_records_{}'.format(args.version))
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))

    if args.testing:
        experiments_table = gat.select_experiments_for_testing(returns = 'dataframe')
    else:
        cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
        cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
        experiments_table = cache.get_ophys_experiment_table()
        experiments_table = experiments_table[(experiments_table.project_code!="VisualBehaviorMultiscope4areasx2d")&(experiments_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()
    print('experiments table loaded')

    # get ROI count for each experiment
    experiments_table['roi_count'] = experiments_table['ophys_experiment_id'].map(lambda oeid: gat.get_roi_count(oeid))
    print('roi counts extracted')

    job_count = 0

    if args.use_previous_fit:
        job_string = "--oeid {} --version {} --use-previous-fit"
    else:
        job_string = "--oeid {} --version {}"

    experiment_ids = experiments_table['ophys_experiment_id'].values
    n_experiment_ids = len(experiment_ids)

    for experiment_id in experiment_ids[int(n_experiment_ids * args.job_start_fraction): int(n_experiment_ids * args.job_end_fraction)]:

        # calculate resource needs based on ROI count
        roi_count = experiments_table.query('ophys_experiment_id == @experiment_id').iloc[0]['roi_count']

        if False:#if args.force_overwrite or not gat.already_fit(experiment_id, args.version):
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(experiment_id, job_count))
            job_title = 'oeid_{}_fit_glm_v_{}'.format(experiment_id, args.version)

            walltime = '{}:00:00'.format(int(np.ceil((calculate_required_walltime(roi_count)))))
            mem = '{}gb'.format(int(np.ceil((calculate_required_mem(roi_count)))))
            job_id = Slurm.JOB_ARRAY_ID
            job_array_id = Slurm.JOB_ARRAY_MASTER_ID
            output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+".out"
    
            # instantiate a SLURM object
            slurm = Slurm(
                cpus_per_task=16,
                job_name=job_title,
                time=walltime,
                mem=mem,
                output= output,
            )

            args_string = job_string.format(experiment_id, args.version)
            slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                )
            )

            time.sleep(0.001)
