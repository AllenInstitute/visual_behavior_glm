import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np

from simple_slurm import Slurm
import visual_behavior.database as db
import visual_behavior_glm.build_dataframes as bd
import psy_output_tools as po

BEHAVIOR_VERSION = 21
parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env-path', type=str, default='visual_behavior', metavar='path to conda environment to use')

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "/home/alex.piet/codebase/GLM/visual_behavior_glm/scripts/response_dataframes.py"
    glm_version = '24_events_all_L2_optimize_by_session'
    stdout_basedir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm"
    stdout_location = os.path.join(stdout_basedir, 'job_records_response_dataframes')
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))

    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    oeids = np.concatenate(summary_df['ophys_experiment_id'].values) 

    job_count = 0

    job_string = "--ophys_experiment_id {}"

    n_cell_ids = len(oeids)

    for oeid in oeids:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(oeid, job_count))
        job_title = 'oeid_{}'.format(oeid)
        walltime = '2:00:00'
        mem = '100gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_"+str(oeid)+".out"
        
        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=4,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output= output,
            partition="braintv"
        )
    
        args_string = job_string.format(oeid)
        slurm.sbatch('{} {} {}'.format(
                python_executable,
                python_file,
                args_string,
            )
        )
        time.sleep(0.001)
