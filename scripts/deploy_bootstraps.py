import os
import argparse
import time
import pandas as pd

from simple_slurm import Slurm
import visual_behavior_glm.PSTH as psth

parser = argparse.ArgumentParser(description='deploy glm fits to cluster')
parser.add_argument('--env-path', type=str, default='visual_behavior', metavar='path to conda environment to use')


def already_fit(row):
    filename = psth.get_hierarchy_filename(
        row.cell_type,
        row.response,
        row['data'],
        row.depth,
        row.nboots,
        row.splits,
        row.extra)
    return os.path.exists(filename) 

def get_bootstrap_jobs():
    nboots=10000
    jobs = [
        {'cell_type':'exc','response':'image','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'sst','response':'image','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'vip','response':'image','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'exc','response':'image','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'sst','response':'image','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'vip','response':'image','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'exc','response':'image','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'sst','response':'image','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'vip','response':'image','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'exc','response':'image','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'sst','response':'image','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'vip','response':'image','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'exc','response':'omission','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'sst','response':'omission','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'vip','response':'omission','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'exc','response':'omission','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'sst','response':'omission','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'vip','response':'omission','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'exc','response':'omission','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'sst','response':'omission','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'vip','response':'omission','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'exc','response':'omission','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'sst','response':'omission','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'vip','response':'omission','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'exc','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'sst','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'vip','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'exc','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'sst','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'vip','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':[],'query':'','extra':''},
        {'cell_type':'exc','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'sst','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'vip','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'exc','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'sst','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'vip','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'','extra':''},
        {'cell_type':'exc','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 1)"','extra':'hit'},
        {'cell_type':'sst','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 1)"','extra':'hit'},
        {'cell_type':'vip','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 1)"','extra':'hit'},
        {'cell_type':'exc','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 1)"','extra':'hit'},
        {'cell_type':'sst','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 1)"','extra':'hit'},
        {'cell_type':'vip','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 1)"','extra':'hit'},
        {'cell_type':'exc','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 0)"','extra':'miss'},
        {'cell_type':'sst','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 0)"','extra':'miss'},
        {'cell_type':'vip','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 0)"','extra':'miss'},
        {'cell_type':'exc','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 0)"','extra':'miss'},
        {'cell_type':'sst','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 0)"','extra':'miss'},
        {'cell_type':'vip','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['visual_strategy_session'],'query':'"(hit == 0)"','extra':'miss'},
        {'cell_type':'exc','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['hit'],'query':'visual_strategy_session','extra':'visual'},
        {'cell_type':'sst','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['hit'],'query':'visual_strategy_session','extra':'visual'},
        {'cell_type':'vip','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['hit'],'query':'visual_strategy_session','extra':'visual'},
        {'cell_type':'exc','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['hit'],'query':'visual_strategy_session','extra':'visual'},
        {'cell_type':'sst','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['hit'],'query':'visual_strategy_session','extra':'visual'},
        {'cell_type':'vip','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['hit'],'query':'visual_strategy_session','extra':'visual'},
        {'cell_type':'exc','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['hit'],'query':'not visual_strategy_session','extra':'timing'},
        {'cell_type':'sst','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['hit'],'query':'not visual_strategy_session','extra':'timing'},
        {'cell_type':'vip','response':'change','data':'events','depth':'layer','nboots':nboots,'splits':['hit'],'query':'not visual_strategy_session','extra':'timing'},
        {'cell_type':'exc','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['hit'],'query':'not visual_strategy_session','extra':'timing'},
        {'cell_type':'sst','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['hit'],'query':'not visual_strategy_session','extra':'timing'},
        {'cell_type':'vip','response':'change','data':'events','depth':'binned_depth','nboots':nboots,'splits':['hit'],'query':'not visual_strategy_session','extra':'timing'},
        ]
    jobs = pd.DataFrame(jobs)
    return jobs

def make_job_string(row):
    arg_string = ''
    arg_string += ' --cell_type {}'.format(row.cell_type)
    arg_string += ' --response {}'.format(row.response)
    arg_string += ' --data {}'.format(row['data'])
    arg_string += ' --depth {}'.format(row.depth)
    arg_string += ' --nboots {}'.format(row.nboots) 
    if len(row.splits) > 0:
        arg_string += ' --splits '+' '.join(row.splits)
    if row.query is not '':
        arg_string += ' --query {}'.format(row.query)    
    if row.extra is not '':
        arg_string += ' --extra {}'.format(row.extra)    
    return arg_string

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "/home/alex.piet/codebase/GLM/visual_behavior_glm/scripts/bootstrap.py"  
    stdout_basedir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm" 
    stdout_location = os.path.join(stdout_basedir, 'job_records_bootstraps')
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))

    job_count = 0
    jobs = get_bootstrap_jobs()

    for index, row in jobs.iterrows():
        if not already_fit(row):
            job_count += 1
            args_string = make_job_string(row)
            print('starting cluster job. job count = {}'.format(job_count))
            print('   ' + args_string)
            job_title = 'bootstraps'
            walltime = '24:00:00'
            mem = '100gb'
            job_id = Slurm.JOB_ARRAY_ID
            job_array_id = Slurm.JOB_ARRAY_MASTER_ID
            output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_bootstrap.out"
            
            # instantiate a SLURM object
            slurm = Slurm(
                cpus_per_task=4,
                job_name=job_title,
                time=walltime,
                mem=mem,
                output= output,
                partition="braintv"
            )

            slurm.sbatch('{} {} {}'.format(
                    python_executable,
                    python_file,
                    args_string,
                )
            )
            time.sleep(0.001)


