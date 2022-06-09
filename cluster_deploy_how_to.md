# This document details my current workflow for deploying model fits for every experiment to the cluster
dougo - 8/7/2020
alexpiet - 06/09/2022

## The basic workflow is as follows:

### Ensure that the run_json for the version you want to deploy is up to date in the frozen model folder
    
    $ import visual_glm_params.GLM_params as glm_params
    $ glm_params.make_run_json(
        <version>,
        label=<text description>,
        username=<your name>,
        src_path=<src_path>,
        TESTING=False,
        include_4x2=<your choice>
        )   
 
### Run a test fit locally

The `scripts/fit_glm.py` script will fit the model and log the results to mongo. To make sure there aren't any code bugs before deploying all 1000+ jobs to the cluster, I've been running a test version locally using the following syntax at the command line:

    $ python fit_glm.py --oeid {OPHYS_EXPERIMENT_ID} --version {GLM VERSION}
    
If that job completes without error, it might be a good idea to check that the dropout summary successfully logged to mongo (and the numbers make sense). See the notebook detailing interactions with mongo to explain how to do that.

### log in to hpc-login to prepare to deploy all jobs

    $ ssh USERNAME@hpc-login.corp.alleninstitute.org

### Activate your environment on the scheduler node

The script that deploys jobs uses the visual_behavior package to get a full list of experiment IDs. So you need to run it from an environment that has `visual_behavior_analysis` installed

    $ source activate {ENVIRONMENT NAME}
    
### Deploy the jobs

The `scripts/deploy_glm_fits.py` script will iterate over the `filtered_ophys_experiment_table` to generate one cluster job per passed experiment. All of the job settings are defined inside of this script. In addition, this script assumes that you have your python executable in `~/.conda/envs/{ENVIRONMENT NAME}/bin/python`

The arguments are as follows:
* --env {YOUR ENVIRONMENT NAME}
* --version {GLM VERSION TO DEPLOY}
* --src-path {PATH TO THE VISUAL_BEHAVIOR_GLM FOLDER IN YOUR HOME DIRECTORY, E.G. /home/dougo/code/visual_behavior_glm}
* --force-overwite {NO FOLLOWING ARGUMENT - A FLAG TO DETERMINE WHETHER TO DEPLOY JOBS FOR ALL EXPERIMENTS, OVERWRITING EXISTING RESULTS. IF NOT CALLED, ONLY JOBS FOR EXPERIMENTS WITH NO CACHED RESULTS WILL BE DEPLOYED.
    
For example, I'd call this script from the hpc-login command line as follows:

    $ python deploy_glm_fits.py --env visual_behavior --version 5_L2_fixed_lambda=1 --src_path /home/dougo/code/visual_behavior_glm 

Or, you can modify the bash script that executes those functions, so you dont need to remember them

    $ ./deploy_glm_fits.sh   
 
### Check job status

You can see the status of all jobs by typing the following at the hpc-login command line

    $ squeue -u {USERNAME}
