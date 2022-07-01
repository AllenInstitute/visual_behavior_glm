#!/bin/bash
#SBATCH --job-name=deploy    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=saaketh.medepalli@alleninstitute.org     # Where to send mail  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=10gb                     # Job memory request (per node)
#SBATCH --time=05:00:00               # Time limit hrs:min:sec
#SBATCH --output=test_%j.log   # Standard output and error log
#SBATCH --partition braintv         # Partition used for processing
#SBATCH --tmp=10G                     # Request the amount of space your jobs needs on /scratch/fast

python deploy_glm_fits.py --version 51_medepalli_omission_specific --env-path /allen/aics/apps/hpc_shared/mod/anaconda3-5.3.0/envs/visual_behavior_glm --src-path /home/saaketh.medepalli/visual_behavior_glm --job-end-fraction 1

 
