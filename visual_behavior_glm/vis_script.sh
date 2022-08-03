#!/bin/bash
#SBATCH --job-name=exp_var               # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=saaketh.medepalli@alleninstitute.org     # Where to send mail  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=128gb                     # Job memory request (per node)
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=test_%j.log   # Standard output and error log
#SBATCH --partition braintv         # Partition used for processing
#SBATCH --tmp=10G                     # Request the amount of space your jobs needs on /scratch/fast

echo "Starting job"
python platform_paper_figure_script.py
