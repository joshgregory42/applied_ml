#!/bin/bash
#SBATCH --nodes=1                           # Number of requested nodes
#SBATCH --ntasks=20        	            # Number of requested cores
#SBATCH --mem=32G                            # Request 4 GB of memory
#SBATCH --time=24:00:00                      # Max walltime in h:m:s
#SBATCH --gres=gpu:1
#SBATCH --partition=aa100                # Alpine testing nodes (see https://tinyurl.com/4betvt6n)
#SBATCH --job-name=large_tune                 # job name
#SBATCH --mail-type=BEGIN,END,FAIL                # Email when job ends and/or fails
#SBATCH --mail-user=jogr4852@colorado.edu   # Email address
#SBATCH --account=ucb482_asc1
#SBATCH --qos=normal                           # QOS (see https://tinyurl.com/4betvt6n)
#SBATCH --output=./output/cnn_large_tune.out	    # Output file name


# Written by:	Shelley Knuth, 24 February 2014
# Updated by:   Andrew Monaghan, 08 March 2018
# Updated by:   Kim Kanigel Winner, 23 June 2018
# Updated by:   Shelley Knuth, 17 May 2019
# Updated by:   Josh Gregory, 29 March 2024
# Purpose:      PyTorch job scripting template

# Purge all existing modules
module purge

# Load the Anaconda module
module load anaconda/2023.09


# deactivate all conda environments, then activate thesis env.
conda deactivate
conda activate applied_ml


python3 final_cnn_large_tune.py