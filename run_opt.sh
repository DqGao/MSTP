#!/bin/bash

#SBATCH --account=xxxx
#SBATCH -p xxxx
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1g
#SBATCH -t 2:00:00
#SBATCH -o OUTPUT/out_%A_%a.txt
#SBATCH -e OUTPUT/err_%A_%a.txt
#SBATCH --array 0-19
#SBATCH -J TrueOptimal

python TrueOptimal.py $SLURM_ARRAY_TASK_ID
