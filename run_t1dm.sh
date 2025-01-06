#!/bin/bash

#SBATCH --account=xxxx
#SBATCH -p xxxx
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem-per-cpu=200M
#SBATCH -t 2:00:00
#SBATCH -o OUTPUT/out_%A_%a.txt
#SBATCH -e OUTPUT/err_%A_%a.txt
#SBATCH --array 0-99
#SBATCH -J T1DM

python T1DM.py $SLURM_ARRAY_TASK_ID
