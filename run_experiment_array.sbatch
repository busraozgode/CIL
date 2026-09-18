#!/bin/bash

#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 0-36:00
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%A_%a.out
#SBATCH -e slurm.%A_%a.err

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "================================"

if [ -f "/home/ozgodeyi/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/ozgodeyi/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/ozgodeyi/anaconda3/bin:$PATH"
fi

conda activate avalanche_r

cd /home/ozgodeyi/CIL

python main.py
