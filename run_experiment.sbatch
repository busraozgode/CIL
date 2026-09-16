#!/bin/bash
#SBATCH -p GPU              # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -t 0-36:00          # max run time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out  # standard output log
#SBATCH -e slurm.%N.%j.err  # standard error log
#SBATCH --gres=gpu:1        # request one GPU

if [ -z "$DATASET" ] || [ -z "$STRATEGY" ]; then
    echo "ERROR: DATASET and STRATEGY must be provided."
    exit 1
fi

# Activate conda environment properly
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

conda activate avalanche_r

# Go to your working directory
cd /home/ozgodeyi/CIL

# Run your script
python main.py --dataset "$DATASET" --strategy "$STRATEGY"