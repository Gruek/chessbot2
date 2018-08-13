#!/bin/bash

#SBATCH --job-name=gbrchip
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

module load tensorflow/1.6.0-py36-gpu
source /home/kru03a/chbot/pyenv/bin/activate
cd /home/kru03a/chbot/chbot2

start=`date +%s`

PATH=$PATH:/home/kru03a/chbot/chbot2 python train.py

end=`date +%s`
runtime=$((end-start))
echo "Total Runtime: $runtime"
