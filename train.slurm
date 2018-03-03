#!/bin/bash

#SBATCH --job-name=DistLearn
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /home/kru03a/chessbot/pyenv/bin/activate
module load cuda/8.0.61
module load cudnn/v6
cd /home/kru03a/chessbot/reinforcement

start=`date +%s`

python train.py

end=`date +%s`
runtime=$((end-start))
echo "Total Runtime: $runtime"
