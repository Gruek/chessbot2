#!/bin/bash

module load tensorflow/1.6.0-py36-gpu
source /home/kru03a/chbot/pyenv/bin/activate
cd /home/kru03a/chbot/chbot2

python -OO -m PyInstaller --add-data="/data/kru03a/chbot/data/model_densenet.h5:data" uci_adapter.py
