#!/bin/bash

source /home/${USER}/.bashrc

source activate dt4sg

python collect_data.py algo_config='pid_expert' ++seed=$1