#!/bin/bash

source /home/${USER}/.bashrc

source activate dt4sg

python collect_data.py algo_config='bb' ++seed=$1