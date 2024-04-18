#!/bin/sh

algo_name=$1
online_algo_name=$2
patient_id=$3
online_seed=$4

python train_offline.py "${algo_name}" "${online_algo_name}" "${patient_id}" "${online_seed}"
