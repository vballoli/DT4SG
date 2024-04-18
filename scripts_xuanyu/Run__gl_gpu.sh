#!/bin/sh

script_name=$1
algo_name=$2
online_algo_name=$3
patient_id=$4
online_seed=$5

job_name="${algo_name}_${online_algo_name}_${patient_id}_${online_seed}"

sbatch --account=stats_dept1 \
  --partition=gpu \
  --gpus-per-task=1 \
  --nodes=1 \
  --mem-per-gpu=6GB \
  --time=00-26:00:00\
  --job-name=$job_name \
  --output=${job_name}_%A.out \
  $script_name $algo_name $online_algo_name $patient_id $online_seed
