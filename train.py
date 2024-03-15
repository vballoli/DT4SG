import gym
from utils import get_env, seeding_function, get_device, hash_seed

import gym
import d3rlpy
import mlxp
import torch
import numpy as np

from simglucose.envs.simglucose_gym_env import T1DSimEnv

@mlxp.launch(config_path='./configs',
            seeding_function=seeding_function)
def train(ctx: mlxp.Context)->None:
    cfg = ctx.config
    logger = ctx.logger

    # Create the environment
    env = T1DSimEnv(patient_name='adolescent#002')
    

    device = get_device()
    # Algorithm
    algo_config = cfg.algo_config
    algo = d3rlpy.algos.CQLConfig(**algo_config).create("cpu")


    

if __name__ == "__main__":
    train()