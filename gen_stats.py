import os
import d3rlpy
import mlxp
import torch
import numpy as np 
import pandas as pd

def f_mr(bt):
    return 1.509*(np.log(bt)**(1.084) - 5.381)

def risk(bt):
    return 10*(3.35506*np.log(bt)**(0.8353) - 3.7932)**2

def HBGI(b: np.ndarray):
    return np.mean(risk(b)*(f_mr(b) > 0))

def LBGI(b: np.ndarray):
    return np.mean(risk(b)*(f_mr(b) < 0))

datasets = ['pid_expert','random', 'sac', 'td3']


cols = ['policy', 'mean', 'variance', 'HBGI', 'LBGI']

seeds = [0,1,2]

stats_df = pd.DataFrame(columns=cols)
for dataset in datasets:
    obs = np.array([], dtype=np.float32)
    rewards = np.array([], dtype=np.float32)
    hbgi =  np.array([], dtype=np.float32)
    lbgi =  np.array([], dtype=np.float32)
    
    with open(f'replays/{dataset}_adolescent#001_{0}.h5', "rb") as f:
        df = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    
    for i in range(len(df.episodes)):
        obs = np.append(obs, df.episodes[i].observations)
        rewards = np.append(rewards, df.episodes[i].rewards)
        hbgi = np.append(hbgi, HBGI(df.episodes[i].observations))
        lbgi = np.append(lbgi, LBGI(df.episodes[i].observations))
    
    for seed in seeds:
        with open(f'replays/{dataset}_adolescent#001_{seed}.h5', "rb") as f:
            df = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
            for i in range(len(df.episodes)):
                obs = np.append(obs, df.episodes[i].observations)
                rewards = np.append(rewards, df.episodes[i].rewards)
                hbgi = np.append(hbgi, HBGI(df.episodes[i].observations))
                lbgi = np.append(lbgi, LBGI(df.episodes[i].observations))
        
        
    mean = np.mean(rewards)
    variance = np.var(rewards)
    hbgi = np.mean(hbgi)
    lbgi = np.mean(lbgi)
    policy = 'pid' if 'pid' in dataset else dataset
    # print(pd.DataFrame([[policy, mean, variance, hbgi, lbgi]], 
    #                                   columns=cols))
    stats_df = pd.concat([stats_df,
                            pd.DataFrame([[policy, mean, variance, hbgi, lbgi]], 
                                        columns=cols)],
                            ignore_index=True)

stats_df.to_csv('stats.csv', index=False)