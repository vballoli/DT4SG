from utils import seeding_function, get_device, get_algo

import os
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace
import mlxp
import torch

from simglucose.envs.simglucose_gym_env import T1DSimEnv
from typing import Any


from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController

from gym.utils import seeding
from datetime import datetime


from pid_params  import pid_params


class T1DSimEnvOurs(T1DSimEnv):

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> torch.Tuple[Any | dict]:
        return self._reset(), {}
    
    def step(self, action: Any) -> torch.Tuple[Any | float | bool | dict]:
        step= self._step(action)
        obs, reward, done, info = step
        return obs, reward, done, False, info
    
    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.integers(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.integers(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4


class RandomPolicy:
    
    def __init__(self, discrete=False) -> None:
        self.discrete = discrete
    
    def policy(self, **kwargs):
        
        if self.discrete:
            grid = np.arange(0.01, 1, step = 0.01)
            return np.random.choice(grid)
        else:
            return np.clip(0, np.abs(np.random.randn()*.5), 1.)


@mlxp.launch(config_path='./configs',
            seeding_function=seeding_function)
def collect(ctx: mlxp.Context)->None:
    cfg = ctx.config
    logger = ctx.logger

    # Create the environment
    env = T1DSimEnvOurs(patient_name=f'{cfg.patient_type}#00{cfg.patient_number}')
    

    device = get_device()
    # Algorithm
    algo_config = cfg.algo_config
    algo_name = algo_config['name']
    # pop the name from the config
    algo_config.pop('name')
    num_steps = algo_config['num_steps']
    # pop the num_steps from the config
    algo_config.pop('num_steps')
    
    
    
    # Define Random policies
    if algo_name == 'random':
        action_size = 1
        action_space = ActionSpace.CONTINUOUS
        policy = RandomPolicy()
    elif algo_name == "discrete_random":
        action_size = len(np.arange(0.01, 1, step = 0.01))
        action_space = ActionSpace.DISCRETE
        policy = RandomPolicy(discrete=True)
    elif algo_name == "pid_expert":
        params = pid_params.get_params()
        patient_name = f"{cfg.patient_type}#{cfg.patient_number}-10"
        p_params = params[patient_name]
        print(p_params)
        policy = PIDController(
            P = p_params["kp"],
            I = p_params["ki"],
            D = p_params["kd"]
        )
    elif algo_name == "pid_base":
        policy = PIDController()
        
    elif algo_name == "bb":
        
        policy = BBController()
        
    
    # Initialize lists 
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    bg_val =  env.reset()
    insulin = env.action_space.sample()
    bg_val, reward, done, _, info = env.step(insulin)
    observations.append(bg_val[0])
    actions.append(float(insulin))

    counter = 0
    print("Starting simulation\n")
    while counter <= num_steps:
        
        if counter % 10000 ==0:
            print(f"Iteration {counter}\n")
        
        insulin = policy.policy(observation = bg_val,  reward=reward, done=done, **info)
        
        if "pid" in algo_name or "bb" in algo_name:
            insulin = insulin.basal
        
        # Update current state and action (s_t, a_t)
        observations.append(bg_val[0])
        actions.append(float(insulin))
        
        bg_val, reward, done, _ , info =  env.step(insulin)
        
        # reward_t = r(s_t, a_t), terminal_t = terminal(s_t,a_t)
        
        rewards.append(reward)
        terminals.append(done)

        if done:
            env.reset()
            insulin = env.action_space.sample()
            bg_val, reward, done, _, info = env.step(insulin)
        
        counter +=1 
        
        
        
        
    dataset = MDPDataset(
        observations = np.array(observations, dtype = np.float32).reshape(-1,1),
        actions = np.array(actions, dtype = np.float32).reshape(-1,1),
        rewards = np.array(rewards, dtype = np.float32).reshape(-1,1),
        terminals = np.array(terminals).reshape(-1,1),
        action_space = action_space,
        action_size = action_size
    )
    
    if not os.path.exists(cfg.replay_path):
        os.makedirs(cfg.replay_path)
    save_path =  os.path.join(cfg.replay_path, f'{algo_name}_{cfg.patient_type}#00{cfg.patient_number}_{cfg.seed}.h5')
    print(f"Saving dataset to: {save_path}")
    with open(save_path, "w+b") as f:
        dataset.dump(f)   
        
        
    
if __name__ == "__main__":
    collect()
        
    
