from utils import seeding_function, get_device, get_algo

import os
import d3rlpy
import mlxp
import torch

from simglucose.envs.simglucose_gym_env import T1DSimEnv
from typing import Any
import numpy as np

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from gym.utils import seeding
from datetime import datetime


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


def get_action(obs, reward, algo_type: str, algo, **kwargs):
    if algo_type == "cql":
        return algo.predict(obs)
    elif algo_type == "dt":
        actor = kwargs['actor']
        return actor.predict(obs, reward)
    elif algo_type == 'td3+bc':
        return algo.predict(obs)
    else:
        raise ValueError(f"Invalid algo type: {algo_type}")
        



@mlxp.launch(config_path='./configs',
            seeding_function=seeding_function)
def test(ctx: mlxp.Context)->None:
    cfg = ctx.config
    logger = ctx.logger

    # Create the environment
    env_ids = [1,2,3]

    envs = [T1DSimEnvOurs(patient_name=f'{cfg.patient_type}#00{env_id}') for env_id in env_ids]

    device = get_device()

    algo_config = cfg.algo_config
    algo_name = algo_config['name']

    algo_config.pop('name')
    
    algo = d3rlpy.load_learnable(cfg.model_path, device=device)
    kwargs = {}
    if algo_name == 'dt':
        kwargs['actor'] = algo.as_stateful_wrapper(target_return=0)

    reward = 0

    trajs = [[], [], []]

    for env_idx, env in enumerate(envs):
        obs, _ = env.reset()
        obs = np.asarray([[obs.CGM]]).reshape(1, 1)
        done = False
        while not done:
            action = get_action(obs, reward, algo_name, algo, **kwargs)
            prev_obs = obs
            obs, reward, done, _, info = env.step(action)
            obs = np.asarray([[obs.CGM]]).reshape(1, 1)
            
            trajs[env_idx].append([prev_obs, action, obs, reward, done, info])

    os.makedirs("trajs/", exist_ok=True)
    os.makedirs(f"trajs/{cfg.model_path.replace('.d3', '')}", exist_ok=True)

    import pickle

    with open(f"trajs/{cfg.model_path.replace('.d3', '')}/trajs.pkl", "wb") as f:
        pickle.dump(trajs, f)
            

if __name__ == '__main__':
    test()