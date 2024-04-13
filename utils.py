import gym
import dataclasses

from gym import error
import numpy as np
import torch
import d3rlpy

import hashlib
import os
import struct


integer_types = (int,)

def _seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.

    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, integer_types):
        a = a % 2**(8 * max_bytes)
    else:
        raise error.Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:

    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928

    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)

    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = _seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])

gym.utils.seeding.hash_seed = hash_seed

def get_env(setting: str, number: int):
    """
    Return the gym environment with the given setting and number
    """
    from simglucose.envs.simglucose_gym_env import T1DSimEnv
    return T1DSimEnv(patient_name=f"{setting}#{number}")

def seeding_function(seed):
    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device() -> torch.device:
    """
    Return the device to be used
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    return device


# Modifying random policies for our purposes 
class OurRandomPolicy(d3rlpy.models.torch.policies.RandomPolicy):
    # Modifying the sample_action method to return actions in the range [0, 4]
    # since insulin doses are in the range [0, 4] mg and can't be negative
    def sample_action(self, x):
        x = np.asarray(x)
        action_shape = (x.shape[0], self._action_size)

        if self._config.distribution == "uniform":
            action = np.random.uniform(0, 4, size=action_shape)
        elif self._config.distribution == "normal":
            action = np.random.normal(
                0.0, self._config.normal_std, size=action_shape
            )
        else:
            raise ValueError(
                f"invalid distribution type: {self._config.distribution}"
            )

        action = np.clip(action, 0, 4.0)

        if self._config.action_scaler:
            action = self._config.action_scaler.reverse_transform_numpy(action)

        return action
@dataclasses.dataclass()
class OurRandomPolicyConfig(d3rlpy.models.torch.policies.RandomPolicyConfig):
    r"""Random Policy for continuous control algorithm.

    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.

    Args:
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        distribution (str): Random distribution. Available options are
            ``['uniform', 'normal']``.
        normal_std (float): Standard deviation of the normal distribution. This
            is only used when ``distribution='normal'``.
    """
    distribution: str = "uniform"
    normal_std: float = 1.0
    
    def create(self, device: DeviceArg = False) -> "RandomPolicy":  # type: ignore
        return OurRandomPolicy(self)


    @staticmethod
    def get_type() -> str:
        return "random_policy"




def get_algo(algo_name: str):
    if algo_name == 'cql':
        return d3rlpy.algos.CQLConfig
    elif algo_name == 'sac':
        return d3rlpy.algos.SACConfig
    elif algo_name == 'td3':
        return d3rlpy.algos.TD3Config
    elif algo_name == 'dt':
        return d3rlpy.algos.DecisionTransformerConfig
    elif algo_name == 'random':
        return d3rlpy.algos.OurRandomPolicyConfig
    elif algo_name == 'discrete_random':
        return d3rlpy.algos.DiscreteRandomPolicyConfig
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
