# from __future__ import annotations
from utils import seeding_function, get_device, get_algo
import d3rlpy
import sys

# with open(f'replays/random_adolescent#001_0.h5', "rb") as f:
#     dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())

# from simglucose.envs.simglucose_gym_env import T1DSimEnv
# from train_online import T1DSimEnvOurs

# from d3rlpy.metrics import EnvironmentEvaluator
# env = T1DSimEnvOurs(patient_name='adolescent#002')

algo_name = str(sys.argv[1])
online_algo_name = str(sys.argv[2])
patient_id = str(sys.argv[3])
online_seed = str(sys.argv[4])
n_steps = 1000000
n_steps_per_epoch = 10000
# For testing
# algo_name = 'cql'
# online_algo_name = 'random'
# patient_id = '001'
# online_seed = '0'
# n_steps = 400
# n_steps_per_epoch = 100
seeding_function(31415926)

# load from HDF5
with open(f'replays/{online_algo_name}_adolescent#{patient_id}_{online_seed}.h5', "rb") as f:
    dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())



device = get_device()
print(device)
algo = get_algo(algo_name)().create(device)

algo.fit(
    dataset,
    n_steps=n_steps,
    n_steps_per_epoch=n_steps_per_epoch,
    experiment_name=f'model_{algo_name}_{online_algo_name}_{patient_id}_{online_seed}',
    with_timestamp=True
)

# save d3 file
algo.save(f'models/model_{algo_name}_{online_algo_name}_{patient_id}_{online_seed}.d3')

# reconstruct full setup from a d3 file
#cql = d3rlpy.load_learnable("model.d3")

# observation = env.reset()
# while True:
#    action = algo.predict([observation])[0]
#    observation, reward, done, _ = env.step(action)
#    if done:
#        break
