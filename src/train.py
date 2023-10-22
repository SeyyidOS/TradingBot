from sb3_contrib import RecurrentPPO
from stable_baselines3.sac import SAC
from environment import BitcoinTradingEnv
import pandas as pd

data = pd.read_csv('../data/btc_data.csv')

TRAIN_SPLIT = int(0.8 * len(data))

train_data_daily = data[:TRAIN_SPLIT]

# Create and wrap the environment
env = BitcoinTradingEnv(train_data_daily)

# Initialize agent
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=25000)

model.save('../checkpoints/model.zip')

