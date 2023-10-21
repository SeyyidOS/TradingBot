from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import BitcoinTradingEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import pandas as pd

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=[64, 'lstm', 64])


data = pd.read_csv('./data/btc_data.csv')
data_daily = data.resample('D').mean()
data_hourly = data.resample('H').mean()

TRAIN_SPLIT = int(0.8 * len(data_daily))

train_data_daily = data_daily[:TRAIN_SPLIT]

train_data_hourly = data_hourly[:TRAIN_SPLIT * 24]  # Rough approximation


# Create and wrap the environment
env = DummyVecEnv([lambda: BitcoinTradingEnv(train_data_hourly)])

# Initialize agent
model = PPO(CustomPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

model.save('./checkpoints/model.zip')

