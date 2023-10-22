from sb3_contrib import RecurrentPPO
from environment import BitcoinTradingEnv
from stable_baselines3.sac import SAC
import pandas as pd

data = pd.read_csv('../data/btc_data.csv')

TRAIN_SPLIT = int(0.8 * len(data))

test_data_daily = data[TRAIN_SPLIT:]

env = BitcoinTradingEnv(test_data_daily)

model = SAC.load('../checkpoints/model.zip')

# Evaluate the model
obs, info = env.reset()
for _ in range(len(env.df) - env.look_back_window):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    if done:
        break

env.render('human')
