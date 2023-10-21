from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import BitcoinTradingEnv
import pandas as pd

data = pd.read_csv('./data/btc_data.csv')
data_daily = data.resample('D').mean()
data_hourly = data.resample('H').mean()

TRAIN_SPLIT = int(0.8 * len(data_daily))

test_data_daily = data_daily[TRAIN_SPLIT:]
test_data_hourly = data_hourly[TRAIN_SPLIT * 24:]

env = DummyVecEnv([lambda: BitcoinTradingEnv(test_data_hourly)])

model = PPO.load('./checkpoints/model.zip')

# Evaluate the model
obs = env.reset()
for _ in range(len(env.envs[0].df) - env.envs[0].look_back_window):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

env.render('human')
