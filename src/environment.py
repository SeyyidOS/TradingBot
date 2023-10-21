import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BitcoinTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100, look_back_window=5):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df
        self.look_back_window = look_back_window
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.btc_held = 0

        # Action: 0->Hold, 1->Buy, 2->Sell
        self.action_space = spaces.Discrete(3)

        # Observations: Open, Close, High, Low, Volume + Owned coins
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(look_back_window, 6), dtype=np.float32)

        self.current_step = look_back_window

        self.history = {
            'net_worth': [],
            'balance': [],
            'btc_price': [],
            'btc_held': []
        }

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self.current_step = self.look_back_window
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step -
                           self.look_back_window: self.current_step, 1:6].values
        obs = np.append(obs, [[self.btc_held]] * self.look_back_window, axis=1)
        return obs

    def step(self, action):
        self.current_step += 1
        prev_net_worth = self.net_worth
        current_price = self.df.iloc[self.current_step]['Close']

        if action == 1:  # Buy
            self.btc_held += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.btc_held * current_price
            self.btc_held = 0

        self.net_worth = self.balance + self.btc_held * current_price
        reward = self.net_worth - prev_net_worth

        self.history['net_worth'].append(self.net_worth)
        self.history['balance'].append(self.balance)
        self.history['btc_price'].append(current_price)
        self.history['btc_held'].append(self.btc_held)

        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        plt.figure(figsize=(15, 6))

        plt.subplot(2, 2, 1)
        plt.plot(self.history['btc_price'], label='BTC Price')
        plt.plot(self.history['net_worth'], label='Net Worth')
        plt.legend()
        plt.title('BTC Price & Net Worth Over Time')

        plt.subplot(2, 2, 2)
        plt.plot(self.history['balance'], label='Balance')
        plt.legend()
        plt.title('Balance Over Time')

        plt.subplot(2, 2, 3)
        plt.plot(self.history['btc_held'], label='BTC Held')
        plt.legend()
        plt.title('BTC Held Over Time')

        plt.tight_layout()
        plt.show()

    def close(self):
        pass
