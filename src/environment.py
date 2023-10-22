from gymnasium import spaces

import matplotlib.pyplot as plt
import numpy as np
import gymnasium


class BitcoinTradingEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100, look_back_window=15):
        super(BitcoinTradingEnv, self).__init__()

        self.df = df.copy()
        self._prep_df()

        self.look_back_window = look_back_window
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.btc_held = 0
        self.transaction_fee_percent = 0.001  # e.g., 0.1% fee

        self.action_space = spaces.Box(
            low=0, high=1, shape=(2, 1), dtype=np.float32)

        # Observations: Open, Close, High, Low, Volume + Owned coins
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(look_back_window, 8), dtype=np.float32)

        self.current_step = look_back_window

        self.history = {
            'net_worth': [],
            'balance': [],
            'btc_price': [],
            'btc_held': []
        }

    def _prep_df(self):
        self.df.loc[:, 'SMA'] = self.df['Close'].rolling(window=15).mean()
        self.df = self.df.fillna(0)

    def _next_observation(self):
        obs = self.df.iloc[self.current_step -
                           self.look_back_window: self.current_step, 1:].values
        obs = np.append(obs, [[self.btc_held]] * self.look_back_window, axis=1)
        mins = np.min(obs, axis=0)
        maxs = np.max(obs, axis=0) + 0.0001
        obs = (obs - mins) / (maxs - mins)
        return obs, {}

    def _append_history(self):
        self.history['net_worth'].append(self.net_worth)
        self.history['balance'].append(self.balance)
        self.history['btc_price'].append(self.df.iloc[self.current_step]['Close'])
        self.history['btc_held'].append(self.btc_held)

    def step(self, action):
        self.current_step += 1
        prev_net_worth = self.net_worth
        current_price = self.df.iloc[self.current_step]['Close']
        final_action = (action[0] - action[1])[0]

        if final_action >= 0:
            buy_amount_btc = self.balance * final_action / current_price
            self.btc_held += buy_amount_btc
            self.balance -= buy_amount_btc * current_price * (1 + self.transaction_fee_percent)
        else:
            sell_amount_btc = self.btc_held * final_action * current_price
            self.btc_held += sell_amount_btc
            self.balance += sell_amount_btc * current_price * (1 + self.transaction_fee_percent)

        self.net_worth = self.balance + self.btc_held * current_price
        reward = self.net_worth - self.initial_balance
        # reward = reward + (self.net_worth - self.initial_balance) if self.net_worth < self.initial_balance else reward

        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        if done:
            if not self.current_step >= len(self.df) - 1:
                reward -= 100

        # if self.current_step % 1000 == 0:
        #     print(f"Step: {self.current_step}, Action: {action}, Reward: {reward}")

        self._append_history()
        obs, info = self._next_observation()
        return obs, reward, done, done, info

    def render(self, mode='human'):
        if mode == 'human':
            plt.figure(figsize=(15, 6))
            plt.subplot(2, 2, 1)
            plt.plot(self.history['btc_price'], label='BTC Price')
            # plt.plot(self.history['net_worth'], label='Net Worth')
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

            plt.subplot(2, 2, 4)
            plt.plot(self.history['net_worth'], label='Net Worth')
            plt.legend()
            plt.title('Net Worth Over Time')

            plt.tight_layout()
            plt.show()
        elif mode == 'rgb_array':
            return np.zeros((400, 600, 3))
        else:
            super(BitcoinTradingEnv, self).render()

    def reset(self, seed=42, options=None):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0
        self.current_step = np.random.randint(self.look_back_window, len(self.df) - self.look_back_window)
        return self._next_observation()

    def close(self):
        pass
