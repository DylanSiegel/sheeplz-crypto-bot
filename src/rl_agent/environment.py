# src/rl_agent/environment.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging

class TradingEnv(gym.Env):
    """
    Custom Trading Environment for OpenAI Gym
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=100000, fee=0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.fee = fee
        self.current_step = 0
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = 0
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: feature vector + position
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.df.shape[1] - 1 + 1,),  # Features + position
            dtype=np.float32
        )
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.df.iloc[self.current_step].drop(['Open time', 'target', 'target_category_15m', 'target_category_1h', 'target_category_4h']).values
        obs = np.append(obs, self.position)
        return obs.astype(np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        reward = 0
        done = False
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.balance -= current_price * (1 + self.fee)
                logging.info(f"Bought at {current_price}")
            elif self.position == -1:
                # Cover short
                profit = self.entry_price - current_price
                reward += profit
                self.balance += self.entry_price * (1 - self.fee)  # Return collateral
                self.position = 1
                self.entry_price = current_price
                logging.info(f"Covered short and bought at {current_price}, Profit: {profit}")
        
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.balance += current_price * (1 - self.fee)
                logging.info(f"Sold at {current_price}")
            elif self.position == 1:
                # Sell long
                profit = current_price - self.entry_price
                reward += profit
                self.balance += current_price * (1 - self.fee)
                self.position = -1
                self.entry_price = current_price
                logging.info(f"Sold long at {current_price}, Profit: {profit}")
        
        # Update net worth
        if self.position == 1:
            self.net_worth = self.balance + (current_price - self.entry_price)
        elif self.position == -1:
            self.net_worth = self.balance + (self.entry_price - current_price)
        else:
            self.net_worth = self.balance
        
        # Calculate reward based on target category
        target = self.df.iloc[self.current_step]['target_category_15m']
        reward += target  # This can be adjusted based on the reward function design
        
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
        
        info = {}
        
        return self._next_observation(), reward, done, info
    
    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Position: {self.position}')
        print(f'Profit: {profit}')
