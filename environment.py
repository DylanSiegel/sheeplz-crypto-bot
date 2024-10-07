# File: environment.py
import numpy as np
import asyncio

class TradingEnvironment:
    def __init__(self, lnn_output_queue, initial_balance=10000, max_leverage=10):
        self.lnn_output_queue = lnn_output_queue
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage

    async def reset(self):
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size (positive for long, negative for short)
        self.entry_price = 0.0
        self.margin = 0.0
        self.equity = self.initial_balance
        self.margin_level = 100.0
        self.history = []
        lnn_output = await self._get_lnn_output()
        self.current_state = self._get_state(lnn_output)
        return self.current_state

    async def step(self, action):
        # action is expected to be a scalar in [-1, 1]
        lnn_output = await self._get_lnn_output()
        market_price = self._get_market_price(lnn_output)
        # Process action
        # action > 0: buy (long), action < 0: sell (short), action == 0: hold
        position_change = action.item() * self.max_leverage  # Scale action to leverage
        # Update position
        self.position += position_change
        self.entry_price = market_price  # Update entry price
        # Compute reward
        reward = self._compute_reward(market_price)
        # Update equity
        self.equity += reward
        self.history.append((self.position, market_price, reward))
        # Check if done (e.g., if balance below zero)
        done = self.equity <= 0
        info = {}
        # Update state
        self.current_state = self._get_state(lnn_output)
        return self.current_state, reward, done, info

    async def _get_lnn_output(self):
        lnn_output = await self.lnn_output_queue.get()
        self.lnn_output_queue.task_done()
        return lnn_output  # Assuming lnn_output is a scalar indicator

    def _get_market_price(self, lnn_output):
        # For simplicity, use the LNN output as a proxy for market price
        # In practice, you would get the market price from actual data
        return lnn_output

    def _compute_reward(self, market_price):
        # Compute reward based on position and market price change
        price_change = market_price - self.entry_price
        reward = self.position * price_change
        return reward

    def _get_state(self, lnn_output):
        # Construct the state representation
        # State includes LNN output, current position, and equity
        state = np.array([lnn_output, self.position, self.equity], dtype=np.float32)
        return state
