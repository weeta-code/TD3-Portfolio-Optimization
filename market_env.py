import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import gym
from gym import spaces

# Initializes the Market Environment when an object is created
# Loads all data (stock pricings) for the agent to trade on.
class MarketEnv(gym.Env):
    # stores full price history, calculates the number of stocks in our portfolio
    # sets a starting port value and then calls reset() to initialize everything properly
    def __init__(self, stock_data, transaction_cost=0.001, risk_free_rate=0.02/252):
        super(MarketEnv, self).__init__()
        self.stock_data = stock_data # full price data of our stock universe
        self.n_stocks = stock_data.shape[1]
        self.starting_cash = 50000 # Example/Placeholder
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate





        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_stocks,), dtype=np.float32
        )



        self.reset()
    # Restarts our environment to the beginning state/Re-initializes our port.
    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.starting_cash
        self.portfolio_weights = np.ones(self.n_stocks) / self.n_stocks # for uniform allocation
        self.portfolio_returns_history = []
        self.previous_weights = self.portfolio_weights.copy()
        return self._get_observation()


    def _calculate_technical_indicators(self, prices):
        # to calculate simple moving averages
        sma_20 = np.mean(prices[-20:], axis=0) if len(prices) >= 20 else np.mean(prices, axis=0)
        sma_50 = np.mean(prices[-50:], axis=0) if len(prices) >= 50 else np.mean(prices, axis=0)

        # to calculate a simplified relative strength index
        returns = np.diff(prices, axis=0) if len(prices) > 1 else np.zeros_like(prices[0])
        up_moves = np.where(returns > 0, returns, 0)
        down_moves = np.where(returns < 0, -returns, 0)
        avg_up = np.mean(up_moves[-14:], axis=0) if len(up_moves) >= 14 else np.mean(up_moves, axis=0)
        avg_down = np.mean(down_moves[-14:], axis=0) if len(down_moves) >= 14 else np.mean(down_moves, axis=0)
        rsi = 100 - (100 / (1 + avg_up / (avg_down + 1e-6)))

        # to ensure all arrays have the same shape
        sma_20 = np.atleast_1d(sma_20)
        sma_50 = np.atleast_1d(sma_50)
        rsi = np.atleast_1d(rsi)

        return np.concatenate([sma_20, sma_50, rsi])



    def _calculate_reward(self, portfolio_return, new_weights):
        lambd = 4.0
        var_penalty = lambd * (portfolio_return ** 2)
        txn_cost = 0.0005 * np.abs(new_weights - self.previous_weights).sum()

        return portfolio_return - var_penalty - txn_cost

    def _sanitize_action(self, raw_action):
        # meant to clip our action to a vector of [0, 1] and renormalize it to exactly 1, restricting the agent to being long-only.
        clipped = np.clip(raw_action, 0, None)
        normed = clipped/ (clipped.sum() + 1e-8)
        return normed.astype(np.float32)

    def step(self, action):

        action = self._sanitize_action(action)
        if self.current_step + 1 >= len(self.stock_data):
            self.done = True
            return self._get_observation(), self.reward, self.done, {}

        self.daily_returns = (self.stock_data[self.current_step + 1] - self.stock_data[self.current_step]) / self.stock_data[self.current_step]
        self.portfolio_return = np.dot(action, self.daily_returns) # weighted sum of individual returns
        self.portfolio_value *= (1 + self.portfolio_return) # cash grows or shrinks based on today's market move

        # to calculate enhanced reward
        self.reward = self._calculate_reward(self.portfolio_return, action)

        self.portfolio_returns_history.append(self.portfolio_return) # placeholder for maybe calculating volatility eventually
        self.previous_weights = action.copy()

        self.current_step += 1
        self.done = False


        action = self._sanitize_action(action)
        if self.current_step + 1 >= len(self.stock_data):
            self.done = True
            return self._get_observation(), self.reward, self.done, {}

        self.daily_returns = (self.stock_data[self.current_step + 1] - self.stock_data[self.current_step]) / self.stock_data[self.current_step]
        self.portfolio_return = np.dot(action, self.daily_returns) # weighted sum of individual returns
        self.portfolio_value *= (1 + self.portfolio_return) # cash grows or shrinks based on today's market move

        # to calculate enhanced reward
        self.reward = self._calculate_reward(self.portfolio_return, action)

        self.portfolio_returns_history.append(self.portfolio_return) # placeholder for maybe calculating volatility eventually
        self.previous_weights = action.copy()

        self.current_step += 1
        self.done = False


        return self._get_observation(), self.reward, self.done, {}

    def render(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumprod(np.array(self.portfolio_returns_history) + 1))
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)





        plt.subplot(1, 2, 2)
        plt.bar(range(self.n_stocks), self.portfolio_weights)
        plt.title("Current Portfolio Weights")
        plt.xlabel("Stock Index")
        plt.ylabel("Weight")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def seed(self, seed_value):

        np.random.seed(seed_value)

        np.random.seed(seed_value)


    def _get_state(self):
        # to get current prices
        current_prices = self.stock_data[self.current_step]



        # to get historical prices for technical indicators
        lookback = 50
        start_idx = max(0, self.current_step - lookback)
        historical_prices = self.stock_data[start_idx:self.current_step + 1]


        # to calculate these technical indicators
        tech_indicators = self._calculate_technical_indicators(historical_prices)



        # to calculate these technical indicators
        tech_indicators = self._calculate_technical_indicators(historical_prices)


        return (current_prices, tech_indicators)

    def _get_observation(self):
        # to return only the current prices as the state for the agent to act on
        current_prices, _ = self._get_state()
        return current_prices
