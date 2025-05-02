import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from market_env import MarketEnv

def create_test_data(n_stocks = 3, n_days = 100):
    # to generate test price data with distinct trends for each stock
    np.random.seed(42)
    
    # to create different trends for each stock
    trends = np.array([0.0002, 0.0, -0.0001])  # one up, one flat, one down
    
    # to generate random walk with drift
    returns = np.random.normal(0, 0.02, (n_days, n_stocks))
    for i in range(n_stocks):
        returns[:, i] += trends[i]  # to add trend
    
    # to convert returns to prices
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    return prices

def run_strategy(env, strategy_func, name):
    state = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    
    while not done:
        action = strategy_func(state)
        state, reward, done, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        
    return np.array(portfolio_values)

def equal_weight_strategy(state):
    # equal weight strategy that maintains equal allocation across all stocks
    n_stocks = len(state)
    weights = np.ones(n_stocks) / n_stocks
    return weights  # always returns equal weights regardless of state

def momentum_strategy(state, lookback=20):
    # momentum strategy that allocates more weight to stocks with positive momentum
    n_stocks = len(state)
    if len(env.portfolio_returns_history) < lookback:
        return np.ones(n_stocks) / n_stocks
    
    # to calculate momentum for each stock using price changes
    momentum = np.zeros(n_stocks)
    for i in range(n_stocks):
        # to get historical prices for this stock
        start_idx = max(0, env.current_step - lookback)
        stock_prices = env.stock_data[start_idx:env.current_step + 1, i]
        
        # to calculate momentum as the percentage change over the lookback period
        if len(stock_prices) > 1:
            momentum[i] = (stock_prices[-1] / stock_prices[0]) - 1
    
    # to amplify the momentum effect
    momentum = momentum * 2  # to increase the spread between weights
    
    # to convert momentum to weights
    weights = np.exp(momentum - np.min(momentum))  # to ensure all weights are positive
    weights = weights / np.sum(weights)  # to normalize to sum to 1
    
    # to add concentration limits
    weights = np.clip(weights, 0.1, 0.6)  # no position smaller than 10% or larger than 60%
    weights = weights / np.sum(weights)  # to renormalize after clipping
    
    return weights

def plot_results(results, title="Portfolio Performance Comparison"):
    plt.figure(figsize=(12, 6))
    
    plt.plot(results['Equal Weight'], 
             label='Equal Weight', 
             color='blue', 
             linewidth=2, 
             linestyle='-')
    plt.plot(results['Momentum'], 
             label='Momentum', 
             color='orange', 
             linewidth=1.5, 
             linestyle='--')
    
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.margins(y=0.1)
    
    print("\nInitial and Final Portfolio Values:")
    print("-" * 50)
    for name, values in results.items():
        print(f"{name}:")
        print(f"  Initial: ${values[0]:.2f}")
        print(f"  Final:   ${values[-1]:.2f}")
        print(f"  Return:  {((values[-1]/values[0] - 1) * 100):.2f}%\n")
    
    plt.show()

def calculate_metrics(values):
    returns = np.diff(values) / values[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_drawdown = np.min((values - np.maximum.accumulate(values)) / np.maximum.accumulate(values))
    return {
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Total Return': (values[-1] / values[0] - 1) * 100
    }

if __name__ == "__main__":
    # to create test data
    prices = create_test_data(n_stocks=3, n_days=100)
    
    # to initialize the environment
    env = MarketEnv(prices)
    
    # to test different strategies
    strategies = {
        'Equal Weight': equal_weight_strategy,
        'Momentum': momentum_strategy
    }
    
    results = {}
    metrics = {}
    
    for name, strategy in strategies.items():
        env.reset()
        values = run_strategy(env, strategy, name)
        results[name] = values
        metrics[name] = calculate_metrics(values)
    
    plot_results(results)
    
    print("\nStrategy Performance Metrics:")
    print("-" * 50)
    for name, metric in metrics.items():
        print(f"\n{name}:")
        for key, value in metric.items():
            print(f"{key}: {value:.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(prices)
    plt.title("Stock Prices")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    returns = np.diff(prices, axis=0) / prices[:-1]
    plt.boxplot(returns)
    plt.title("Returns Distribution")
    plt.ylabel("Daily Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()