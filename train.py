import os
# Set environment variable to handle OpenMP error on macOS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import matplotlib.pyplot as plt
from market_env import MarketEnv
from td3 import TD3Agent, Actor, Critic
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import random

def get_random_sp500(n: int = 25):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    symbols = pd.read_html(url, header=0)[0]["Symbol"].tolist()   
    return random.sample(symbols, n)                               

def get_stock_data(tickers=[get_random_sp500()], 
                   start_date='2020-01-01',
                   end_date='2023-12-31'):
    """Download and prepare stock data for training"""
    data = []
    for ticker in tickers:
        try:
            stock = yf.download(ticker, start=start_date, end=end_date)
            # Use 'Close' price and handle missing data
            prices = stock['Close'].fillna(method='ffill').values
            data.append(prices)
            print(f"Successfully downloaded data for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            return None
    
    # Stack the price data and check for validity
    price_data = np.column_stack(data)
    if np.isnan(price_data).any():
        print("Warning: NaN values found in price data")
        # Fill NaN values with forward fill
        price_data = pd.DataFrame(price_data).fillna(method='ffill').values
    
    return price_data

def train_agent(env, agent, n_episodes=100, max_steps=1500, eval_freq=10):
    """Main training loop"""
    # Track metrics
    episode_rewards = []
    portfolio_values = []
    best_reward = -np.inf
    training_start_time = datetime.now()
    episode_sharpes = []
    sharpe_per_episodes = []
    
    print("\nStarting training at:", training_start_time.strftime("%H:%M:%S"))
    print(f"Training for {n_episodes} episodes, max {max_steps} steps per episode")
    print("=" * 50)
    
    try:
        for episode in range(n_episodes):
            print(f"\nStarting episode {episode + 1}")
            episode_start_time = datetime.now()
            
            print("Resetting environment...")
            state = env.reset()
            print(f"Initial state shape: {state.shape}")
            
            episode_reward = 0
            episode_values = [env.portfolio_value]
            start_value = env.portfolio_value
            
            for step in range(max_steps):
                if step % 10 == 0:  # Print every 10 steps
                    print(f"Step {step}/{max_steps}", end='\r')
                
                # Select action with noise for exploration
                action = agent.select_action(state, add_noise=True)
                print(f"\nStep {step + 1}: Selected action shape: {action.shape}")
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                print(f"Received reward: {reward:.4f}")
                
                # Store transition in replay buffer
<<<<<<< HEAD
                agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train agent
=======
                agent.replay_buffer.add(state, action, reward, next_state, done)             # Train agent
>>>>>>> 6e48420 (Initial commit)
                if len(agent.replay_buffer.state) > 1000:
                    print("Training on batch...", end='\r')
                    agent.train(batch_size=100)
                
                episode_reward += reward
                episode_values.append(env.portfolio_value)
                state = next_state

                returns = np.array(env.portfolio_returns_history)
                ep_sharpe = ((returns.mean() - env.risk_free_rate) / (returns.std() + 1e-8)) * np.sqrt(252)
                
                if done:
                    print("\nEpisode finished early at step", step + 1)
                    break
            
            # Track performance
            episode_rewards.append(episode_reward)
            portfolio_values.append(episode_values)
            episode_sharpes.append(ep_sharpe)
            
            # Calculate time elapsed
            episode_time = datetime.now() - episode_start_time
            total_time = datetime.now() - training_start_time
            
            # Print episode summary
            end_value = env.portfolio_value
            episode_return = ((end_value / start_value) - 1) * 100
            
            print(f"\nEpisode {episode + 1}/{n_episodes} - Time: {episode_time.seconds}s (Total: {total_time.seconds}s)")
            print(f"Steps completed: {step + 1}")
            print(f"Portfolio: ${end_value:.2f} (Return: {episode_return:.1f}%)")
            print(f"Episode Sharpe: {ep_sharpe:.2f}")
            print(f"Total Reward: {episode_reward:.2f}")
            
            # Evaluate and save best model
            if episode % eval_freq == 0:
                avg_reward = np.mean(episode_rewards[-eval_freq:])
                avg_sharpe = np.mean(episode_sharpes[-eval_freq:])
                print("\nEvaluation:")
                print(f"Average Reward (last {eval_freq} episodes): {avg_reward:.2f}")
                print(f"Average Sharpe (last {eval_freq} episodes): {avg_sharpe:.2f}") 
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save_checkpoint("best_model.pth")
                    print("New best model saved!")
                
                # Plot training progress
                plot_training_progress(episode_rewards, portfolio_values[-1])
                plt.savefig(f'training_progress_ep_{episode}.png')
                plt.close()
                print(f"Progress plot saved as training_progress_ep_{episode}.png")
                print("-" * 50)
        
        total_training_time = datetime.now() - training_start_time
        print(f"\nTotal training time: {total_training_time.seconds} seconds")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def plot_training_progress(rewards, latest_portfolio_values):
    """Visualize training metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot episode rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot latest episode's portfolio value
    ax2.plot(latest_portfolio_values)
    ax2.set_title('Latest Episode Portfolio Value')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True)
    
    plt.tight_layout()

def evaluate_benchmark(agent_returns, spy_returns):
    def sharpe(r):
        return ((r.mean()) / (r.std() + 1e-8)) * np.sqrt(252)
    
    print("\nBenchmarking vs SPY")
    print("-" * 40)
    print(f"Agent Sharpe Ratio: {sharpe(agent_returns):.2f}")
    print(f"SPY Sharpe Ratio:   {sharpe(spy_returns):.2f}")
    print(f"Agent Total Return: {np.prod(1 + agent_returns) - 1:.2%}")
    print(f"SPY Total Return:   {np.prod(1 + spy_returns) - 1:.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.cumprod(1 + agent_returns), label="TD3 Agent")
    plt.plot(np.cumprod(1 + spy_returns), label="SPY")
    plt.legend()
    plt.title("Cumulative Returns: TD3 vs SPY")
    plt.grid(True)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value (normalized)")
    plt.tight_layout()
    plt.savefig("benchmark_vs_spy.png")
    print("Benchmark plot saved as benchmark_vs_spy.png")
    plt.show()


if __name__ == "__main__":
    print("\nStarting program initialization...")
    
    # Get training data
    print("\nDownloading stock data...")
    stock_data = get_stock_data(
        tickers=[get_random_sp500()],
        start_date='2023-01-01',  # Using just 1 year of data for testing
        end_date='2023-12-31'
    )
    if stock_data is None:
        print("Failed to get stock data. Exiting...")
        exit(1)
    
    print(f"Successfully loaded stock data with shape: {stock_data.shape}")
    
    # Initialize environment
    print("\nInitializing environment...")
    try:
        env = MarketEnv(stock_data)
        print("Environment initialized successfully")
        print(f"Initial portfolio value: ${env.portfolio_value:.2f}")
        print(f"State space dimension: {env.observation_space.shape}")
        print(f"Action space dimension: {env.action_space.shape}")
    except Exception as e:
        print(f"Error initializing environment: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Initialize agent
    print("\nInitializing agent...")
    state_dim = len(stock_data[0])  # Number of stocks
    action_dim = state_dim  # Portfolio weights
    max_action = 1.0  # Weights sum to 1
    
    try:
        print(f"Creating TD3Agent with state_dim={state_dim}, action_dim={action_dim}")
        print("Initializing Actor network...")
        actor = Actor(state_dim, action_dim)
        print("Actor network initialized")
        
        print("Initializing Critic network...")
        critic = Critic(state_dim, action_dim)
        print("Critic network initialized")
        
        print("Creating TD3Agent instance...")
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            actor=actor,  # Pass initialized actor network
            critic=critic,  # Pass initialized critic network
            actor_lr=1e-3,
            critic_lr=1e-3,
            buffer_size=10000  # Reduced buffer size for testing
        )
        print("Agent initialized successfully")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test environment reset
    print("\nTesting environment reset...")
    try:
        state = env.reset()
        print(f"Reset successful. Initial state shape: {state.shape}")
        print(f"Initial state values: {state}")
    except Exception as e:
        print(f"Error during environment reset: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test single action
    print("\nTesting single action...")
    try:
        action = agent.select_action(state, add_noise=True)
        print(f"Action selected successfully. Action shape: {action.shape}")
        print(f"Action values: {action}")
        
        returns = np.array(env.portfolio_returns_history)
        sharpe = ((returns.mean() - env.risk_free_rate) / (returns.std() + 1e-8)) * np.sqrt(252)
        next_state, reward, done, _ = env.step(action)
        print(f"Step executed successfully")
        print(f"Next state shape: {next_state.shape}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
    except Exception as e:
        print(f"Error during action test: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\nAll initialization tests passed. Starting training...")
    print("=" * 50)
    
    try:
        # Train the agent (minimal episodes and steps for testing)
        train_agent(env, agent, n_episodes=10, max_steps=1000, eval_freq=2)

        # Benchmarking: Load SPY data and calculate performance metrics
        # spy_data = yf.download('SPY', start='2023-01-01', end='2023-12-31')
        # spy_returns = spy_data['Close'].pct_change().dropna()

        spy_data = yf.download('SPY', start='2023-01-01', end='2023-12-31')[['Close']]  # Enforce DataFrame with 1 column
        spy_returns = spy_data['Close'].pct_change().dropna()  # This is now a Series


        # Calculate portfolio returns
        portfolio_returns = np.array(env.portfolio_returns_history)
        sharpe_ratio = ((portfolio_returns.mean() - env.risk_free_rate) / (portfolio_returns.std() + 1e-8)) * np.sqrt(252)
        
        # Compare with SPY
        spy_sharpe = ((spy_returns.mean() - env.risk_free_rate) / (spy_returns.std() + 1e-8)) * np.sqrt(252)
        

        print(f"Agent Sharpe Ratio: {sharpe_ratio:.2f}")
        # print(f"SPY Sharpe Ratio: {spy_sharpe:.2f}")
        print(f"SPY Sharpe Ratio: {spy_sharpe.iloc[0]:.2f}")


        if sharpe_ratio > spy_sharpe.iloc[0]:
            print("Agent outperformed SPY!")
        else:
            print("SPY outperformed the agent.")
        
        # Save final model
        agent.save_checkpoint("final_model.pth")
        
        print("\nTraining completed!")
        print("Final model saved as 'final_model.pth'")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
