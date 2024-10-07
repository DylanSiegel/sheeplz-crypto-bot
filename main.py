# File: main.py
import asyncio
import logging
from data.mexc_websocket_connector import MexcWebsocketConnector
from data.data_processor import DataProcessor
from error_handler import ErrorHandler
from dotenv import load_dotenv
import os
from agent import PPOAgent
from environment import TradingEnvironment
import torch

# Constants
INPUT_SIZE = 10  # Should match INPUT_SIZE in data_processor.py
ACTION_SIZE = 1  # Scalar action

def load_configuration():
    load_dotenv(os.path.join(os.path.dirname(__file__), 'configs/.env'))
    symbols = os.getenv("SYMBOLS", "BTCUSDT").split(",")
    timeframes = os.getenv("TIMEFRAMES", "Min1").split(",")
    return symbols, timeframes

def validate_configuration(symbols, timeframes):
    if not symbols:
        raise ValueError("SYMBOLS must be defined in the .env file.")
    if not timeframes:
        raise ValueError("TIMEFRAMES must be defined in the .env file.")

async def train_agent(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state = await env.reset()
        done = False
        trajectories = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'values': []}
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = await env.step(action)
            _, _, value = agent.model(torch.FloatTensor(state).to(agent.device))
            trajectories['states'].append(state)
            trajectories['actions'].append(action)
            trajectories['log_probs'].append(log_prob)
            trajectories['values'].append(value.item())
            trajectories['rewards'].append(reward)
            state = next_state
        # Compute returns and update agent
        next_value = 0 if done else agent.model(torch.FloatTensor(state).to(agent.device))[-1].item()
        returns = agent.compute_returns(trajectories['rewards'], [1]*len(trajectories['rewards']), trajectories['values'], next_value)
        trajectories['returns'] = returns
        agent.update(trajectories)

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbosity
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Main")

    symbols, timeframes = load_configuration()
    try:
        validate_configuration(symbols, timeframes)
    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
        return
    logger.info(f"Loaded configuration: Symbols={symbols}, Timeframes={timeframes}")

    # Initialize components
    data_queue = asyncio.Queue()
    lnn_output_queue = asyncio.Queue()
    error_handler = ErrorHandler()

    processor = DataProcessor(data_queue, lnn_output_queue, error_handler, symbols, timeframes)
    connector = MexcWebsocketConnector(data_queue, symbols, timeframes, error_handler)

    # Initialize agent and environment
    state_size = 3  # lnn_output, position, equity
    action_size = ACTION_SIZE
    agent = PPOAgent(state_size, action_size)
    env = TradingEnvironment(lnn_output_queue)

    # Create tasks
    connector_task = asyncio.create_task(connector.connect())
    processor_task = asyncio.create_task(processor.run())
    training_task = asyncio.create_task(train_agent(agent, env))

    tasks = [connector_task, processor_task, training_task]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Pipeline terminated by user.")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Data pipeline terminated gracefully.")
    except Exception as e:
        error_handler.handle_error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
