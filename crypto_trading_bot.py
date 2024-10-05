# crypto_trading_bot.py
import asyncio
import logging
import tracemalloc
import os
from typing import Tuple, Optional, Dict, List, Any
import torch
import numpy as np
import aiohttp
import joblib
from sklearn.preprocessing import MinMaxScaler

from data.mexc_data_ingestion import DataIngestion, Config as DataIngestionConfig
from models.agents.agent import TradingAgent
from models.gmn.gmn import CryptoGMN
from models.lnn.lnn_model import LiquidNeuralNetwork
from models.utils.risk_management import RiskManager
from models.utils.config import Config

# Load configuration
config = Config("configs/config.yaml")

# Configure logging
log_file_path = os.path.join(os.getcwd(), "logs", "trading_bot.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Initialize scaler path (should be defined before use in functions)
scaler_path = "models/lnn/scaler.joblib"

async def main():
    """Main function to run the crypto trading bot."""
    try:
        timeframes = config.timeframes
        indicators = config.indicators
        max_history_length = config.max_history_length

        # Initialize GMN
        gmn = CryptoGMN(timeframes, indicators, max_history_length=max_history_length)

        # Initialize Data Ingestion Config and DataIngestion
        data_ingestion_config = DataIngestionConfig(
            symbol=config.symbol,
            timeframes=config.timeframes,
            private_channels=config.private_channels,
            reconnect_delay=config.reconnect_delay,
            # ... add other data ingestion config parameters as needed
        )
        data_ingestion = DataIngestion(gmn, data_ingestion_config)

        # Initialize Risk Manager
        risk_manager = RiskManager(config.risk_parameters)

        # Load or initialize and fit scaler  (moved here)
        scaler = await load_or_fit_scaler(scaler_path, gmn, config.training_history_length) # Modified

        # Initialize or train LNN Model
        lnn_model_path = config.lnn_model_path
        model = await load_or_train_lnn(gmn, lnn_model_path, config, scaler) # Function handles both loading and training

        if model is None:  # Check if model loading/training failed
            logging.error("Failed to load or train LNN model. Exiting.")
            await shutdown(gmn=gmn, data_ingestion=data_ingestion, risk_manager=risk_manager, scaler=scaler)
            return

        # Initialize TradingAgent *after* model is successfully loaded or trained
        agent = TradingAgent(timeframes, indicators, model, config, risk_manager, scaler)

        tracemalloc.start()  # Start tracemalloc after objects are initialized
        tasks = [
            asyncio.create_task(data_ingestion.connect()),
            asyncio.create_task(agent_loop(agent, gmn))
        ]


        try:
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logging.info("Main tasks cancelled.")
        except Exception as e:
            logging.exception(f"Unhandled exception in main loop: {e}") # Log traceback
        finally:
            # Memory profiling
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logging.info("Top 10 memory allocations:")
            for stat in top_stats[:10]:
                logging.info(stat)

            tracemalloc.stop()
            await shutdown(gmn=gmn, data_ingestion=data_ingestion, risk_manager=risk_manager, scaler=scaler, agent=agent)

    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user.")
        # If tasks are running, cancel them here before shutdown.


async def agent_loop(agent: TradingAgent, gmn: CryptoGMN):
    """Agent loop."""
    while True:
        try: # Handle exceptions within the agent loop
            market_data = gmn.get_all_data() # More efficient

            if not all(market_data.values()): # Check if all timeframes have data
                await asyncio.sleep(config.agent_loop_delay)
                continue

            await agent.make_decision(market_data)

        except Exception as e:
            logging.error(f"Error in agent loop: {e}", exc_info=True)  # Include traceback
        await asyncio.sleep(config.agent_loop_delay)





async def load_or_train_lnn(gmn, model_path, config, scaler):
    """Loads or trains the LNN model."""
    try:
        model = LiquidNeuralNetwork(len(config.timeframes) * len(config.indicators), config.lnn_hidden_size, 1)
        model.load_state_dict(torch.load(model_path, map_location=config.device)) # Load on specified device
        model.to(config.device) # Move model to device after loading
        model.eval()
        logging.info("Loaded pre-trained LNN model.")
        return model
    except FileNotFoundError:
        logging.info("Pre-trained LNN model not found. Training a new model...")
        return await train_and_save_lnn(gmn, model_path, config, scaler)
    except Exception as e:
        logging.error(f"Error loading LNN model: {e}", exc_info=True)
        return None


async def train_and_save_lnn(gmn, model_path, config, scaler):
    """Trains and saves LNN model."""

    try:
        X_train, y_train = await prepare_lnn_training_data(gmn, config.training_history_length, scaler) # Pass scaler here

        if X_train is None or y_train is None:
            return None # Return None on failure


        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=config.device) # Create on correct device
        y_train = torch.tensor(y_train, dtype=torch.float32, device=config.device) # Use config.device here

        input_size = X_train.shape[1]
        model = LiquidNeuralNetwork(input_size, config.lnn_hidden_size, 1).to(config.device)  # Send to device
        # ... (rest of training loop, including criterion, optimizer)


        # Save the trained model
        torch.save(model.state_dict(), model_path)  # Use state_dict
        model.eval()
        logging.info(f"LNN model trained and saved to {model_path}")

        return model

    except Exception as e:
        logging.error(f"Error during LNN training: {e}", exc_info=True) # Log traceback
        return None



async def prepare_lnn_training_data(gmn: CryptoGMN, history_length: int, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray]: # Changed
    """Prepares training data."""
    try:
        market_data = gmn.get_all_data()
        # ... (data preparation logic, same as before)


        # Fit and Transform with the scaler
        X_scaled = scaler.fit_transform(X) # Fit the scaler here
        # ... (rest of the method)

    except Exception as e:
        logging.error(f"Error preparing LNN training data: {e}", exc_info=True)  # More detailed
        return None, None


async def load_or_fit_scaler(scaler_path: str, gmn: CryptoGMN, history_length: int) -> MinMaxScaler:  # Changed
    """Loads scaler or fits a new one."""
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}")
    except FileNotFoundError:
        logging.info("Scaler file not found. Fitting a new scaler...")
        # Prepare data for scaler fitting
        X, _ = await prepare_lnn_training_data(gmn, history_length, None)  # Use unscaled data
        if X is None:
            raise ValueError("Failed to prepare data for scaler fitting.")

        scaler = MinMaxScaler()
        scaler.fit(X)
        joblib.dump(scaler, scaler_path)
        logging.info(f"New scaler fitted and saved to {scaler_path}")

    return scaler


async def shutdown(gmn: CryptoGMN = None, data_ingestion: DataIngestion = None, risk_manager: RiskManager = None, scaler: MinMaxScaler = None, agent: Optional[TradingAgent] = None):
    # ... (same as before)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user.")