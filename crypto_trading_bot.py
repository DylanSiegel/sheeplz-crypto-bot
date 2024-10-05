# File: crypto_trading_bot.py

import asyncio
import logging
import tracemalloc
import os
from typing import Tuple, Optional, Dict
import torch
import numpy as np
import aiohttp  # Asynchronous HTTP client
import joblib    # For saving and loading scalers

from data.mexc_data_ingestion import DataIngestion
from models.agents.agent import TradingAgent
from models.gmn.gmn import CryptoGMN
from models.lnn.lnn_model import LiquidNeuralNetwork
from models.utils.risk_management import RiskManager
from models.utils.config import Config
from sklearn.preprocessing import MinMaxScaler

# Load configuration
config = Config("configs/config.yaml")

# Configure logging
log_file_path = os.path.join(os.getcwd(), "logs", "trading_bot.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# Initialize scaler and save path
scaler_path = "models/lnn/scaler.joblib"

async def main():
    """Main function to run the crypto trading bot."""
    timeframes = config.timeframes
    indicators = config.indicators
    max_history_length = config.max_history_length

    # Initialize GMN
    gmn = CryptoGMN(timeframes, indicators, max_history_length=max_history_length)

    # Initialize Data Ingestion
    data_ingestion = DataIngestion(gmn, config)

    # Initialize Risk Manager
    risk_manager = RiskManager(config.risk_parameters)

    # Load or initialize scaler
    scaler = await load_scaler(scaler_path)

    # Initialize or train LNN Model
    lnn_model_path = config.lnn_model_path
    try:
        model_state_dict = torch.load(lnn_model_path, map_location=torch.device('cpu'))
        input_size = len(timeframes) * len(indicators)  # Input size based on features
        model = LiquidNeuralNetwork(input_size, config.lnn_hidden_size, 1)
        model.load_state_dict(model_state_dict)
        model.eval()
        logging.info("Loaded pre-trained LNN model.")
    except FileNotFoundError:
        logging.info("Pre-trained LNN model not found. Training a new model...")
        try:
            model = await train_and_save_lnn(gmn, lnn_model_path, config=config, scaler=scaler)
            if model is None:
                logging.error("Failed to train LNN model. Exiting.")
                await shutdown(gmn, data_ingestion, risk_manager, scaler)
                return
        except Exception as e:
            logging.error(f"Error preparing training data or training the model: {e}")
            await shutdown(gmn, data_ingestion, risk_manager, scaler)
            return
    except Exception as e:
        logging.error(f"Error loading LNN model: {e}")
        await shutdown(gmn, data_ingestion, risk_manager, scaler)
        return

    # Initialize TradingAgent with the loaded scaler
    agent = TradingAgent(timeframes, indicators, model, config, risk_manager, scaler)

    tracemalloc.start()

    # Create asyncio tasks
    tasks = [
        asyncio.create_task(data_ingestion.connect()),
        asyncio.create_task(agent_loop(agent, gmn))
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logging.info("Main tasks have been cancelled.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    finally:
        # Memory profiling
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        logging.info("Top 10 memory allocations:")
        for stat in top_stats[:10]:
            logging.info(stat)

        tracemalloc.stop()
        await shutdown(gmn, data_ingestion, risk_manager, scaler, agent)

async def shutdown(gmn: CryptoGMN, data_ingestion: DataIngestion, risk_manager: RiskManager, scaler: MinMaxScaler, agent: Optional[TradingAgent] = None):
    """Gracefully shuts down all components."""
    try:
        gmn.shutdown()
    except Exception as e:
        logging.error(f"Error shutting down GMN: {e}")

    try:
        await data_ingestion.close()
    except AttributeError:
        # If DataIngestion does not have a close method
        pass
    except Exception as e:
        logging.error(f"Error shutting down Data Ingestion: {e}")

    try:
        await agent.close()
    except AttributeError:
        # If TradingAgent does not have a close method
        pass
    except Exception as e:
        logging.error(f"Error shutting down Trading Agent: {e}")

    logging.info("Shutdown complete.")

async def agent_loop(agent: TradingAgent, gmn: CryptoGMN):
    """The main agent loop that retrieves data, makes decisions, and executes trades."""
    while True:
        market_data = {}
        for timeframe in agent.timeframes:
            market_data[timeframe] = {}
            for indicator in agent.indicators:
                try:
                    data = gmn.get_data(timeframe, indicator)
                    if data is not None:
                        market_data[timeframe][indicator] = data
                except ValueError as e:
                    logging.error(f"Error getting data for {timeframe} {indicator} from GMN: {e}")

        try:
            await agent.make_decision(market_data)
        except Exception as e:
            logging.error(f"Error in agent loop: {e}")

        await asyncio.sleep(config.agent_loop_delay)

async def train_and_save_lnn(
    gmn: CryptoGMN, 
    model_path: str, 
    config: Config, 
    scaler: MinMaxScaler
) -> Optional[LiquidNeuralNetwork]:
    """Trains a new LNN model and saves it to the specified path."""
    try:
        X_train, y_train = await prepare_lnn_training_data(gmn, config.training_history_length)
        if X_train is None or y_train is None:
            logging.error("Failed to prepare LNN training data.")
            return None

        input_size = X_train.shape[1]
        model = LiquidNeuralNetwork(input_size, config.lnn_hidden_size, 1)
        criterion = torch.nn.BCEWithLogitsLoss()  # Suitable for binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lnn_learning_rate)

        epochs = config.lnn_training_epochs
        batch_size = config.config.get("lnn_batch_size", 32)  # Optional: Add batch_size to config.yaml
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                # Move data to the same device as model if using GPU
                outputs = model(batch_X.unsqueeze(1))  # Shape: (batch_size, 1)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        model.eval()
        logging.info(f"LNN model trained and saved to {model_path}")

        # Save the scaler for future use
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")

        return model

    except Exception as e:
        logging.error(f"Error during LNN training: {e}")
        return None

async def prepare_lnn_training_data(gmn: CryptoGMN, history_length: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares training data for the LNN model.
    Generates feature vectors and corresponding labels based on historical market data.
    """
    try:
        market_data = gmn.get_all_data()

        # Check if sufficient data is available
        if not market_data or len(market_data['1m']['price']) < history_length + 1:
            logging.error("Not enough data to prepare training dataset.")
            return None, None

        X = []
        y = []

        # Prepare feature vectors and labels
        for i in range(history_length, len(market_data['1m']['price']) - 1):
            features = []

            for timeframe in gmn.timeframes:
                for indicator in gmn.indicators:
                    data_series = market_data[timeframe].get(indicator)
                    if data_series and len(data_series) > i:
                        value = data_series[i]
                        if isinstance(value, dict):
                            # Flatten dictionary values
                            features.extend(list(value.values()))
                        else:
                            features.append(value)
                    else:
                        features.append(0.0)  # Placeholder for missing data

            # Target: Future price change (binary classification)
            future_price = market_data['1m']['price'][i + 1]  # Next minute's price
            current_price = market_data['1m']['price'][i]
            price_change = (future_price - current_price) / current_price

            y.append(1 if price_change > 0 else 0)  # Binary label

            X.append(features)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Scale features using the same scaler as during training
        # This scaler should be fitted during training and loaded here
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

        logging.info("LNN training data prepared successfully.")
        return X_scaled, y

    except Exception as e:
        logging.error(f"Error preparing LNN training data: {e}")
        return None, None

async def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Loads an existing scaler or initializes a new one."""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}.")
    else:
        scaler = MinMaxScaler()
        logging.info("Initialized new MinMaxScaler.")
    return scaler

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user.")
