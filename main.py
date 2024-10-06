import asyncio
import logging
import tracemalloc
import os
from typing import Optional, List

import torch
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from data.ingestion.mexc_data_ingestion import DataIngestion
from data.config import Config as DataIngestionConfig
from models.agents.agent import TradingAgent
from models.gmn.gmn import CryptoGMN
from models.lnn.lnn_model import LiquidNeuralNetwork
from models.utils.risk_management import RiskManager
from models.utils.config import Config

config = Config("configs/config.yaml")

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

scaler_path = "models/lnn/scaler.joblib"

bot_tasks: List[asyncio.Task] = []

async def main():
    global bot_tasks

    try:
        timeframes = config.timeframes
        max_history_length = config.max_history_length

        gmn = CryptoGMN(timeframes, config.indicators, max_history_length=max_history_length, config=config)

        data_ingestion_config = DataIngestionConfig(
            symbol=config.symbol,
            timeframes=config.timeframes,
            private_channels=config.private_channels,
            reconnect_delay=config.reconnect_delay,
            max_retry_attempts=config.max_retry_attempts
        )
        data_ingestion = DataIngestion(gmn, data_ingestion_config)

        risk_manager = RiskManager(config.risk_parameters)
        scaler = await load_scaler(scaler_path)  # No need to pass gmn and history_length
        model = await load_or_train_lnn(gmn, config.lnn_model_path, config, scaler)

        if model is None:
            logging.error("Failed to load or train LNN model. Exiting.")
            await shutdown()  # Call simplified shutdown
            return

        agent = TradingAgent(timeframes, config.indicators, model, config, risk_manager, scaler)
        tracemalloc.start()
        bot_tasks = [
            asyncio.create_task(data_ingestion.connect()),
            asyncio.create_task(agent_loop(agent, gmn))
        ]

        try:
            await asyncio.gather(*bot_tasks)
        except asyncio.CancelledError:
            logging.info("Main tasks cancelled.")
        except Exception as e:
            logging.exception(f"Unhandled exception in main loop: {e}")
        finally:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logging.info("Top 10 memory allocations:")
            for stat in top_stats[:10]:
                logging.info(stat)

            tracemalloc.stop()
            await shutdown()

    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user.")


async def agent_loop(agent: TradingAgent, gmn: CryptoGMN):
    while True:
        try:
            market_data = gmn.get_all_data()
            if not all(market_data.values()):
                await asyncio.sleep(config.agent_loop_delay)
                continue
            await agent.make_decision(market_data)
        except Exception as e:
            logging.error(f"Error in agent loop: {e}", exc_info=True)
        await asyncio.sleep(config.agent_loop_delay)


async def load_or_train_lnn(gmn, model_path, config, scaler):
    try:
        input_size = len(config.timeframes) * len(config.indicators)
        model = LiquidNeuralNetwork(input_size, config.lnn_hidden_size, 1).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        logging.info("Loaded pre-trained LNN model.")
        return model
    except FileNotFoundError:
        logging.info("Pre-trained LNN model not found. Training a new model...")
        return await train_and_save_lnn(gmn, model_path, config, scaler)
    except Exception as e:
        logging.exception(f"Error loading LNN model: {e}")
        return None


async def train_and_save_lnn(gmn, model_path, config, scaler):
    try:
        X_train, y_train = await prepare_lnn_training_data(gmn, config.training_history_length, scaler)
        if X_train is None or y_train is None:
            return None
        X_train = torch.tensor(X_train, dtype=torch.float32, device=config.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=config.device)
        input_size = X_train.shape[1]
        model = LiquidNeuralNetwork(input_size, config.lnn_hidden_size, 1).to(config.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lnn_learning_rate)
        epochs = config.lnn_training_epochs
        batch_size = 32
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X.unsqueeze(1))
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), model_path)
        model.eval()
        logging.info(f"LNN model trained and saved to {model_path}")
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
        return model
    except Exception as e:
        logging.error(f"Error during LNN training: {e}", exc_info=True)
        return None


async def prepare_lnn_training_data(gmn: CryptoGMN, history_length: int, scaler: MinMaxScaler):
    try:
        market_data = gmn.get_all_data()
        if not market_data or len(market_data['1m']['price']) < history_length + 1:
            logging.error("Not enough data to prepare training dataset.")
            return None, None

        X = []
        y = []
        for i in range(history_length, len(market_data['1m']['price']) - 1):
            features = []
            for timeframe in gmn.timeframes:
                for indicator in gmn.indicators:
                    data_series = market_data[timeframe].get(indicator)
                    if data_series and len(data_series) > i:
                        value = data_series[i]
                        if isinstance(value, dict):
                            features.extend(list(value.values()))
                        else:
                            features.append(value)
                    else:
                        features.append(0.0)

            future_price = market_data['1m']['price'][i + 1]
            current_price = market_data['1m']['price'][i]
            price_change = (future_price - current_price) / current_price
            y.append(1 if price_change > 0 else 0)
            X.append(features)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X_scaled = scaler.fit_transform(X)  # Fit and transform inside the function
        return X_scaled, y

    except Exception as e:
        logging.error(f"Error preparing LNN training data: {e}", exc_info=True)
        return None, None


async def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Loads a saved scaler or creates a new one."""
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}")
        return scaler
    except FileNotFoundError:
        logging.warning(f"Scaler file not found at {scaler_path}. Initializing a new scaler.")
        return MinMaxScaler()


async def shutdown():
    """Shuts down the application gracefully."""
    global bot_tasks
    for task in bot_tasks:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logging.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Trading bot stopped by user. Shutting down...")
        asyncio.run(shutdown())