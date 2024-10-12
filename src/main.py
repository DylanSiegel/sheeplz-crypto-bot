# src/main.py

import asyncio
import sys  # Added missing import
from loguru import logger
from src.websocket_client import BinanceFuturesWebSocketClient
from src.real_time_lnn import RealTimeLNN
from src.utils import load_config
import os

def setup_logging(config: dict):
    logger.remove()
    logger.add(
        sys.stderr,
        level=config['logging']['level'],
        format=config['logging']['format']
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/system.log",
        rotation=config['logging']['rotation'],
        retention=config['logging']['retention'],
        compression=config['logging']['compression'],
        level=config['logging']['level'],
        format=config['logging']['format'],
    )

async def main():
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config)
    logger.info("System initialization started.")

    # Initialize RealTimeLNN
    lnn = RealTimeLNN(config)

    # Initialize WebSocket client
    client = BinanceFuturesWebSocketClient(
        symbols=config['symbols'],
        on_message=lnn.process_message,
        config=config
    )

    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        await client.close()
        lnn.save_model()

if __name__ == "__main__":
    asyncio.run(main())
