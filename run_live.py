"""Minimal live-run script that wires your agent to the Kraken client.

This script is intentionally minimal and defaults to DRY_RUN=true. Replace
the state construction (`make_state_from_ticker`) to match the inputs your
agent expects (same features/shape used during training).

Usage: set environment variables (see `.env.example`) then run:
    python run_live.py
"""
import os
import time
import logging
from typing import Any

import numpy as np
import torch

from config import EnvironmentConfig
from exchanges.kraken_client import KrakenClient
from exchanges.position_manager import PositionManager, PositionLimits
from integrations.tradingview_adapter import PriceBuffer, VolumeBuffer, combine_indicators_to_action
from agent import MetaSACAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')


def make_state_from_ticker(ticker: dict, config: EnvironmentConfig) -> Any:
    """Build a placeholder state for the agent from a ticker.

    IMPORTANT: Replace this with the same feature pipeline used during
    training/backtests. The agent expects a np.ndarray of shape
    (sequence_length, state_dim) or similar. Here we build a simple
    zero-filled window with the last price in the close column as one
    feature to allow the demo to run.
    """
    last_price = float(ticker.get('last', 0.0) or 0.0)
    state = np.zeros((config.window_size, config.state_dim), dtype=np.float32)
    # put normalized price in first column as an example
    state[:, 0] = last_price
    return state


def main():
    SYMBOL = os.getenv('SYMBOL', 'XBT/USD')  # change to your preferred symbol
    MAX_ORDER_USD = float(os.getenv('MAX_ORDER_USD', '10'))
    DRY_RUN = bool(str(os.getenv('DRY_RUN', 'true')).lower() in ('1', 'true', 'yes'))

    # init client
    kraken = KrakenClient(dry_run=DRY_RUN)
    # init a simple position manager with conservative default limits
    pm_limits = PositionLimits(max_notional_usd=float(os.getenv('MAX_ACCOUNT_NOTIONAL_USD', '1000')), min_order_usd=float(os.getenv('MIN_ORDER_USD', '1.0')))
    posman = PositionManager(pm_limits)

    # load agent (model weights optional)
    cfg = EnvironmentConfig()
    agent = MetaSACAgent(cfg, env=None)
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        try:
            agent.load_state_dict(torch.load(model_path, map_location=cfg.device))
            logger.info(f"Loaded agent state from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

    logger.info(f"Starting live loop for {SYMBOL} (DRY_RUN={DRY_RUN})")
    time_step = 0
    # small price/volume buffers used by the TradingView adapter
    price_buffer = PriceBuffer(size=int(os.getenv('TV_WINDOW', '20')))
    volume_buffer = VolumeBuffer(size=int(os.getenv('TV_WINDOW', '20')))

    try:
        while True:
            try:
                ticker = kraken.fetch_ticker(SYMBOL)
            except Exception as e:
                logger.error(f"Failed to fetch ticker {SYMBOL}: {e}")
                time.sleep(1.0)
                continue

            state = make_state_from_ticker(ticker, cfg)
            # update price buffer
            last_price = float(ticker.get('last', 0.0) or 0.0)
            last_vol = float(ticker.get('volume', 0.0) or 0.0)
            price_buffer.add(last_price)
            volume_buffer.add(last_vol)

            # create dummy graph inputs required by agent.select_action
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            graph_node_features = torch.zeros((1, cfg.graph_input_dim), dtype=torch.float32)

            action = agent.select_action(state, time_step=time_step, edge_index=edge_index, graph_node_features=graph_node_features, eval=True)

            # agent.select_action may return a scalar or vector depending on actor
            a_value = float(np.asarray(action).squeeze().item()) if hasattr(action, 'squeeze') or isinstance(action, (list, tuple, np.ndarray)) else float(action)

            # Get TradingView-derived signal and convert to an action
            # derive TV action via ported indicators combiner
            tv_action = combine_indicators_to_action(price_buffer.to_list(), volume_buffer.to_list())
            # textual TV signal for logging
            if tv_action > 0.05:
                tv_signal = 'buy'
            elif tv_action < -0.05:
                tv_signal = 'sell'
            else:
                tv_signal = 'none'

            # safety deadband
            deadband = float(os.getenv('DEADBAND', '0.02'))
            # combine agent action with TradingView adapter action conservatively
            # weighted average: agent (0.7), tradingview (0.3)
            combined_action = float(max(min(0.7 * a_value + 0.3 * tv_action, 1.0), -1.0))

            if abs(combined_action) < deadband:
                logger.info(f"Combined action {combined_action:.4f} within deadband {deadband}; no trade (agent={a_value:.4f}, tv={tv_action:.4f}, signal={tv_signal})")
            else:
                order_info = kraken.action_to_order(combined_action, SYMBOL, max_order_usd=MAX_ORDER_USD)
                if order_info.get('side') is None or order_info.get('amount', 0) <= 0:
                    logger.info(f"No order created from combined action {combined_action}")
                else:
                    logger.info(f"Placing order (pre-checks): {order_info} (agent={a_value:.4f}, tv={tv_action:.4f}, signal={tv_signal})")
                    # Safety: check position manager limits
                    if posman.would_exceed_limits(order_info['side'], order_info['amount'], order_info['price']):
                        logger.warning("Order would exceed configured position limits; skipping")
                    else:
                        resp = kraken.create_market_order(SYMBOL, order_info['side'], order_info['amount'])
                        logger.info(f"Order response: {resp}")
                        # In dry-run mode, record_trade is optional; here we record for simulation
                        if DRY_RUN:
                            posman.record_trade(order_info['side'], order_info['amount'], order_info['price'])
                            logger.info(f"Simulated position: {posman.current_position()}")

            time_step += 1
            time.sleep(float(os.getenv('POLL_INTERVAL', '1.0')))

    except KeyboardInterrupt:
        logger.info("Live loop stopped by user")


if __name__ == '__main__':
    main()
