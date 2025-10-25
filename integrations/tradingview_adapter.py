"""Lightweight adapter to 'wire' the vendored Tradingview scripts into the bot.

This adapter does NOT execute Pine scripts. Instead it exposes a tiny API that
loads available Tradingview script filenames (from vendor/Tradingview) and
provides a deterministic, explainable "signal" based on recent price history.

You can later replace the heuristic here with a proper translator or an RPC
call to a TradingView execution environment.
"""
from collections import deque
from typing import Deque, List, Optional
import os

from integrations.indicators_tv import (
    watchtower_signal,
    bot_activity_idiot_light,
    believe_it_meter,
    livermore_3_points,
    auto_fib_levels,
)

VENDOR_TV_PATH = os.path.join(os.path.dirname(__file__), '..', 'vendor', 'Tradingview')


def available_scripts() -> List[str]:
    path = os.path.abspath(VENDOR_TV_PATH)
    try:
        files = [f for f in os.listdir(path) if f.lower().endswith('.pine') or f.lower().endswith('.txt') or f.lower().endswith('')]
        return files
    except Exception:
        return []


def combine_indicators_to_action(prices: List[float], volumes: List[float]) -> float:
    """Combine ported indicators into a normalized TV action in [-1,1].

    Strategy (defaults):
      - watchtower_signal: weight 0.4 (discrete)
      - believe_it_meter: weight 0.35 (continuous)
      - livermore_3_points: weight 0.15 (discrete)
      - auto_fib proximity: weight 0.1 (discrete)
      - bot_activity spike reinforces believe_it direction

    Returns a float in [-1,1].
    """
    w_wt = 0.4
    w_bim = 0.35
    w_liv = 0.15
    w_fib = 0.1

    # compute base indicators
    wt = watchtower_signal(prices)
    bim = believe_it_meter(prices)
    liv = livermore_3_points(prices)
    vol_sig = bot_activity_idiot_light(volumes)
    fib = auto_fib_levels(prices)

    # discrete mappings
    wt_val = 1.0 if wt == 'buy' else (-1.0 if wt == 'sell' else 0.0)
    liv_val = 1.0 if liv == 'buy' else (-1.0 if liv == 'sell' else 0.0)

    # fib proximity: if last price is below 0.382 level -> buy, above 0.618 -> sell
    fib_val = 0.0
    if fib:
        last = prices[-1]
        lvl382 = fib.get('0.382')
        lvl618 = fib.get('0.618')
        if lvl382 is not None and last < lvl382:
            fib_val = 1.0
        elif lvl618 is not None and last > lvl618:
            fib_val = -1.0

    # volume spike reinforcement
    if vol_sig == 'spike':
        if bim > 0:
            bim = min(1.0, bim + 0.3)
        elif bim < 0:
            bim = max(-1.0, bim - 0.3)

    # combine weighted sum
    raw = w_wt * wt_val + w_bim * float(bim) + w_liv * liv_val + w_fib * fib_val

    # normalize to [-1,1] (weights sum to 1.0)
    action = max(min(raw, 1.0), -1.0)
    return float(action)


class PriceBuffer:
    def __init__(self, size: int = 20):
        self.size = size
        self.buf: Deque[float] = deque(maxlen=size)

    def add(self, price: float) -> None:
        self.buf.append(float(price))

    def to_list(self) -> List[float]:
        return list(self.buf)

class VolumeBuffer:
    def __init__(self, size: int = 20):
        self.size = size
        self.buf: Deque[float] = deque(maxlen=size)

    def add(self, vol: float) -> None:
        self.buf.append(float(vol))

    def to_list(self) -> List[float]:
        return list(self.buf)
