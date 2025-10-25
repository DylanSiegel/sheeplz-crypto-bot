"""Ported TradingView-style indicators (simplified, deterministic implementations).

Includes implementations for:
 - WatchTower (EMA crossover momentum)
 - Bot Activity Idiot Light (volume spike detector)
 - Believe-It-Meter (confidence combining RSI + momentum)
 - Livermore 3-Points (simple reversal pattern detector)
 - Auto-Fib (swing high/low and Fibonacci levels)

These are intentionally conservative, deterministic, and unit-testable.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np


def ema(series: List[float], period: int) -> List[float]:
    arr = np.asarray(series, dtype=float)
    if len(arr) == 0:
        return []
    alpha = 2 / (period + 1)
    out = [arr[0]]
    for v in arr[1:]:
        out.append((1 - alpha) * out[-1] + alpha * v)
    return out


def watchtower_signal(prices: List[float], short=8, long=21, threshold_pct=0.001) -> Optional[str]:
    """Detect momentum via EMA crossover: buy when short EMA sufficiently above long EMA, sell when below.

    threshold_pct: fractional threshold relative to long EMA (e.g., 0.001 ~ 0.1%)
    """
    if len(prices) < max(short, long) + 1:
        return None
    short_ema = ema(prices, short)[-1]
    long_ema = ema(prices, long)[-1]
    if short_ema > long_ema * (1 + threshold_pct):
        return 'buy'
    if short_ema < long_ema * (1 - threshold_pct):
        return 'sell'
    return None


def bot_activity_idiot_light(volumes: List[float], window=20, spike_factor=3.0) -> Optional[str]:
    """Detect large volume spikes compared to recent average; returns 'buy' if volume spike and rising, 'sell' if spike and falling price implied externally.

    Here we only inspect volumes; caller may combine with price direction.
    """
    if len(volumes) < window + 1:
        return None
    window_avg = float(np.mean(volumes[-window - 1:-1]))
    last = float(volumes[-1])
    if window_avg <= 0:
        return None
    if last > window_avg * spike_factor:
        return 'spike'
    return None


def rsi(prices: List[float], period: int = 14) -> List[float]:
    arr = np.asarray(prices, dtype=float)
    if len(arr) < period + 1:
        return []
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    out = []
    if avg_loss == 0:
        out.append(100.0)
    else:
        rs = avg_gain / avg_loss
        out.append(100 - (100 / (1 + rs)))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100 - (100 / (1 + rs)))
    return out


def believe_it_meter(prices: List[float], period=14) -> float:
    """Return a confidence score in [-1,1] combining RSI and short/long momentum.

    Positive => bullish confidence, Negative => bearish.
    """
    if len(prices) < period + 5:
        return 0.0
    r = rsi(prices, period)
    if not r:
        rscore = 0.0
    else:
        last_rsi = r[-1]
        rscore = (last_rsi - 50.0) / 50.0

    # momentum via EMA slopes
    short = ema(prices, 5)
    long = ema(prices, 20)
    if not short or not long:
        mscore = 0.0
    else:
        mscore = (short[-1] - short[-2]) - (long[-1] - long[-2])
        # normalize by average price magnitude
        avgp = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        if avgp != 0:
            mscore = mscore / avgp

    # combine with weights
    score = 0.6 * rscore + 0.4 * np.tanh(mscore * 10)
    # clamp
    return float(max(min(score, 1.0), -1.0))


def livermore_3_points(prices: List[float], window=10) -> Optional[str]:
    """Simple detection of a 3-point reversal pattern (very simplified).

    If a local low followed by higher low and a higher high occurs -> 'buy'
    If the inverse occurs -> 'sell'
    """
    # allow short sequences (just inspect last three points if available)
    if len(prices) < 3:
        return None
    # find last three points
    p3 = prices[-3:]
    a, b, c = p3[0], p3[1], p3[2]
    if a < b and b < c:
        return 'buy'
    if a > b and b > c:
        return 'sell'
    return None


def auto_fib_levels(prices: List[float], lookback=50) -> Dict[str, float]:
    """Compute swing high/low over lookback and return Fibonacci retracement levels.

    Returns a dict with keys: high, low, 0.236, 0.382, 0.5, 0.618, 0.786
    """
    if len(prices) == 0:
        return {}
    window = prices[-lookback:] if len(prices) >= lookback else prices
    high = float(np.max(window))
    low = float(np.min(window))
    diff = high - low
    if diff == 0:
        # return flat levels
        return {"high": high, "low": low, "0.236": high, "0.382": high, "0.5": high, "0.618": high, "0.786": high}
    levels = {
        "high": high,
        "low": low,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
    }
    return levels


__all__ = [
    'watchtower_signal',
    'bot_activity_idiot_light',
    'believe_it_meter',
    'livermore_3_points',
    'auto_fib_levels',
]
