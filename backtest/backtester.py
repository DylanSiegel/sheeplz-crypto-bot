"""Simple backtester for 1-minute signals using vendored/ported indicators.

This backtester is intentionally simple and deterministic:
- Accepts a sequence of OHLCV bars
- Computes indicators per bar
- Generates signals via a configurable combiner of indicators
- Simulates market execution at next bar's open price with slippage and fees
- Uses PositionManager to enforce limits
"""
from typing import List, Dict, Any, Optional
import numpy as np
from exchanges.position_manager import PositionManager, PositionLimits
from integrations.indicators_tv import (
    watchtower_signal,
    bot_activity_idiot_light,
    believe_it_meter,
    livermore_3_points,
    auto_fib_levels,
)


class Backtester:
    def __init__(self, ohlcv: List[Dict[str, float]], starting_cash: float = 10000.0,
                 max_order_usd: float = 10.0, fee_pct: float = 0.0005, slippage_pct: float = 0.0005):
        """ohlcv: list of dicts with keys ['open','high','low','close','volume'] ordered oldest->newest"""
        self.ohlcv = ohlcv
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.max_order_usd = float(max_order_usd)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.posman = PositionManager(PositionLimits(max_notional_usd=starting_cash*0.5, min_order_usd=1.0))
        self.trades: List[Dict[str, Any]] = []

    def _price_series(self) -> List[float]:
        return [bar['close'] for bar in self.ohlcv]

    def run(self) -> Dict[str, Any]:
        prices = []
        volumes = []
        for i in range(len(self.ohlcv)-1):
            bar = self.ohlcv[i]
            prices.append(bar['close'])
            volumes.append(bar.get('volume', 0.0))

            # compute indicators
            wt = watchtower_signal(prices)
            vol_sig = bot_activity_idiot_light(volumes)
            bim = believe_it_meter(prices)
            liv = livermore_3_points(prices)
            fib = auto_fib_levels(prices)

            # combine signals: simple rule
            votes = []
            if wt == 'buy':
                votes.append(1)
            if wt == 'sell':
                votes.append(-1)
            if liv == 'buy':
                votes.append(1)
            if liv == 'sell':
                votes.append(-1)
            # treat volume spike as reinforcing the last price direction via bim
            if vol_sig == 'spike':
                if bim > 0:
                    votes.append(1)
                elif bim < 0:
                    votes.append(-1)

            combined = 0
            if votes:
                combined = sum(votes) / len(votes)

            # translate combined vote to order
            if combined > 0:
                side = 'buy'
            elif combined < 0:
                side = 'sell'
            else:
                side = None

            if side is not None:
                # simulate order execution at next bar open price with slippage
                next_bar = self.ohlcv[i+1]
                exec_price = next_bar['open'] * (1 + (self.slippage_pct if side == 'buy' else -self.slippage_pct))
                usd_notional = min(self.max_order_usd, self.starting_cash * 0.01)
                base_amount = usd_notional / exec_price
                fee = usd_notional * self.fee_pct

                # check position manager
                if self.posman.would_exceed_limits(side, base_amount, exec_price):
                    # skip
                    continue

                # execute
                self.trades.append({
                    'idx': i,
                    'side': side,
                    'price': exec_price,
                    'amount': base_amount,
                    'fee': fee,
                })
                # update cash
                self.cash -= usd_notional + fee if side == 'buy' else -usd_notional + fee
                # record trade in posman
                self.posman.record_trade(side, base_amount, exec_price)

        # At end, compute unrealized PnL based on last close
        last_price = self.ohlcv[-1]['close']
        position = self.posman.current_position()['base']
        unrealized = position * last_price
        net_worth = self.cash + unrealized
        pnl = net_worth - self.starting_cash
        return {
            'starting_cash': self.starting_cash,
            'ending_cash': self.cash,
            'unrealized': unrealized,
            'net_worth': net_worth,
            'pnl': pnl,
            'trades_count': len(self.trades),
            'trades': self.trades,
        }
