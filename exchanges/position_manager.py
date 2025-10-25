"""Simple position manager to track simulated positions and enforce safety limits.

This is intentionally small: it keeps a running position in base units, enforces
maximum position exposure in USD, and provides helper checks before placing
orders. It works in paper/dry-run mode and can be extended later to persist
positions or sync with exchange state.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionLimits:
    max_notional_usd: float = 1000.0  # maximum USD exposure
    max_base_amount: Optional[float] = None  # optional cap on base currency
    min_order_usd: float = 1.0


class PositionManager:
    def __init__(self, limits: Optional[PositionLimits] = None):
        self.limits = limits or PositionLimits()
        # Track position as signed base amount (positive = long/buy, negative = short/sell)
        self.position_base = 0.0
        self.avg_entry_price = None

    def current_position(self):
        return {"base": self.position_base, "avg_entry_price": self.avg_entry_price}

    def would_exceed_limits(self, side: str, amount_base: float, price: float) -> bool:
        """Return True if executing this order would exceed configured limits."""
        usd_notional = abs(amount_base) * price
        if usd_notional < self.limits.min_order_usd:
            return True

        # compute prospective position
        prospective_base = self.position_base + (amount_base if side == "buy" else -amount_base)
        prospective_notional = abs(prospective_base) * price

        if self.limits.max_notional_usd is not None and prospective_notional > self.limits.max_notional_usd:
            return True

        if self.limits.max_base_amount is not None and abs(prospective_base) > self.limits.max_base_amount:
            return True

        return False

    def record_trade(self, side: str, amount_base: float, price: float) -> None:
        """Record an executed trade (updates position). Assumes trade executed successfully."""
        # update avg entry price via simple weighted average for the position
        signed_amount = amount_base if side == "buy" else -amount_base
        if self.position_base == 0 or (self.position_base > 0 and signed_amount > 0) or (self.position_base < 0 and signed_amount < 0):
            # extending position in same direction or opening new
            total_base = self.position_base + signed_amount
            if total_base == 0:
                # flat
                self.avg_entry_price = None
                self.position_base = 0.0
                return
            if self.avg_entry_price is None:
                self.avg_entry_price = price
            else:
                # weighted average price
                prev_notional = abs(self.position_base) * (self.avg_entry_price or price)
                add_notional = abs(signed_amount) * price
                self.avg_entry_price = (prev_notional + add_notional) / (abs(self.position_base) + abs(signed_amount))
            self.position_base = total_base
        else:
            # reducing or flipping position
            self.position_base += signed_amount
            if abs(self.position_base) < 1e-12:
                self.position_base = 0.0
                self.avg_entry_price = None
