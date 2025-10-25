"""Simple Kraken connector wrapper using ccxt with a DRY-RUN safety mode.

This wrapper intentionally provides a small, well-documented surface so the
bot can be wired to live markets while defaulting to safe behavior (no live
orders unless DRY_RUN is disabled via env/config).

Note: This uses the ccxt library. For Kraken Futures or testnet-specific
endpoints you may need to adjust `ccxt` exchange id or urls.
"""
import os
import time
import logging
from typing import Any, Dict, Optional

import ccxt

logger = logging.getLogger(__name__)


class KrakenClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 testnet: bool = False, dry_run: bool = True, rate_limit_sleep: float = 0.2):
        api_key = api_key or os.getenv("KRAKEN_API_KEY")
        api_secret = api_secret or os.getenv("KRAKEN_API_SECRET")
        self.dry_run = bool(str(os.getenv("DRY_RUN", str(dry_run))).lower() in ("1", "true", "yes"))
        self.rate_limit_sleep = float(os.getenv("RATE_LIMIT_SLEEP", rate_limit_sleep))

        if api_key and api_secret:
            self.client = ccxt.kraken({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            })
        else:
            # Unauthenticated client for public data
            self.client = ccxt.kraken({"enableRateLimit": True})

        # If testnet requested, try to set test urls if provided by ccxt mapping
        if testnet:
            try:
                # Some ccxt exchanges provide a `test` url mapping
                self.client.urls["api"] = self.client.urls.get("test", self.client.urls.get("api"))
            except Exception:
                logger.warning("Could not switch client to testnet urls; check ccxt support for kraken testnet")

        # Load market metadata into a cache (best-effort)
        try:
            self.markets = self.client.load_markets()
        except Exception as e:
            logger.warning(f"Failed to load markets: {e}")
            self.markets = {}

    def _sleep(self) -> None:
        # Small sleep to help respect rate limits; ccxt also enforces rate-limiting
        time.sleep(self.rate_limit_sleep)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        self._sleep()
        return self.client.fetch_ticker(symbol)

    def fetch_balance(self) -> Dict[str, Any]:
        if self.dry_run:
            logger.info("DRY RUN: returning simulated balance")
            # Minimal simulated balance for dry-run
            return {"total": {"USD": 1000.0}}
        self._sleep()
        return self.client.fetch_balance()

    def fetch_open_orders(self, symbol: Optional[str] = None) -> Any:
        if self.dry_run:
            logger.info("DRY RUN: fetch_open_orders -> []")
            return []
        self._sleep()
        return self.client.fetch_open_orders(symbol)

    def cancel_order(self, order_id: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        if self.dry_run:
            logger.info(f"DRY RUN cancel_order: {order_id}")
            return {"info": {"dry_run": True}}
        self._sleep()
        return self.client.cancel_order(order_id, params or {})

    def create_market_order(self, symbol: str, side: str, amount: float, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Place a market order (or simulate it in dry-run).

        Args:
            symbol: market symbol accepted by ccxt (e.g. 'BTC/USD' or 'XBT/USD')
            side: 'buy' or 'sell'
            amount: amount in base currency (e.g. BTC)
        """
        params = params or {}
        if self.dry_run:
            logger.info(f"DRY RUN ORDER: {side} {amount} {symbol} (market) params={params}")
            return {"info": {"dry_run": True}, "symbol": symbol, "side": side, "amount": amount}
        try:
            self._sleep()
            order = self.client.create_order(symbol, "market", side, amount, None, params)
            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    # Small utility to convert an agent action (-1..1) into (side, amount_base)
    def action_to_order(self, action: float, symbol: str, max_order_usd: float = 100.0, price: Optional[float] = None) -> Dict[str, Any]:
        """Converts a normalized action to an order dict.

        - action in [-1,1]
        - maps magnitude to USD notional up to max_order_usd
        - returns dict {side, amount, price, usd_notional}
        """
        if abs(action) < 1e-6:
            return {"side": None, "amount": 0.0, "price": price, "usd_notional": 0.0}

        side = "buy" if action > 0 else "sell"
        magnitude = min(abs(action), 1.0)
        usd = magnitude * float(max_order_usd)

        # fetch price if not provided
        if price is None:
            try:
                ticker = self.fetch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or ticker.get("info", {}).get("price"))
            except Exception:
                raise RuntimeError("Unable to fetch price for symbol to convert USD notional to base amount")

        amount = usd / float(price)

        # try to round to market precision if available
        market = self.markets.get(symbol)
        if market:
            precision = market.get("precision", {})
            base_prec = precision.get("amount")
            if base_prec is not None:
                amount = float(round(amount, base_prec))

        return {"side": side, "amount": amount, "price": price, "usd_notional": usd}
