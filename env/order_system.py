# env/order_system.py

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from env.config import EnvironmentConfig

logger = logging.getLogger(__name__)

@dataclass
class Order:
    order_id: str
    agent_id: int
    side: str  # "buy" / "sell"
    order_type: str  # "market" / "limit" / "stop_limit"
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"  # "pending"/"open"/"filled"/"canceled"/"rejected"
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    created_at: int = 0
    expire_at: Optional[int] = None
    time_in_force: str = "GTC"

    def __post_init__(self):
        if self.order_type in ["limit", "stop_limit"] and self.price is None:
            raise ValueError("Price must be specified for limit or stop-limit orders.")
        if self.order_type == "stop_limit" and self.stop_price is None:
            raise ValueError("Stop price must be specified for stop-limit orders.")

class OrderBook:
    """
    Simplified in-memory order book to match "market" orders or store "limit" orders.
    """
    def __init__(self):
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}

    def add_order(self, order: Order):
        if order.side == "buy":
            levels = self.bids
        else:
            levels = self.asks
        if order.price not in levels:
            levels[order.price] = 0.0
        levels[order.price] += order.size

    def remove_order(self, order: Order):
        if order.side == "buy":
            levels = self.bids
        else:
            levels = self.asks
        if order.price in levels:
            levels[order.price] -= order.size
            if levels[order.price] <= 0:
                del levels[order.price]

    def match_order(self, order: Order) -> List[Tuple[Order, float, float]]:
        """
        Attempt to fill a market/limit order from the opposite side of the book.
        Returns fill events: (matching_order, filled_size, fill_price).
        For simplicity, matching_order is None. We'll only store fill_size/fill_price.
        """
        fills = []
        if order.side == "buy":
            levels = self.asks
            best_price_func = min
        else:
            levels = self.bids
            best_price_func = max

        while True:
            if not levels:
                break

            best_price = best_price_func(levels.keys())

            # If limit/stop-limit and can't fill at best_price => done
            if order.order_type in ["limit", "stop_limit"]:
                if order.side == "buy" and (order.price < best_price):
                    break
                if order.side == "sell" and (order.price > best_price):
                    break

            best_level_size = levels[best_price]
            fill_size = min(order.size, best_level_size)

            fills.append((None, fill_size, best_price))
            order.size -= fill_size
            levels[best_price] -= fill_size

            if levels[best_price] <= 0:
                del levels[best_price]
            if order.size <= 0:
                break

        return fills

class OrderManager:
    """
    Manages creation and execution of orders in the environment's order book.
    """
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.order_book = OrderBook()
        self.orders: Dict[str, Order] = {}
        self.next_order_id = 1

    def create_order_id(self, agent_id: int) -> str:
        order_id = f"order-{agent_id}-{self.next_order_id}"
        self.next_order_id += 1
        return order_id

    def place_order(
        self,
        agent_id: int,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        expire_at: Optional[int] = None
    ) -> str:
        oid = self.create_order_id(agent_id)
        new_order = Order(
            order_id=oid,
            agent_id=agent_id,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            stop_price=stop_price,
            created_at=0,
            time_in_force=time_in_force,
            expire_at=expire_at
        )

        if order_type == "market":
            # Immediately try to match
            fills = self.order_book.match_order(new_order)
            if fills:
                self.execute_fills(new_order, fills)
            else:
                # No liquidity => reject
                new_order.status = "rejected"
                logger.warning(f"Market order {oid} rejected, no liquidity.")
        else:
            # Add to order book
            self.order_book.add_order(new_order)
            new_order.status = "open"
            logger.info(f"Order {oid} placed: {side} {order_type} size={size}, price={price}")

        self.orders[oid] = new_order
        return oid if new_order.status != "rejected" else ""

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order and order.status == "open":
            self.order_book.remove_order(order)
            order.status = "canceled"
            logger.info(f"Order {order_id} canceled.")
            return True
        logger.warning(f"Unable to cancel order {order_id}. Not open or not found.")
        return False

    def execute_fills(self, order: Order, fills: List[Tuple[Order, float, float]]):
        for _, fill_size, fill_price in fills:
            order.filled_size += fill_size
            order.average_fill_price = fill_price
            logger.info(f"Order {order.order_id} partially filled {fill_size} @ {fill_price}")

        if order.filled_size >= order.size - 1e-12:
            # Fully filled
            order.status = "filled"
            logger.info(f"Order {order.order_id} fully filled.")

    def update_orders(self, current_price: float, current_time_step: int):
        """
        Update existing orders (limit/stop-limit) if they can now be matched, or if they expire.
        """
        to_remove = []
        for oid, o in self.orders.items():
            if o.status == "open":
                if o.expire_at is not None and current_time_step >= o.expire_at:
                    # Expire
                    self.order_book.remove_order(o)
                    o.status = "canceled"
                    to_remove.append(oid)
                    logger.info(f"Order {oid} expired at step={current_time_step}.")
                    continue
                # Attempt matching
                fills = self.order_book.match_order(o)
                if fills:
                    self.execute_fills(o, fills)
                    if o.status == "filled":
                        to_remove.append(oid)
        for oid in to_remove:
            del self.orders[oid]

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
