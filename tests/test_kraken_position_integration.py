from exchanges.kraken_client import KrakenClient
from exchanges.position_manager import PositionManager, PositionLimits


def test_kraken_action_and_position_integration():
    kc = KrakenClient(dry_run=True)
    limits = PositionLimits(max_notional_usd=10000.0, min_order_usd=1.0)
    pm = PositionManager(limits)

    # create order via action helper
    order = kc.action_to_order(0.5, 'XBT/USD', max_order_usd=200.0, price=20000.0)
    assert order['side'] == 'buy'
    assert order['usd_notional'] == 100.0

    # ensure position manager allows this order
    assert pm.would_exceed_limits(order['side'], order['amount'], order['price']) is False

    # simulate execution and record
    resp = kc.create_market_order('XBT/USD', order['side'], order['amount'])
    assert resp.get('info', {}).get('dry_run') is True
    pm.record_trade(order['side'], order['amount'], order['price'])
    pos = pm.current_position()
    assert pos['base'] > 0
