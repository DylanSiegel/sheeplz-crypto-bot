from exchanges.position_manager import PositionManager, PositionLimits


def test_position_limits_min_order():
    limits = PositionLimits(max_notional_usd=100.0, min_order_usd=10.0)
    pm = PositionManager(limits)
    # An order below min USD should be flagged
    assert pm.would_exceed_limits('buy', amount_base=0.0001, price=100.0) is True


def test_position_limits_max_notional():
    limits = PositionLimits(max_notional_usd=500.0, min_order_usd=1.0)
    pm = PositionManager(limits)
    # If current position + order would exceed max notional, it should return True
    assert pm.would_exceed_limits('buy', amount_base=10.0, price=100.0) is True


def test_record_trade_updates_position():
    limits = PositionLimits(max_notional_usd=10000.0, min_order_usd=1.0)
    pm = PositionManager(limits)
    pm.record_trade('buy', amount_base=1.0, price=100.0)
    pos = pm.current_position()
    assert pos['base'] == 1.0
    assert pos['avg_entry_price'] == 100.0
