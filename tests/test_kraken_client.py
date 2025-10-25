import pytest

from exchanges.kraken_client import KrakenClient


def test_dry_run_create_order():
    client = KrakenClient(dry_run=True)
    resp = client.create_market_order('XBT/USD', 'buy', 0.001)
    assert isinstance(resp, dict)
    assert resp.get('info', {}).get('dry_run') is True


def test_action_to_order_with_price():
    client = KrakenClient(dry_run=True)
    res = client.action_to_order(0.5, 'XBT/USD', max_order_usd=200.0, price=20000.0)
    assert res['side'] == 'buy'
    assert res['usd_notional'] == pytest.approx(0.5 * 200.0)
    assert res['amount'] == pytest.approx((0.5 * 200.0) / 20000.0)
