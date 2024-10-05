# File: tests/test_risk_management.py

import pytest
from models.utils.risk_management import RiskManager

@pytest.fixture
def risk_manager():
    params = {
        "max_drawdown": 0.1,
        "max_position_size": 0.05
    }
    return RiskManager(params)

def test_risk_within_limits(risk_manager):
    current_drawdown = 0.05
    current_position = 'long'
    market_data = {}
    assert risk_manager.check_risk(current_drawdown, current_position, market_data) == True

def test_risk_exceeds_drawdown(risk_manager):
    current_drawdown = 0.15
    current_position = 'long'
    market_data = {}
    assert risk_manager.check_risk(current_drawdown, current_position, market_data) == False

def test_risk_without_position(risk_manager):
    current_drawdown = 0.05
    current_position = None
    market_data = {}
    assert risk_manager.check_risk(current_drawdown, current_position, market_data) == True

def test_risk_edge_case(risk_manager):
    current_drawdown = 0.1
    current_position = 'short'
    market_data = {}
    assert risk_manager.check_risk(current_drawdown, current_position, market_data) == False  # Assuming >= is considered exceeding
