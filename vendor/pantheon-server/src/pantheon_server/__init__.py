"""
Pantheon Server package initialization
"""

__version__ = "0.1.0"
__author__ = "SpartanDigital"
__description__ = "FastAPI + Streamlit server for cryptocurrency analysis using pantheon-legends"

# Core imports for easy access
from .services.coinbase_service import CoinbaseService
from .services.market_analyzer import PantheonMarketAnalyzer

__all__ = [
    "CoinbaseService",
    "PantheonMarketAnalyzer",
]
