from typing import Dict, List
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def preprocess_market_data(self, raw_data: Dict) -> Dict:
        """Process raw market data into standardized format"""
        processed_data = {
            'close_price': float(raw_data['last_price']),
            'volume': float(raw_data['volume_24h']),
            'bid_ask_spread': float(raw_data['best_ask']) - float(raw_data['best_bid']),
            'funding_rate': float(raw_data['funding_rate']),
            'open_interest': float(raw_data['open_interest']),
            'leverage_ratio': float(raw_data['leverage_ratio']),
            'market_depth_ratio': self._calculate_depth_ratio(raw_data),
            'taker_buy_ratio': float(raw_data['taker_buy_volume']) / 
                              (float(raw_data['taker_buy_volume']) + 
                               float(raw_data['taker_sell_volume']))
        }
        return processed_data