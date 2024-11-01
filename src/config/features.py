from pydantic import BaseModel, Field
from typing import List

class FeatureConfig(BaseModel):
    """Feature extraction configuration.

    This class defines parameters for extracting features from market data.  These features are used as input to the trading agent's neural network.

    Attributes:
        rolling_window_size (int): The size of the rolling window used for calculating features.  A larger window considers more historical data but might make the features less responsive to recent changes.
        technical_indicators (List[str]): A list of technical indicators to calculate.  These indicators provide insights into market trends and momentum.  Supported indicators depend on your `ta` library version.
        market_features (List[str]): A list of raw market features to include. These are typically directly from the market data (e.g., price, volume).
        normalization_method (str): The method used for normalizing features.  Normalization is essential to prevent features with larger scales from dominating the model's learning.


    """
    rolling_window_size: int = Field(100, description="Size of rolling window for features")
    technical_indicators: List[str] = Field(default=["rsi", "macd", "bbands", "volatility"], description="List of technical indicators to use")
    market_features: List[str] = Field(default=["price", "volume", "spread", "depth", "funding_rate"], description="List of market features to use")
    normalization_method: str = Field("standard", description="Feature normalization method")