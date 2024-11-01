from pydantic import BaseModel, Field

class RiskConfig(BaseModel):
    """Risk management configuration.

    This class defines parameters to control risk during trading.  These parameters are crucial for preventing excessive losses.

    Attributes:
        max_position_size (float): The maximum fraction of the account balance that can be used for a single position.  A value between 0 and 1.  For example, 0.1 means a maximum of 10% of the account balance can be risked on any one trade.
        stop_loss_pct (float): The percentage loss at which an open position will be automatically closed (stop-loss order).  A value between 0 and 1.  For example, 0.02 means a stop-loss order will be triggered if the position incurs a 2% loss.
        take_profit_pct (float): The percentage profit at which an open position will be automatically closed (take-profit order). A value between 0 and 1. For example, 0.04 means a take-profit order will be triggered if the position achieves a 4% profit.
        max_leverage (float): The maximum leverage allowed for trades.  This magnifies both profits and losses.  Higher leverage increases risk.
        min_trade_interval (int): The minimum time (in seconds) between consecutive trades to avoid excessive trading frequency.  This helps to avoid over-trading and potential slippage.
        max_drawdown (float): The maximum acceptable drawdown (percentage loss from peak equity) before trading is stopped.  This parameter helps to prevent catastrophic losses.
        position_sizing_method (str): The method used for determining position sizes.  This could be 'kelly' (Kelly criterion), 'fixed', or another custom method.

    """
    max_position_size: float = Field(0.1, description="Maximum position size as fraction of capital")
    stop_loss_pct: float = Field(0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(0.04, description="Take profit percentage")
    max_leverage: float = Field(5.0, description="Maximum allowed leverage")
    min_trade_interval: int = Field(5, description="Minimum time interval between trades (seconds)")
    max_drawdown: float = Field(0.2, description="Maximum allowed drawdown")
    position_sizing_method: str = Field("kelly", description="Position sizing method")