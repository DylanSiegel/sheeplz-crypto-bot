import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque
from loguru import logger
import torch

@dataclass
class RiskConfig:
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_leverage: float = 5.0
    min_trade_interval: float = 1.0  # seconds
    max_drawdown: float = 0.2
    position_sizing_method: str = "adaptive_kelly"
    num_threads: int = 24  # Optimized for Ryzen 9 7900X
    risk_check_batch_size: int = 1000
    use_cuda: bool = True

class Position:
    def __init__(self, size: float, entry_price: float, entry_time: float):
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = None
        self.take_profit = None
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

class EnhancedRiskManager:
    """Hardware-optimized risk management system"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
        
        # Position management
        self.positions: Dict[str, Position] = {}
        self.position_lock = threading.Lock()
        
        # Performance tracking
        self.trades_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        self.last_trade_time = 0
        
        # Risk metrics
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.win_rate = 0.5  # Initial estimate
        
        # Batch processing buffers
        self.pending_actions = []
        self.pending_prices = []
        self.pending_timestamps = []
        
        # CUDA streams for parallel computation
        self.streams = None
        if self.device.type == 'cuda':
            self.streams = [torch.cuda.Stream() for _ in range(2)]

    def validate_action_batch(self, 
                            actions: List[int],
                            prices: List[float],
                            account_balance: float) -> List[Tuple[bool, str]]:
        """Validate multiple actions in parallel"""
        if self.device.type == 'cuda':
            return self._validate_action_batch_gpu(actions, prices, account_balance)
        else:
            return self._validate_action_batch_cpu(actions, prices, account_balance)

    @torch.compile
    def _validate_action_batch_gpu(self,
                                 actions: List[int],
                                 prices: List[float],
                                 account_balance: float) -> List[Tuple[bool, str]]:
        """GPU-accelerated batch action validation"""
        with torch.cuda.amp.autocast():
            # Convert inputs to tensors
            actions_tensor = torch.tensor(actions, device=self.device)
            prices_tensor = torch.tensor(prices, device=self.device)
            
            # Calculate position sizes
            position_sizes = self._calculate_position_sizes_gpu(
                actions_tensor, prices_tensor, account_balance
            )
            
            # Check position limits
            position_valid = torch.abs(position_sizes) <= (
                self.config.max_position_size * account_balance
            )
            
            # Check leverage limits
            leverage = torch.abs(position_sizes * prices_tensor) / account_balance
            leverage_valid = leverage <= self.config.max_leverage
            
            # Check trading frequency
            current_time = time.time()
            time_valid = (current_time - self.last_trade_time) >= self.config.min_trade_interval
            
            # Combine all checks
            valid = position_valid & leverage_valid & torch.full_like(
                position_valid, time_valid, dtype=torch.bool
            )
            
            # Generate messages
            messages = []
            for i, (p_valid, l_valid, is_valid) in enumerate(
                zip(position_valid.cpu(), leverage_valid.cpu(), valid.cpu())
            ):
                if not is_valid:
                    if not time_valid:
                        messages.append("Trading too frequently")
                    elif not p_valid:
                        messages.append("Position size exceeds maximum")
                    elif not l_valid:
                        messages.append("Leverage exceeds maximum")
                else:
                    messages.append("")
            
            return list(zip(valid.cpu().tolist(), messages))

    def _validate_action_batch_cpu(self,
                                 actions: List[int],
                                 prices: List[float],
                                 account_balance: float) -> List[Tuple[bool, str]]:
        """CPU-based parallel action validation"""
        def validate_single(args):
            action, price = args
            return self._validate_single_action(action, price, account_balance)
        
        # Process validations in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            results = list(executor.map(validate_single, zip(actions, prices)))
        
        return results

    def _validate_single_action(self,
                              action: int,
                              current_price: float,
                              account_balance: float) -> Tuple[bool, str]:
        """Validate a single action with all risk checks"""
        current_time = time.time()
        
        # Check trading frequency
        if current_time - self.last_trade_time < self.config.min_trade_interval:
            return False, "Trading too frequently"
        
        # Calculate position size
        position_size = self._calculate_adaptive_position_size(
            action, current_price, account_balance
        )
        
        # Check position limits
        if abs(position_size) > self.config.max_position_size * account_balance:
            return False, "Position size exceeds maximum"
        
        # Check leverage
        leverage = abs(position_size * current_price) / account_balance
        if leverage > self.config.max_leverage:
            return False, "Leverage exceeds maximum"
        
        # Check drawdown
        if self.current_drawdown > self.config.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        return True, ""

    @torch.compile
    def _calculate_position_sizes_gpu(self,
                                    actions: torch.Tensor,
                                    prices: torch.Tensor,
                                    account_balance: float) -> torch.Tensor:
        """Calculate position sizes using GPU acceleration"""
        # Base position size
        base_size = account_balance * self.config.max_position_size
        
        # Calculate adaptive Kelly criterion
        kelly_fraction = self._calculate_kelly_fraction_gpu