import torch
import math
from typing import Dict, Iterator, Optional
from torch.optim.optimizer import Optimizer

class ModifiedAdam(Optimizer):
    """
    Implementation of Adam optimizer modified for spherical optimization.
    Implements deformed gradient correction and proper scaling on the hypersphere.
    """
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)
    
    def _compute_scaled_grad(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        eps: float
    ) -> torch.Tensor:
        """Compute gradient scaled properly for spherical optimization"""
        # Project gradient onto tangent space
        r = torch.norm(p)
        u = p / (r + eps)
        grad_tang = grad - torch.sum(grad * u) * u
        
        # Scale gradient by radius
        scaled_grad = grad_tang / (r + eps)
        return scaled_grad
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']: