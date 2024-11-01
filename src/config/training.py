from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    """Training configuration.

    This class contains hyperparameters for the reinforcement learning training process.

    Attributes:
        num_episodes (int): The total number of training episodes.
        gamma (float): The discount factor used in the reinforcement learning algorithm.  This determines the importance of future rewards. A value closer to 1 emphasizes long-term rewards.
        gae_lambda (float): The lambda parameter for Generalized Advantage Estimation (GAE).  This parameter controls the bias-variance tradeoff in GAE.
        entropy_coef (float): The coefficient for the entropy bonus in the loss function.  This encourages exploration during training.
        value_loss_coef (float): The coefficient for the value function loss in the loss function.  This balances the policy loss and value function loss.
        max_grad_norm (float): The maximum norm for gradient clipping to prevent exploding gradients during training.  Gradient clipping helps stabilize training.
        update_interval (int): The number of steps between policy updates.  More frequent updates might lead to faster learning but can be more noisy.
        num_minibatches (int): The number of minibatches used for each policy update. This parameter is used in algorithms like PPO to reduce computational cost.
        warmup_steps (int): The number of initial steps during which the policy is not updated.  This allows the agent to gather some initial experience before updating the policy.

    """
    num_episodes: int = Field(10000, description="Number of training episodes")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    entropy_coef: float = Field(0.01, description="Entropy coefficient")
    value_loss_coef: float = Field(0.5, description="Value loss coefficient")
    max_grad_norm: float = Field(0.5, description="Maximum gradient norm")
    update_interval: int = Field(2048, description="Steps between policy updates")
    num_minibatches: int = Field(4, description="Number of minibatches per update")
    warmup_steps: int = Field(1000, description="Number of warmup steps")