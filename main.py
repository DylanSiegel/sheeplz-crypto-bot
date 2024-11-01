import logging
from src.config import TradingConfig
from src.models.agent import Agent
from src.data.features import FeatureExtractor
from src.env.bybit_env import BybitFuturesEnv
from src.utils.risk_manager import RiskManager

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = TradingConfig()
    
    # Initialize components
    feature_extractor = FeatureExtractor(config)
    risk_manager = RiskManager(config)
    env = BybitFuturesEnv(config)
    
    # Initialize agent
    agent = Agent(
        input_size=config.feature_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    )
    
    logger.info("Starting training...")
    
    # Training loop
    for episode in range(config.num_episodes):
        state, info = env.reset()
        hidden_states = agent.init_hidden()
        episode_reward = 0
        
        while True:
            # Get action from agent
            action, value, hidden_states = agent.act(state, hidden_states)
            
            # Validate action with risk manager
            is_valid, message = risk_manager.validate_action(
                action, env.current_price, env.account_balance
            )
            
            if not is_valid:
                logger.warning(f"Action rejected: {message}")
                action = 2  # Default to HOLD
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
            
            state = next_state
            
        logger.info(f"Episode {episode} completed with reward {episode_reward}")

if __name__ == "__main__":
    main()