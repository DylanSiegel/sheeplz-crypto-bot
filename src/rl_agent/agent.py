# src/rl_agent/agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import yaml
import logging

# Import custom policy
from .custom_policy import LiquidNNPolicy

def load_config():
    """
    Load configuration from YAML file.
    """
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Define directories from config
MODEL_DIR = '../../models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_agent(env, total_timesteps=100000, save_freq=10000):
    """
    Train the PPO agent.
    
    Parameters:
    - env: The trading environment.
    - total_timesteps (int): Total number of training timesteps.
    - save_freq (int): Frequency (in timesteps) to save the model.
    """
    try:
        model = PPO(LiquidNNPolicy, env, verbose=1)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=MODEL_DIR,
                                                 name_prefix='ppo_trading_model')
        eval_callback = EvalCallback(env, best_model_save_path=MODEL_DIR,
                                     log_path=MODEL_DIR, eval_freq=5000,
                                     deterministic=True, render=False)
        
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
        
        # Save the final model
        model.save(os.path.join(MODEL_DIR, 'ppo_trading_final'))
        logging.info("Training completed and model saved.")
    except Exception as e:
        logging.error(f"Error during agent training: {e}")
