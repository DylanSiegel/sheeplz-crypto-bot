# File: train.py (Optimized Training Loop)

import numpy as np
import torch
from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from agent import MetaSACAgent
from reward import calculate_reward

from typing import Callable
import random
import logging

def get_noise_schedule(initial_noise: float, final_noise: float, decay_steps: int) -> Callable[[int], float]:
    """
    Creates a linear noise schedule.

    Args:
        initial_noise: Initial noise value.
        final_noise: Final noise value.
        decay_steps: Number of steps to decay noise over.

    Returns:
        A function that takes the current step and returns the noise value.
    """
    def noise_fn(step: int) -> float:
        if step > decay_steps:
            return final_noise
        return initial_noise - (initial_noise - final_noise) * (step / decay_steps)
    return noise_fn

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MetaSACTrainer")

    # Initialize configuration
    config = EnvironmentConfig(
        state_dim=50,
        action_dim=5,
        hidden_dim=128,
        attention_dim=64,
        num_mlp_layers=3,
        dropout_rate=0.1,
        time_encoding_dim=16,
        custom_layers=["KLinePatternLayer", "VolatilityTrackingLayer", "FractalDimensionLayer"],
        window_size=20,
        num_hyperparams=10,
        graph_input_dim=10,
        graph_hidden_dim=32,
        num_graph_layers=2,
        ensemble_size=3,
        weight_decay=1e-5
    )

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create mock historical data
    mock_data = np.random.randn(2000, config.state_dim).astype(np.float32)

    # Initialize environment
    env = HistoricalEnvironment(mock_data)

    # Initialize agent
    agent = MetaSACAgent(config, env)

    # Warm-up replay buffer with random actions
    initial_steps = 200
    state = env.reset()
    for step in range(initial_steps):
        action = np.random.uniform(-1, 1, config.action_dim)
        next_state, reward, done, _info = env.step(action, step)
        agent.replay_buffer.add(state, action, reward, next_state, done, step)
        if done:
            state = env.reset()
        else:
            state = next_state

    # Define noise schedule
    noise_schedule = get_noise_schedule(initial_noise=0.2, final_noise=0.01, decay_steps=10000)

    # Training loop parameters
    num_epochs = 100
    updates_per_epoch = 100

    for epoch in range(num_epochs):
        epoch_losses = {}
        for _update in range(updates_per_epoch):
            # Example meta input: could be based on market indicators
            meta_input = np.random.randn(config.batch_size, config.meta_input_dim).astype(np.float32)
            time_memory = list(range(config.window_size))  # Example time memory

            losses = agent.update_params_with_training_time_search(
                replay_buffer=agent.replay_buffer,
                meta_input=meta_input,
                time_memory=time_memory,
                update_steps=1,
                search_algorithm="best-of-n",  # or "beam-search"
                num_samples=4,
                beam_width=3,
                search_depth=5,
                use_d_search=False,
                exploration_noise_std_fn=noise_schedule
            )

            # Aggregate losses for the epoch
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)

        # Compute average losses for the epoch
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, Total Steps={agent.train_steps}")
        for loss_name, loss_value in avg_losses.items():
            logger.info(f"  {loss_name}: {loss_value:.4f}")
            agent.writer.add_scalar(f"Epoch/Loss/{loss_name}", loss_value, epoch+1)

        # Save model checkpoints periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"metasac_epoch_{epoch+1}.pth"
            agent.save(checkpoint_path)
            logger.info(f"Model checkpoint saved at {checkpoint_path}")

    # Final model save
    agent.save("metasac_final.pth")
    logger.info("Final model saved as metasac_final.pth")

    # Close the TensorBoard writer
    agent.writer.close()

if __name__ == "__main__":
    main()
