# train.py
import numpy as np
import torch
from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from agent import MetaSACAgent

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

if __name__ == "__main__":
    # 1) Create config and mock data
    config = EnvironmentConfig()
    np.random.seed(42)
    mock_data = np.random.randn(2000, config.state_dim).astype(np.float32)

    # 2) Create environment
    env = HistoricalEnvironment(mock_data)

    # 3) Create agent
    agent = MetaSACAgent(config, env)

    # 4) Warm-up replay buffer
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

    # 5) Noise schedule
    noise_schedule = get_noise_schedule(initial_noise=0.2, final_noise=0.01, decay_steps=10000)

    # 6) Training loop
    num_epochs = 5
    updates_per_epoch = 10
    for epoch in range(num_epochs):


        for _update in range(updates_per_epoch):
            # Example meta input: random
            meta_input = np.random.randn(config.batch_size, config.meta_input_dim).astype(np.float32)
            time_memory = [0]

            agent.update_params_with_training_time_search(
                agent.replay_buffer,
                meta_input=meta_input,
                time_memory=time_memory,
                update_steps=1,
                search_algorithm="best-of-n",
                num_samples=4,
                beam_width=3,
                search_depth=5,
                use_d_search=False,
                exploration_noise_std_fn=noise_schedule
            )
        print(f"Epoch {epoch+1}/{num_epochs} completed, total steps={agent.train_steps}.")

    # 7) Save model
    agent.save("metasac_final.pth")