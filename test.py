import unittest
import numpy as np
import torch
import os
from typing import Tuple, List, Dict
import logging
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# Import your SAC implementation
from meta_sac import MetaSACAgent, MetaSACConfig, ReplayBuffer, SinusoidalTimeEncoding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEnvironment:
    """
    Custom environment for testing SAC agent.
    Simulates a simple control task where the goal is to move a point mass to a target.
    """
    def __init__(self, state_dim: int = 10, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = 200
        self.target_state = np.zeros(state_dim)  # Target is at origin
        self.step_count = 0
        self.state = None
        self.action_scale = 2.0  # Scale factor for actions
        
        # Dynamic parameters
        self.inertia = 0.8  # How much previous state affects next state
        self.action_influence = 0.2  # How much actions affect the state
        self.noise_scale = 0.01  # Environmental noise

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Initialize state randomly but not too far from origin
        self.state = np.random.randn(self.state_dim)
        self.step_count = 0
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in environment given an action.
        
        Args:
            action: Array of action values in [-1, 1]
            
        Returns:
            (next_state, reward, done, info)
        """
        # Clip actions to valid range
        action = np.clip(action, -1, 1) * self.action_scale
        
        # Update state based on action and dynamics
        action_effect = np.zeros(self.state_dim)
        action_effect[:self.action_dim] = action
        
        self.state = (
            self.inertia * self.state + 
            self.action_influence * action_effect +
            self.noise_scale * np.random.randn(self.state_dim)
        )
        
        # Calculate reward based on distance to target and action magnitude
        distance_to_target = np.linalg.norm(self.state - self.target_state)
        action_penalty = 0.1 * np.linalg.norm(action)  # Small penalty for large actions
        reward = -distance_to_target - action_penalty
        
        # Update step count and check if episode is done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Add some success and failure conditions
        if distance_to_target < 0.1:  # Success condition
            reward += 10
            done = True
        elif distance_to_target > 20:  # Failure condition
            reward -= 10
            done = True
            
        return self.state.copy(), reward, done, {
            'distance': distance_to_target,
            'step': self.step_count
        }

class TestSAC(unittest.TestCase):
    """Test suite for SAC agent."""
    
    def setUp(self):
        """Setup test environment and agent."""
        self.config = MetaSACConfig(
            state_dim=10,
            action_dim=2,
            hidden_dim=64,
            attention_dim=32,
            meta_input_dim=5,
            batch_size=64,
            lr=3e-4,
            meta_lr=1e-4,
            time_encoding_dim = 20
        )
        self.env = CustomEnvironment(self.config.state_dim, self.config.action_dim)
        self.agent = MetaSACAgent(self.config)
        self.replay_buffer = ReplayBuffer(100000)
        self.time_encoding_dim = 20
        
        # Training parameters
        self.num_eval_episodes = 5
        self.num_train_episodes = 50
        self.eval_interval = 10
        
        # Create directory for saving results
        self.results_dir = 'test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def generate_time_encoding(self) -> int:
        """Generate time encoding vector."""
        return np.random.randint(0, 1000)
    
    def evaluate_policy(self, num_episodes: int = 5) -> Tuple[float, List[float]]:
        """
        Evaluate current policy.
        
        Returns:
            Tuple of (mean reward, list of episode rewards)
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            time_step = 0
            while not done:
                action = self.agent.select_action(state, time_step, eval=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                time_step+=1
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), episode_rewards

    def test_basic_functionality(self):
        """Test basic environment and agent functionality."""
        # Test environment
        state = self.env.reset()
        self.assertEqual(state.shape, (self.config.state_dim,), "Environment state shape is incorrect")
        
        # Test agent action selection
        time_step = self.generate_time_encoding()
        action = self.agent.select_action(state, time_step)
        self.assertEqual(action.shape, (self.config.action_dim,), "Action shape is incorrect")
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1), "Action values out of range")
    
    def test_time_encoding(self):
        """Test time encoding mechanism."""
        time_encoding_dim = self.config.time_encoding_dim
        time_encoding = SinusoidalTimeEncoding(time_encoding_dim)
        
        # Test for a specific time step
        time_step = torch.tensor([5])
        encoding = time_encoding(time_step)
        
        self.assertEqual(encoding.shape, torch.Size([1, time_encoding_dim]), "Incorrect time encoding shape")
        self.assertTrue(torch.all(encoding >= -1.0) and torch.all(encoding <= 1.0), "Time encoding values are out of range")

    def test_learning(self):
        """Test if agent can learn and improve."""
        initial_mean, initial_rewards = self.evaluate_policy()
        logger.info(f"Initial performance: {initial_mean:.2f}")
        
        eval_rewards = []
        training_rewards = []
        distances = []
        alphas = []
        critic1_losses = []
        critic2_losses = []
        actor_losses = []

        # Training loop
        for episode in range(self.num_train_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_distances = []
            done = False
            meta_input = np.random.randn(self.config.meta_input_dim)
            time_memory = []
            time_step = 0
            
            while not done:
                # Get action from agent
                time_memory.append(time_step)
                action = self.agent.select_action(state, time_step)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_distances.append(info['distance'])
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update agent if enough samples
                if len(self.replay_buffer) >= self.config.batch_size:
                    update_info = self.agent.update_params(
                        self.replay_buffer,
                        meta_input,
                        time_memory[-self.config.batch_size:]
                    )
                    
                    # Verify no NaN values in losses
                    for key, value in update_info.items():
                        self.assertFalse(np.isnan(value), f"NaN detected in {key}")
                    
                    # Store losses and alpha values
                    if "alpha" in update_info:
                      alphas.append(update_info["alpha"])
                    if "critic1_loss" in update_info:
                        critic1_losses.append(update_info["critic1_loss"])
                    if "critic2_loss" in update_info:
                        critic2_losses.append(update_info["critic2_loss"])
                    if "actor_loss" in update_info:
                         actor_losses.append(update_info["actor_loss"])
                
                state = next_state
                time_step += 1

            training_rewards.append(episode_reward)
            distances.append(np.mean(episode_distances))
            
            # Evaluate policy periodically
            if episode % self.eval_interval == 0:
                mean_reward, _ = self.evaluate_policy()
                eval_rewards.append(mean_reward)
                logger.info(f"Episode {episode}, Eval reward: {mean_reward:.2f}")
        
        final_mean, final_rewards = self.evaluate_policy()
        logger.info(f"Final performance: {final_mean:.2f}")
        
        # Plot results
        self._plot_results(training_rewards, eval_rewards, distances, alphas, critic1_losses, critic2_losses, actor_losses)
        
        # Assert improvement
        self.assertGreater(final_mean, initial_mean, 
                          "Agent failed to improve performance")
        
        #Assert that the alpha is decreasing
        if len(alphas) > 10:
            self.assertLess(np.mean(alphas[-10:]), np.mean(alphas[:10]), "Alpha not decreasing")

    def test_reproducibility(self):
        """Test if training is reproducible with same seed."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Get results from two identical training runs
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            np.random.seed(42)
            
            agent = MetaSACAgent(self.config)
            env = CustomEnvironment(self.config.state_dim, self.config.action_dim)
            
            state = env.reset()
            actions = []
            time_step = 0

            for _ in range(10):
                action = agent.select_action(state, time_step, eval=True)
                actions.append(action)
                state, _, done, _ = env.step(action)
                time_step+=1
                if done:
                    break
            
            results.append(actions)
        
        # Compare results
        for a1, a2 in zip(results[0], results[1]):
            np.testing.assert_array_almost_equal(a1, a2, err_msg="Actions are not reproducible")

    def test_deterministic_eval(self):
        """Test action selection is deterministic in eval mode."""
        state = self.env.reset()
        time_step = self.generate_time_encoding()
        
        actions = []
        for _ in range(10):
            action = self.agent.select_action(state, time_step, eval=True)
            actions.append(action)
        
        # Compare all actions to first action
        for idx, action in enumerate(actions[1:]):
            np.testing.assert_array_almost_equal(action, actions[0], err_msg=f"Deterministic eval failed on action {idx +1}")

    def test_save_load(self):
        """Test model saving and loading."""
        # Train agent briefly
        torch.manual_seed(42)
        np.random.seed(42)
        state = self.env.reset()
        initial_actions = []
        time_step = 0
        
        for _ in range(10):
            action = self.agent.select_action(state, time_step, eval=True)
            initial_actions.append(action)
            state, _, done, _ = self.env.step(action)
            time_step +=1
            if done:
                state = self.env.reset()
                time_step = 0
        
        # Save model
        save_path = os.path.join(self.results_dir, "test_model.pth")
        self.agent.save(save_path)
        
        # Create new agent and load saved model
        torch.manual_seed(42)
        np.random.seed(42)
        loaded_agent = MetaSACAgent(self.config)
        loaded_agent.load(save_path)
        
        # Compare actions
        state = self.env.reset()
        time_step = 0

        for initial_action in initial_actions:
            loaded_action = loaded_agent.select_action(state, time_step, eval=True)
            np.testing.assert_array_almost_equal(initial_action, loaded_action, err_msg="Action is different after loading")
            state, _, done, _ = self.env.step(loaded_action)
            time_step += 1
            if done:
                state = self.env.reset()
                time_step = 0

    def _plot_results(self, training_rewards: List[float], eval_rewards: List[float], 
                     distances: List[float], alphas: List[float], critic1_losses: List[float],
                     critic2_losses: List[float], actor_losses: List[float]) -> None:
        """Plot and save training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot training rewards
        plt.figure(figsize=(10, 5))
        plt.plot(training_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(self.results_dir, f'training_rewards_{timestamp}.png'))
        plt.close()
        
        # Plot evaluation rewards
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.num_train_episodes, self.eval_interval), eval_rewards)
        plt.title('Evaluation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(self.results_dir, f'eval_rewards_{timestamp}.png'))
        plt.close()
        
        # Plot distances to target
        plt.figure(figsize=(10, 5))
        plt.plot(distances)
        plt.title('Average Distance to Target')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        plt.savefig(os.path.join(self.results_dir, f'distances_{timestamp}.png'))
        plt.close()

        # Plot alpha
        plt.figure(figsize=(10,5))
        plt.plot(alphas)
        plt.title('Alpha over training')
        plt.xlabel("Training steps")
        plt.ylabel("Alpha")
        plt.savefig(os.path.join(self.results_dir, f"alpha_{timestamp}.png"))
        plt.close()

        # Plot critic 1 losses
        plt.figure(figsize=(10, 5))
        plt.plot(critic1_losses)
        plt.title('Critic 1 Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.results_dir, f'critic1_loss_{timestamp}.png'))
        plt.close()

        # Plot critic 2 losses
        plt.figure(figsize=(10, 5))
        plt.plot(critic2_losses)
        plt.title('Critic 2 Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.results_dir, f'critic2_loss_{timestamp}.png'))
        plt.close()

        # Plot actor losses
        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.results_dir, f'actor_loss_{timestamp}.png'))
        plt.close()

    def tearDown(self):
        """Cleanup after tests."""
        # Close any open plots
        plt.close('all')

if __name__ == '__main__':
    unittest.main()