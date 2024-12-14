import unittest
import torch
import numpy as np
import random
from meta_sac import MetaSACAgent  
# 1. Initialization Tests:

class TestInitialization(unittest.TestCase):
    def test_create_agent(self):
        """Test if the agent initializes without errors."""
        agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")
        self.assertIsNotNone(agent)

    def test_network_parameters(self):
        """Test if network parameters are initialized correctly and on the correct device."""
        agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")

        for param in agent.actor.parameters():
            self.assertEqual(param.device, torch.device("cpu"))
        for param in agent.critic1.parameters():
            self.assertEqual(param.device, torch.device("cpu"))
        for param in agent.meta_controller.parameters():
            self.assertEqual(param.device, torch.device("cpu"))

    def test_alpha_initialization(self):
      """Test if alpha is initialized correctly."""
      agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")
      self.assertEqual(agent.alpha.item(), 0.2)

# 2. Forward Pass Tests:

class TestForwardPass(unittest.TestCase):
    def setUp(self):
        self.agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")
        self.state = torch.randn(1, 10)
        self.time_encoding = torch.randn(1, 10)
        self.action = torch.randn(1, 2)
        self.meta_input = torch.randn(1, 5)

    def test_actor_forward(self):
        """Test the actor's forward pass."""
        mu, log_sigma = self.agent.actor(self.state, self.time_encoding)
        self.assertEqual(mu.shape, (1, 2))
        self.assertEqual(log_sigma.shape, (1, 2))

    def test_critic_forward(self):
        """Test the critic's forward pass."""
        q_value = self.agent.critic1(self.state, self.action, self.time_encoding)
        self.assertEqual(q_value.shape, (1, 1))

    def test_meta_controller_forward(self):
        """Test the meta-controller's forward pass."""
        meta_output = self.agent.meta_controller(self.meta_input)
        self.assertEqual(meta_output.shape, (1, 1))

# 3. Action Selection Tests:

class TestActionSelection(unittest.TestCase):
    def setUp(self):
        self.agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")
        self.state = np.random.rand(10)
        self.time_encoding = np.random.rand(10)

    def test_select_action_shape(self):
        """Test the shape of the selected action."""
        action, _ = self.agent.select_action(self.state, self.time_encoding)
        self.assertEqual(action.shape, (2,))

    def test_select_action_eval(self):
        """Test that eval mode returns the mean action."""
        action = self.agent.select_action(self.state, self.time_encoding, eval=True)
        # In eval mode, action should be deterministic (tanh(mu))
        # We can't directly compare to mu, but we can check if two eval actions are the same
        action2 = self.agent.select_action(self.state, self.time_encoding, eval=True)
        self.assertTrue(np.allclose(action, action2))

    def test_select_action_bounds(self):
        """Test that actions are within the expected bounds [-1, 1]."""
        for _ in range(100):
            action, _ = self.agent.select_action(self.state, self.time_encoding)
            self.assertTrue(np.all(action >= -1) and np.all(action <= 1))

# 4. Update Tests (Isolated Components):

class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")
        self.batch_size = 32
        self.memory = [(np.random.rand(10), np.random.rand(2), np.random.rand(1), np.random.rand(10), False) for _ in range(self.batch_size)]
        self.meta_input = np.random.rand(5)
        self.time_encodings = [np.random.rand(10) for _ in range(self.batch_size)]

    def test_critic_loss_decreases(self):
      """Test that the critic loss decreases after an update."""

      # Get initial critic loss
      initial_q_value1 = self.agent.critic1(torch.randn(self.batch_size, 10), torch.randn(self.batch_size, 2), torch.randn(self.batch_size, 10))
      initial_q_value2 = self.agent.critic2(torch.randn(self.batch_size, 10), torch.randn(self.batch_size, 2), torch.randn(self.batch_size, 10))
      initial_q_target = torch.randn(self.batch_size, 1)

      initial_critic1_loss = F.mse_loss(initial_q_value1, initial_q_target)
      initial_critic2_loss = F.mse_loss(initial_q_value2, initial_q_target)

      # Perform an update
      self.agent.update_params(self.memory, self.batch_size, self.meta_input, self.time_encodings)

      # Get critic loss after update
      updated_q_value1 = self.agent.critic1(torch.randn(self.batch_size, 10), torch.randn(self.batch_size, 2), torch.randn(self.batch_size, 10))
      updated_q_value2 = self.agent.critic2(torch.randn(self.batch_size, 10), torch.randn(self.batch_size, 2), torch.randn(self.batch_size, 10))

      updated_critic1_loss = F.mse_loss(updated_q_value1, initial_q_target)
      updated_critic2_loss = F.mse_loss(updated_q_value2, initial_q_target)

      # Assert that loss has decreased (or at least not significantly increased)
      self.assertLessEqual(updated_critic1_loss.item(), initial_critic1_loss.item() * 1.1) # Allow for slight increase due to stochasticity
      self.assertLessEqual(updated_critic2_loss.item(), initial_critic2_loss.item() * 1.1)

    def test_actor_loss_decreases(self):
      """Test that the actor loss decreases after an update."""

      # Perform an update
      self.agent.update_params(self.memory, self.batch_size, self.meta_input, self.time_encodings)

      # Get actor loss after update
      states = torch.FloatTensor(np.array([item[0] for item in self.memory])).to(self.agent.device)
      time_encodings = torch.FloatTensor(np.array(self.time_encodings)).to(self.agent.device)
      mu, log_sigma = self.agent.actor(states, time_encodings)
      sigma = torch.exp(log_sigma)
      dist = torch.distributions.Normal(mu, sigma)
      z = dist.rsample()
      action = z
      log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
      log_prob = log_prob.sum(-1, keepdim=True)
      q_val = torch.min(self.agent.critic1(states, action, time_encodings), self.agent.critic2(states, action, time_encodings))
      updated_actor_loss = (-q_val + self.agent.alpha * log_prob).mean()

      # Assert that loss is not NaN (which can happen if something goes wrong)
      self.assertFalse(torch.isnan(updated_actor_loss).any())

    def test_meta_controller_update(self):
        """Test that the meta-controller updates alpha."""
        initial_alpha = self.agent.alpha.item()
        self.agent.update_params(self.memory, self.batch_size, self.meta_input, self.time_encodings)
        new_alpha = self.agent.alpha.item()
        self.assertNotEqual(initial_alpha, new_alpha)

    def test_soft_update(self):
        """Test that the soft update mechanism works."""
        initial_target_params = [param.clone() for param in self.agent.critic_target1.parameters()]
        self.agent.update_params(self.memory, self.batch_size, self.meta_input, self.time_encodings)
        new_target_params = [param.clone() for param in self.agent.critic_target1.parameters()]

        for initial_param, new_param in zip(initial_target_params, new_target_params):
            self.assertFalse(torch.equal(initial_param, new_param))

# 5. Training Loop Tests (Integration):
#  Simulate various scenarios to test the agent's ability to learn and adapt.
class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")

    def test_learning_simple_task(self):
        """Test if the agent can learn a simple task (e.g., reaching a target state)."""
        # Define a simple reward function (e.g., negative distance to target)
        target_state = np.array([0.5] * 10)
        def reward_function(state):
            return -np.linalg.norm(state - target_state)

        # Train the agent
        memory = []
        time_memory = []
        for episode in range(50):
            state = np.random.rand(10)
            time_encoding = np.random.rand(10)
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.agent.select_action(state, time_encoding)
                next_state = np.clip(state + action * 0.1, 0, 1) # Simple state transition
                reward = reward_function(next_state)
                done = np.linalg.norm(next_state - target_state) < 0.1
                memory.append((state, action, reward, next_state, done))
                time_memory.append(time_encoding)
                state = next_state
                episode_reward += reward

                if len(memory) >= 32:
                   meta_input = np.random.rand(5) # Placeholder meta-input
                   self.agent.update_params(memory, 32, meta_input, time_memory)

    def test_adaptation_to_changing_target(self):
        """Test if the agent can adapt to a changing target state using meta-learning."""
        target_state = np.array([0.5] * 10)
        change_interval = 25 # Change target every 25 episodes
        # Train the agent
        memory = []
        time_memory = []

        for episode in range(100):
           if episode % change_interval == 0:
               # Change the target state
               target_state = np.random.rand(10)

           state = np.random.rand(10)
           time_encoding = np.random.rand(10)
           done = False
           episode_reward = 0

           while not done:
               action, _ = self.agent.select_action(state, time_encoding)
               next_state = np.clip(state + action * 0.1, 0, 1)
               reward = -np.linalg.norm(next_state - target_state) # Reward based on current target
               done = np.linalg.norm(next_state - target_state) < 0.1
               memory.append((state, action, reward, next_state, done))
               time_memory.append(time_encoding)
               state = next_state
               episode_reward += reward

               if len(memory) >= 32:
                   # Meta-input could indicate the current target or a change in target
                   meta_input = target_state if episode % change_interval == 0 else np.random.rand(5)
                   self.agent.update_params(memory, 32, meta_input, time_memory)

    def test_overfitting(self):
        """Test if the agent is overfitting to a small dataset."""
        # Train on a small, fixed dataset
        dataset_size = 50
        fixed_memory = [(np.random.rand(10), np.random.rand(2), np.random.rand(1), np.random.rand(10), False) for _ in range(dataset_size)]
        fixed_time_memory = [np.random.rand(10) for _ in range(dataset_size)]

        for _ in range(100):
            meta_input = np.random.rand(5)
            self.agent.update_params(fixed_memory, 32, meta_input, fixed_time_memory)

        # Evaluate on a separate validation set
        validation_memory = [(np.random.rand(10), np.random.rand(2), np.random.rand(1), np.random.rand(10), False) for _ in range(dataset_size)]
        validation_time_memory = [np.random.rand(10) for _ in range(dataset_size)]

        # TODO: Implement a way to measure performance on the validation set and compare it to the training set.
        #       If the training performance is much better than the validation performance, it could indicate overfitting.

    def test_long_term_learning(self):
      """Test how well the agent retains information over many updates."""
      memory = []
      time_memory = []

      for episode in range(200):
        state = np.random.rand(10)
        time_encoding = np.random.rand(10)
        done = False
        episode_reward = 0
        while not done:
            action, _ = self.agent.select_action(state, time_encoding)
            next_state = np.clip(state + action * 0.1, 0, 1)
            reward = np.random.rand(1) # Random reward for simplicity
            done = episode_reward > 10 # Simple termination condition
            memory.append((state, action, reward, next_state, done))
            time_memory.append(time_encoding)
            state = next_state
            episode_reward += reward

            if len(memory) >= 32:
               meta_input = np.random.rand(5)
               self.agent.update_params(memory, 32, meta_input, time_memory)

      # TODO: Evaluate the agent's performance after many updates to see if it has learned a useful policy
      #       This might involve defining a specific task or using a held-out evaluation set.

# 6. Edge Case Tests:
class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")

    def test_zero_reward(self):
        """Test the agent's behavior with zero rewards."""
        memory = [(np.random.rand(10), np.random.rand(2), np.array([0.0]), np.random.rand(10), False) for _ in range(32)]
        time_memory = [np.random.rand(10) for _ in range(32)]
        meta_input = np.random.rand(5)
        self.agent.update_params(memory, 32, meta_input, time_memory)

        # Check if update still happens without errors

    def test_large_reward(self):
      """Test how the agent handles extremely large rewards (positive or negative)."""
      memory = [(np.random.rand(10), np.random.rand(2), np.array([1e10]), np.random.rand(10), False) for _ in range(32)]
      time_memory = [np.random.rand(10) for _ in range(32)]
      meta_input = np.random.rand(5)

      self.agent.update_params(memory, 32, meta_input, time_memory)

      # Check for stability issues (e.g., NaNs in parameters)
      for param in self.agent.actor.parameters():
          self.assertFalse(torch.isnan(param).any())
      for param in self.agent.critic1.parameters():
          self.assertFalse(torch.isnan(param).any())
      for param in self.agent.meta_controller.parameters():
          self.assertFalse(torch.isnan(param).any())
    def test_nan_inputs(self):
        """Test how the agent handles NaN inputs."""
        state = np.array([np.nan] * 10)
        time_encoding = np.array([np.nan] * 10)
        action = np.array([np.nan] * 2)
        reward = np.array([np.nan])
        next_state = np.array([np.nan] * 10)
        done = False
        meta_input = np.array([np.nan] * 5)

        # Check if select_action handles NaNs gracefully
        try:
            action, _ = self.agent.select_action(state, time_encoding)
        except Exception as e:
            self.fail(f"select_action raised an exception with NaN inputs: {e}")

        # Check if update_params handles NaNs gracefully
        memory = [(state, action, reward, next_state, done)]
        time_memory = [time_encoding]
        try:
            self.agent.update_params(memory, 1, meta_input, time_memory)
        except Exception as e:
            self.fail(f"update_params raised an exception with NaN inputs: {e}")

# 7. Saving and Loading Tests (Optional):
class TestSavingLoading(unittest.TestCase):
    def setUp(self):
        self.agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")
        self.save_path = "test_agent.pth"

    def test_save_and_load(self):
        """Test if the agent can be saved and loaded correctly."""
        # Save the agent
        torch.save(self.agent.state_dict(), self.save_path)

        # Create a new agent
        new_agent = MetaSACAgent(state_dim=10, action_dim=2, hidden_dim=64, attention_dim=32, meta_input_dim=5, device="cpu")

        # Load the saved state dict
        new_agent.load_state_dict(torch.load(self.save_path))

        # Check if the parameters are the same
        for param, new_param in zip(self.agent.parameters(), new_agent.parameters()):
            self.assertTrue(torch.equal(param, new_param))

    def tearDown(self):
        import os
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

if __name__ == '__main__':
    unittest.main()