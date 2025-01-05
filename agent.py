# File: agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from networks import (
    MetaSACActorEnhanced, 
    MetaSACCriticEnhanced, 
    MetaController, 
    PolicyDistillerEnsemble, 
    MarketModeClassifier, 
    HighLevelPolicy,
    MarketGraphModel
)
from replay_buffer import ReplayBuffer
from reward import calculate_reward

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class SACLoss(nn.Module):
    """Soft Actor-Critic Loss Function."""
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, q_pred: torch.Tensor, q_target: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, log_prob: torch.Tensor):
        # Critic Loss: MSE between predicted Q and target Q
        critic_loss = F.mse_loss(q_pred, q_target)

        # Actor Loss: minimize alpha * log_prob - Q
        actor_loss = (self.alpha * log_prob - q_pred).mean()

        # Entropy Loss: maximize entropy (minimize negative entropy)
        entropy_loss = -self.alpha * log_prob.mean()

        total_loss = critic_loss + actor_loss + entropy_loss
        return total_loss, critic_loss, actor_loss, entropy_loss

class TradingLoss(nn.Module):
    """Specialized loss function tailored for trading tasks."""
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted returns or actions
            targets: Actual returns or actions
        Returns:
            Computed loss
        """
        # Example: Mean Squared Error with additional penalty for volatility
        mse_loss = F.mse_loss(predictions, targets)
        volatility_penalty = torch.mean(torch.var(predictions, dim=1))
        loss = mse_loss + 0.1 * volatility_penalty
        return loss

class MetaSACAgent(nn.Module):
    """Meta Soft Actor-Critic (SAC) agent with advanced enhancements."""
    def __init__(self, config: EnvironmentConfig, env: HistoricalEnvironment):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.env = env

        # Graph Neural Network to model relationships between market entities
        self.market_gnn = MarketGraphModel(config).to(self.device)

        # Actor & Critics with Enhancements (Recurrent Layers, Transformer Layers, Residual Connections, etc.)
        self.actor = MetaSACActorEnhanced(config).to(self.device)
        self.critic1 = MetaSACCriticEnhanced(config).to(self.device)
        self.critic2 = MetaSACCriticEnhanced(config).to(self.device)
        self.critic_target1 = MetaSACCriticEnhanced(config).to(self.device)
        self.critic_target2 = MetaSACCriticEnhanced(config).to(self.device)

        # Initialize target networks
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # Meta-Controller for dynamic hyperparameter adjustment
        self.meta_controller = MetaController(config).to(self.device)
        
        # Policy Distiller with Ensemble of Specialist Policies
        self.specialist_policies = [MetaSACActorEnhanced(config).to(self.device) for _ in range(config.ensemble_size)]
        self.policy_distiller = PolicyDistillerEnsemble(self.specialist_policies, config).to(self.device)

        # Market Mode Classifier & High-Level Policy for hierarchical decision-making
        self.market_mode_classifier = MarketModeClassifier(
            input_dim=config.state_dim, 
            hidden_dim=config.hidden_dim, 
            output_dim=config.num_market_modes
        ).to(self.device)
        self.high_level_policy = HighLevelPolicy(config.state_dim, config.hidden_dim).to(self.device)

        # Optimizers with Advanced Techniques (AdamW)
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.critic1_optimizer = AdamW(self.critic1.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.critic2_optimizer = AdamW(self.critic2.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.meta_optimizer = AdamW(self.meta_controller.parameters(), lr=config.meta_lr, weight_decay=config.weight_decay)
        self.distiller_optimizer = AdamW(self.policy_distiller.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.market_mode_optimizer = AdamW(self.market_mode_classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.high_level_optimizer = AdamW(self.high_level_policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Learning Rate Schedulers
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode="min", factor=0.5, patience=5)
        self.critic1_scheduler = ReduceLROnPlateau(self.critic1_optimizer, mode="min", factor=0.5, patience=5)
        self.critic2_scheduler = ReduceLROnPlateau(self.critic2_optimizer, mode="min", factor=0.5, patience=5)
        self.high_level_scheduler = ReduceLROnPlateau(self.high_level_optimizer, mode="min", factor=0.5, patience=5)
        self.meta_scheduler = ReduceLROnPlateau(self.meta_optimizer, mode="min", factor=0.5, patience=5)
        self.distiller_scheduler = ReduceLROnPlateau(self.distiller_optimizer, mode="min", factor=0.5, patience=5)

        # Entropy coefficient alpha with automatic adjustment
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=torch.float32, device=self.device))
        self.target_entropy = -float(config.action_dim) * config.target_entropy_factor

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)

        # Logging
        self.writer = SummaryWriter()
        self.train_steps = 0

        # Move all modules to device
        self.to(self.device)

    def select_action(self, state: np.ndarray, time_step: int, edge_index: torch.Tensor, graph_node_features: torch.Tensor, eval=False) -> np.ndarray:
        """
        Selects action using either the actor or policy distiller.

        Args:
            state (np.ndarray): Current state.
            time_step (int): Current time step.
            edge_index (torch.Tensor): Graph edges.
            graph_node_features (torch.Tensor): Graph node features.
            eval (bool): Whether to select action in evaluation mode.

        Returns:
            np.ndarray: Selected action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, seq_length, state_dim)
        time_tensor = torch.tensor([time_step], dtype=torch.float32).to(self.device)  # (1,)
        graph_node_features = graph_node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        # Process graph data
        graph_embedding = self.market_gnn(graph_node_features, edge_index)  # (num_nodes, hidden_dim)
        graph_embedding = torch.mean(graph_embedding, dim=0, keepdim=True).repeat(state_tensor.size(0), 1, 1)  # (1, 1, hidden_dim)
        state_tensor = torch.cat([state_tensor, graph_embedding], dim=1)  # (1, seq_length +1, hidden_dim)

        if eval:
            with torch.no_grad():
                mu, log_sigma = self.actor(state_tensor, time_tensor)
                action = mu
        else:
            with torch.no_grad():
                mu, log_sigma = self.policy_distiller(state_tensor, time_tensor)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                action = torch.tanh(z)
        
        action = action.clamp(-1, 1)
        return action.cpu().numpy().squeeze(0)

    def compute_q_targets(self, 
                          rewards: torch.Tensor, 
                          next_states: torch.Tensor, 
                          time_steps: torch.Tensor, 
                          dones: torch.Tensor,
                          r_scaling: torch.Tensor,
                          market_mode_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the target Q-values using the target critic networks.

        Args:
            rewards (torch.Tensor): Rewards from the environment.
            next_states (torch.Tensor): Next states.
            time_steps (torch.Tensor): Time steps.
            dones (torch.Tensor): Done flags.
            r_scaling (torch.Tensor): Reward scaling factors.
            market_mode_probs (torch.Tensor): Market mode probabilities.

        Returns:
            torch.Tensor: Target Q-values.
        """
        with torch.no_grad():
            next_mu, next_log_sigma = self.actor(next_states, time_steps)
            sigma = torch.exp(next_log_sigma)
            dist = torch.distributions.Normal(next_mu, sigma)
            next_actions = torch.tanh(dist.rsample())
            log_probs = dist.log_prob(next_actions) - torch.log(1 - next_actions.pow(2) + self.config.epsilon)
            next_log_probs = log_probs.sum(-1, keepdim=True)

            q_target1 = self.critic_target1(next_states, next_actions, time_steps)
            q_target2 = self.critic_target2(next_states, next_actions, time_steps)
            q_target = torch.min(q_target1, q_target2)

            scaled_rewards = (
                r_scaling[:, 0:1]*rewards*market_mode_probs[:, 0:1] +
                r_scaling[:, 1:2]*rewards*market_mode_probs[:, 1:2] +
                r_scaling[:, 2:3]*rewards*market_mode_probs[:, 2:3]
            )
            y = scaled_rewards + (1.0 - dones)*self.config.gamma*(q_target - self.alpha*next_log_probs)
            return y

    def update_critics(self, states, actions, time_steps, q_targets) -> Tuple[float, float]:
        """
        Updates the critic networks using Mean Squared Error loss.

        Args:
            states (torch.Tensor): Current states.
            actions (torch.Tensor): Actions taken.
            time_steps (torch.Tensor): Time steps.
            q_targets (torch.Tensor): Target Q-values.

        Returns:
            Tuple[float, float]: Critic1 and Critic2 losses.
        """
        q1 = self.critic1(states, actions, time_steps)
        q2 = self.critic2(states, actions, time_steps)
        critic1_loss = F.mse_loss(q1, q_targets)
        critic2_loss = F.mse_loss(q2, q_targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.max_grad_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.max_grad_norm)
        self.critic2_optimizer.step()

        return critic1_loss.item(), critic2_loss.item()

    def update_actor(self, states, time_steps) -> float:
        """
        Updates the actor network to maximize expected return.

        Args:
            states (torch.Tensor): Current states.
            time_steps (torch.Tensor): Time steps.

        Returns:
            float: Actor loss.
        """
        mu, log_sigma = self.policy_distiller(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)

        q_val1 = self.critic1(states, actions, time_steps)
        q_val2 = self.critic2(states, actions, time_steps)
        q_val = torch.min(q_val1, q_val2)
        actor_loss = (self.alpha * log_probs - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_distiller(self, states, time_steps) -> float:
        """
        Updates the policy distiller network.

        Args:
            states (torch.Tensor): Current states.
            time_steps (torch.Tensor): Time steps.

        Returns:
            float: Distiller loss.
        """
        mu, log_sigma = self.policy_distiller(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)

        q_val1 = self.critic1(states, actions, time_steps)
        q_val2 = self.critic2(states, actions, time_steps)
        q_val = torch.min(q_val1, q_val2)
        distiller_loss = (self.alpha * log_probs - q_val).mean()

        self.distiller_optimizer.zero_grad()
        distiller_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_distiller.parameters(), self.config.max_grad_norm)
        self.distiller_optimizer.step()

        return distiller_loss.item()

    def update_market_mode_classifier(self, states, market_modes) -> float:
        """
        Updates the market mode classifier network.

        Args:
            states (torch.Tensor): Current states.
            market_modes (torch.Tensor): Market mode labels.

        Returns:
            float: Market mode classification loss.
        """
        preds = self.market_mode_classifier(states)
        loss = F.cross_entropy(preds, market_modes)
        self.market_mode_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.market_mode_classifier.parameters(), self.config.max_grad_norm)
        self.market_mode_optimizer.step()
        return loss.item()

    def update_meta_controller(self, meta_input_tensor, log_probs, rewards) -> Tuple[float, torch.Tensor]:
        """
        Updates the meta-controller network.

        Args:
            meta_input_tensor (torch.Tensor): Inputs for the meta-controller.
            log_probs (torch.Tensor): Log probabilities from the policy.
            rewards (torch.Tensor): Rewards from the environment.

        Returns:
            Tuple[float, torch.Tensor]: Meta-controller loss and reward scaling factors.
        """
        mean_r = torch.mean(rewards, dim=0, keepdim=True)
        var_r = torch.var(rewards, dim=0, keepdim=True)
        reward_stats = torch.cat([mean_r, var_r], dim=-1).to(self.device)

        batch_size = meta_input_tensor.size(0)
        reward_stats = reward_stats.repeat(batch_size, 1)

        out = self.meta_controller(meta_input_tensor, reward_stats)
        # Assuming out contains scaling factors at specific indices
        # Adjust indices based on actual implementation
        r_scaling = out[:, :self.config.num_hyperparams + 3]

        # Example meta-controller loss (placeholder)
        meta_loss = F.mse_loss(out, torch.ones_like(out))  # Placeholder loss

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), self.config.max_grad_norm)
        self.meta_optimizer.step()

        return meta_loss.item(), r_scaling

    def update_high_level_policy(self, states, advantages) -> float:
        """
        Updates the high-level policy network.

        Args:
            states (torch.Tensor): Current states.
            advantages (torch.Tensor): Advantage estimates.

        Returns:
            float: High-level policy loss.
        """
        probs = self.high_level_policy(states)
        log_probs = torch.log(probs + 1e-10)
        policy_loss = -(log_probs * advantages).mean()

        self.high_level_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), self.config.max_grad_norm)
        self.high_level_optimizer.step()
        return policy_loss.item()

    def soft_update(self, target_net, source_net) -> None:
        """
        Performs a soft update of target network parameters.

        Args:
            target_net (nn.Module): Target network.
            source_net (nn.Module): Source network.
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - self.config.tau) + source_param.data*self.config.tau)

    def perform_best_of_n_search(
        self,
        initial_state: torch.Tensor,
        num_samples: int,
        search_depth: int,
        time_memory: List[int],
        exploration_noise_std: float = 0.0
    ) -> List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, int]]]:
        """
        Samples trajectories using Best-of-N search.

        Args:
            initial_state (torch.Tensor): Initial state tensor.
            num_samples (int): Number of samples.
            search_depth (int): Depth of search.
            time_memory (List[int]): Historical time steps.
            exploration_noise_std (float): Standard deviation for exploration noise.

        Returns:
            List[List[Tuple]]: List of trajectories.
        """
        all_trajectories = []
        for _ in range(num_samples):
            state = initial_state.clone()
            trajectory = []
            time_idx = time_memory[-1]
            for _step in range(search_depth):
                with torch.no_grad():
                    mu, log_sigma = self.actor(state.unsqueeze(0), torch.tensor([time_idx], device=self.device))
                    sigma = torch.exp(log_sigma)
                    dist = torch.distributions.Normal(mu, sigma)
                    z = dist.rsample()
                    action = torch.tanh(z).squeeze(0)
                    if exploration_noise_std > 0.0:
                        noise = torch.randn_like(action) * exploration_noise_std
                        action = torch.clamp(action + noise, -1.0, 1.0)

                action_np = action.cpu().numpy()
                next_state_np, reward, done, _info = self.env.step(action_np, time_idx)
                next_state = torch.FloatTensor(next_state_np).to(self.device)
                trajectory.append((state.cpu().numpy(), action_np, reward, next_state_np, done, time_idx))

                state = next_state
                time_idx += 1
                if done:
                    break
            all_trajectories.append(trajectory)
        return all_trajectories

    def perform_beam_search(
        self,
        initial_state: torch.Tensor,
        beam_width: int,
        search_depth: int,
        time_memory: List[int],
        exploration_noise_std: float = 0.0
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, int]]:
        """
        Performs Beam Search to find the best trajectory.

        Args:
            initial_state (torch.Tensor): Initial state tensor.
            beam_width (int): Beam width.
            search_depth (int): Depth of search.
            time_memory (List[int]): Historical time steps.
            exploration_noise_std (float): Standard deviation for exploration noise.

        Returns:
            List[Tuple]: Best trajectory.
        """
        beam = [(initial_state.clone(), [], 0.0, time_memory[-1])]
        final_trajs = []

        for _ in range(search_depth):
            new_beam = []
            for current_state, partial_traj, cum_reward, t_step in beam:
                if partial_traj and partial_traj[-1][4] is True:
                    final_trajs.append((current_state, partial_traj, cum_reward, t_step))
                    continue
                with torch.no_grad():
                    mu, log_sigma = self.actor(current_state.unsqueeze(0), torch.tensor([t_step], device=self.device))
                    sigma = torch.exp(log_sigma)
                    dist = torch.distributions.Normal(mu, sigma)
                    candidate_actions = []
                    for _i in range(beam_width):
                        z = dist.rsample()
                        a = torch.tanh(z).squeeze(0)
                        if exploration_noise_std > 0.0:
                            noise = torch.randn_like(a) * exploration_noise_std
                            a = torch.clamp(a + noise, -1.0, 1.0)
                        candidate_actions.append(a)

                for a_tensor in candidate_actions:
                    a_np = a_tensor.cpu().numpy()
                    ns_np, r, d, _info = self.env.step(a_np, t_step)
                    ns = torch.FloatTensor(ns_np).to(self.device)
                    new_traj = partial_traj.copy()
                    new_traj.append((current_state.cpu().numpy(), a_np, r, ns_np, d, t_step))
                    new_beam.append((ns, new_traj, cum_reward + r, t_step + 1))

            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_width]

        final_trajs.extend(beam)
        if final_trajs:
            best_traj = max(final_trajs, key=lambda x: x[2])[1]
            return best_traj
        else:
            return []

    def update_params_with_training_time_search(
        self,
        replay_buffer: ReplayBuffer,
        meta_input: np.ndarray,
        time_memory: List[int],
        update_steps: int =1,
        search_algorithm: str = "best-of-n",
        num_samples: int = 4,
        beam_width: int = 3,
        search_depth: int = 5,
        use_d_search: bool = False,
        exploration_noise_std_fn: Callable[[int], float] = lambda step: 0.0
    ) -> Dict[str, float]:
        """
        Updates parameters using search-generated data.

        Args:
            replay_buffer (ReplayBuffer): Experience replay buffer.
            meta_input (np.ndarray): Input for the meta-controller.
            time_memory (List[int]): Historical time steps.
            update_steps (int): Number of update iterations.
            search_algorithm (str): Search algorithm type ("best-of-n" or "beam-search").
            num_samples (int): Number of samples for best-of-n search.
            beam_width (int): Beam width for beam search.
            search_depth (int): Depth of search.
            use_d_search (bool): Whether to use depth-based search.
            exploration_noise_std_fn (Callable[[int], float]): Function to determine exploration noise.

        Returns:
            Dict[str, float]: Dictionary of loss values for logging.
        """
        final_info = {}
        for _ in range(update_steps):
            if len(replay_buffer) < self.config.batch_size:
                logger.warning("Not enough samples in replay buffer.")
                continue

            batch = replay_buffer.sample(self.config.batch_size)
            sampled_states = torch.FloatTensor([b[0] for b in batch]).to(self.device)

            search_trajectories = []
            for state in sampled_states:
                exploration_noise_std = exploration_noise_std_fn(self.train_steps)
                if search_algorithm == "best-of-n":
                    all_trajectories = self.perform_best_of_n_search(
                        state, num_samples, search_depth, time_memory, exploration_noise_std
                    )
                    if use_d_search:
                        for traj in all_trajectories:
                            search_trajectories.extend(traj)
                    else:
                        if all_trajectories:
                            best_traj = max(all_trajectories, key=lambda t: sum(step[2] for step in t))
                            search_trajectories.extend(best_traj)
                elif search_algorithm == "beam-search":
                    best_traj = self.perform_beam_search(
                        state, beam_width, search_depth, time_memory, exploration_noise_std
                    )
                    search_trajectories.extend(best_traj)
                else:
                    logger.error(f"Unknown search algorithm: {search_algorithm}")
                    continue

            if not search_trajectories:
                logger.warning("No trajectories found during search.")
                continue

            # Extract search trajectory data
            states_search = torch.FloatTensor([traj[0] for traj in search_trajectories]).to(self.device)
            actions_search = torch.FloatTensor([traj[1] for traj in search_trajectories]).to(self.device)
            rewards_search = torch.FloatTensor([traj[2] for traj in search_trajectories]).unsqueeze(1).to(self.device)
            next_states_search = torch.FloatTensor([traj[3] for traj in search_trajectories]).to(self.device)
            dones_search = torch.FloatTensor([traj[4] for traj in search_trajectories]).unsqueeze(1).to(self.device)
            time_steps_search = torch.FloatTensor([traj[5] for traj in search_trajectories]).to(self.device)

            # Market mode classification (using existing classifier)
            random_modes = torch.randint(0, self.config.num_market_modes, (states_search.shape[0],)).to(self.device)
            market_mode_loss = self.update_market_mode_classifier(states_search, random_modes)

            # Meta-Controller update
            meta_input_tensor = torch.FloatTensor(meta_input).to(self.device)
            with torch.no_grad():
                mu, log_sigma = self.actor(states_search, time_steps_search)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                actions_ = torch.tanh(z)
                log_probs = dist.log_prob(z) - torch.log(1 - actions_.pow(2) + self.config.epsilon)
                log_probs = log_probs.sum(-1, keepdim=True)

            meta_loss_val, r_scaling = self.update_meta_controller(meta_input_tensor, log_probs, rewards_search)
            market_mode_probs = self.market_mode_classifier(states_search)

            # Compute Q targets
            q_targets = self.compute_q_targets(rewards_search, next_states_search, time_steps_search, dones_search, r_scaling, market_mode_probs)

            # Update Critic Networks
            critic1_loss, critic2_loss = self.update_critics(states_search, actions_search, time_steps_search, q_targets)

            # Update Actor Network
            actor_loss_val = self.update_actor(states_search, time_steps_search)

            # Update Policy Distiller
            distiller_loss_val = self.update_distiller(states_search, time_steps_search)

            # Update High-Level Policy
            advantages = q_targets - torch.mean(q_targets, dim=0, keepdim=True)
            high_level_loss_val = self.update_high_level_policy(states_search, advantages)

            # Soft-update target critics
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            # Update Learning Rate Schedulers
            self.actor_scheduler.step(actor_loss_val)
            self.critic1_scheduler.step(critic1_loss)
            self.critic2_scheduler.step(critic2_loss)
            self.high_level_scheduler.step(high_level_loss_val)
            self.meta_scheduler.step(meta_loss_val)
            self.distiller_scheduler.step(distiller_loss_val)

            self.train_steps += 1

            # Logging
            self.writer.add_scalar("Loss/actor", actor_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/critic1", critic1_loss, self.train_steps)
            self.writer.add_scalar("Loss/critic2", critic2_loss, self.train_steps)
            self.writer.add_scalar("Loss/meta", meta_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/distiller", distiller_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/market_mode", market_mode_loss, self.train_steps)
            self.writer.add_scalar("Loss/high_level_policy", high_level_loss_val, self.train_steps)
            self.writer.add_scalar("Params/alpha", self.alpha.item(), self.train_steps)

            final_info = {
                "actor_loss": actor_loss_val,
                "critic1_loss": critic1_loss,
                "critic2_loss": critic2_loss,
                "meta_loss": meta_loss_val,
                "distiller_loss": distiller_loss_val,
                "market_mode_loss": market_mode_loss,
                "high_level_loss": high_level_loss_val,
                "alpha": self.alpha.item()
            }

        return final_info

    def update_params(self, replay_buffer: ReplayBuffer, meta_input: np.ndarray, time_memory: List[int], update_steps: int =1) -> Dict[str, float]:
        """
        Updates parameters without using search.

        Args:
            replay_buffer (ReplayBuffer): Experience replay buffer.
            meta_input (np.ndarray): Input for the meta-controller.
            time_memory (List[int]): Historical time steps.
            update_steps (int): Number of update iterations.

        Returns:
            Dict[str, float]: Dictionary of loss values for logging.
        """
        final_info = {}
        for _ in range(update_steps):
            if len(replay_buffer) < self.config.batch_size:
                logger.warning("Not enough samples in replay buffer.")
                continue

            batch = replay_buffer.sample(self.config.batch_size)
            states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
            actions = torch.FloatTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
            dones = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1).to(self.device)
            time_steps = torch.FloatTensor([b[5] for b in batch]).to(self.device)

            # Market mode classification (using existing classifier)
            random_modes = torch.randint(0, self.config.num_market_modes, (states.shape[0],)).to(self.device)
            market_mode_loss = self.update_market_mode_classifier(states, random_modes)

            # Meta-Controller update
            meta_input_tensor = torch.FloatTensor(meta_input).to(self.device)
            with torch.no_grad():
                mu, log_sigma = self.actor(states, time_steps)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                actions_ = torch.tanh(z)
                log_probs = dist.log_prob(z) - torch.log(1 - actions_.pow(2) + self.config.epsilon)
                log_probs = log_probs.sum(-1, keepdim=True)

            meta_loss_val, r_scaling = self.update_meta_controller(meta_input_tensor, log_probs, rewards)
            market_mode_probs = self.market_mode_classifier(states)

            # Compute Q targets
            q_targets = self.compute_q_targets(rewards, next_states, time_steps, dones, r_scaling, market_mode_probs)

            # Update Critic Networks
            critic1_loss, critic2_loss = self.update_critics(states, actions, time_steps, q_targets)

            # Update Actor Network
            actor_loss_val = self.update_actor(states, time_steps)

            # Update Policy Distiller
            distiller_loss_val = self.update_distiller(states, time_steps)

            # Update High-Level Policy
            advantages = q_targets - torch.mean(q_targets, dim=0, keepdim=True)
            high_level_loss_val = self.update_high_level_policy(states, advantages)

            # Soft-update target critics
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            # Update Learning Rate Schedulers
            self.actor_scheduler.step(actor_loss_val)
            self.critic1_scheduler.step(critic1_loss)
            self.critic2_scheduler.step(critic2_loss)
            self.high_level_scheduler.step(high_level_loss_val)
            self.meta_scheduler.step(meta_loss_val)
            self.distiller_scheduler.step(distiller_loss_val)

            self.train_steps += 1

            # Logging
            self.writer.add_scalar("Loss/actor", actor_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/critic1", critic1_loss, self.train_steps)
            self.writer.add_scalar("Loss/critic2", critic2_loss, self.train_steps)
            self.writer.add_scalar("Loss/meta", meta_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/distiller", distiller_loss_val, self.train_steps)
            self.writer.add_scalar("Loss/market_mode", market_mode_loss, self.train_steps)
            self.writer.add_scalar("Loss/high_level_policy", high_level_loss_val, self.train_steps)
            self.writer.add_scalar("Params/alpha", self.alpha.item(), self.train_steps)

            final_info = {
                "actor_loss": actor_loss_val,
                "critic1_loss": critic1_loss,
                "critic2_loss": critic2_loss,
                "meta_loss": meta_loss_val,
                "distiller_loss": distiller_loss_val,
                "market_mode_loss": market_mode_loss,
                "high_level_loss": high_level_loss_val,
                "alpha": self.alpha.item()
            }

        return final_info

    def save(self, path: str) -> None:
        """
        Saves the agent's state dictionaries.

        Args:
            path (str): Path to save the model.
        """
        try:
            torch.save({
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic_target1": self.critic_target1.state_dict(),
                "critic_target2": self.critic_target2.state_dict(),
                "meta_controller": self.meta_controller.state_dict(),
                "policy_distiller": self.policy_distiller.state_dict(),
                "market_mode_classifier": self.market_mode_classifier.state_dict(),
                "high_level_policy": self.high_level_policy.state_dict(),
                "alpha": self.alpha.detach().cpu().numpy(),
                "train_steps": self.train_steps
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load(self, path: str) -> None:
        """
        Loads the agent's state dictionaries.

        Args:
            path (str): Path from which to load the model.
        """
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic1.load_state_dict(ckpt["critic1"])
            self.critic2.load_state_dict(ckpt["critic2"])
            self.critic_target1.load_state_dict(ckpt["critic_target1"])
            self.critic_target2.load_state_dict(ckpt["critic_target2"])
            self.meta_controller.load_state_dict(ckpt["meta_controller"])
            self.policy_distiller.load_state_dict(ckpt["policy_distiller"])
            self.market_mode_classifier.load_state_dict(ckpt["market_mode_classifier"])
            self.high_level_policy.load_state_dict(ckpt["high_level_policy"])
            self.alpha.data.copy_(torch.tensor(ckpt["alpha"], dtype=torch.float32, device=self.device))
            self.train_steps = ckpt["train_steps"]
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
