# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Callable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import EnvironmentConfig
from env.environment import HistoricalEnvironment
from networks import (
    MetaSACActor, 
    MetaSACCritic, 
    MetaController, 
    PolicyDistiller, 
    MarketModeClassifier, 
    HighLevelPolicy
)
from replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

class MetaSACAgent(nn.Module):
    """Meta Soft Actor-Critic (SAC) agent with training-time search."""
    def __init__(self, config: EnvironmentConfig, env: HistoricalEnvironment):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.env = env

        # Actor & Critics
        self.actor = MetaSACActor(config).to(self.device)
        self.critic1 = MetaSACCritic(config).to(self.device)
        self.critic2 = MetaSACCritic(config).to(self.device)
        self.critic_target1 = MetaSACCritic(config).to(self.device)
        self.critic_target2 = MetaSACCritic(config).to(self.device)

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # Meta-Controller
        self.meta_controller = MetaController(config).to(self.device)
        
        # Policy Distiller & Specialists
        self.specialist_policies = [MetaSACActor(config).to(self.device) for _ in range(2)]
        self.policy_distiller = PolicyDistiller(self.specialist_policies).to(self.device)

        # Market Mode Classifier & High-Level Policy
        self.market_mode_classifier = MarketModeClassifier(
            config.state_dim, config.hidden_dim, config.num_market_modes
        ).to(self.device)
        self.high_level_policy = HighLevelPolicy(config.state_dim, config.hidden_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=config.lr)
        self.meta_optimizer = Adam(self.meta_controller.parameters(), lr=config.meta_lr)
        self.distiller_optimizer = Adam(self.policy_distiller.parameters(), lr=config.lr)
        self.market_mode_optimizer = Adam(self.market_mode_classifier.parameters(), lr=config.lr)
        self.high_level_optimizer = Adam(self.high_level_policy.parameters(), lr=config.lr)

        # LR Schedulers
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode="min", factor=0.5, patience=5)
        self.critic1_scheduler = ReduceLROnPlateau(self.critic1_optimizer, mode="min", factor=0.5, patience=5)
        self.critic2_scheduler = ReduceLROnPlateau(self.critic2_optimizer, mode="min", factor=0.5, patience=5)
        self.high_level_scheduler = ReduceLROnPlateau(self.high_level_optimizer, mode="min", factor=0.5, patience=5)
        self.meta_scheduler = ReduceLROnPlateau(self.meta_optimizer, mode="min", factor=0.5, patience=5)
        self.distiller_scheduler = ReduceLROnPlateau(self.distiller_optimizer, mode="min", factor=0.5, patience=5)

        # Entropy coefficient alpha
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=torch.float32, device=self.device))
        self.target_entropy = -float(config.action_dim) * config.target_entropy_factor

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)

        # Logging
        self.writer = SummaryWriter()
        self.train_steps = 0

    def select_action(self, state: np.ndarray, time_step: int, eval=False) -> np.ndarray:
        """Selects action with policy or policy distiller (training)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        time_tensor = torch.tensor([time_step], dtype=torch.float32).to(self.device)
        if eval:
            with torch.no_grad():
                mu, _log_sigma = self.actor(state_tensor, time_tensor)
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
        """Computes Q-targets with reward scaling."""
        with torch.no_grad():
            mu, log_sigma = self.actor(next_states, time_steps)
            sigma = torch.exp(log_sigma)
            dist = torch.distributions.Normal(mu, sigma)
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
        """Updates critic networks."""
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
        """Updates actor network."""
        mu, log_sigma = self.actor(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)

        q_val = torch.min(
            self.critic1(states, actions, time_steps),
            self.critic2(states, actions, time_steps)
        )
        actor_loss = (self.alpha * log_probs - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_distiller(self, states, time_steps) -> float:
        """Updates policy distiller."""
        mu, log_sigma = self.policy_distiller(states, time_steps)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        actions = torch.tanh(z)
        log_probs = dist.log_prob(z) - torch.log(1 - actions.pow(2) + self.config.epsilon)
        log_probs = log_probs.sum(-1, keepdim=True)

        q_val = torch.min(
            self.critic1(states, actions, time_steps),
            self.critic2(states, actions, time_steps)
        )
        distiller_loss = (self.alpha * log_probs - q_val).mean()

        self.distiller_optimizer.zero_grad()
        distiller_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_distiller.parameters(), self.config.max_grad_norm)
        self.distiller_optimizer.step()

        return distiller_loss.item()

    def update_market_mode_classifier(self, states, market_modes) -> float:
        """Updates market mode classifier."""
        preds = self.market_mode_classifier(states)
        loss = F.cross_entropy(preds, market_modes)
        self.market_mode_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.market_mode_classifier.parameters(), self.config.max_grad_norm)
        self.market_mode_optimizer.step()
        return loss.item()

    def update_meta_controller(self, meta_input_tensor, log_probs, rewards) -> Tuple[float, torch.Tensor]:
        """Updates meta-controller."""
        mean_r = torch.mean(rewards, dim=0, keepdim=True)
        var_r = torch.var(rewards, dim=0, keepdim=True)
        reward_stats = torch.cat([mean_r, var_r], dim=-1).to(self.device)
        batch_size = meta_input_tensor.size(0)
        reward_stats = reward_stats.repeat(batch_size, 1)

        out = self.meta_controller(meta_input_tensor, reward_stats)
        learning_rate_alpha = out[2]

        alpha_target_loss = -(learning_rate_alpha - torch.log(self.alpha)) * (log_probs + self.target_entropy).detach().mean()

        self.meta_optimizer.zero_grad()
        alpha_target_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), self.config.max_grad_norm)
        self.meta_optimizer.step()

        r_scaling = torch.stack([out[5], out[6], out[7]], dim=-1)
        return alpha_target_loss.item(), r_scaling

    def update_high_level_policy(self, states, advantages) -> float:
        """Updates high-level policy."""
        probs = self.high_level_policy(states)
        log_probs = torch.log(probs + 1e-10)
        policy_loss = -(log_probs * advantages).mean()

        self.high_level_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), self.config.max_grad_norm)
        self.high_level_optimizer.step()
        return policy_loss.item()

    def soft_update(self, target_net, source_net) -> None:
        """Soft updates target network parameters."""
        for tp, sp in zip(target_net.parameters(), source_net.parameters()):
            tp.data.copy_(tp.data*(1-self.config.tau) + sp.data*self.config.tau)

    def update_params_with_training_time_search(
        self,
        replay_buffer: ReplayBuffer,
        meta_input: np.ndarray,
        time_memory: List[int],
        update_steps: int = 1,
        search_algorithm: str = "best-of-n",
        num_samples: int = 4,
        beam_width: int = 3,
        search_depth: int = 5,
        use_d_search: bool = False,
        exploration_noise_std_fn: Callable[[int], float] = lambda step: 0.0
    ) -> Dict[str, float]:
        """Updates parameters using search-generated data."""
        final_info = {}
        for _ in range(update_steps):
            if len(replay_buffer) < self.config.batch_size:
                return final_info

            batch, _, _ = replay_buffer.sample(self.config.batch_size)
            sampled_states = torch.FloatTensor(np.stack([b[0] for b in batch])).to(self.device)

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
                        best_traj = max(all_trajectories, key=lambda t: sum(step[2] for step in t))
                        search_trajectories.extend(best_traj)
                elif search_algorithm == "beam-search":
                    best_traj = self.perform_beam_search(
                        state, beam_width, search_depth, time_memory, exploration_noise_std
                    )
                    search_trajectories.extend(best_traj)
                else:
                    raise ValueError(f"Unknown search algorithm: {search_algorithm}")

            if not search_trajectories:
                continue

            states_search = torch.FloatTensor([st[0] for st in search_trajectories]).to(self.device)
            actions_search = torch.FloatTensor([st[1] for st in search_trajectories]).to(self.device)
            rewards_search = torch.FloatTensor([st[2] for st in search_trajectories]).unsqueeze(1).to(self.device)
            next_states_search = torch.FloatTensor([st[3] for st in search_trajectories]).to(self.device)
            dones_search = torch.FloatTensor([st[4] for st in search_trajectories]).unsqueeze(1).to(self.device)
            time_steps_search = torch.FloatTensor([st[5] for st in search_trajectories]).to(self.device)

            # Market mode classification (random labels for demonstration)
            random_modes = torch.randint(0, self.config.num_market_modes, (states_search.size(0),)).to(self.device)
            market_mode_loss = self.update_market_mode_classifier(states_search, random_modes)

            # Meta-Controller
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
            q_targets = self.compute_q_targets(rewards_search, next_states_search, time_steps_search, dones_search, r_scaling, market_mode_probs)

            # Critics
            critic1_loss, critic2_loss = self.update_critics(states_search, actions_search, time_steps_search, q_targets)

            # Actor & Distiller
            actor_loss_val = self.update_actor(states_search, time_steps_search)
            distiller_loss_val = self.update_distiller(states_search, time_steps_search)

            # High-level policy
            advantages = q_targets - torch.mean(q_targets, dim=0, keepdim=True)
            high_level_loss_val = self.update_high_level_policy(states_search, advantages)

            # Soft-update critics
            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            # LR schedulers
            self.actor_scheduler.step(actor_loss_val)
            self.critic1_scheduler.step(critic1_loss)
            self.critic2_scheduler.step(critic2_loss)
            self.high_level_scheduler.step(high_level_loss_val)
            self.meta_scheduler.step(meta_loss_val)
            self.distiller_scheduler.step(distiller_loss_val)

            self.train_steps += 1
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

    def perform_best_of_n_search(
        self,
        initial_state: torch.Tensor,
        num_samples: int,
        search_depth: int,
        time_memory: List[int],
        exploration_noise_std: float = 0.0
    ) -> List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, int]]]:
        """Samples trajectories using Best-of-N search."""
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
        """Performs Beam Search."""
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
        best_traj = max(final_trajs, key=lambda x: x[2])[1]
        return best_traj

    def update_params(self, replay_buffer, meta_input, time_memory, update_steps=1):
        """Updates parameters without search."""
        final_info = {}
        for _ in range(update_steps):
            if len(replay_buffer) < self.config.batch_size:
                return final_info

            batch, _, _ = replay_buffer.sample(self.config.batch_size)
            states = torch.FloatTensor(np.stack([b[0] for b in batch])).to(self.device)
            actions = torch.FloatTensor(np.stack([b[1] for b in batch])).to(self.device)
            rewards = torch.FloatTensor(np.stack([b[2] for b in batch])).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.stack([b[3] for b in batch])).to(self.device)
            dones = torch.FloatTensor(np.stack([b[4] for b in batch])).unsqueeze(1).to(self.device)
            time_steps = torch.FloatTensor(np.stack([b[5] for b in batch])).to(self.device)

            random_modes = torch.randint(0, self.config.num_market_modes, (states.shape[0],)).to(self.device)
            market_mode_loss = self.update_market_mode_classifier(states, random_modes)

            with torch.no_grad():
                mu, log_sigma = self.actor(states, time_steps)
                sigma = torch.exp(log_sigma)
                dist = torch.distributions.Normal(mu, sigma)
                z = dist.rsample()
                actions_ = torch.tanh(z)
                log_probs = dist.log_prob(z) - torch.log(1 - actions_.pow(2) + self.config.epsilon)
                log_probs = log_probs.sum(-1, keepdim=True)

            meta_input_tensor = torch.FloatTensor(meta_input).to(self.device)
            meta_loss_val, r_scaling = self.update_meta_controller(meta_input_tensor, log_probs, rewards)
            market_mode_probs = self.market_mode_classifier(states)

            q_targets = self.compute_q_targets(rewards, next_states, time_steps, dones, r_scaling, market_mode_probs)

            critic1_loss, critic2_loss = self.update_critics(states, actions, time_steps, q_targets)
            actor_loss_val = self.update_actor(states, time_steps)
            distiller_loss_val = self.update_distiller(states, time_steps)

            advantages = q_targets - torch.mean(q_targets, dim=0, keepdim=True)
            high_level_loss_val = self.update_high_level_policy(states, advantages)

            self.soft_update(self.critic_target1, self.critic1)
            self.soft_update(self.critic_target2, self.critic2)

            self.actor_scheduler.step(actor_loss_val)
            self.critic1_scheduler.step(critic1_loss)
            self.critic2_scheduler.step(critic2_loss)
            self.high_level_scheduler.step(high_level_loss_val)
            self.meta_scheduler.step(meta_loss_val)
            self.distiller_scheduler.step(distiller_loss_val)

            self.train_steps += 1
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
        """Saves the agent's state dicts."""
        try:
            torch.save({
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic_target1": self.critic_target1.state_dict(),
                "critic_target2": self.critic_target2.state_dict(),
                "meta_controller": self.meta_controller.state_dict(),
                "distiller": self.policy_distiller.state_dict(),
                "market_mode": self.market_mode_classifier.state_dict(),
                "high_level": self.high_level_policy.state_dict(),
                "alpha": self.alpha.detach().cpu().numpy(),
                "train_steps": self.train_steps
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save: {str(e)}")
            raise

    def load(self, path: str) -> None:
        """Loads the agent's state dicts."""
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic1.load_state_dict(ckpt["critic1"])
            self.critic2.load_state_dict(ckpt["critic2"])
            self.critic_target1.load_state_dict(ckpt["critic_target1"])
            self.critic_target2.load_state_dict(ckpt["critic_target2"])
            self.meta_controller.load_state_dict(ckpt["meta_controller"])
            self.policy_distiller.load_state_dict(ckpt["distiller"])
            self.market_mode_classifier.load_state_dict(ckpt["market_mode"])
            self.high_level_policy.load_state_dict(ckpt["high_level"])
            self.alpha.data.copy_(torch.tensor(ckpt["alpha"], dtype=torch.float32, device=self.device))
            self.train_steps = ckpt["train_steps"]
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load: {str(e)}")
            raise
