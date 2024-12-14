import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# ------------------------- Helper Classes -------------------------

class APELU(nn.Module):
    def __init__(self, alpha_init=0.01, beta_init=1.0):
        super(APELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * x * torch.exp(self.beta * x))

# ------------------------- Modern MLP -------------------------
class ModernMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.1):
        super(ModernMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim))
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x


# ------------------------- Adaptive Modulation MLP -------------------------
class TimeAwareBias(nn.Module):
    def __init__(self, input_dim, time_encoding_dim=10, hidden_dim=20):
        super().__init__()
        self.time_embedding = nn.Linear(time_encoding_dim, hidden_dim)
        self.time_projection = nn.Linear(hidden_dim, input_dim)
        self.activation = APELU()

    def forward(self, time_encoding):
        # time_encoding: [batch_size, time_encoding_dim]
        x = self.time_embedding(time_encoding)
        x = self.activation(x)
        x = self.time_projection(x)
        return x  # [batch_size, input_dim]


class AdaptiveModulationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.1, time_encoding_dim=10):
        super(AdaptiveModulationMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.modulations = nn.ParameterList()
        self.time_biases = nn.ModuleList()

        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i != num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim))
                self.modulations.append(nn.Parameter(torch.ones(hidden_dim)))
                self.time_biases.append(TimeAwareBias(hidden_dim, time_encoding_dim))
        self.activation = APELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, time_encoding):
        # x: [batch_size, feature_dim]
        # time_encoding: [batch_size, time_encoding_dim]
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                modulation_factor = self.modulations[i] + self.time_biases[i](time_encoding)
                # Ensure broadcasting works correctly:
                # modulation_factor: [batch_size, hidden_dim]
                # x: [batch_size, hidden_dim] (after first layer)
                x = x * modulation_factor
                x = layer(x)
                x = self.norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
            else:
                # final layer
                x = x * modulation_factor if i > 0 else x  # if needed
                x = layer(x)
        return x

# ------------------------- Attention Mechanism -------------------------
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, input_dim)

    def forward(self, input):
        # input = [batch_size, seq_len, input_dim]
        query = self.query_proj(input)
        key = self.key_proj(input)
        value = self.value_proj(input)

        scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(scores / (input.shape[-1]**0.5), dim=-1)
        context_vector = torch.matmul(attention_weights, value)  # [batch_size, seq_len, attention_dim]
        output = self.out_proj(context_vector)
        return output


# ------------------------- SAC Actor Network -------------------------
class MetaSACActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers=3, dropout_rate=0.1, time_encoding_dim=10):
        super(MetaSACActor, self).__init__()
        self.attention = Attention(input_dim, attention_dim)
        self.mlp = AdaptiveModulationMLP(input_dim, hidden_dim, 2*action_dim, num_mlp_layers, dropout_rate, time_encoding_dim)
        self.action_dim = action_dim

    def forward(self, x, time_encoding):
        # Expecting x: [batch_size, state_dim], turn into [batch_size, seq_len=1, state_dim]
        # For attention, we need a seq_len dimension. If seq_len=1:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention(x)  # [batch_size, 1, input_dim]
        x = x.squeeze(1)  # [batch_size, input_dim]
        x = self.mlp(x, time_encoding)  # [batch_size, 2*action_dim]
        mu, log_sigma = x[:, :self.action_dim], x[:, self.action_dim:]
        return torch.tanh(mu), log_sigma

# ------------------------- SAC Critic Network -------------------------
class MetaSACCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers=3, dropout_rate=0.1, time_encoding_dim=10):
        super(MetaSACCritic, self).__init__()
        self.attention = Attention(input_dim + action_dim, attention_dim)
        self.mlp = AdaptiveModulationMLP(input_dim + action_dim, hidden_dim, 1, num_mlp_layers, dropout_rate, time_encoding_dim)
        self.action_dim = action_dim

    def forward(self, state, action, time_encoding):
        # state: [batch_size, state_dim]
        # action: [batch_size, action_dim]
        # Combine: [batch_size, state_dim + action_dim]
        x = torch.cat([state, action], dim=-1)
        x = x.unsqueeze(1)  # [batch_size, seq_len=1, state_dim+action_dim]
        x = self.attention(x)  # [batch_size, 1, input_dim+action_dim]
        x = x.squeeze(1)       # [batch_size, input_dim+action_dim]
        q_value = self.mlp(x, time_encoding)  # [batch_size, 1]
        return q_value

# ------------------------- Meta Controller -------------------------
class MetaController(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_mlp_layers=3, dropout_rate=0.1):
        super().__init__()
        self.mlp = ModernMLP(input_dim, hidden_dim, output_dim, num_mlp_layers, dropout_rate)

    def forward(self, x):
        return self.mlp(x)

# ------------------------- Meta-SAC Agent -------------------------
class MetaSACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, attention_dim, meta_input_dim, time_encoding_dim=10, num_mlp_layers=3, dropout_rate=0.1, lr=1e-3, alpha=0.2, gamma=0.99, tau=0.005, meta_lr=1e-4, device="cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = nn.Parameter(torch.tensor(alpha)).to(device)
        self.target_entropy = -torch.prod(torch.tensor(action_dim).float()).to(device)
        self.time_encoding_dim = time_encoding_dim

        self.actor = MetaSACActor(state_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers, dropout_rate, time_encoding_dim).to(device)
        self.critic1 = MetaSACCritic(state_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers, dropout_rate, time_encoding_dim).to(device)
        self.critic2 = MetaSACCritic(state_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers, dropout_rate, time_encoding_dim).to(device)
        self.critic_target1 = MetaSACCritic(state_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers, dropout_rate, time_encoding_dim).to(device)
        self.critic_target2 = MetaSACCritic(state_dim, action_dim, hidden_dim, attention_dim, num_mlp_layers, dropout_rate, time_encoding_dim).to(device)

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.meta_input_dim = meta_input_dim
        self.meta_controller = MetaController(meta_input_dim, hidden_dim, 1, num_mlp_layers, dropout_rate).to(device)
        self.meta_optimizer = optim.Adam(self.meta_controller.parameters(), lr=meta_lr)

    def select_action(self, state, time_encoding, eval=False):
        # state: [state_dim]
        # time_encoding: [time_encoding_dim]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)        # [1, state_dim]
        time_encoding = torch.FloatTensor(time_encoding).unsqueeze(0).to(self.device)  # [1, time_encoding_dim]

        mu, log_sigma = self.actor(state, time_encoding)
        if eval:
            return mu.detach().cpu().numpy()[0]

        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()

        action = z
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]

    def soft_update(self, target_network, source_network):
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_params(self, memory, batch_size, meta_input, time_memory):
        if len(memory) < batch_size:
            return

        # Sample indices
        indices = random.sample(range(len(memory)), batch_size)
        batch = [memory[idx] for idx in indices]
        batch_time_encodings = [time_memory[idx] for idx in indices]

        states = torch.FloatTensor(np.array([item[0] for item in batch])).to(self.device)        # [batch_size, state_dim]
        actions = torch.FloatTensor(np.array([item[1] for item in batch])).to(self.device)       # [batch_size, action_dim]
        rewards = torch.FloatTensor(np.array([item[2] for item in batch])).unsqueeze(1).to(self.device) # [batch_size, 1]
        next_states = torch.FloatTensor(np.array([item[3] for item in batch])).to(self.device)   # [batch_size, state_dim]
        dones = torch.FloatTensor(np.array([item[4] for item in batch])).unsqueeze(1).to(self.device) # [batch_size, 1]

        meta_input = torch.FloatTensor(meta_input).unsqueeze(0).to(self.device)  # [1, meta_input_dim]
        batch_time_encodings = torch.FloatTensor(np.array(batch_time_encodings)).to(self.device)  # [batch_size, time_encoding_dim]

        # Compute target Q values
        with torch.no_grad():
            # Select next actions
            next_actions = []
            next_log_probs = []
            # Compute next actions for each next_state individually
            next_states_np = next_states.cpu().numpy()
            batch_time_encodings_np = batch_time_encodings.cpu().numpy()
            for ns, te in zip(next_states_np, batch_time_encodings_np):
                a, lp = self.select_action(ns, te)
                next_actions.append(a)
                next_log_probs.append(lp)

            next_actions = torch.FloatTensor(next_actions).to(self.device)   # [batch_size, action_dim]
            next_log_probs = torch.FloatTensor(next_log_probs).to(self.device) # [batch_size, 1]

            q_target1 = self.critic_target1(next_states, next_actions, batch_time_encodings)
            q_target2 = self.critic_target2(next_states, next_actions, batch_time_encodings)
            q_target_min = torch.min(q_target1, q_target2)
            q_target = rewards + (1.0 - dones) * self.gamma * (q_target_min - self.alpha * next_log_probs)

        q_value1 = self.critic1(states, actions, batch_time_encodings)
        q_value2 = self.critic2(states, actions, batch_time_encodings)

        critic1_loss = F.mse_loss(q_value1, q_target)
        critic2_loss = F.mse_loss(q_value2, q_target)

        # Update Critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor
        mu, log_sigma = self.actor(states, batch_time_encodings)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()

        action = z
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        q_val = torch.min(self.critic1(states, action, batch_time_encodings), self.critic2(states, action, batch_time_encodings))
        actor_loss = (-q_val + self.alpha * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Meta controller Loss
        meta_output = self.meta_controller(meta_input)  # output should be log(alpha)
        alpha_target_loss = - (meta_output - torch.log(self.alpha)) * (log_prob + self.target_entropy).detach().mean()

        self.meta_optimizer.zero_grad()
        alpha_target_loss.backward()
        self.meta_optimizer.step()

        self.soft_update(self.critic_target1, self.critic1)
        self.soft_update(self.critic_target2, self.critic2)

        with torch.no_grad():
            new_alpha = torch.exp(self.meta_controller(meta_input))
            self.alpha.copy_(new_alpha)


# ------------------------- Main -------------------------
if __name__ == '__main__':
    # Sample Usage:
    state_dim = 10 # example dimension
    action_dim = 2
    hidden_dim = 64
    attention_dim = 32
    meta_input_dim = 5
    batch_size = 32
    time_encoding_dim = 10
    memory = []
    time_memory = []

    agent = MetaSACAgent(state_dim, action_dim, hidden_dim, attention_dim, meta_input_dim, time_encoding_dim, device="cpu")

    for i in range(1000):
        state = np.random.rand(state_dim)
        time_encoding = np.random.rand(time_encoding_dim)
        action, _ = agent.select_action(state, time_encoding)
        action = np.clip(action, -1, 1)  # ensure action is within bounds after tanh
        next_state = np.random.rand(state_dim)
        reward = np.random.rand(1)
        done = False
        meta_input = np.random.rand(meta_input_dim)

        memory.append((state, action, reward, next_state, done))
        time_memory.append(time_encoding)

        agent.update_params(memory, batch_size, meta_input, time_memory)
        print("Alpha:", agent.alpha.item())
