import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass

###############################################################################
# 1) Model Configuration Dataclass
###############################################################################
@dataclass
class ModelConfig:
    """
    Configuration for the improved DRL trading brain.
    """
    # Input dims
    price_feat_dim: int = 6
    text_vocab_size: int = 0
    text_embed_dim: int = 32

    # Core architecture
    d_model: int = 64
    memory_dim: int = 64
    num_heads: int = 2
    seq_len: int = 128
    recursion_steps: int = 2

    # Action space
    num_actions: int = 4  # For discrete. If continuous, you'd do something else
    smoothing: float = 0.1  # label smoothing

    # Dropouts & etc.
    dropout: float = 0.1
    use_causal_mask: bool = True
    use_rotary_embeddings: bool = True

    # RL-specific
    use_value_uncertainty: bool = True  # If True => mean, logvar. Else => single head


###############################################################################
# 2) Rotary Positional Embeddings (Temporal)
###############################################################################
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embeddings for enhanced temporal awareness (paper: https://arxiv.org/abs/2104.09864).
    """
    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        seq_pos = torch.arange(max_seq_len).float()
        sincos = torch.einsum("i,j->ij", seq_pos, inv_freq)
        emb = torch.cat((sincos.sin(), sincos.cos()), dim=-1)  # [max_seq_len, dim]
        self.register_buffer("emb", emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, nHeads, seq_len, head_dim]
        returns: same shape, but multiplied by the rotation embedding
        """
        B, nH, S, d = x.shape
        emb = self.emb[:S, :].unsqueeze(0).unsqueeze(1)  # [1,1,S,d]
        return x * emb


###############################################################################
# 3) TPA Attention (with optional Causal Mask + Rotary)
###############################################################################
class TPAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_len: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_rotary: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_causal_mask = use_causal_mask
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary = RotaryPositionalEmbedding(self.head_dim, seq_len) if use_rotary else None

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

        if use_causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, S, d_model]
        key_padding_mask: [B, S], True for positions to mask out
        """
        B, S, _ = x.shape
        x_ln = self.layernorm(x)

        q = self.q_proj(x_ln).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B,nH,S,h]
        k = self.k_proj(x_ln).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_ln).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            q = self.rotary(q)
            k = self.rotary(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.use_causal_mask:
            cm = self.causal_mask[:S, :S]
            scores = scores.masked_fill(cm.unsqueeze(0).unsqueeze(0), float('-inf'))
        if key_padding_mask is not None:
            # Expand to [B, 1, 1, S]
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # [B,nH,S,h]
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return x + out


###############################################################################
# 4) DualMemoryModule (Short + Long) with gating
###############################################################################
class DualMemoryModule(nn.Module):
    def __init__(self, d_model: int, memory_dim: int, dropout: float = 0.1):
        super().__init__()
        self.short_lstm = nn.LSTM(
            input_size=d_model, hidden_size=memory_dim,
            num_layers=1, batch_first=True, dropout=dropout
        )
        # Long-term memory
        self.long_mem = nn.Parameter(torch.zeros(memory_dim))
        self.long_mem_prior = nn.Parameter(torch.zeros(memory_dim))

        self.gate_net = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.ReLU(),
            nn.Linear(memory_dim * 2, memory_dim * 2),
            nn.LayerNorm(memory_dim * 2),
            nn.Sigmoid()
        )
        self.compress = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.long_mem, mean=0.0, std=0.02)
        nn.init.normal_(self.long_mem_prior, mean=0.0, std=0.02)
        for name, param in self.gate_net.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: [B, S, d_model]
        hidden: (h,c) => LSTM states
        returns: (final_mem: [B,memory_dim], new_hidden)
        """
        B, S, _ = x.shape
        lstm_out, (h, c) = self.short_lstm(x, hidden)
        short_mem = lstm_out[:, -1, :]
        long_mem_total = self.long_mem + self.long_mem_prior
        long_mem_batch = long_mem_total.unsqueeze(0).expand(B, -1)

        combined = torch.cat([short_mem, long_mem_batch], dim=-1)
        gates = self.gate_net(combined)
        short_gate, long_gate = gates.chunk(2, dim=-1)
        gated = torch.cat([short_mem * short_gate, long_mem_batch * long_gate], dim=-1)
        final_mem = self.compress(gated)
        return final_mem, (h, c)


###############################################################################
# 5) RecursiveReasoner
###############################################################################
class RecursiveReasoner(nn.Module):
    def __init__(self, mem_dim: int, recursion_steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.recursion_steps = recursion_steps
        self.transform = nn.Sequential(
            nn.Linear(mem_dim, mem_dim * 2),
            nn.LayerNorm(mem_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mem_dim * 2, mem_dim),
            nn.LayerNorm(mem_dim)
        )
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, mem_dim]
        for _ in range(self.recursion_steps):
            delta = self.transform(x)
            x = x + self.res_scale * delta
        return x


###############################################################################
# 6) ImprovedUnifiedTradingBrain - DRL version
###############################################################################
class ImprovedUnifiedTradingBrain(nn.Module):
    """
    Integrates:
    - Price encoder (+ optional text)
    - TPA (attention)
    - DualMemory (LSTM + param)
    - RecursiveReasoner
    - Policy head (action distribution)
    - Value head (for DRL training)

    For discrete action spaces => action_logits
    For continuous => output (mu, log_sigma)
    """
    def __init__(self, config: ModelConfig, continuous_actions: bool = False):
        super().__init__()
        self.config = config
        self.continuous_actions = continuous_actions

        # Price encoder
        self.price_encoder = nn.Sequential(
            nn.Linear(config.price_feat_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        # Optional text
        if config.text_vocab_size > 0:
            self.text_encoder = nn.Sequential(
                nn.Embedding(config.text_vocab_size, config.text_embed_dim),
                nn.Linear(config.text_embed_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        else:
            self.text_encoder = None

        # TPA
        self.tpa = TPAAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            seq_len=config.seq_len,
            dropout=config.dropout,
            use_causal_mask=config.use_causal_mask,
            use_rotary=config.use_rotary_embeddings
        )

        # Memory
        self.memory = DualMemoryModule(config.d_model, config.memory_dim, config.dropout)
        self.reasoner = RecursiveReasoner(config.memory_dim, config.recursion_steps, config.dropout)

        # Output
        self.output_net = nn.Sequential(
            nn.Linear(config.memory_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        if continuous_actions:
            # For continuous, we produce mu, log_sigma for each action dimension
            self.policy_head = nn.Linear(config.d_model, config.num_actions * 2)
        else:
            # For discrete, we produce a single set of logits for num_actions
            self.policy_head = nn.Linear(config.d_model, config.num_actions)

        # Value head
        if config.use_value_uncertainty:
            # Mean & logvar
            self.value_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 2)
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 1)
            )

        self._init_weights()

    def _init_weights(self):
        def _init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.apply(_init_layer)

    def forward(
        self,
        price_input: torch.Tensor,
        text_input: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          - policy_out: if discrete => [B, num_actions], if continuous => [B, num_actions*2]
          - value_out: if use_value_uncertainty => [B,2], else => [B,1]
          - new_hidden: updated LSTM states
        """
        # 1) Price encoding
        x = self.price_encoder(price_input)

        # 2) Optional text
        if self.text_encoder is not None and text_input is not None:
            txt = self.text_encoder(text_input)
            txt_ctx = txt.mean(dim=1, keepdim=True)
            x = x + txt_ctx

        # 3) TPA
        x = self.tpa(x, key_padding_mask=key_padding_mask)
        # 4) Memory
        mem_out, new_hidden = self.memory(x, hidden)
        # 5) Reason
        refined = self.reasoner(mem_out)

        # 6) Output
        final = self.output_net(refined)
        policy_out = self.policy_head(final)
        value_out = self.value_head(final)

        return {
            "policy_out": policy_out,
            "value_out": value_out,
            "hidden": new_hidden
        }


###############################################################################
# 7) DRL Loss Functions (SAC-like or PPO-like) - Minimal Examples
###############################################################################

def sac_loss(
    model_outputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_value: torch.Tensor,
    alpha: float,
    gamma: float = 0.99,
    continuous: bool = False
) -> torch.Tensor:
    """
    Basic placeholder for an actor/critic loss in a SAC-like setting.
    This is extremely simplified. In a real codebase, you'd separate
    actor and critic updates, handle Q-targets with target networks, etc.
    """
    policy_out = model_outputs["policy_out"]  # [B, action_dim*2] or [B, action_dim]
    value_out = model_outputs["value_out"]    # [B,2] or [B,1]

    # 1) Critic loss. Suppose we interpret 'value_out' as a Q-value for this example.
    if value_out.shape[-1] == 2:
        # (mean, logvar)
        mean, logvar = value_out[...,0], value_out[...,1]
        var = torch.exp(logvar)
        # negative log-likelihood of the Q-target => or direct MSE if simpler
        q_target = rewards + gamma * next_value
        diff = q_target - mean
        critic_loss = 0.5 * (logvar + diff.pow(2)/var).mean()
    else:
        q_pred = value_out.squeeze(-1)
        q_target = rewards + gamma * next_value
        critic_loss = F.mse_loss(q_pred, q_target)

    # 2) Actor/policy loss. If discrete => cross-entropy or policy gradient approach.
    # If continuous => diagonal Gaussian approach.
    if continuous:
        # policy_out => [B, 2*action_dim] => (mu, log_sigma)
        B, out_dim = policy_out.shape
        a_dim = out_dim // 2
        mu = policy_out[:,:a_dim]
        log_sigma = policy_out[:,a_dim:]
        dist = torch.distributions.Normal(mu, torch.exp(log_sigma))
        z = dist.rsample()
        # For example, if actions in [-1,1], we do a Tanh transform:
        act_pred = torch.tanh(z)
        # sample logp
        logp = dist.log_prob(z).sum(-1, keepdim=True) - torch.log(1 - act_pred.pow(2) + 1e-6).sum(-1, keepdim=True)
        # Q-value from the model's value_out (this is not fully correct for SAC, just a placeholder):
        # Typically you'd have separate Q networks...
        actor_loss = (alpha * logp - q_target).mean()  # super simplified
    else:
        # Discrete => cross-entropy with actions or advantage-based policy gradient
        actor_loss = F.cross_entropy(policy_out, actions)

    total_loss = critic_loss + actor_loss
    return total_loss


###############################################################################
# 8) Example Train / Inference Loop
###############################################################################

def train_step(
    model: ImprovedUnifiedTradingBrain,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    alpha: float = 0.2,
    gamma: float = 0.99,
    continuous: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> float:
    """
    Minimal example of how you might do a single DRL update step.
    In real code, you'd separate actor + critic updates, handle target networks, etc.
    """
    model.train()
    for k,v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    # Forward
    out = model(
        price_input=batch["price"],  # [B,S,features]
        text_input=batch.get("text")
    )

    # Suppose we have "actions" => discrete or continuous, "rewards", "next_value"
    loss = sac_loss(
        out,
        actions=batch["actions"],
        rewards=batch["rewards"],
        next_value=batch["next_value"],
        alpha=alpha,
        gamma=gamma,
        continuous=continuous
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def inference_step(
    model: ImprovedUnifiedTradingBrain,
    price_input: torch.Tensor,
    text_input: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    continuous: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    """
    Inference that returns an action.
    If continuous => sample from Normal. If discrete => sample from softmax.
    """
    model.eval()
    price_input = price_input.to(device)
    if text_input is not None:
        text_input = text_input.to(device)

    out = model(price_input, text_input)

    policy_out = out["policy_out"]
    if continuous:
        # policy_out => [B, 2*action_dim]
        B, out_dim = policy_out.shape
        a_dim = out_dim // 2
        mu = policy_out[:,:a_dim]
        log_sigma = policy_out[:,a_dim:]
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        if temperature <= 0:
            # deterministic => just mu
            z = mu
        else:
            z = dist.rsample() * temperature
        action = torch.tanh(z)
    else:
        # Discrete
        if temperature <= 0:
            action = torch.argmax(policy_out, dim=-1)
        else:
            scaled_logits = policy_out / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return action


###############################################################################
# 9) Demo / Main
###############################################################################
if __name__ == "__main__":
    config = ModelConfig(
        price_feat_dim=6,
        text_vocab_size=1000,
        d_model=64,
        memory_dim=64,
        num_heads=2,
        seq_len=128,
        recursion_steps=2,
        num_actions=4,
        smoothing=0.1,
        dropout=0.1,
        use_causal_mask=True,
        use_rotary_embeddings=True,
        use_value_uncertainty=True  # e.g. mean+logvar
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For demonstration, let's do discrete actions
    model = ImprovedUnifiedTradingBrain(config, continuous_actions=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Example batch
    B = 32
    batch = {
        "price": torch.randn(B, 10, config.price_feat_dim),  # 10-step sequence
        "actions": torch.randint(0, config.num_actions, (B,)),  # discrete
        "rewards": torch.randn(B, 1),
        "next_value": torch.zeros(B, 1)  # placeholder
    }

    loss_val = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        alpha=0.2,
        gamma=0.99,
        continuous=False
    )
    print(f"Train Step Loss: {loss_val:.4f}")

    # Inference
    test_price = torch.randn(1, 10, config.price_feat_dim)
    act = inference_step(model, price_input=test_price, temperature=1.0, continuous=False)
    print(f"Inferred Action: {act.item()}")
