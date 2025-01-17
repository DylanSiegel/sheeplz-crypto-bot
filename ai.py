import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################
# Global Config (adjust to your needs)
#################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Speed optimization for CNN/LSTM ops
USE_MIXED_PRECISION = True            # Enable PyTorch AMP for half-precision

#################################################################
# Unified AI Brain
#################################################################
class UnifiedTradingBrain(nn.Module):
    """
    A single compact model that fuses:
      - Multi-modal input encoding (price + optional text).
      - Minimal TPA-based Transformer for temporal modeling.
      - A small memory mechanism (short + long).
      - Internal mini-loop for recursive reasoning.
      - Outputs both policy logits (for discrete actions) and a value estimate.
    
    Designed to have a relatively low parameter count for edge devices,
    but still incorporate advanced architectural ideas.
    """
    def __init__(
        self,
        price_feat_dim=6,          # e.g., OHLCV + 1 extra feature
        text_vocab_size=0,         # 0 => no text input by default
        text_embed_dim=32,         # used if text_vocab_size > 0
        d_model=64,                # hidden dimension throughout
        num_heads=2,               # heads for TPA
        memory_dim=64,             # dimension for both short & long memory
        recursion_steps=2,         # small recursion count
        num_actions=4              # buy, sell, hold, close, etc.
    ):
        super().__init__()

        self.text_vocab_size = text_vocab_size
        self.d_model = d_model
        self.recursion_steps = recursion_steps

        # ----------------------------
        # 1) Multi-Modal Input Embedding
        # ----------------------------
        self.price_encoder = nn.Sequential(
            nn.Linear(price_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        if text_vocab_size > 0:
            self.text_emb = nn.Embedding(text_vocab_size, text_embed_dim)
            self.text_fc = nn.Sequential(
                nn.Linear(text_embed_dim, d_model),
                nn.ReLU()
            )
        else:
            self.text_emb = None

        # ----------------------------
        # 2) Minimal TPA-based Transformer
        #    For demonstration, we'll do a single TPA-like attention block
        # ----------------------------
        # Factorizable Q,K,V
        # We keep it extra minimal to reduce complexity
        self.q_fc = nn.Linear(d_model, d_model, bias=False)
        self.k_fc = nn.Linear(d_model, d_model, bias=False)
        self.v_fc = nn.Linear(d_model, d_model, bias=False)
        self.num_heads = num_heads

        # For final feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        # ----------------------------
        # 3) Memory (short + long)
        #    For simplicity, we keep a single learnable vector for LTM
        # ----------------------------
        self.long_term_memory = nn.Parameter(torch.zeros(memory_dim))
        # Short-term memory can be an LSTM:
        self.short_term_lstm = nn.LSTM(d_model, memory_dim, batch_first=True)

        # ----------------------------
        # 4) Recursive Reasoning "Mini-Loop"
        # ----------------------------
        self.reflection_fc = nn.Linear(memory_dim + d_model, d_model)

        # ----------------------------
        # 5) Policy and Value Heads
        # ----------------------------
        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        # Basic initialization
        nn.init.normal_(self.long_term_memory, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, price_input, text_input=None):
        """
        price_input: [B, seq_len, price_feat_dim]
        text_input: [B, text_len], optional

        Returns:
          action_logits: [B, num_actions]
          state_value: [B, 1]
        """
        B, seq_len, _ = price_input.shape

        # ------------------------------------------------------
        # 1) Multi-modal Embedding
        # ------------------------------------------------------
        price_enc = self.price_encoder(price_input)  # [B, seq_len, d_model]
        if self.text_emb is not None and text_input is not None:
            # Basic approach: embed text, mean-pool
            text_embedded = self.text_emb(text_input)         # [B, text_len, text_embed_dim]
            text_vec = text_embedded.mean(dim=1)              # [B, text_embed_dim]
            text_vec = self.text_fc(text_vec)                 # [B, d_model]
            # Optionally broadcast to each time step
            text_vec = text_vec.unsqueeze(1).expand(-1, seq_len, -1)
            price_enc = price_enc + text_vec

        # ------------------------------------------------------
        # 2) Minimal TPA-based Self-Attention for Time
        # ------------------------------------------------------
        # Shape: [B, seq_len, d_model]
        # We'll reshape for multi-head attention in a simplistic way:
        # Q, K, V => [B, seq_len, d_model]
        Q = self.q_fc(price_enc)
        K = self.k_fc(price_enc)
        V = self.v_fc(price_enc)

        # Split heads
        def split_heads(x, n_heads):
            # x: [B, seq_len, d_model]
            B, S, D = x.shape
            head_dim = D // n_heads
            x = x.view(B, S, n_heads, head_dim).transpose(1, 2)  # [B, n_heads, seq_len, head_dim]
            return x

        Q_ = split_heads(Q, self.num_heads)
        K_ = split_heads(K, self.num_heads)
        V_ = split_heads(V, self.num_heads)

        # Scaled dot-product
        head_dim = self.d_model // self.num_heads
        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / (head_dim**0.5)  # [B, n_heads, S, S]
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_)  # [B, n_heads, S, head_dim]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, n_heads, head_dim]
        attn_output = attn_output.view(B, seq_len, self.d_model)

        # Residual + FFN
        x = price_enc + attn_output
        x = x + self.ffn(x)  # Another residual

        # ------------------------------------------------------
        # 3) Memory: short-term (LSTM) + long-term (param)
        # ------------------------------------------------------
        # short-term LSTM
        # x shape: [B, seq_len, d_model]
        lstm_out, (h, c) = self.short_term_lstm(x)
        # We'll use the final hidden state as short-term summary
        short_summary = lstm_out[:, -1, :]  # [B, memory_dim]

        # long-term memory (single vector)
        # Expand to batch dimension and add to short summary
        long_mem = self.long_term_memory.unsqueeze(0).expand(B, -1)  # [B, memory_dim]
        # We won't do a gating step here (simplification)
        mem_combined = torch.cat([short_summary, long_mem], dim=-1)  # [B, memory_dim * 2]

        # ------------------------------------------------------
        # 4) Recursive Reasoning "Mini-Loop"
        # ------------------------------------------------------
        # We'll do 'recursion_steps' of refining the final representation
        # Start with an integrated representation
        reasoning_rep = torch.zeros(B, self.d_model, device=DEVICE)

        for _ in range(self.recursion_steps):
            # simple reflection
            fused = self.reflection_fc(mem_combined)
            # combine with prior reasoning
            reasoning_rep = reasoning_rep + fused

        # ------------------------------------------------------
        # 5) Policy & Value Heads
        # ------------------------------------------------------
        # We'll produce final [B, d_model] representation to feed these heads
        final_rep = reasoning_rep  # [B, d_model]
        action_logits = self.policy_head(final_rep)  # [B, num_actions]
        state_value = self.value_head(final_rep)      # [B, 1]

        return action_logits, state_value


#################################################################
# Example: Minimal Training / Inference Loops
#################################################################

def train_unified_brain(dataloader, max_epochs=1):
    """
    Minimal training loop for demonstration.
    Replace the dataloader with your real data, which should yield:
      - price_input: [B, seq_len, price_feat_dim]
      - text_input (optional): [B, text_len]
      - action_labels: [B] or [B, 1]
      - value_targets: [B, 1] for supervised or RL-like training
    """
    model = UnifiedTradingBrain().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    model.train()
    for epoch in range(max_epochs):
        for batch_idx, batch in enumerate(dataloader):
            price_input = batch["price"].to(DEVICE)                # [B, seq_len, feat_dim]
            text_input = batch.get("text", None)
            if text_input is not None:
                text_input = text_input.to(DEVICE)
            action_labels = batch["actions"].to(DEVICE)            # [B]
            value_targets = batch["values"].to(DEVICE)             # [B, 1]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                action_logits, predicted_value = model(price_input, text_input)

                # Example losses
                # 1) Policy supervised or cross-entropy
                policy_loss = F.cross_entropy(action_logits, action_labels)

                # 2) Value regression (MSE or L1)
                value_loss = F.mse_loss(predicted_value, value_targets)

                total_loss = policy_loss + 0.1 * value_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

    return model


def inference_unified_brain(model, price_input, text_input=None):
    """
    Single inference call. 
    price_input: [1, seq_len, feat_dim]
    text_input: [1, text_len] or None
    """
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            action_logits, state_value = model(price_input, text_input)
            # Choose action
            action = torch.argmax(action_logits, dim=-1).item()
            val = state_value.item()
    return action, val


#################################################################
# Usage Example (Pseudocode)
#################################################################
if __name__ == "__main__":
    # Suppose you have a DataLoader that yields the required fields
    # dataloader = ...

    # model = train_unified_brain(dataloader, max_epochs=5)

    # # Inference on a single example
    # test_price_input = torch.randn(1, 10, 6, device=DEVICE)
    # chosen_action, value_est = inference_unified_brain(model, test_price_input)
    # print(f"Action: {chosen_action}, Value: {value_est:.4f}")
    pass
