import torch
import torch.nn as nn
import torch.nn.functional as F

class HypersphericalEncoder(nn.Module):
    def __init__(self, projection_dim=128, sequence_length=60, n_price_features=5, n_indicator_features=7, temperature=0.07, device='cuda'):
        super().__init__()
        self.projection_dim = projection_dim
        self.sequence_length = sequence_length
        self.n_price_features = n_price_features
        self.n_indicator_features = n_indicator_features
        self.temperature = temperature
        self.device = device

        self.gru = nn.GRU(
            input_size=n_price_features + n_indicator_features,
            hidden_size=projection_dim * 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        ).to(device)

        self.projection1 = nn.Linear(projection_dim * 4, projection_dim * 2).to(device)
        self.layer_norm = nn.LayerNorm(projection_dim * 2).to(device)
        self.projection2 = nn.Linear(projection_dim * 2, projection_dim).to(device)

        self._init_weights()
        self.eval()
        torch.set_grad_enabled(False)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection1.weight)
        nn.init.zeros_(self.projection1.bias)
        nn.init.xavier_uniform_(self.projection2.weight)
        nn.init.zeros_(self.projection2.bias)

    def forward(self, x, lengths=None):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if lengths is not None:
                packed_features = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                packed_output, gru_hidden = self.gru(packed_features)
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                output, gru_hidden = self.gru(x)

            gru_hidden = torch.cat((gru_hidden[-2], gru_hidden[-1]), dim=1)
            hidden = F.relu(self.layer_norm(self.projection1(gru_hidden)))
            projected = self.projection2(hidden)
            normalized = F.normalize(projected / self.temperature, p=2, dim=1)

            return normalized, output

    @torch.no_grad()
    def encode_batch(self, sequences):
        sequences = sequences.to(self.device, non_blocking=True)
        return self(sequences)
