"""
EncDec-AD model architecture (Malhotra et al., 2016).

Corresponds to Chapter 4.3 (Model Architecture) of the dissertation.
An LSTM encoder compresses each window to a fixed-length latent vector;
an LSTM decoder reconstructs the window in reverse temporal order (the
reversal trick from Malhotra et al., 2016), which is then flipped back
for loss computation against the original window.
"""

import torch
import torch.nn as nn

from config import HIDDEN_SIZE


class LSTMEncoder(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, seq_len, h_0, c_0):
        batch_size = h_0.size(1)
        n_features = self.output_layer.out_features
        decoder_input = torch.zeros(batch_size, 1, n_features, device=h_0.device)
        hidden, cell = h_0, c_0
        outputs = []
        for _ in range(seq_len):
            out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            step_output = self.output_layer(out)
            outputs.append(step_output)
            decoder_input = step_output
        return torch.cat(outputs, dim=1)  # reverse-order reconstruction


class EncDecAD(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size)
        self.decoder = LSTMDecoder(n_features, hidden_size)

    def forward(self, x):
        seq_len = x.size(1)
        h_n, c_n = self.encoder(x)
        reconstruction_reversed = self.decoder(seq_len, h_n, c_n)
        return torch.flip(reconstruction_reversed, dims=[1])


def extract_latents(model, windows, device):
    """Runs windows through the encoder only and returns the final hidden state
    h_n as a flat (n_windows, hidden_size) array — used for the latent space
    visualisation in Figure 6.1 (Section 6.2.1)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)
        h_n, _ = model.encoder(x)
    return h_n.squeeze(0).cpu().numpy()
