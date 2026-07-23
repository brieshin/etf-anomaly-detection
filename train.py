"""
Training loop with early stopping.

Corresponds to Chapter 4.3 / Chapter 5.1 (Training Results) of the
dissertation. With the default config, training stops at epoch 44
with a best validation loss of approximately 0.798 (Case reproduced
exactly with SEED=0, END_DATE="2026-07-01").
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import HIDDEN_SIZE, BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE, DEVICE
from model import EncDecAD


def train_model(train_windows, val_windows, n_features, device=DEVICE):
    model = EncDecAD(n_features=n_features, hidden_size=HIDDEN_SIZE).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_tensor = torch.from_numpy(val_windows).to(device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimiser.step()
            train_loss_sum += loss.item() * batch.size(0)
        train_loss = train_loss_sum / len(train_windows)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = nn.functional.mse_loss(val_recon, val_tensor).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    return model, history


def reconstruction_errors(model, windows, device=DEVICE):
    """Per-timestep, per-asset absolute reconstruction error."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)
        recon = model(x)
        errors = torch.abs(x - recon).cpu().numpy()
    return errors
