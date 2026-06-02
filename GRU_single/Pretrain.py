import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def _create_lagged(X_np, feat_indices, tgt_indices, lag, horizon):
    X_lag, y_lag = [], []
    for i in range(lag, len(X_np) - horizon):
        X_lag.append(X_np[i - lag : i, feat_indices])
        y_lag.append(X_np[i + horizon, tgt_indices])
    return np.array(X_lag), np.array(y_lag)


def pretrain(X_scaled, train_cut_1, train_cut_2, model, feature_cols, target_cols,
             lag, horizon, n_epochs, batch_size, lr, loss_fn, device='cpu'):
    '''
    Teacher-forcing pre-training on both training chunks.
    Each chunk is windowed independently to avoid the discontinuity.

    horizon: how many steps ahead to predict (1 = next step, 20 = trend prediction)
    '''
    feat_indices = [list(X_scaled.columns).index(f) for f in feature_cols]
    tgt_indices  = [list(X_scaled.columns).index(t) for t in target_cols]
    X_np = X_scaled.values

    X1, y1 = _create_lagged(X_np[:train_cut_1],  feat_indices, tgt_indices, lag, horizon)
    X2, y2 = _create_lagged(X_np[train_cut_2:],  feat_indices, tgt_indices, lag, horizon)

    X_train = torch.tensor(np.concatenate([X1, X2]), dtype=torch.float32)
    y_train = torch.tensor(np.concatenate([y1, y2]), dtype=torch.float32)

    loader    = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}  loss: {train_losses[-1]:.4f}")

    return train_losses
