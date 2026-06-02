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
             lag, horizon, n_epochs, batch_size, lr, loss_fn, device='cpu', sign_alpha=0.0):
    '''
    Teacher-forcing pre-training on both training chunks.
    Each chunk is windowed independently to avoid the discontinuity.

    horizon: how many steps ahead to predict (1 = next step, 20 = trend prediction)
    sign_alpha: weight for sign loss penalising wrong direction of change over the horizon
    '''
    feat_indices = [list(X_scaled.columns).index(f) for f in feature_cols]
    tgt_indices  = [list(X_scaled.columns).index(t) for t in target_cols]
    tgt_in_feat  = [feature_cols.index(t) for t in target_cols]
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

            if sign_alpha > 0.0:
                last_ctx   = x_batch[:, -1, tgt_in_feat]
                delta_true = y_batch - last_ctx
                delta_pred = pred    - last_ctx
                sign_loss  = torch.mean((torch.sign(delta_pred) - torch.sign(delta_true)) ** 2)
                loss = loss + sign_alpha * sign_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}  loss: {train_losses[-1]:.4f}")

    return train_losses
