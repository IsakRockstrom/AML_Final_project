import torch
import numpy as np


def _sample_transitions(X, on_to_off_threshold, off_to_on_threshold, search_ahead, lag):
    transition_starts_on_to_off = []
    transition_starts_off_to_on = []
    normal_starts = []
    for w in range(lag, len(X) - lag - search_ahead):
        future = X['AMOC'].iloc[w + lag : w + lag + search_ahead]

        on_to_off = future.iloc[-1] - future.iloc[0] < 0 and future.max() - future.min() > on_to_off_threshold
        off_to_on = future.iloc[-1] - future.iloc[0] > 0 and future.max() - future.min() > off_to_on_threshold

        if on_to_off:
            transition_starts_on_to_off.append(w)
        if off_to_on:
            transition_starts_off_to_on.append(w)
        if not on_to_off and not off_to_on:
            normal_starts.append(w)

    return transition_starts_on_to_off, transition_starts_off_to_on, normal_starts


def train_rollout(X, transition_prob,
                  model, feature_cols, target_cols,
                  train_steps, lag, train_horizon, loss_fn, lr,
                  transitions, normal, grad_clip=0.5):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    feat_indices = [list(X.columns).index(f) for f in feature_cols]
    tgt_indices  = [list(X.columns).index(t) for t in target_cols]
    X_np = X.values.astype(np.float32)

    train_losses = []
    rng = np.random.default_rng()

    for step in range(train_steps):
        w = rng.choice(transitions) if rng.random() < transition_prob else rng.choice(normal)

        # Keep window as a tensor throughout — no numpy mid-loop, no detach
        window = torch.tensor(X_np[w : w + lag][:, feat_indices])          
        y_w    = torch.tensor(X_np[w + lag : w + lag + train_horizon][:, tgt_indices]) 

        loss = torch.zeros(1)
        for j in range(train_horizon):
            pred   = model(window.unsqueeze(0)).squeeze(0)                
            loss   = loss + loss_fn(pred, y_w[j])
            window = torch.cat([window[1:], pred.unsqueeze(0)], dim=0)   

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item() / train_horizon
        train_losses.append(loss_val)
        if not np.isfinite(loss_val):
            print(f"Diverged at step {step}")
            break

    return train_losses


def eval_rollout(X, transition_prob, on_to_off_threshold, off_to_on_threshold, search_ahead,
                 model, feature_cols, target_cols,
                 lag, eval_horizon, loss_fn,
                 transitions, normal):

    model.eval()
    rng = np.random.default_rng()

    w = rng.choice(transitions) if rng.random() < transition_prob else rng.choice(normal)

    feat_indices = [list(X.columns).index(f) for f in feature_cols]
    tgt_indices  = [list(X.columns).index(t) for t in target_cols]
    X_np = X.values.astype(np.float32)

    window = torch.tensor(X_np[w : w + lag][:, feat_indices])
    y_w    = torch.tensor(X_np[w + lag : w + lag + eval_horizon][:, tgt_indices])

    eval_loss = 0.0
    with torch.no_grad():
        for j in range(eval_horizon):
            pred      = model(window.unsqueeze(0)).squeeze(0)
            eval_loss += loss_fn(pred, y_w[j]).item()
            window     = torch.cat([window[1:], pred.unsqueeze(0)], dim=0)

    model.train()
    return eval_loss / eval_horizon
