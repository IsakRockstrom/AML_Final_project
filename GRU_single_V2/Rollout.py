import torch
import numpy as np


def _sample_transitions(X, threshold, search_ahead, lag, horizon):
    transition_starts = []
    normal_starts     = []
    for w in range(lag + horizon, len(X) - search_ahead):
        future = X['AMOC'].iloc[w : w + search_ahead]
        if future.max() - future.min() > threshold:
            transition_starts.append(w)
        else:
            normal_starts.append(w)
    return transition_starts, normal_starts


def _balance_transitions(X, transitions, lag, search_ahead):
    on_to_off, off_to_on = [], []
    for w in transitions:
        future = X['AMOC'].iloc[w + lag : w + lag + search_ahead]
        if future.iloc[-1] - future.iloc[0] < 0:
            on_to_off.append(w)
        else:
            off_to_on.append(w)
    n = min(len(on_to_off), len(off_to_on))
    rng = np.random.default_rng()
    return (list(rng.choice(on_to_off, n, replace=False)) +
            list(rng.choice(off_to_on, n, replace=False)))


def cumsum_loss(preds, targets):
    """Integral penalty: cumsum divergence between predicted and true trajectories.
    preds, targets: lists of tensors of shape [n_targets], one per rollout step.
    Returns a scalar tensor.
    """
    preds_t   = torch.stack(preds)    # [steps, n_targets]
    targets_t = torch.stack(targets)  # [steps, n_targets]
    cum_pred  = torch.cumsum(preds_t,   dim=0)
    cum_true  = torch.cumsum(targets_t, dim=0)
    return torch.mean((cum_pred - cum_true) ** 2)


def _predict(X_w, model, feat_indices):
    x = torch.tensor(X_w[:, feat_indices], dtype=torch.float32).unsqueeze(0)
    return model(x).squeeze(0)  # [n_targets] or scalar if output_size=1


def train_rollout(X, train_cut_1, train_cut_2, transition_prob, threshold, search_ahead, horizon,
                  model, feature_cols, target_cols,
                  train_steps, lag, loss_fn, lr, integral_alpha = 0.0):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    transitions_1, normal_1 = _sample_transitions(X[:train_cut_1], threshold, search_ahead, lag, horizon)
    transitions_2, normal_2 = _sample_transitions(X[train_cut_2:], threshold, search_ahead, lag, horizon)
    transitions = transitions_1 + [w + train_cut_2 for w in transitions_2]
    #transitions = _balance_transitions(X, transitions, lag, search_ahead)
    normal      = normal_1      + [w + train_cut_2 for w in normal_2]

    feat_indices = [list(X.columns).index(f) for f in feature_cols]
    tgt_indices  = [list(X.columns).index(t) for t in target_cols]

    train_losses = []
    rng = np.random.default_rng()

    for _ in range(train_steps):
        w = rng.choice(transitions) if rng.random() < transition_prob else rng.choice(normal)

        X_w = X[w - lag - horizon : w - horizon].values                     # [lag, n_cols]
        y_w = X[w : w + search_ahead].values    # [train_horizon, n_cols]

        loss = torch.tensor(0.0)
        preds = []
        targets = []
        for j in range(search_ahead):
            pred   = _predict(X_w, model, feat_indices)      # [n_targets]
            target = torch.tensor(y_w[j, tgt_indices], dtype=torch.float32)
            preds.append(pred)
            targets.append(target)
            loss  += loss_fn(pred, target)
            X_w = np.roll(X_w, -1, axis=0)
            X_w[-1, tgt_indices] = pred.detach().numpy()

        loss = loss / search_ahead + integral_alpha * cumsum_loss(preds, targets)       

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

    return train_losses


def eval_rollout(X, train_cut_1, train_cut_2, transition_prob, threshold, search_ahead,
                 model, feature_cols, target_cols,
                 lag, horizon, loss_fn, integral_alpha = 0.0):

    model.eval()
    rng = np.random.default_rng()
    transitions, normal = _sample_transitions(X[train_cut_1:train_cut_2], threshold, search_ahead, lag, horizon)

    w = rng.choice(transitions) if rng.random() < transition_prob else rng.choice(normal)

    X_eval       = X[train_cut_1:train_cut_2].values
    feat_indices = [list(X.columns).index(f) for f in feature_cols]
    tgt_indices  = [list(X.columns).index(t) for t in target_cols]

    X_w = X_eval[w - lag - horizon: w - horizon]
    y_w = X_eval[w : w + search_ahead]

    eval_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for j in range(search_ahead):
            pred   = _predict(X_w, model, feat_indices)
            target = torch.tensor(y_w[j, tgt_indices], dtype=torch.float32)
            preds.append(pred)
            targets.append(target)
            eval_loss += loss_fn(pred, target).item()
            X_w = np.roll(X_w, -1, axis=0)
            X_w[-1, tgt_indices] = pred.numpy()

    eval_loss = eval_loss / search_ahead + integral_alpha * cumsum_loss(preds, targets).item()

    model.train()
    return eval_loss 
