import torch
from torch import nn
import numpy as np


class Predictor(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, h_n = self.gru(x)
        y = self.head(out[:, -1, :]).squeeze(-1)  
        return y

def _sample_transitions(X, threshold, search_ahead, lag):
    transition_starts = []
    normal_starts     = []
    for w in range(lag, len(X) - lag - search_ahead):
        future = X['AMOC'].iloc[w + lag : w + lag + search_ahead]
        if future.max() - future.min() > threshold:
            transition_starts.append(w)
        else:
            normal_starts.append(w)
    return transition_starts, normal_starts


def _predict(X_w, model, feat_indices):
    x = torch.tensor(X_w[:, feat_indices], dtype=torch.float32).unsqueeze(0)
    return model(x).squeeze()


def train_rollout(X, train_cut_1, train_cut_2, transition_prob, threshold, search_ahead, models, features, train_steps, lag, train_horizon, loss_fn, lrs):

    for model in models.values():
        model.train()

    optimizers = {name: torch.optim.Adam(model.parameters(), lr=lrs[name]) for name, model in models.items()}

    transitions_1, normal_1 = _sample_transitions(X[:train_cut_1], threshold, search_ahead, lag)
    transitions_2, normal_2 = _sample_transitions(X[train_cut_2:], threshold, search_ahead, lag)
    transitions = transitions_1 + [w + train_cut_2 for w in transitions_2]
    normal      = normal_1      + [w + train_cut_2 for w in normal_2]

    col_idx  = {name: list(X.columns).index(name) for name in models}
    feat_idx = {name: [list(X.columns).index(f) for f in features[name]] for name in models}
    model_names = list(models.keys())

    train_losses = []
    rng = np.random.default_rng()

    for i in range(train_steps):

        if rng.random() < transition_prob:
            w = rng.choice(transitions)
        else:
            w = rng.choice(normal)

        X_w = X[w : w + lag].values          # [lag, n_features]
        y_w = X[w + lag : w + lag + train_horizon].values  # [train_horizon, n_features]

        loss = torch.tensor(0.0)
        for j in range(train_horizon):
            new_vals = np.empty(len(model_names))
            for k, name in enumerate(model_names):
                pred = _predict(X_w, models[name], feat_idx[name])
                loss += loss_fn(pred, torch.tensor(y_w[j, col_idx[name]], dtype=torch.float32))
                new_vals[k] = pred.detach().item()
            X_w = np.roll(X_w, -1, axis=0)
            X_w[-1] = new_vals
        loss.backward()

        for name in models:
            torch.nn.utils.clip_grad_norm_(models[name].parameters(), 1.0)
            optimizers[name].step()
            optimizers[name].zero_grad()

        train_losses.append(loss.item() / train_horizon)

    return train_losses
    


def eval_rollout(X, train_cut_1, train_cut_2, transition_prob, threshold, search_ahead, models, features, lag, eval_horizon, loss_fn):
    for name in models:
        models[name].eval()

    rng = np.random.default_rng()
    transitions, normal = _sample_transitions(X[train_cut_1:train_cut_2], threshold, search_ahead, lag)

    if rng.random() < transition_prob:
            w = rng.choice(transitions)
    else:
        w = rng.choice(normal)
    
    X_eval = X[train_cut_1:train_cut_2].values
    col_idx  = {name: list(X.columns).index(name) for name in models}
    feat_idx = {name: [list(X.columns).index(f) for f in features[name]] for name in models}
    model_names = list(models.keys())

    X_w = X_eval[w : w + lag].copy()                    # [lag, n_features]
    y_w = X_eval[w + lag : w + lag + eval_horizon]      # [eval_horizon, n_features]

    eval_loss = 0.0

    with torch.no_grad():
        for j in range(eval_horizon):
            new_vals = np.empty(len(model_names))
            for k, name in enumerate(model_names):
                pred = _predict(X_w, models[name], feat_idx[name])
                eval_loss += loss_fn(pred, torch.tensor(y_w[j, col_idx[name]], dtype=torch.float32)).item()
                new_vals[k] = pred.item()
            X_w = np.roll(X_w, -1, axis=0)
            X_w[-1] = new_vals
        
    for name in models:
        models[name].train()

    return eval_loss / eval_horizon




        
            


