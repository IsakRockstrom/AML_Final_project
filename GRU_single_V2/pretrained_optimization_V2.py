import optuna
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import json

from model_stuff import Predictor
from Rollout import train_rollout, _predict
from Pretrain import pretrain
from SingleModel import compute_covariances, _sample_noise

# ── Data setup ────────────────────────────────────────────────────────────────
X_df = pd.read_csv('../data/CombinedDatasetLongUnsmooth.csv')
X_df = X_df.drop(columns=['AMOC']).rename(columns={'AMOC_40N': 'AMOC'})
model_vars = ['AMOC', 'SFWF', 'PD_200m']
feature_cols = target_cols = model_vars
train_cut_1, train_cut_2 = 12500, 18500
train_idx = list(range(train_cut_1)) + list(range(train_cut_2, len(X_df)))

y_scalers = {}
X_scaled  = X_df[model_vars].copy().astype(float)
for var in model_vars:
    sc = StandardScaler().fit(X_df[var].values[train_idx].reshape(-1, 1))
    X_scaled[var] = sc.transform(X_df[var].values.reshape(-1, 1))
    y_scalers[var] = sc

amoc_std = y_scalers['AMOC'].scale_[0]
X_np     = X_scaled.values

# ── Fixed eval windows (global indices into X_scaled) ─────────────────────────
EVAL_WINDOWS = [12900, 13600, 15250, 15650, 16350, 17900]
EVAL_HORIZON = 400
DEVICE       = 'cpu'

feat_indices = [list(X_scaled.columns).index(f) for f in feature_cols]
tgt_indices  = [list(X_scaled.columns).index(t) for t in target_cols]

cov_on, cov_off = compute_covariances(X_df, model_vars, threshold=13.0)
amoc_idx = model_vars.index('AMOC')
threshold_scaled_eval = 13.0 / amoc_std


def _eval_fixed(model, start_idx, lag, horizon, rho=0.3, k=0.01, n_samples=3):
    model.eval()
    y_w     = X_np[start_idx : start_idx + EVAL_HORIZON]
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for _ in range(n_samples):
        X_w        = X_np[start_idx - lag - horizon : start_idx - horizon].copy()
        noise_prev = {v: 0.0 for v in model_vars}
        sample_loss = 0.0

        with torch.no_grad():
            # warm-up: roll forward horizon steps without computing loss
            for _ in range(horizon):
                x    = torch.tensor(X_w[:, feat_indices], dtype=torch.float32).unsqueeze(0)
                pred = model(x).squeeze(0).numpy()
                X_w  = np.roll(X_w, -1, axis=0)
                X_w[-1, tgt_indices] = pred

            # eval: horizon steps of rollout with noise
            for j in range(EVAL_HORIZON):
                x    = torch.tensor(X_w[:, feat_indices], dtype=torch.float32).unsqueeze(0)
                pred = model(x).squeeze(0).numpy()

                noise      = _sample_noise(pred[amoc_idx], threshold_scaled_eval,
                                           model_vars, cov_on, cov_off, noise_prev, rho, k)
                pred_noisy = pred + np.array([noise[v] for v in model_vars])
                noise_prev = noise

                sample_loss += loss_fn(
                    torch.tensor(pred_noisy[tgt_indices], dtype=torch.float32),
                    torch.tensor(y_w[j, tgt_indices],    dtype=torch.float32),
                ).item()

                X_w = np.roll(X_w, -1, axis=0)
                X_w[-1, tgt_indices] = pred_noisy

        total_loss += sample_loss / EVAL_HORIZON

    model.train()
    return total_loss / n_samples


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    lag               = trial.suggest_categorical('lag',               [100, 150, 200])
    horizon           = trial.suggest_categorical('horizon',           [0, 15, 30])
    n_pretrain_epochs = trial.suggest_categorical('n_pretrain_epochs', [15, 20, 30])
    train_steps       = trial.suggest_categorical('train_steps',       [200, 300])
    lr_pretrain       = trial.suggest_float('lr_pretrain', 1e-4, 5e-3, log=True)
    lr_rollout        = trial.suggest_float('lr_rollout',  1e-5, 5e-4, log=True)
    transition_prob   = trial.suggest_categorical('transition_prob',   [0.7, 0.9])
    integral_alpha    = trial.suggest_float('integral_alpha', 0.0, 1.0)
    sign_alpha        = trial.suggest_float('sign_alpha',     0.0, 1.0)
    hidden_size       = trial.suggest_categorical('hidden_size', [64, 128])
    dropout           = trial.suggest_float('dropout', 0.0, 0.3)

    transition_config = trial.suggest_categorical('transition_config', ['200_12', '250_13'])
    search_ahead      = int(transition_config.split('_')[0])
    threshold_sv      = float(transition_config.split('_')[1])
    threshold_scaled  = threshold_sv / amoc_std

    model = Predictor(
        n_features  = len(feature_cols),
        hidden_size = hidden_size,
        num_layers  = 1,
        dropout     = dropout,
        output_size = len(target_cols),
    ).to(DEVICE)

    # Stage 1: teacher-forcing pre-train
    pretrain(
        X_scaled, train_cut_1, train_cut_2, model,
        feature_cols, target_cols,
        lag=lag, horizon=horizon,
        n_epochs=n_pretrain_epochs, batch_size=64,
        lr=lr_pretrain, loss_fn=nn.MSELoss(),
        device=DEVICE, sign_alpha=sign_alpha,
    )

    # Stage 2: rollout fine-tune
    train_rollout(
        X_scaled, train_cut_1, train_cut_2,
        transition_prob, threshold_scaled, search_ahead, horizon,
        model, feature_cols, target_cols,
        train_steps=train_steps, lag=lag,
        loss_fn=nn.MSELoss(), lr=lr_rollout,
        integral_alpha=integral_alpha,
    )

    # Eval on fixed windows with MSE for fair comparison across trials
    eval_loss = np.mean([_eval_fixed(model, w, lag, horizon) for w in EVAL_WINDOWS])

    print(f"Trial {trial.number}: lag={lag}, horizon={horizon}, pretrain_ep={n_pretrain_epochs}, "
          f"steps={train_steps}, lr_pre={lr_pretrain:.1e}, lr_roll={lr_rollout:.1e}, "
          f"tp={transition_prob}, config={transition_config}, "
          f"int_alpha={integral_alpha:.2f}, sign_alpha={sign_alpha:.2f}, "
          f"hidden={hidden_size}, dropout={dropout:.2f}, eval={eval_loss:.4f}")
    return eval_loss


study = optuna.create_study(
    direction      = 'minimize',
    storage        = 'sqlite:///optuna_pretrained_v3.db',
    study_name     = 'pretrained_v3',
    load_if_exists = True,
)

# Warm start with previous best params mapped to new search space
study.enqueue_trial({
    "lag":               150,
    "horizon":           0,
    "n_pretrain_epochs": 20,
    "train_steps":       200,
    "lr_pretrain":       0.00196,
    "lr_rollout":        9.94e-05,
    "transition_prob":   0.7,
    "integral_alpha":    0.0,
    "sign_alpha":        0.0,
    "hidden_size":       64,
    "dropout":           0.027,
    "transition_config": "200_12",
})

study.optimize(objective, n_trials=100)

print(f"\nBest value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

with open('best_params_pretrained_v3.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
