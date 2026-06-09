import optuna
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import json

from GRU_single.model_stuff import Predictor
from Rollout import train_rollout, _predict
from Pretrain import pretrain
from SingleModel import compute_covariances, _sample_noise


X_df       = pd.read_csv('../data/CESMDataset_Try.csv')
model_vars = ['AMOC', 'SFWF', 'PD_200m']
feature_cols = target_cols = model_vars
train_cut_1, train_cut_2 = 5000, 11000
train_idx = list(range(train_cut_1)) + list(range(train_cut_2, len(X_df)))

y_scalers = {}
X_scaled  = X_df[model_vars].copy().astype(float)
for var in model_vars:
    sc = StandardScaler().fit(X_df[var].values[train_idx].reshape(-1, 1))
    X_scaled[var] = sc.transform(X_df[var].values.reshape(-1, 1))
    y_scalers[var] = sc

amoc_std = y_scalers['AMOC'].scale_[0]
X_np     = X_scaled.values


EVAL_WINDOWS  = [6300, 9000, 6600, 9900]   
EVAL_HORIZON  = 400
DEVICE        = 'cpu'

feat_indices = [list(X_scaled.columns).index(f) for f in feature_cols]
tgt_indices  = [list(X_scaled.columns).index(t) for t in target_cols]

cov_on, cov_off = compute_covariances(X_df, model_vars, threshold=14.0)
amoc_idx = model_vars.index('AMOC')
threshold_scaled_eval = 13.0 / amoc_std

def _eval_fixed(model, start_idx, lag, rho=0.3, k=0.02, n_samples=3):
    model.eval()
    y_w    = X_np[start_idx : start_idx + EVAL_HORIZON]
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for _ in range(n_samples):
        X_w        = X_np[start_idx - lag : start_idx].copy()
        noise_prev = {v: 0.0 for v in model_vars}
        sample_loss = 0.0

        with torch.no_grad():
            for j in range(EVAL_HORIZON):
                x    = torch.tensor(X_w[:, feat_indices], dtype=torch.float32).unsqueeze(0)
                pred = model(x).squeeze(0).numpy()

                noise      = _sample_noise(pred[amoc_idx], threshold_scaled_eval,
                                           model_vars, cov_on, cov_off, noise_prev, rho, k)
                pred_noisy = pred + np.array([noise[v] for v in model_vars])
                noise_prev = noise

                sample_loss += loss_fn(
                    torch.tensor(pred_noisy[tgt_indices], dtype=torch.float32),
                    torch.tensor(y_w[j, tgt_indices], dtype=torch.float32),
                ).item()

                X_w = np.roll(X_w, -1, axis=0)
                X_w[-1, tgt_indices] = pred_noisy

        total_loss += sample_loss / EVAL_HORIZON

    model.train()
    return total_loss / n_samples



def objective(trial):
    lag               = trial.suggest_categorical('lag',               [150, 200])
    n_pretrain_epochs = trial.suggest_categorical('n_pretrain_epochs', [10, 20, 30])
    lr_pretrain       = trial.suggest_float('lr_pretrain', 1e-4, 1e-2, log=True)
    train_horizon     = trial.suggest_categorical('train_horizon',     [100, 120, 150])
    train_steps       = trial.suggest_categorical('train_steps',       [150, 200, 250])
    lr_rollout        = trial.suggest_float('lr_rollout', 1e-5, 1e-3, log=True)
    transition_prob   = trial.suggest_categorical('transition_prob',   [0.7, 0.9])
    dropout           = trial.suggest_float('dropout', 0.0, 0.3)
    loss_name         = 'mse'

   
    transition_config = trial.suggest_categorical('transition_config', ['200_10', '200_12', '250_12', '300_14'])
    search_ahead      = int(transition_config.split('_')[0])
    threshold_sv      = float(transition_config.split('_')[1])
    threshold_scaled  = threshold_sv / amoc_std

    loss_fn = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.HuberLoss(delta=0.5)}[loss_name]

    hidden_size       = trial.suggest_categorical('hidden_size', [64, 128])
    num_layers       = trial.suggest_categorical('num_layers', [1, 2])

    model = Predictor(
        n_features  = len(feature_cols),
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
        output_size = len(target_cols),
    ).to(DEVICE)

    
    pretrain(
        X_scaled, train_cut_1, train_cut_2, model,
        feature_cols, target_cols,
        lag=lag, horizon=1,
        n_epochs=n_pretrain_epochs, batch_size=64,
        lr=lr_pretrain, loss_fn=nn.MSELoss(),
        device=DEVICE,
    )

  
    train_rollout(
        X_scaled, train_cut_1, train_cut_2,
        transition_prob, threshold_scaled, search_ahead,
        model, feature_cols, target_cols,
        train_steps=train_steps, lag=lag,
        train_horizon=train_horizon, loss_fn=loss_fn, lr=lr_rollout,
    )

    
    eval_loss = np.mean([_eval_fixed(model, w, lag) for w in EVAL_WINDOWS])

    print(f"Trial {trial.number}: pretrain_ep={n_pretrain_epochs}, horizon={train_horizon}, "
          f"steps={train_steps}, lr_pre={lr_pretrain:.1e}, lr_roll={lr_rollout:.1e}, "
          f"tp={transition_prob}, config={transition_config}, loss={loss_name}, "
          f"dropout={dropout:.2f}, eval={eval_loss:.4f}")
    return eval_loss


study = optuna.create_study(
    direction    = 'minimize',
    storage    = 'sqlite:///optuna_pretrained_v2.db',
    study_name = 'pretrained_v2',
    load_if_exists = True,
)

study.enqueue_trial({
    "lag": 150,
    "n_pretrain_epochs": 20,
    "lr_pretrain": 0.00196,
    "train_horizon": 120,
    "train_steps": 200,
    "lr_rollout": 9.94e-05,
    "transition_prob": 0.7,
    "dropout": 0.027,
    "loss": "mse",
    "transition_config": "200_12",
    "hidden_size": 64,
    "num_layers": 1,
})
study.optimize(objective, n_trials=50)


print(f"\nBest value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

with open('best_params_pretrained_v1.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
