import optuna
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import json

from model_stuff import Predictor
from Rollout import train_rollout, eval_rollout

X_df       = pd.read_csv('../data/CESMDataset_Try.csv')
model_vars = ['AMOC', 'SFWF', 'PD_200m']
features   = {name: [v for v in model_vars if v != name] for name in model_vars}
n_features = len(model_vars) - 1

train_cut_1  = 5000
train_cut_2  = 11000
eval_horizon = 500
train_steps  = 200
device       = 'cpu'

train_idx = list(range(train_cut_1)) + list(range(train_cut_2, len(X_df)))

scalers  = {}
X_scaled = X_df[model_vars].copy().astype(float)
for var in model_vars:
    sc = StandardScaler().fit(X_df[var].values[train_idx].reshape(-1, 1))
    X_scaled[var] = sc.transform(X_df[var].values.reshape(-1, 1))
    scalers[var]  = sc

amoc_std = scalers['AMOC'].scale_[0]


def objective(trial):
    lag             = trial.suggest_categorical('lag',             [50, 100, 150])
    train_horizon   = trial.suggest_categorical('train_horizon',   [20, 50, 80])

    transition_config = trial.suggest_categorical('transition_config', ['100_7', '150_7', '150_9', '200_9'])
    search_ahead = int(transition_config.split('_')[0])
    threshold_sv = float(transition_config.split('_')[1])

    transition_prob = trial.suggest_categorical('transition_prob', [0.5, 0.7])
    loss_name       = trial.suggest_categorical('loss',            ['mse', 'mae', 'huber'])

    threshold_scaled = threshold_sv / amoc_std
    loss_fn = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.HuberLoss(delta=0.5)}[loss_name]

    model_params = {
        name: {
            'dropout': trial.suggest_float(f'dropout_{name}', 0.0, 0.3),
            'lr':      trial.suggest_float(f'lr_{name}',      1e-4, 1e-2, log=True),
        }
        for name in model_vars
    }

    models = {
        name: Predictor(
            n_features=n_features,
            hidden_size=64,
            num_layers=1,
            dropout=model_params[name]['dropout'],
        ).to(device)
        for name in model_vars
    }

    lrs = {name: model_params[name]['lr'] for name in model_vars}

    train_rollout(
        X_scaled, train_cut_1, train_cut_2,
        transition_prob, threshold_scaled, search_ahead,
        models, features,
        train_steps=train_steps,
        lag=lag,
        train_horizon=train_horizon,
        loss_fn=loss_fn,
        lrs=lrs,
    )

    
    eval_loss = np.mean([
        eval_rollout(
            X_scaled, train_cut_1, train_cut_2,
            transition_prob=1.0,
            threshold=14.0 / amoc_std,   
            search_ahead=eval_horizon,
            models=models, features=features,
            lag=lag, eval_horizon=eval_horizon,
            loss_fn=nn.MSELoss(),
        )
        for _ in range(5)
    ])


    print(f"Trial {trial.number}: lag={lag}, horizon={train_horizon}, search={search_ahead}, "
          f"thr={threshold_sv}, tp={transition_prob}, loss={loss_name}, eval={eval_loss:.4f}")
    return eval_loss


study = optuna.create_study(
    direction='minimize',
    storage='sqlite:///optuna_rollout_v2.db',
    study_name='rollout_v2',
    load_if_exists=True,
)
study.optimize(objective, n_trials=50)

print(f"\nBest value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

with open('best_params_rollout_v2.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
