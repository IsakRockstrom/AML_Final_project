import optuna
import pandas as pd
import numpy as np
import torch
from torch import nn

from data_helpers import prepare_this_data, torchify_this_data
from model_stuff import model_setup

X          = pd.read_csv('../data/CESMDataset_Try.csv')
variables  = ['AMOC', 'SFWF', 'PD_200m']
train_cut_1 = 5000
train_cut_2 = 11000
batch_size  = 128
n_epochs    = 20
device      = 'cpu'


def objective(trial):

    lag     = trial.suggest_categorical('lag',     [50, 100, 200])
    horizon = trial.suggest_categorical('horizon', [1, 10, 50])

    total_loss = 0.0

    for target in variables:
        feature_cols = [v for v in variables if v != target]

        params = {
            'n_features':  len(feature_cols),
            'hidden_size': trial.suggest_categorical(f'hidden_size_{target}', [32, 64, 128, 256]),
            'num_layers':  trial.suggest_int(f'num_layers_{target}', 1, 3),
            'dropout':     trial.suggest_float(f'dropout_{target}', 0.0, 0.5),
            'lr':          trial.suggest_float(f'lr_{target}', 1e-4, 1e-2, log=True),
            'loss_fn':     nn.MSELoss(),
        }

        X_train_lag, y_train_lag, X_eval_lag, y_eval_lag, _, _ = prepare_this_data(
            X, target, feature_cols, train_cut_1, train_cut_2, lag, horizon
        )
        _, _, train_loader, X_eval_t, _, eval_loader = torchify_this_data(
            batch_size, X_train_lag, y_train_lag, X_eval_lag, y_eval_lag
        )

        model, _, _ = model_setup(device, params, n_epochs, train_loader, eval_loader, eval=False)

        model.eval()
        with torch.no_grad():
            preds = model(X_eval_t.to(device)).cpu().numpy()

        eval_mse = np.mean((preds - y_eval_lag) ** 2)
        total_loss += eval_mse
        print(f"  {target}: eval MSE = {eval_mse:.4f}")

    avg_loss = total_loss / len(variables)
    print(f"Trial {trial.number} done — avg eval MSE: {avg_loss:.4f}")
    return avg_loss


study = optuna.create_study(
    direction='minimize',
    storage='sqlite:///optuna_individual.db',
    study_name='individual_models',
    load_if_exists=True,
)
study.optimize(objective, n_trials=30)

print(f"\nBest value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

import json
with open('best_params_individual.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
