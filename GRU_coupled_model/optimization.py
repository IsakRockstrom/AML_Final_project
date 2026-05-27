import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


from data_helpers import prepare_this_data, torchify_this_data
from model_stuff import model_setup
from CoupledModel import CoupledModel
from objective_function import objective_function

X = pd.read_csv('../data/CESMDataset_Try.csv')

train_cut_1 = len(X)
train_cut_2 = len(X)
batch_size = 128
steps = len(X)
device = "cpu"
models = ['NN_AMOC', 'NN_SFWF', 'NN_PD_200m']

AMOC_features = ['SFWF', 'PD_200m']
SFWF_features = ['AMOC', 'PD_200m']
PD_200m_features = ['SFWF', 'AMOC']

threshold = 13
on = X['AMOC'] > threshold
off = X['AMOC'] < threshold
AMOC_on = X[on]['AMOC']
AMOC_off = X[off]['AMOC']
SFWF_on = X[on]['SFWF']
SFWF_off = X[off]['SFWF']
PD_200m_on = X[on]['PD_200m']
PD_200m_off = X[off]['PD_200m']

data_on  = np.column_stack([AMOC_on, SFWF_on, PD_200m_on])
data_off = np.column_stack([AMOC_off, SFWF_off, PD_200m_off])

cov_on    = np.cov(data_on.T)      
cov_off   = np.cov(data_off.T)



def objective(trial):

    print('starting trial')

    params = {}
    for model in models:
        params[model] = {
            'hidden_size': trial.suggest_categorical(f'hidden_size_{model}', [32, 64, 128, 256]),
            'num_layers':  trial.suggest_int(f'num_layers_{model}', 1, 3),
            'dropout':     trial.suggest_float(f'dropout_{model}', 0.0, 0.5),
            'lr':          trial.suggest_float(f'lr_{model}', 1e-4, 1e-2, log=True),
            'loss_fn':     nn.MSELoss(),
            'n_features':   2
        }
    
    universal_params = {
        'lag':         trial.suggest_categorical(f'lag', [20, 50, 100, 200]),
        'horizon':     trial.suggest_categorical(f'horizon', [1, 20, 50, 100]),
        'rho':         trial.suggest_float(f'rho', 0, 1),
        'k':           trial.suggest_float(f'k', 0, 1.5), 
    }

    model_configs = {
    'NN_AMOC':    {'target_col': 'AMOC',    'feature_cols': ['SFWF', 'PD_200m']},
    'NN_SFWF':    {'target_col': 'SFWF',    'feature_cols': ['AMOC', 'PD_200m']},
    'NN_PD_200m': {'target_col': 'PD_200m', 'feature_cols': ['SFWF', 'AMOC']},
    }

    n_epochs = 20
    trained_models = {}

    for model_name, config in model_configs.items():
        print(f'training model: {model_name}')
        X_train_lag, y_train_lag, X_eval_lag, y_eval_lag, y_scaler, x_scaler = prepare_this_data(
            X, config['target_col'], config['feature_cols'],
            train_cut_1, train_cut_2,
            universal_params['lag'], universal_params['horizon']
        )
        X_train_t, y_train_t, train_loader, X_eval_t, y_eval_t, eval_loader = torchify_this_data(
            batch_size, X_train_lag, y_train_lag, X_eval_lag, y_eval_lag
        )
        model, _, _ = model_setup(device, params[model_name], n_epochs, train_loader, eval_loader, X_eval_t, y_eval_lag)
        trained_models[model_name] = {
            'model':    model,
            'y_scaler': y_scaler,
            'x_scaler': x_scaler,
        }


    coupled = CoupledModel.from_trained(trained_models)

    n_runs = 5
    start_indices = np.linspace(
    universal_params['lag'] + universal_params['horizon'],
    len(X) - universal_params['lag'] - universal_params['horizon'] - 1,
    n_runs,
    dtype=int
    )

    all_AMOC = []
    for start_idx in start_indices:
        print(f'running coupled model trial: {start_idx}')
        AMOC, _, _ = coupled.predict(device, 
                                    X, 
                                    start_idx, 
                                    universal_params['lag'], 
                                    universal_params['horizon'], 
                                    AMOC_features, 
                                    SFWF_features, 
                                    PD_200m_features, 
                                    steps, 
                                    threshold, 
                                    cov_on, 
                                    cov_off, 
                                    universal_params['rho'], 
                                    universal_params['k'])
        all_AMOC.extend(AMOC)



    AMOC_pred = np.array(all_AMOC)
    AMOC_true = X['AMOC'].values
    print('computing objective')
    obj = objective_function(AMOC_pred, AMOC_true)

    return obj

study = optuna.create_study(direction='minimize', storage='sqlite:///optuna_study.db', study_name='coupled_model', load_if_exists=True)
study.optimize(objective, n_trials=90)

best_params = study.best_params
with open('best_params.json', 'w') as f:
    import json
    json.dump(best_params, f, indent=2)
