import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

''' 
Data processor to handle cases:

Data cuts:
1. There is no train/eval split
2. There is a single cut
3. There are two train cuts, for selecting a chunk of training data: [....|train|....]
Discontinuities for case 3 are handled in _create lagged

Objective:
1. Predicting absolute values
2. Predicting deltas, i.e the change in fex AMOC from t to t+1
'''

def _create_lagged(X, y, lag, horizon, train_cut_1 = None, delta = False):

    X_lag, y_lag = [], []

    for i in range(lag, len(X) - horizon):
        X_lag.append(X[i-lag:i])
        y_lag.append(y[i + horizon])

    X_lag = np.array(X_lag)
    y_lag = np.array(y_lag)

    if delta:
        y_lag = y_lag - y[lag - 1 : len(X) - horizon - 1]

    if train_cut_1 is not None:
        mask = np.ones(len(X_lag), dtype=bool)
        mask[train_cut_1 - lag + 1 : train_cut_1] = False
        X_lag = X_lag[mask]
        y_lag = y_lag[mask]

    return X_lag, y_lag

def prepare_this_data(df, target_col, feature_cols, train_cut_1, train_cut_2, lag, horizon, delta = False):

    X = df[feature_cols].values
    y = df[target_col].values

    X_eval = np.array([])
    y_eval = np.array([])

    if train_cut_1 is None and train_cut_2 is None:
        X_train, y_train = X, y
    
    elif train_cut_2 is None:
        X_train, y_train = X[:train_cut_1], y[:train_cut_1]
        X_eval, y_eval = X[train_cut_1:], y[train_cut_1:]
        
    else: 
        X_train, y_train = np.concatenate([X[:train_cut_1], X[train_cut_2:]]), np.concatenate([y[:train_cut_1], y[train_cut_2:]])
        X_eval, y_eval = X[train_cut_1:train_cut_2], y[train_cut_1:train_cut_2]

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_eval_scaled = x_scaler.transform(X_eval) if len(X_eval) > 0 else X_eval
    y_eval_scaled = y_scaler.transform(y_eval.reshape(-1, 1)).ravel() if len(y_eval) > 0 else y_eval

    X_train_lag, y_train_lag = _create_lagged(X_train_scaled, y_train_scaled, lag, horizon, train_cut_1=train_cut_1, delta = delta)
    X_eval_lag,   y_eval_lag   = _create_lagged(X_eval_scaled, y_eval_scaled, lag, horizon, delta = delta)

    return X_train_lag, y_train_lag, X_eval_lag, y_eval_lag, y_scaler, x_scaler 


def torchify_this_data(batch_size, X_train_lag, y_train_lag, X_eval_lag=None, y_eval_lag=None):

    X_train_t = torch.tensor(X_train_lag, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_lag, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_eval_lag is not None and len(X_eval_lag) > 0:
        X_eval_t = torch.tensor(X_eval_lag, dtype=torch.float32)
        y_eval_t = torch.tensor(y_eval_lag, dtype=torch.float32)
        eval_dataset = TensorDataset(X_eval_t, y_eval_t)
        eval_loader  = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    else:
        X_eval_t, y_eval_t, eval_loader = None, None, None

    return X_train_t, y_train_t, train_loader, X_eval_t, y_eval_t, eval_loader




