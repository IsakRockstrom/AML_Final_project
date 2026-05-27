import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def _create_lagged(X, y, lag, horizon, train_cut_1 = None):
    X_lag, y_lag = [], []
    for i in range(lag, len(X) - horizon):
        X_lag.append(X[i-lag:i])
        y_lag.append(y[i + horizon])

    X_lag = np.array(X_lag)
    y_lag = np.array(y_lag)

    if train_cut_1 is not None:
        mask = np.ones(len(X_lag), dtype=bool)
        mask[train_cut_1 - lag + 1 : train_cut_1] = False
        X_lag = X_lag[mask]
        y_lag = y_lag[mask]

    return X_lag, y_lag

def prepare_this_data(df, target_col, feature_cols, train_cut_1, train_cut_2, lag, horizon):

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, y_train = np.concatenate([X[:train_cut_1], X[train_cut_2:]]), np.concatenate([y[:train_cut_1], y[train_cut_2:]])
    X_eval, y_eval = X[train_cut_1:train_cut_2], y[train_cut_1:train_cut_2]

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_eval_scaled = x_scaler.transform(X_eval) if len(X_eval) > 0 else X_eval
    y_eval_scaled = y_scaler.transform(y_eval.reshape(-1, 1)).ravel() if len(y_eval) > 0 else y_eval

    X_train_lag, y_train_lag = _create_lagged(X_train_scaled, y_train_scaled, lag, horizon, train_cut_1=train_cut_1)
    X_eval_lag,   y_eval_lag   = _create_lagged(X_eval_scaled, y_eval_scaled, lag, horizon)

    return X_train_lag, y_train_lag, X_eval_lag, y_eval_lag, y_scaler, x_scaler 

def torchify_this_data(batch_size, X_train_lag, y_train_lag, X_eval_lag, y_eval_lag ):
    
    X_train_t = torch.tensor(X_train_lag, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_lag, dtype=torch.float32)

    X_eval_t  = torch.tensor(X_eval_lag,  dtype=torch.float32)
    y_eval_t  = torch.tensor(y_eval_lag,  dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    eval_dataset  = TensorDataset(X_eval_t,  y_eval_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False)

    return X_train_t, y_train_t, train_loader, X_eval_t, y_eval_t, eval_loader 



