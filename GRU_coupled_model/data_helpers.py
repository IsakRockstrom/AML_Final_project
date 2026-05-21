import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader



def _create_lagged(X, y, lag, horizon):
    X_lag, y_lag = [], []
    for i in range(lag, len(X) - horizon):
        X_lag.append(X[i-lag:i])
        y_lag.append(y[i + horizon])
    return np.array(X_lag), np.array(y_lag)

def prepare_this_data(df, target_col, feature_cols, train_end, test_end, lag, horizon):

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:test_end], y[train_end:test_end]
    X_eval, y_eval = X[test_end:], y[test_end:]

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled  = x_scaler.transform(X_test)
    X_eval_scaled  = x_scaler.transform(X_eval)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    y_eval_scaled  = y_scaler.transform(y_eval.reshape(-1, 1)).ravel()

    X_train_lag, y_train_lag = _create_lagged(X_train_scaled, y_train_scaled, lag, horizon)
    X_test_lag,  y_test_lag  = _create_lagged(X_test_scaled, y_test_scaled, lag, horizon)
    X_eval_lag,   y_eval_lag   = _create_lagged(X_eval_scaled, y_eval_scaled, lag, horizon)

    return X_train_lag, y_train_lag, X_test_lag, y_test_lag, X_eval_lag, y_eval_lag, y_scaler, x_scaler 

def torchify_this_data(batch_size, X_train_lag, y_train_lag, X_test_lag, y_test_lag, X_eval_lag, y_eval_lag ):
    
    X_train_t = torch.tensor(X_train_lag, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_lag, dtype=torch.float32)

    X_eval_t  = torch.tensor(X_eval_lag,  dtype=torch.float32)
    y_eval_t  = torch.tensor(y_eval_lag,  dtype=torch.float32)

    X_test_t  = torch.tensor(X_test_lag,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_lag,  dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    eval_dataset  = TensorDataset(X_eval_t,  y_eval_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return X_train_t, y_train_t, train_loader, X_test_t, y_test_t, test_loader, X_eval_t, y_eval_t, eval_loader 



