import pandas as pd
import torch
from statsmodels.tsa.stattools import acf
import numpy as np

#Easy csv loading and torching of the data

def load_and_transform(path, features):
    df = pd.read_csv(path)
    df = df[features].dropna().reset_index(drop=True)
    stats = {}
    for col in features:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = (df[col] - mu) / (sigma) #Constant features may be problematic due to division by zero but not relevant for this assignment
        stats[col] = {"mean": mu, "std": sigma}
    return df, stats


def torch_data(df, input_cols, target_col, train_frac):
    U = torch.tensor(df[input_cols].values[:-1], dtype=torch.float32)
    Y = torch.tensor(df[[target_col]].values[1:], dtype=torch.float32)
    split = int(len(U) * train_frac)
    U_train, U_test = U[:split], U[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    return U_train, U_test, Y_train, Y_test


def estimate_covariances(residuals, df_raw, features, threshold_feature, threshold, washout, train_frac=0.8):
    split = int(len(df_raw) * train_frac)
    # Residual row i corresponds to predicting target index washout+i+1 (Y = df[1:]),
    # so align the regime mask to the TARGET, i.e. slice [washout+1 : split+1].
    amoc_raw = df_raw[threshold_feature].values[washout + 1: split + 1]
    # guard against any length mismatch from the +1 shift at the boundary
    n = min(len(amoc_raw), len(residuals))
    amoc_raw  = amoc_raw[:n]
    residuals = residuals[:n]

    on_mask  = amoc_raw > threshold
    off_mask = ~on_mask
    cov_on  = np.cov(residuals[on_mask].T)
    cov_off = np.cov(residuals[off_mask].T)

    rhos = []
    for j, _ in enumerate(features):
        r = acf(residuals[:, j], nlags=1, fft=True)[1]
        rhos.append(r)
    rho = float(np.mean(rhos))

    print(f"  Regime ON samples: {on_mask.sum()}  OFF samples: {off_mask.sum()}")
    print(f"  cov_on  diag (scaled): {np.diag(cov_on)}")
    print(f"  cov_off diag (scaled): {np.diag(cov_off)}")
    print(f"  on/off ratio: {np.diag(cov_on) / np.diag(cov_off)}")
    print(f"  mean rho: {rho:.4f}")

    return cov_on, cov_off, rho
