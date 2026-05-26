import pandas as pd
import torch

#Easy csv loading and torching of the data

def load_and_torch(path, input_cols, target_col, train_frac):
    df = pd.read_csv(path)
    cols = input_cols + [target_col]
    df = df[cols].dropna().reset_index(drop=True)
    stats = {}
    for col in cols:
        mu, sigma = df[col].mean(), df[col].std()
        df[col] = (df[col] - mu) / (sigma) #Constant features may be problematic due to division by zero but not relevant for this assignment
        stats[col] = {"mean": mu, "std": sigma}

    U = torch.tensor(df[input_cols].values[:-1], dtype=torch.float32)
    Y = torch.tensor(df[[target_col]].values[1:], dtype=torch.float32)
    split = int(len(U) * train_frac)
    U_train, U_test = U[:split], U[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    return U_train, U_test, Y_train, Y_test, stats