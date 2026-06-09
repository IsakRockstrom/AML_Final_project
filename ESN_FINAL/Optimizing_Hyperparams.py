import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import itertools
from random import sample


DATA_PATH = "datasets/CombinedDatasetLongUnsmooth.csv"
TRAIN_FRAC = 0.8
WASHOUT = 500
THRESHOLD_HIGH = 11.5
THRESHOLD_LOW =  9.5


INPUT_STRUCTURE = {
    "AMOC":    ["PD_200m", "ICEFRAC"],
    "SFWF":    ["AMOC",    "ICEFRAC"],
    "PD_200m": ["AMOC",    "SFWF",   "ICEFRAC"],
    "ICEFRAC": ["AMOC",    "PD_200m", "SFWF"],
}


def load_and_transform(path, features):
    df = pd.read_csv(path)[features].dropna().reset_index(drop=True)
    stats = {}
    for col in features:
        mu, sigma = df[col].mean(), df[col].std()
        assert sigma > 0, f"Column '{col}' has zero variance"
        df[col] = (df[col] - mu) / sigma
        stats[col] = {"mean": float(mu), "std": float(sigma)}
    return df, stats


def torch_data(df, input_cols, target_col, train_frac):
    U = torch.tensor(df[input_cols].values[:-1], dtype=torch.float32)
    Y = torch.tensor(df[[target_col]].values[1:], dtype=torch.float32)
    split = int(len(U) * train_frac)
    return U[:split], U[split:], Y[:split], Y[split:]


def calculate_nmse(pred, true):
    if isinstance(pred, torch.Tensor):
        return (((pred - true) ** 2).mean() / true.var()).item()
    return float(np.mean((pred - true) ** 2) / np.var(true))


def calculate_rmse(pred, true):
    if isinstance(pred, torch.Tensor):
        pred, true = pred.detach().numpy(), true.detach().numpy()
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def calculate_mae(pred, true):
    if isinstance(pred, torch.Tensor):
        pred, true = pred.detach().numpy(), true.detach().numpy()
    return float(np.mean(np.abs(pred - true)))


class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs=1,
                 spectral_radius=0.9, sparsity=0.9, leaking_rate=0.3,
                 input_scaling=1.0, ridge_alpha=1e-6, seed=42):
        self.n_inputs        = n_inputs
        self.n_reservoir     = n_reservoir
        self.n_outputs       = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity        = sparsity
        self.leaking_rate    = leaking_rate
        self.input_scaling   = input_scaling
        self.ridge_alpha     = ridge_alpha
        torch.manual_seed(seed)
        self._build_reservoir()
        self.W_out = None
        self._x = torch.zeros(n_reservoir)


    def _build_reservoir(self):
        N = self.n_reservoir
        self.W_in = (torch.rand(N, self.n_inputs) * 2 - 1) * self.input_scaling
        mask = (torch.rand(N, N) > self.sparsity).float()
        W_raw = (torch.rand(N, N) * 2 - 1) * mask
        rho = torch.linalg.eigvals(W_raw).abs().max().real
        self.W = W_raw * (self.spectral_radius / rho)


    def _update_state(self, x, u):
        pre = self.W @ x + self.W_in @ u
        return (1 - self.leaking_rate) * x + self.leaking_rate * torch.tanh(pre)


    def _step(self, u):
        self._x = self._update_state(self._x, u)
        return self._x


    def reset_state(self):
        self._x = torch.zeros(self.n_reservoir)


    def _collect_states(self, U, washout):
        self.reset_state()
        states = []
        for t in range(len(U)):
            x = self._step(U[t])
            if t >= washout:
                states.append(x.clone())
        return torch.stack(states)


    def fit(self, U, Y, washout=100):
        X_res      = self._collect_states(U, washout)
        X_ext      = torch.cat([X_res, U[washout:]], dim=1)
        Y_tgt      = Y[washout:]
        n          = X_ext.shape[1]
        A          = X_ext.T @ X_ext + self.ridge_alpha * torch.eye(n)
        self.W_out = torch.linalg.solve(A, X_ext.T @ Y_tgt)
        return self

    def predict_sequence(self, U, washout=0):
        X_res = self._collect_states(U, washout)
        X_ext = torch.cat([X_res, U[washout:]], dim=1)
        return X_ext @ self.W_out



def optimise_hyperparams(target_col, input_cols, df, seeds, n_iter=50,
                          washout=WASHOUT, train_frac=TRAIN_FRAC):

    param_grid = {
        "n_reservoir":     [500, 1000, 2000],
        "spectral_radius": [0.5, 0.7, 0.9, 0.95, 0.99],
        "leaking_rate":    [0.001, 0.003, 0.005, 0.01, 0.05, 0.1],
        "input_scaling":   [0.1, 0.5, 1.0, 2.0],
        "ridge_alpha":     [1e-4, 1e-2, 1e-1, 1.0],
        "sparsity":        [0.9],
    }

    all_configs = [
        dict(zip(param_grid.keys(), v))
        for v in itertools.product(*param_grid.values())
    ]
    sampled = sample(all_configs, min(n_iter, len(all_configs)))
    U_train, U_test, Y_train, Y_test = torch_data(df, input_cols, target_col, train_frac)

    all_results = []

    for i, config in enumerate(sampled):
        seed_metrics = []
        for seed in seeds:
            esn = EchoStateNetwork(
                n_inputs=len(input_cols), n_outputs=1, seed=seed, **config
            )
            esn.fit(U_train, Y_train, washout=washout)
            pred = esn.predict_sequence(U_test, washout=0)
            seed_metrics.append({
                "nmse": calculate_nmse(pred, Y_test),
                "rmse": calculate_rmse(pred.detach().numpy(),
                                       Y_test.detach().numpy()),
                "mae":  calculate_mae (pred.detach().numpy(),
                                       Y_test.detach().numpy()),
            })

        mean_nmse = np.mean([m["nmse"] for m in seed_metrics])
        std_nmse  = np.std ([m["nmse"] for m in seed_metrics])
        mean_rmse = np.mean([m["rmse"] for m in seed_metrics])
        mean_mae  = np.mean([m["mae"]  for m in seed_metrics])

        all_results.append({
            "config":    config,
            "mean_nmse": mean_nmse,
            "std_nmse":  std_nmse,
            "mean_rmse": mean_rmse,
            "mean_mae":  mean_mae,
        })

        print(f"  [{i+1:>3}/{len(sampled)}]  "
              f"NMSE={mean_nmse:.5f} ±{std_nmse:.5f}  "
              f"lr={config['leaking_rate']}  "
              f"rho={config['spectral_radius']}  "
              f"lam={config['ridge_alpha']}  "
              f"N={config['n_reservoir']}")

    all_results.sort(key=lambda x: x["mean_nmse"])
    best = all_results[0]

    print(f"\n  Best NMSE : {best['mean_nmse']:.5f} ± {best['std_nmse']:.5f}")
    print(f"  Best RMSE : {best['mean_rmse']:.4f}")
    print(f"  Best MAE  : {best['mean_mae']:.4f}")
    print(f"  Best config: {best['config']}")

    return best["config"], all_results


def optimise_all(df, seeds=list(range(10)), n_iter=50):

    best_configs = {}
    all_results  = {}

    for target, inputs in INPUT_STRUCTURE.items():
        best_configs[target], all_results[target] = optimise_hyperparams(
            target_col = target,
            input_cols = inputs,
            df         = df,
            seeds      = seeds,
            n_iter     = n_iter,
        )


    print(f"\n  {'Variable':10s}  {'NMSE':>10}  {'Config'}")
    for target, config in best_configs.items():
        nmse = all_results[target][0]["mean_nmse"]
        print(f"  {target:10s}  {nmse:>10.5f}  {config}")

    return best_configs, all_results


if __name__ == "__main__":

    features = list(set(
        [f for inputs in INPUT_STRUCTURE.values() for f in inputs]
        + list(INPUT_STRUCTURE.keys())
    ))

    df, stats = load_and_transform(DATA_PATH, features)

    best_configs, all_results = optimise_all(
        df     = df,
        seeds  = list(range(10)),
        n_iter = 50,
    )


    for target, config in best_configs.items():
        print(f"    \"{target}\": dict(", end="")
        print(", ".join(f"{k}={v}" for k, v in config.items()), end="")
        print("),")
    print("}")