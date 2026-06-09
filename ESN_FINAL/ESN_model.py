import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.signal import welch


DATA_PATH = "datasets/CombinedDatasetLongUnsmooth.csv"
TRAIN_FRAC = 0.8
WASHOUT = 500
THRESHOLD_HIGH = 12
THRESHOLD_LOW = 9
RHO = 0.5
K = 0.5

INPUT_STRUCTURE = {
    "AMOC": ["PD_200m", "ICEFRAC"],
    "SFWF": ["AMOC",    "ICEFRAC"],
    "PD_200m": ["AMOC",    "SFWF",   "ICEFRAC"],
    "ICEFRAC": ["AMOC",    "PD_200m", "SFWF"],
}

CONFIGS = {
    "AMOC":dict(n_reservoir=2000, spectral_radius=0.5, sparsity=0.9,
                    leaking_rate=0.05, input_scaling=1.0, ridge_alpha=1e-1),
    "SFWF":dict(n_reservoir=500,  spectral_radius=0.7, sparsity=0.9,
                    leaking_rate=0.1,  input_scaling=1.0, ridge_alpha=1e-1),
    "PD_200m":dict(n_reservoir=500,  spectral_radius=0.95, sparsity=0.9,
                    leaking_rate=0.01, input_scaling=1.0, ridge_alpha=1),
    "ICEFRAC":dict(n_reservoir=500,  spectral_radius=0.7, sparsity=0.9,
                    leaking_rate=0.1,  input_scaling=1.0, ridge_alpha=1e-1),
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


def inverse_transform(y, stats, col):
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()
    return y * stats[col]["std"] + stats[col]["mean"]


#ERRORS

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

#ESN ARCHITECTURE

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs=1,
                 spectral_radius=0.9, sparsity=0.9, leaking_rate=0.3,
                 input_scaling=1.0, ridge_alpha=1e-6, seed=42):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha
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
        X_res = self._collect_states(U, washout)
        X_ext = torch.cat([X_res, U[washout:]], dim=1)
        Y_tgt = Y[washout:]
        n = X_ext.shape[1]
        A = X_ext.T @ X_ext + self.ridge_alpha * torch.eye(n)
        self.W_out = torch.linalg.solve(A, X_ext.T @ Y_tgt)
        return self

    def predict_one(self, u):
        x = self._step(u)
        x_ext = torch.cat([x, u])
        return x_ext @ self.W_out

    def predict_sequence(self, U, washout=0):
        X_res = self._collect_states(U, washout)
        X_ext = torch.cat([X_res, U[washout:]], dim=1)
        return X_ext @ self.W_out

    @property
    def actual_spectral_radius(self):
        return torch.linalg.eigvals(self.W).abs().max().real


#STATE CHECKER AND NOISE

def get_regime_hysteresis(amoc_vals, threshold_high, threshold_low):
    regime = np.zeros(len(amoc_vals), dtype=int)
    current = 1 if amoc_vals[0] > threshold_high else 0
    for t, val in enumerate(amoc_vals):
        if current == 0 and val > threshold_high:
            current = 1
        elif current == 1 and val < threshold_low:
            current = 0
        regime[t] = current
    return regime


def sample_noise(current_regime, cov_on, cov_off, noise_prev, rho, k):
    n_vars = cov_on.shape[0]
    cov = (cov_on if current_regime == 1 else cov_off) * k
    stds = np.sqrt(np.diag(cov))
    raw = np.random.multivariate_normal(mean=np.zeros(n_vars), cov=cov)
    clipped = np.where(np.abs(raw) < stds, raw, stds * np.sign(raw))
    return np.sqrt(1 - rho ** 2) * clipped + rho * noise_prev


def estimate_covariances(residuals, df_raw, features, threshold_feature,
                          threshold_high, threshold_low, washout, train_frac=0.8):
    n = len(features)
    amoc_vals = df_raw[threshold_feature].values
    regime = np.array(get_regime_hysteresis(amoc_vals, threshold_high, threshold_low))
    regime = regime[int(washout):int(washout) + len(residuals)]
    on_mask = regime == 1
    off_mask = regime == 0
    if on_mask.sum() < n + 1 or off_mask.sum() < n + 1:
        raise ValueError("Not enough samples in one regime — adjust thresholds.")
    cov_on  = np.cov(residuals[on_mask].T)
    cov_off = np.cov(residuals[off_mask].T)
    rhos = []
    for j, feature in enumerate(features):
        r = acf(residuals[:, j], nlags=1, fft=True)[1]
        rhos.append(r)
        print(f"  AR(1) rho [{feature}]: {r:.4f}")
    rho = float(np.mean(rhos))
    print(f"  Mean rho: {rho:.4f}")
    return cov_on, cov_off, rho

def print_regime_stats(values, threshold_high, threshold_low, label=""):
    regime = get_regime_hysteresis(np.array(values), threshold_high, threshold_low)
    frac_high = regime.mean()
    changes = np.diff(regime)
    t_idx = np.where(changes != 0)[0]
    res_times = np.diff(t_idx)
    high_t = res_times[regime[t_idx[:-1]] == 1]
    low_t = res_times[regime[t_idx[:-1]] == 0]


#COUPLED ESN ARCHITECTURE

class CoupledESN:
    def __init__(self, input_structure, configs):
        self.input_structure  = input_structure
        self.features = list(input_structure.keys())
        self.esns = {}
        self.test_data = {}
        self.feature_pointers = {}
        self.stats = None
        for i, feature in enumerate(self.features):
            self.feature_pointers[feature] = i
        seeds = np.arange(69, 69 + len(self.features))
        for feature, seed in zip(self.features, seeds):
            n_inputs = len(input_structure[feature])
            self.esns[feature] = EchoStateNetwork(
                n_inputs=n_inputs, **configs[feature], seed=int(seed))

    def fit(self, df, stats, washout, train_frac=0.8):
        self.stats = stats
        for feature, esn in self.esns.items():
            input_cols = self.input_structure[feature]
            U_train, U_test, Y_train, Y_test = torch_data(df, input_cols, feature, train_frac)
            esn.fit(U_train, Y_train, washout=washout)
            self.test_data[feature] = {
                "features": {"inputs": input_cols, "output": feature},
                "data":     {"U_train": U_train, "U_test": U_test,
                             "Y_train": Y_train, "Y_test": Y_test}}
        return self

    def get_residuals(self, washout):
        if not self.test_data:
            raise RuntimeError("Call fit() before get_residuals().")
        n = len(self.esns)
        T = len(self.test_data[self.features[0]]["data"]["U_train"]) - washout
        residuals = np.zeros((T, n))
        for j, feature in enumerate(self.features):
            U_train = self.test_data[feature]["data"]["U_train"]
            Y_train = self.test_data[feature]["data"]["Y_train"]
            pred = self.esns[feature].predict_sequence(U_train, washout=washout)
            true = Y_train[washout:]
            res = (pred - true).detach().numpy().reshape(-1)
            residuals[:, j] = res * self.stats[feature]["std"]
        return residuals

    def evaluate(self, washout=0):
        if not self.test_data:
            raise RuntimeError("Call fit() before evaluate().")
        results = {}
        for feature, esn in self.esns.items():
            U_test = self.test_data[feature]["data"]["U_test"]
            Y_test = self.test_data[feature]["data"]["Y_test"]
            pred = esn.predict_sequence(U_test, washout=washout)
            results[feature] = (pred, Y_test[washout:])
        return results

    def rollout(self, init_window, steps, threshold_high, threshold_low,
                threshold_feature, cov_on, cov_off, rho, k=1.0):
        if self.stats is None:
            raise RuntimeError("Call fit() before rollout().")
        stats = self.stats
        for esn in self.esns.values():
            esn.reset_state()
        for t in range(len(init_window)):
            row = init_window[t]
            for feature, esn in self.esns.items():
                inputs = self.input_structure[feature]
                u = torch.tensor(
                    [row[self.feature_pointers[inp]] for inp in inputs],
                    dtype=torch.float32
                )
                esn._step(u)
        last = init_window[-1]
        current_scaled = {}
        for feature, esn in self.esns.items():
            inputs = self.input_structure[feature]
            u      = torch.tensor(
                [last[self.feature_pointers[inp]] for inp in inputs],
                dtype=torch.float32
            )
            current_scaled[feature] = esn.predict_one(u).item()

        def unscale(val, feature):
            return val * stats[feature]["std"] + stats[feature]["mean"]

        feature_list = list(self.esns.keys())
        results = {f: [unscale(current_scaled[f], f)] for f in feature_list}
        noise_prev = np.zeros(len(feature_list))
        amoc_init = unscale(current_scaled[threshold_feature], threshold_feature)
        current_regime = 1 if amoc_init > threshold_high else 0

        PHYSICAL_BOUNDS = {f: (-3.0, 3.0) for f in feature_list}

        for _ in range(steps):
            amoc_unscaled = unscale(current_scaled[threshold_feature], threshold_feature)
            if current_regime == 0 and amoc_unscaled > threshold_high:
                current_regime = 1
            elif current_regime == 1 and amoc_unscaled < threshold_low:
                current_regime = 0
            noise = sample_noise(current_regime, cov_on, cov_off, noise_prev, rho, k)
            for i, feature in enumerate(feature_list):
                current_scaled[feature] += noise[i] / stats[feature]["std"]
                lo, hi = PHYSICAL_BOUNDS[feature]
                current_scaled[feature] = float(np.clip(current_scaled[feature], lo, hi))
            noise_prev = noise
            next_scaled = {}
            for feature, esn in self.esns.items():
                inputs = self.input_structure[feature]
                u = torch.tensor(
                    [current_scaled[inp] for inp in inputs],
                    dtype=torch.float32
                )
                next_scaled[feature] = esn.predict_one(u).item()
            current_scaled = next_scaled
            for feature in feature_list:
                results[feature].append(unscale(current_scaled[feature], feature))
        return results


#ENSEMBLE OF FEATURES

def run_ensemble(model, df, df_raw, n_ensemble, steps,
                 threshold_high, threshold_low, threshold_feature,
                 cov_on, cov_off, rho, k, washout):
    features = model.features
    amoc_vals = df_raw[threshold_feature].values
    transitions = np.where(np.diff((amoc_vals > threshold_low).astype(int)))[0]
    low_starts = transitions[::2] [:n_ensemble // 2]
    high_starts = transitions[1::2][:n_ensemble // 2]
    start_idxs = np.concatenate([low_starts, high_starts])
    ensemble = {f: [] for f in features}
    for start in start_idxs:
        init_start = max(0, start - washout + 20)
        init_window = df.values[init_start:init_start + washout]
        if len(init_window) < washout:
            continue
        result = model.rollout(
            init_window=init_window, steps=steps,
            threshold_high=threshold_high, threshold_low=threshold_low,
            threshold_feature=threshold_feature,
            cov_on=cov_on, cov_off=cov_off, rho=rho, k=k,
        )
        for f in features:
            ensemble[f].append(result[f])
    return {f: np.array(v) for f, v in ensemble.items()}


#TEACHER FORCED ESN

def run_esn_ar1(INPUT_COLS, TARGET_COL, seed,
                U_train, U_test, Y_train, Y_test, stats,
                WASHOUT=500, plot=False,
                ESN_CONFIG=None,
                threshold_high=THRESHOLD_HIGH, threshold_low=THRESHOLD_LOW,
                rho=0.5, k=0.5,
                cov_on=None, cov_off=None):
    if ESN_CONFIG is None:
        ESN_CONFIG = dict(n_reservoir=2000, spectral_radius=0.5, sparsity=0.9,
                          leaking_rate=0.05, input_scaling=1.0, ridge_alpha=1e-1)
    if cov_on  is None: cov_on  = np.array([[1.0]])
    if cov_off is None: cov_off = np.array([[1.0]])

    esn = EchoStateNetwork(n_inputs=len(INPUT_COLS), n_outputs=1, **ESN_CONFIG, seed=seed)
    esn.fit(U_train, Y_train, washout=WASHOUT)

    mu  = stats[TARGET_COL]["mean"]
    std = stats[TARGET_COL]["std"]
    threshold_high_scaled = (threshold_high - mu) / std
    threshold_low_scaled  = (threshold_low  - mu) / std

    X_res = esn._collect_states(U_test, washout=0)
    X_ext = torch.cat([X_res, U_test], dim=1)
    Y_pred_clean = X_ext @ esn.W_out

    T = len(Y_pred_clean)
    noise_prev = np.zeros(1)
    current_regime = 1 if Y_pred_clean[0].item() > threshold_high_scaled else 0
    Y_pred_noisy = torch.zeros_like(Y_pred_clean)

    for t in range(T):
        pred_scaled = Y_pred_clean[t].item()
        if current_regime == 0 and pred_scaled > threshold_high_scaled:
            current_regime = 1
        elif current_regime == 1 and pred_scaled < threshold_low_scaled:
            current_regime = 0
        cov_t = (cov_on if current_regime == 1 else cov_off) * k
        std_t = float(np.sqrt(cov_t[0, 0]))
        raw = np.random.normal(0, std_t)
        clipped = np.clip(raw, -std_t, std_t)
        noise = float(np.sqrt(1 - rho ** 2) * clipped + rho * noise_prev[0])
        noise_prev = np.array([noise])
        Y_pred_noisy[t] = pred_scaled + noise

    Y_pred_raw = inverse_transform(Y_pred_noisy, stats, TARGET_COL).squeeze()
    Y_test_raw = inverse_transform(Y_test, stats, TARGET_COL).squeeze()

    metrics = {
        "nmse": calculate_nmse(Y_pred_noisy, Y_test),
        "rmse": calculate_rmse(Y_pred_raw,   Y_test_raw),
        "mae":  calculate_mae (Y_pred_raw,   Y_test_raw),
    }
    return esn, metrics, Y_pred_noisy

#PLOTS

def plot_evaluation(eval_results, stats, features, out_path="coupled_eval.png"):
    n_vars = len(features)
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(n_vars, 1, figsize=(13, 4 * n_vars), sharex=True)
    if n_vars == 1: axes = [axes]
    fig.suptitle("Coupled ESN — teacher-forced evaluation", fontsize=13)
    for ax, color, (var, (pred, true)) in zip(axes, colors, eval_results.items()):
        mu  = stats[var]["mean"]
        std = stats[var]["std"]
        p = pred.detach().numpy().squeeze() * std + mu
        t = true.detach().numpy().squeeze() * std + mu
        err = calculate_nmse(pred, true)
        ax.plot(t, label="Target", color="black", lw=1.2, alpha=0.7)
        ax.plot(p, label="ESN", color=color, lw=1.2, linestyle="--")
        ax.set_ylabel(var)
        ax.set_title(f"{var}  (NMSE={err:.5f})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Test time step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ensemble(ensemble, features, stats, out_path="coupled_ensemble.png"):
    n_vars = len(features)
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(n_vars, 1, figsize=(13, 4 * n_vars), sharex=True)
    if n_vars == 1: axes = [axes]
    fig.suptitle("Coupled ESN — ensemble rollout", fontsize=13)
    display_bounds = {
        "AMOC":    (4.0,    22.0),
        "SFWF":    (-3e-5,  1e-4),
        "PD_200m": (1.0282, 1.0292),
        "ICEFRAC": (0.0,    0.8),
    }
    for ax, color, feature in zip(axes, colors, features):
        trajs = ensemble[feature]
        t = np.arange(trajs.shape[1])
        for traj in trajs:
            ax.plot(t, traj, color=color, lw=0.5, alpha=0.3)
        ax.plot(t, trajs.mean(axis=0), color="black", lw=1.5, label="mean")
        ax.set_ylabel(feature)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if feature in display_bounds:
            ax.set_ylim(*display_bounds[feature])
    axes[-1].set_xlabel("Rollout step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_distribution_and_psd(Y_pred_raw, Y_test_raw, target_col,
                               dt_years=1/12, out_path=None, clip_bounds=None):
    if isinstance(Y_pred_raw, torch.Tensor):
        Y_pred_raw = Y_pred_raw.detach().numpy().squeeze()
    if isinstance(Y_test_raw, torch.Tensor):
        Y_test_raw = Y_test_raw.detach().numpy().squeeze()
    if clip_bounds is not None:
        lo, hi = clip_bounds
        Y_pred_clipped = Y_pred_raw[(Y_pred_raw >= lo) & (Y_pred_raw <= hi)]
        print(f"  Clipped {len(Y_pred_raw) - len(Y_pred_clipped)} "
              f"/ {len(Y_pred_raw)} values outside ({lo}, {hi})")
    else:
        Y_pred_clipped = Y_pred_raw
    w_dist = wasserstein_distance(Y_test_raw, Y_pred_clipped)
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    bins = np.linspace(
        min(Y_test_raw.min(), Y_pred_clipped.min()),
        max(Y_test_raw.max(), Y_pred_clipped.max()), 40
    )
    axes[0].hist(Y_test_raw,    bins=bins, density=True, alpha=0.4,
                 color="gray", label="True (hist)")
    axes[0].hist(Y_pred_clipped, bins=bins, density=True, alpha=0.4,
                 color="red",  label="Model (hist)")
    x_grid = np.linspace(bins[0], bins[-1], 500)
    axes[0].plot(x_grid, gaussian_kde(Y_test_raw)(x_grid),
                 color="black", lw=2, linestyle="-",  label="True (fit)")
    axes[0].plot(x_grid, gaussian_kde(Y_pred_clipped)(x_grid),
                 color="red",   lw=2, linestyle="--", label="Model (fit)")
    axes[0].set_title(f"Distribution — Wasserstein distance = {w_dist:.3f}")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    nperseg = len(Y_test_raw)
    fs = 1.0 / dt_years
    f_t, psd_t = welch(Y_test_raw,  fs=fs, nperseg=nperseg)
    f_m, psd_m = welch(Y_pred_raw,  fs=fs, nperseg=nperseg)
    axes[1].loglog(f_t[1:], psd_t[1:], color="black", lw=1.2,
                   linestyle="-",  label="True")
    axes[1].loglog(f_m[1:], psd_m[1:], color="red",   lw=1.2,
                   linestyle="--", label="Model", alpha=0.85)
    axes[1].set_title(f"Power Spectral Density — {target_col}")
    axes[1].set_xlabel("Frequency [cycles / year]")
    axes[1].set_ylabel("PSD")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Wasserstein distance: {w_dist:.4f}")
    return w_dist

def run_noise_ablation(model, df, df_raw, stats, cov_on, cov_off,
                       threshold_high, threshold_low, threshold_feature,
                       washout, steps=6000):
    amoc_vals = df_raw[threshold_feature].values
    regime = get_regime_hysteresis(amoc_vals, threshold_high, threshold_low)
    changes = np.diff(regime)
    low_to_high = np.where(changes ==  1)[0]
    high_to_low = np.where(changes == -1)[0]
    print(f"  Low→High transitions: {len(low_to_high)}")
    print(f"  High→Low transitions: {len(high_to_low)}")
    low_end = low_to_high[0]
    low_start = max(0, low_end - washout)
    init_low = df.values[low_start:low_end]
    high_start = low_to_high[0] + 1
    high_end = min(len(df), high_start + washout)
    init_high = df.values[high_start:high_end]
    if len(init_low)  < washout: init_low  = df.values[:washout]
    if len(init_high) < washout: init_high = df.values[:washout]
    print(f"  Low  IC — AMOC range: {amoc_vals[low_start:low_end].min():.2f} "
          f"to {amoc_vals[low_start:low_end].max():.2f} Sv")
    print(f"  High IC — AMOC range: {amoc_vals[high_start:high_end].min():.2f} "
          f"to {amoc_vals[high_start:high_end].max():.2f} Sv")

    def rollout_pair(k, rho):
        low  = model.rollout(init_low,  steps=steps,
                             threshold_high=threshold_high, threshold_low=threshold_low,
                             threshold_feature=threshold_feature,
                             cov_on=cov_on, cov_off=cov_off, rho=rho, k=k)
        high = model.rollout(init_high, steps=steps,
                             threshold_high=threshold_high, threshold_low=threshold_low,
                             threshold_feature=threshold_feature,
                             cov_on=cov_on, cov_off=cov_off, rho=rho, k=k)
        return low, high

    low_nonoise, high_nonoise = rollout_pair(k=0.0, rho=0.0)
    low_noise, high_noise = rollout_pair(k=0.3, rho=0.0)
    low_ar1, high_ar1 = rollout_pair(k=0.5, rho=0.5)

    true_amoc = amoc_vals[:steps]
    t = np.arange(steps)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("ESN coupled rollout — role of noise", fontsize=13)
    configs = [
        (low_nonoise, high_nonoise, "No noise (k=0)"),
        (low_noise,   high_noise,   "White noise (k=0.3, ρ=0)"),
        (low_ar1,     high_ar1,     "AR(1) noise (k=0.5, ρ=0.5)"),
    ]
    for ax, (res_low, res_high, title) in zip(axes, configs):
        ax.plot(t, true_amoc, color="gray", lw=0.8, alpha=0.5, label="True AMOC")
        ax.plot(t, np.array(res_low ["AMOC"][:steps]), color="green", lw=1.2, label="Low IC")
        ax.plot(t, np.array(res_high["AMOC"][:steps]), color="red",   lw=1.2, label="High IC")
        ax.axhline(threshold_high, color="black", lw=0.8, linestyle="--", alpha=0.5)
        ax.axhline(threshold_low,  color="black", lw=0.8, linestyle=":",  alpha=0.5)
        ax.set_ylim(4.0, 22.0)
        ax.set_title(title)
        ax.set_ylabel("AMOC [Sv]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Rollout step")
    plt.tight_layout()
    plt.savefig("esn_noise_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    return {"no_noise": (low_nonoise, high_nonoise),
            "white_noise": (low_noise, high_noise),
            "ar1_noise": (low_ar1, high_ar1)}


def compare_noise_configs(model, init_window, df_raw, stats,
                           cov_on, cov_off, threshold_high, threshold_low,
                           threshold_feature, n_runs=50, steps=6000):
    configs = [
        {"k": 0.0, "rho": 0.0, "label": "No noise"},
        {"k": 0.3, "rho": 0.0, "label": "White noise (k=0.3)"},
        {"k": 0.3, "rho": 0.5, "label": "AR(1) (k=0.3, ρ=0.5)"},
        {"k": 0.5, "rho": 0.5, "label": "AR(1) (k=0.5, ρ=0.5)"},
    ]
    real_frac = get_regime_hysteresis(
        df_raw[threshold_feature].values, threshold_high, threshold_low
    ).mean()
    print(f"  Real high fraction: {real_frac:.3f}")
    print(f"\n  {'Config':<25} {'Mean frac':>10} {'Std frac':>10} {'Mean transitions':>18}")
    print(f"  {'-'*65}")
    results = {}
    for cfg in configs:
        fracs, n_trans = [], []
        for _ in range(n_runs):
            result = model.rollout(
                init_window, steps=steps,
                threshold_high=threshold_high, threshold_low=threshold_low,
                threshold_feature=threshold_feature,
                cov_on=cov_on, cov_off=cov_off,
                rho=cfg["rho"], k=cfg["k"],
            )
            traj = np.array(result[threshold_feature])
            regime = get_regime_hysteresis(traj, threshold_high, threshold_low)
            fracs.append(regime.mean())
            n_trans.append(np.abs(np.diff(regime)).sum())
        results[cfg["label"]] = {"fracs": fracs, "n_trans": n_trans}
        print(f"  {cfg['label']:<25} {np.mean(fracs):>10.3f} "
              f"{np.std(fracs):>10.3f} {np.mean(n_trans):>18.1f}")
    return results, real_frac

def evaluate_transitions(model, df, df_raw, stats, cov_on, cov_off,
                          threshold_high, threshold_low, threshold_feature,
                          washout=500, steps_after=500, k=0.5, rho=0.5):
    """
    For every regime transition, initialise the model washout steps before
    the transition and run forward for steps_after steps.
    Plots predicted vs true AMOC for all low→high and high→low transitions.
    """
    INIT_BEFORE = 50

    amoc_vals = df_raw[threshold_feature].values
    regime    = get_regime_hysteresis(amoc_vals, threshold_high, threshold_low)
    changes   = np.diff(regime)

    low_to_high = np.where(changes ==  1)[0]
    high_to_low = np.where(changes == -1)[0]

    print(f"  Low→High transitions: {len(low_to_high)}")
    print(f"  High→Low transitions: {len(high_to_low)}")

    def run_transition(trans_idx):
        window_end   = trans_idx - INIT_BEFORE
        window_start = max(0, window_end - washout)
        if window_end - window_start < washout:
            return None, None
        init_window  = df.values[window_start:window_end]
        total_steps  = INIT_BEFORE + steps_after
        result       = model.rollout(
            init_window       = init_window,
            steps             = total_steps,
            threshold_high    = threshold_high,
            threshold_low     = threshold_low,
            threshold_feature = threshold_feature,
            cov_on            = cov_on,
            cov_off           = cov_off,
            rho               = rho,
            k                 = k,
        )
        true_start  = window_end
        true_end    = min(len(amoc_vals), true_start + total_steps)
        true_window = amoc_vals[true_start:true_end]
        pred_window = np.array(result[threshold_feature][:len(true_window)])
        return pred_window, true_window

    def plot_transitions(indices, direction, out_path):
        results = []
        for idx in indices:
            pred, true = run_transition(idx)
            if pred is not None:
                results.append((pred, true))

        if not results:
            print(f"  No valid {direction} transitions found.")
            return

        n     = len(results)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        t     = np.arange(INIT_BEFORE + steps_after)

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4 * nrows),
                                  sharex=True, sharey=True)
        axes = np.array(axes).flatten()
        fig.suptitle(f"Transition evaluation — {direction} "
                     f"({INIT_BEFORE} steps before, {steps_after} after)",
                     fontsize=13)

        for i, (pred, true) in enumerate(results):
            ax = axes[i]
            ax.plot(t[:len(true)], true, color="black", lw=1.2,
                    alpha=0.8, label="True")
            ax.plot(t[:len(pred)], pred, color="red",   lw=1.2,
                    linestyle="--", label="Model")
            ax.axvline(INIT_BEFORE, color="blue", lw=0.8,
                       linestyle=":", alpha=0.7, label="Transition point")
            ax.axhline(threshold_high, color="gray", lw=0.7, linestyle="--")
            ax.axhline(threshold_low,  color="gray", lw=0.7, linestyle=":")
            ax.set_ylim(4.0, 20.0)
            ax.set_title(f"Event {i+1}")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)

        # hide unused axes
        for j in range(len(results), len(axes)):
            axes[j].set_visible(False)

        for ax in axes[(nrows-1)*ncols:]:
            ax.set_xlabel("Steps from init")
        for ax in axes[::ncols]:
            ax.set_ylabel("AMOC [Sv]")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved → {out_path}")

    plot_transitions(low_to_high, "Low → High", "transitions_low_to_high.png")
    plot_transitions(high_to_low, "High → Low", "transitions_high_to_low.png")


if __name__ == "__main__":

    features = list(INPUT_STRUCTURE.keys())
    df_raw = pd.read_csv(DATA_PATH)[features]
    df, stats = load_and_transform(DATA_PATH, features)
    model = CoupledESN(INPUT_STRUCTURE, CONFIGS)
    model.fit(df, stats, washout=WASHOUT)
    residuals = model.get_residuals(washout=WASHOUT)
    cov_on, cov_off, rho = estimate_covariances(
        residuals = residuals,
        df_raw = df_raw,
        features = features,
        threshold_feature = "AMOC",
        threshold_high = THRESHOLD_HIGH,
        threshold_low = THRESHOLD_LOW,
        washout = WASHOUT,
        train_frac = TRAIN_FRAC,
    )
    cov_on_amoc = np.array([[cov_on [0, 0]]])
    cov_off_amoc = np.array([[cov_off[0, 0]]])

    
    eval_results = model.evaluate()
    print(f"\n  {'Variable':10s}  {'NMSE':>10}  {'RMSE':>10}  {'MAE':>10}")
    for var, (pred, true) in eval_results.items():
        mu, std = stats[var]["mean"], stats[var]["std"]
        p_raw = pred.detach().numpy().squeeze() * std + mu
        t_raw = true.detach().numpy().squeeze() * std + mu
        print(f"  {var:10s}  {calculate_nmse(pred, true):>10.5f}  "
              f"{calculate_rmse(p_raw, t_raw):>10.4f}  "
              f"{calculate_mae(p_raw, t_raw):>10.4f}")

    plot_evaluation(eval_results, stats, features, out_path="esn_teacher_forced.png")
    pred_amoc, true_amoc = eval_results["AMOC"]
    mu, std = stats["AMOC"]["mean"], stats["AMOC"]["std"]
    plot_distribution_and_psd(
        Y_pred_raw = pred_amoc.detach().numpy().squeeze() * std + mu,
        Y_test_raw = true_amoc.detach().numpy().squeeze() * std + mu,
        target_col = "AMOC",
        dt_years = 1/12,
        out_path = "esn_tf_dist_psd.png",
    )


    U_train_amoc = model.test_data["AMOC"]["data"]["U_train"]
    U_test_amoc  = model.test_data["AMOC"]["data"]["U_test"]
    Y_train_amoc = model.test_data["AMOC"]["data"]["Y_train"]
    Y_test_amoc  = model.test_data["AMOC"]["data"]["Y_test"]
    esn_ar1, metrics_ar1, Y_pred_noisy = run_esn_ar1(
        INPUT_COLS = INPUT_STRUCTURE["AMOC"],
        TARGET_COL = "AMOC",
        seed = 0,
        U_train = U_train_amoc,
        U_test = U_test_amoc,
        Y_train = Y_train_amoc,
        Y_test = Y_test_amoc,
        stats = stats,
        WASHOUT = WASHOUT,
        ESN_CONFIG = CONFIGS["AMOC"],
        threshold_high = THRESHOLD_HIGH,
        threshold_low  = THRESHOLD_LOW,
        rho = RHO,
        k = K,
        cov_on = cov_on_amoc,
        cov_off = cov_off_amoc,
    )

    nmse_clean = calculate_nmse(eval_results["AMOC"][0], eval_results["AMOC"][1])
    print(f"  Clean ESN  — NMSE={nmse_clean:.5f}")
    print(f"  AR(1) ESN  — NMSE={metrics_ar1['nmse']:.5f}  "
          f"RMSE={metrics_ar1['rmse']:.4f}  MAE={metrics_ar1['mae']:.4f}")

    Y_pred_noisy_raw = inverse_transform(Y_pred_noisy, stats, "AMOC").squeeze()
    Y_test_raw_amoc  = inverse_transform(Y_test_amoc,  stats, "AMOC").squeeze()

    plot_distribution_and_psd(
        Y_pred_raw = Y_pred_noisy_raw,
        Y_test_raw = Y_test_raw_amoc,
        target_col = "AMOC",
        dt_years = 1/12,
        clip_bounds = (4.0, 17.0),
        out_path = "esn_ar1_tf_dist_psd.png",
    )


    run_noise_ablation(
        model = model,
        df = df,
        df_raw = df_raw,
        stats = stats,
        cov_on = cov_on,
        cov_off = cov_off,
        threshold_high = THRESHOLD_HIGH,
        threshold_low = THRESHOLD_LOW,
        threshold_feature = "AMOC",
        washout = WASHOUT,
        steps = 6000,
    )


    amoc_vals = df_raw["AMOC"].values
    transitions = np.where(np.diff((amoc_vals > THRESHOLD_LOW).astype(int)))[0]
    init_start = max(0, transitions[0] - WASHOUT + 50)
    init_window = df.values[init_start:init_start + WASHOUT]

    noise_results, real_frac = compare_noise_configs(
        model  = model,
        init_window = init_window,
        df_raw = df_raw,
        stats = stats,
        cov_on = cov_on,
        cov_off = cov_off,
        threshold_high = THRESHOLD_HIGH,
        threshold_low = THRESHOLD_LOW,
        threshold_feature = "AMOC",
        n_runs = 50,
        steps = 6000,
    )

    
    ensemble = run_ensemble(
        model = model,
        df = df,
        df_raw = df_raw,
        n_ensemble = 50,
        steps = 2000,
        threshold_high = THRESHOLD_HIGH,
        threshold_low = THRESHOLD_LOW,
        threshold_feature = "AMOC",
        cov_on = cov_on,
        cov_off = cov_off,
        rho = RHO,
        k = K,
        washout = WASHOUT,
    )
    plot_ensemble(ensemble, features, stats, out_path="esn_ensemble.png")

    print("\nEvaluating transitions...")
    evaluate_transitions(
        model             = model,
        df                = df,
        df_raw            = df_raw,
        stats             = stats,
        cov_on            = cov_on,
        cov_off           = cov_off,
        threshold_high    = THRESHOLD_HIGH,
        threshold_low     = THRESHOLD_LOW,
        threshold_feature = "AMOC",
        washout           = WASHOUT,
        steps_after       = 500,
        k                 = K,
        rho               = RHO,
    )
    amoc_true = df_raw["AMOC"].values
    amoc_ensemble_flat = np.concatenate(ensemble["AMOC"])


    w_full = plot_distribution_and_psd(
        Y_pred_raw = amoc_ensemble_flat,
        Y_test_raw = amoc_true,
        target_col = "AMOC",
        dt_years   = 1/12,
        out_path   = "esn_ensemble_dist_psd_full.png",
    )


    w_clip = plot_distribution_and_psd(
        Y_pred_raw  = amoc_ensemble_flat,
        Y_test_raw  = amoc_true,
        target_col  = "AMOC",
        dt_years    = 1/12,
        clip_bounds = (4.0, 19.0),
        out_path    = "esn_ensemble_dist_psd_clipped.png",
    )


    frac_high_real = get_regime_hysteresis(
        amoc_true, THRESHOLD_HIGH, THRESHOLD_LOW
    ).mean()
    frac_high_ens  = np.mean([
        get_regime_hysteresis(np.array(traj), THRESHOLD_HIGH, THRESHOLD_LOW).mean()
        for traj in ensemble["AMOC"]
    ])

    print(f"\n  Real high fraction     : {frac_high_real:.3f}")
    print(f"  Ensemble high fraction : {frac_high_ens:.3f}")
    print(f"  Wasserstein (full)     : {w_full:.4f} Sv")
    print(f"  Wasserstein (clipped)  : {w_clip:.4f} Sv")

    print_regime_stats(amoc_true,          THRESHOLD_HIGH, THRESHOLD_LOW, "Real data")
    print_regime_stats(amoc_ensemble_flat, THRESHOLD_HIGH, THRESHOLD_LOW, "Ensemble (flattened)")

    high_res, low_res = [], []
    for traj in ensemble["AMOC"]:
        regime  = get_regime_hysteresis(np.array(traj), THRESHOLD_HIGH, THRESHOLD_LOW)
        changes = np.diff(regime)
        t_idx   = np.where(changes != 0)[0]
        if len(t_idx) < 2:
            continue
        res = np.diff(t_idx)
        high_res.extend(res[regime[t_idx[:-1]] == 1].tolist())
        low_res.extend (res[regime[t_idx[:-1]] == 0].tolist())

    if high_res: print(f"  Ensemble mean high residence: {np.mean(high_res):.0f} steps")
    if low_res:  print(f"  Ensemble mean low  residence: {np.mean(low_res):.0f} steps")
