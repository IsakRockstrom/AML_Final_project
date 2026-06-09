import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: prevent OpenMP conflict
import pandas as pd
from LT_data import load_and_transform, estimate_covariances
from Coupled_ESN_Architecture import CoupledESN
from Plot_Coupled import plot_evaluation, plot_rollout
import numpy as np
from TEP_data import calculate_nmse, calculate_mae, calculate_rmse
import matplotlib.pyplot as plt


def run_coupled_ESN(data_path = "datasets/CESMDataset_Try.csv", washout = 200, train_frac = 0.8, threshold_feature = "AMOC", amoc_threshold = 13.0, k = 1, steps = 2000):
    features = ["AMOC", "SFWF", "PD_200m"]
    df_raw = pd.read_csv(data_path)[features]
    configs = [
        dict(n_reservoir=500, spectral_radius=0.9, sparsity=0.9, leaking_rate=0.3, input_scaling=0.5, ridge_alpha=1e-1),  # AMOC
        dict(n_reservoir=500, spectral_radius=0.9, sparsity=0.9, leaking_rate=0.3, input_scaling=0.5, ridge_alpha=1e-1),  # SFWF
        dict(n_reservoir=500, spectral_radius=0.9, sparsity=0.9, leaking_rate=0.5, input_scaling=0.5, ridge_alpha=1e-2),  # PD_200m
    ]
    # compute how long each regime visit lasts
    above = (df_raw["AMOC"].values > amoc_threshold).astype(int)
    changes = np.diff(above)
    transition_indices = np.where(changes != 0)[0]

    residence_times = np.diff(transition_indices)
    high_times = residence_times[above[transition_indices[:-1]] == 1]
    low_times  = residence_times[above[transition_indices[:-1]] == 0]

    print(f"Mean high regime residence: {high_times.mean():.0f} steps")
    print(f"Mean low  regime residence: {low_times.mean():.0f} steps")


    df, stats = load_and_transform(data_path, features)
    # find all transition points
    amoc_vals   = df_raw["AMOC"].values
    transitions = np.where(np.diff((amoc_vals > amoc_threshold).astype(int)))[0]
    print("Transition points:", transitions)

    # take the first low→high transition
    # the window should end just after the transition crosses the threshold
    transition_idx = transitions[0]  # index where crossing happens
    init_start     = max(0, transition_idx - washout + 50)  # end 50 steps past transition
    init_window    = df.values[init_start:init_start + washout]

    print(f"Init window: steps {init_start} to {init_start + washout}")
    print(f"AMOC range in window: {df_raw['AMOC'].values[init_start:init_start+washout].min():.2f} "
        f"to {df_raw['AMOC'].values[init_start:init_start+washout].max():.2f}")
    model = CoupledESN(features, configs)
    model.fit(features, df, stats, washout = washout)
    residuals = model.get_residuals(washout = washout)
    threshold_scaled = (13.0 - stats["AMOC"]["mean"]) / stats["AMOC"]["std"]
    print(threshold_scaled)  # should be between -1 and +1
    print("residuals shape:", residuals.shape)  # should be (T, n_features)
    print("residuals dtype:", residuals.dtype)
    print("NaNs in residuals:", np.isnan(residuals).any())
    print("Infs in residuals:", np.isinf(residuals).any())
    cov_on, cov_off, rho = estimate_covariances(residuals, df_raw, features, threshold_feature, amoc_threshold, washout, train_frac)
    print("cov_on  diag:", np.diag(cov_on))
    print("cov_off diag:", np.diag(cov_off))
    print("on/off ratio:", np.diag(cov_on) / np.diag(cov_off))

    def run_ensemble(model, df, df_raw, stats, features, n_ensemble, steps, 
                 amoc_threshold, threshold_feature, cov_on, cov_off, rho, k, washout):

        amoc_vals   = df_raw["AMOC"].values
        transitions = np.where(np.diff((amoc_vals > amoc_threshold).astype(int)))[0]
        
        # sample init windows: half from low regime, half from high regime
        low_starts  = transitions[::2][:n_ensemble//2]   # entering low
        high_starts = transitions[1::2][:n_ensemble//2]  # entering high
        start_idxs  = np.concatenate([low_starts, high_starts])

        ensemble = {f: [] for f in features}

        for start in start_idxs:
            init_start  = max(0, start - washout + 20)
            init_window = df.values[init_start:init_start + washout]
            if len(init_window) < washout:
                continue
            result = model.rollout(
                init_window, steps, amoc_threshold, threshold_feature,
                cov_on, cov_off, rho, k
            )
            for f in features:
                ensemble[f].append(result[f])

        return {f: np.array(v) for f, v in ensemble.items()}


    def plot_ensemble(ensemble, features, out_path="esn_ensemble.png"):
        n_vars  = len(features)
        colors  = plt.cm.tab10.colors
        fig, axes = plt.subplots(n_vars, 1, figsize=(13, 4 * n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]
        fig.suptitle("Coupled ESN — ensemble rollout", fontsize=13)

        for ax, color, feature in zip(axes, colors, features):
            trajs = ensemble[feature]           # (n_ensemble, steps+1)
            t     = np.arange(trajs.shape[1])
            for traj in trajs:
                ax.plot(t, traj, color=color, lw=0.5, alpha=0.3)
            ax.plot(t, trajs.mean(axis=0), color="black", lw=1.5, label="mean")
            ax.set_ylabel(feature)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Rollout step")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Ensemble plot saved → {out_path}")
    eval_results = model.evaluate()
    for var, (pred, true) in eval_results.items():
        print(f"  {var:10s}  NMSE = {calculate_nmse(pred, true):.5f}")
    plot_evaluation(eval_results, stats, features, "esn_coupled_eval.png")
    ensemble = run_ensemble(
    model, df, df_raw, stats, features,
    n_ensemble        = 20,
    steps             = 2000,
    amoc_threshold    = amoc_threshold,
    threshold_feature = threshold_feature,
    cov_on            = cov_on,
    cov_off           = cov_off,
    rho               = rho,
    k                 = k,
    washout           = washout,
    )
    plot_ensemble(ensemble, features, "esn_ensemble.png")

    result = model.rollout(init_window, steps, amoc_threshold, threshold_feature, cov_on, cov_off, rho, k)
    plot_rollout(result, stats, features, "esn_coupled_rollout.png")
    # fraction of time spent in high regime across ensemble
    amoc_ensemble = ensemble["AMOC"]  # (n_ensemble, steps+1)
    frac_high = (amoc_ensemble > amoc_threshold).mean()
    print(f"Fraction of time in high regime: {frac_high:.3f}")

    # compare to real data
    frac_high_real = (df_raw["AMOC"].values > amoc_threshold).mean()
    print(f"Real data fraction in high regime: {frac_high_real:.3f}")
    print(f"Number of high regime visits: {len(high_times)}")
    print(f"Number of low  regime visits: {len(low_times)}")

    # expected fraction from residence times
    expected_frac_high = high_times.mean() / (high_times.mean() + low_times.mean())
    print(f"Expected high fraction from mean residence: {expected_frac_high:.3f}")
    print(f"Actual high fraction: {frac_high_real:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(high_times, bins=30, color="#1f77b4", edgecolor="black")
    axes[0].set_title("High regime residence times")
    axes[0].set_xlabel("Steps")
    axes[1].hist(low_times,  bins=30, color="#d62728", edgecolor="black")
    axes[1].set_title("Low regime residence times")
    axes[1].set_xlabel("Steps")
    plt.tight_layout()
    plt.savefig("residence_times.png", dpi=150)

    # only count as "switched to high" when AMOC > 15
    # only count as "switched to low"  when AMOC < 11
    threshold_high = 15.0  # must exceed this to enter high regime
    threshold_low  = 11.0  # must drop below this to enter low regime

    def get_regime_hysteresis(amoc_vals, threshold_high, threshold_low):
        regime = np.zeros(len(amoc_vals), dtype=int)  # 0 = low, 1 = high
        current = 0 if amoc_vals[0] < threshold_high else 1
        for t, val in enumerate(amoc_vals):
            if current == 0 and val > threshold_high:
                current = 1
            elif current == 1 and val < threshold_low:
                current = 0
            regime[t] = current
        return regime

    regime = get_regime_hysteresis(df_raw["AMOC"].values, threshold_high=15.0, threshold_low=11.0)

    frac_high_hysteresis = regime.mean()
    print(f"High regime fraction (hysteresis): {frac_high_hysteresis:.3f}")

    # recompute residence times
    changes = np.diff(regime)
    transition_indices = np.where(changes != 0)[0]
    residence_times = np.diff(transition_indices)
    high_times = residence_times[regime[transition_indices[:-1]] == 1]
    low_times  = residence_times[regime[transition_indices[:-1]] == 0]

    print(f"Number of high regime visits: {len(high_times)}")
    print(f"Number of low  regime visits: {len(low_times)}")
    print(f"Mean high residence: {high_times.mean():.0f} steps")
    print(f"Mean low  residence: {low_times.mean():.0f} steps")
    # track regime each step, then:
    print("regime flips during rollout:", np.sum(np.abs(np.diff(regime))))

run_coupled_ESN()