import numpy as np
import matplotlib.pyplot as plt

#File used for calculating errors, transforming data to a normalized distribution and plotting 
#the predicted results as well as errors hence the name T(Transform) E(Errors) P(Plot)

def inverse_transform(y, stats, col):
    return (y.detach().numpy() * stats[col]["std"] + stats[col]["mean"])


def calculate_nmse(pred, true):
    return (((pred - true) ** 2).mean() / true.var()).item()


def calculate_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def calculate_mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def plot_results(Y_pred_raw, Y_test_raw, target_col, metrics):
    n_show = len(Y_test_raw)
    t = np.arange(len(Y_test_raw))
    fig, axes = plt.subplots(3, 1, figsize = (13, 10))
    ax = axes[0]
    ax.set_title(f"Prediction vs target  (first {n_show} test steps)")
    ax.plot(t[:n_show], Y_test_raw[:n_show], label="Target", color="#1f77b4", lw=1.5)
    ax.plot(t[:n_show], Y_pred_raw[:n_show], label="ESN output", color="#d62728", lw=1.5, linestyle="--", alpha=0.85)
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(alpha = 0.3)
    ax = axes[1]
    ax.set_title("Target vs Predicted")
    ax.scatter(Y_test_raw, Y_pred_raw, s=3, alpha=0.25, color="#2ca02c")
    lo, hi = Y_test_raw.min(), Y_test_raw.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"True  {target_col}")
    ax.set_ylabel(f"Predicted  {target_col}")
    ax.grid(alpha=0.3)
    ax = axes[2]
    residuals = Y_pred_raw - Y_test_raw
    ax.plot(t, residuals, color="#7f7f7f", lw=0.7)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Test time step")
    ax.set_ylabel("Error")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("esn_results.png", dpi=300, bbox_inches="tight")