import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: prevent OpenMP conflict
import matplotlib.pyplot as plt
from TEP_data import calculate_nmse





def plot_evaluation(eval_results, stats, features, out_path):
    """Plot teacher-forced test predictions for all variables."""
    n_vars = len(features)
    colors = plt.cm.tab10.colors  # auto-generate colors for any number of features

    fig, axes = plt.subplots(n_vars, 1, figsize=(13, 4 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]  # ensure iterable for single-variable case
    fig.suptitle("Coupled ESN — teacher-forced test evaluation", fontsize=13)

    for ax, color, (var, (pred, true)) in zip(axes, colors, eval_results.items()):
        mu  = stats[var]["mean"]
        std = stats[var]["std"]
        p = pred.detach().numpy().squeeze() * std + mu
        t = true.detach().numpy().squeeze() * std + mu
        err = calculate_nmse(pred, true)

        ax.plot(t, label="Target", color="black",  lw=1.2, alpha=0.7)
        ax.plot(p, label="ESN",    color=color,    lw=1.2, linestyle="--")
        ax.set_ylabel(var)
        ax.set_title(f"{var}  (NMSE = {err:.5f})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Test time step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Evaluation plot saved → {out_path}")


def plot_rollout(rollout_result: dict, stats: dict, features: list, out_path: str):
    """Plot one autoregressive rollout trajectory."""
    n_vars = len(features)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(n_vars, 1, figsize=(13, 4 * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]
    fig.suptitle("Coupled ESN — autoregressive rollout (with noise)", fontsize=13)

    for ax, color, (var, values) in zip(axes, colors, rollout_result.items()):
        ax.plot(values, color=color, lw=1.2)
        ax.set_ylabel(var)
        ax.set_title(var)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Rollout step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Rollout plot saved → {out_path}")