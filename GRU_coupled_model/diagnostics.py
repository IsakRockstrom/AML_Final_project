import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.optimize import curve_fit


def _double_gaussian(x, mu1, sigma1, w1, mu2, sigma2, w2):
    g1 = w1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
    g2 = w2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
    return g1 + g2


def _fit_double_gaussian(data, n_bins=100):
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2

    mu1_0 = np.percentile(data, 20)
    mu2_0 = np.percentile(data, 80)
    p0 = [mu1_0, 1.0, 0.5, mu2_0, 1.0, 0.5]
    bounds = (
        [-np.inf, 1e-3, 0, -np.inf, 1e-3, 0],
        [ np.inf, np.inf, 1,  np.inf, np.inf, 1],
    )
    popt, _ = curve_fit(_double_gaussian, centres, counts, p0=p0, bounds=bounds, maxfev=10000)
    return popt, centres, counts


def plot_distribution(true_data, pred_data, var_name='AMOC', units='Sv',
                      true_color='tab:orange', pred_color='tab:blue', ax=None, save_path=None):
    
    true_data = np.asarray(true_data).ravel()
    pred_data = np.asarray(pred_data).ravel()

    wd = wasserstein_distance(true_data, pred_data)

    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    x_min = min(true_data.min(), pred_data.min())
    x_max = max(true_data.max(), pred_data.max())
    x_grid = np.linspace(x_min, x_max, 500)

    ax.hist(true_data, bins=100, density=True, alpha=0.25, color=true_color, label='True (hist)')
    ax.hist(pred_data, bins=100, density=True, alpha=0.25, color=pred_color, label='Model (hist)')

    try:
        p_true, _, _ = _fit_double_gaussian(true_data)
        ax.plot(x_grid, _double_gaussian(x_grid, *p_true),
                color=true_color, lw=2, label='True (fit)')
    except RuntimeError:
        pass

    try:
        p_pred, _, _ = _fit_double_gaussian(pred_data)
        ax.plot(x_grid, _double_gaussian(x_grid, *p_pred),
                color=pred_color, lw=2, linestyle='--', label='Model (fit)')
    except RuntimeError:
        pass

    ax.set_xlabel(f'{var_name} [{units}]')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution — Wasserstein distance = {wd:.3f} {units}')
    ax.legend()

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()

    return wd


def plot_power_spectrum(true_data, pred_data, var_name='AMOC', units='Sv',
                        true_color='tab:orange', pred_color='tab:blue',
                        fs=1.0, ax=None, save_path=None):

    true_data = np.asarray(true_data, dtype=float).ravel()
    pred_data = np.asarray(pred_data, dtype=float).ravel()

    def _psd(x):
        x = x - x.mean()
        N = len(x)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        psd = (np.abs(X) ** 2) / (N * fs)
        psd[1:-1] *= 2.0         
        return freqs[1:], psd[1:] 

    freqs_true, psd_true = _psd(true_data)
    freqs_pred, psd_pred = _psd(pred_data)

    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.loglog(freqs_true, psd_true, color=true_color, lw=2,          label='True')
    ax.loglog(freqs_pred, psd_pred, color=pred_color, lw=2, ls='--', label='Model')

    ax.set_xlabel('Frequency [cycles / year]')
    ax.set_ylabel(f'PSD [{units}² / year]')
    ax.set_title(f'Power Spectral Density — {var_name}')
    ax.legend()

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
