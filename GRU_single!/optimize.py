import optuna
import pandas as pd
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

from model_stuff import Predictor
from Pretrain import pretrain
from Rollout import train_rollout
from SingleModel import SingleModel


X_df = pd.read_csv('../data/CombinedDatasetLongUnsmooth.csv')
X_df = X_df.drop(columns=['AMOC']).rename(columns={'AMOC_40N': 'AMOC'})

model_vars   = ['AMOC', 'SFWF', 'PD_200m']
feature_cols = target_cols = model_vars
train_cut_1, train_cut_2 = 12500, 18500
train_idx = list(range(train_cut_1)) + list(range(train_cut_2, len(X_df)))

y_scalers = {}
X_scaled  = X_df[model_vars].copy().astype(float)
for var in model_vars:
    sc = StandardScaler().fit(X_df[var].values[train_idx].reshape(-1, 1))
    X_scaled[var] = sc.transform(X_df[var].values.reshape(-1, 1))
    y_scalers[var] = sc

x_scaler = StandardScaler().fit(X_df[feature_cols].values[train_idx])
amoc_std  = y_scalers['AMOC'].scale_[0]


LAG            = 150
EVAL_START     = train_cut_1
EVAL_STEPS     = 30000
PRED_THRESHOLD = 14.0

PSD_F_LOW  = 1e-4
PSD_F_HIGH = 1e-2

TRUE_AMOC = X_df['AMOC'].values[:EVAL_STEPS]

ROLLOUT_SEED = 42
EVAL_SEEDS   = [0, 1, 2]

NOISE_CONFIGS = [
    {'k': 0.01, 'rho': 0.8},
    {'k': 0.01, 'rho': 0.9},
    {'k': 0.05, 'rho': 0.7},
    {'k': 0.05, 'rho': 0.8},
    {'k': 0.10, 'rho': 0.6},
    {'k': 0.10, 'rho': 0.7},
    {'k': 0.20, 'rho': 0.4},
    {'k': 0.20, 'rho': 0.5},
]


def _psd(x, fs=1.0):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = len(x)
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / (N * fs)
    psd[1:-1] *= 2.0
    return np.fft.rfftfreq(N, d=1.0 / fs)[1:], psd[1:]


def psd_log_mse(true, pred):
    n = min(len(true), len(pred))
    freqs, psd_t = _psd(true[:n])
    _,     psd_p = _psd(pred[:n])
    eps  = 1e-10
    mask = (freqs >= PSD_F_LOW) & (freqs <= PSD_F_HIGH) & (psd_t > eps) & (psd_p > eps)
    return float(np.mean((np.log(psd_p[mask]) - np.log(psd_t[mask])) ** 2))


def compute_losses(true_amoc, pred_amoc):
    wd   = wasserstein_distance(true_amoc, pred_amoc)
    pmse = psd_log_mse(true_amoc, pred_amoc)
    return wd, pmse


def _eval_best_noise(single, pred_lag):
    best_combined = float('inf')
    best_wd       = float('inf')
    best_pmse     = float('inf')
    best_cfg      = NOISE_CONFIGS[0]

    best_seed = EVAL_SEEDS[0]

    for cfg in NOISE_CONFIGS:
        wd_runs, pmse_runs = [], []
        for eval_seed in EVAL_SEEDS:
            preds = single.predict(
                X_df, EVAL_START, feature_cols, target_cols,
                lag=pred_lag, steps=EVAL_STEPS,
                threshold=PRED_THRESHOLD, rho=cfg['rho'], k=cfg['k'],
                seed=eval_seed,
            )
            pred_amoc = np.array(preds['AMOC'])
            wd, pmse  = compute_losses(TRUE_AMOC, pred_amoc)
            wd_runs.append(wd)
            pmse_runs.append(pmse)

        mean_wd   = float(np.mean(wd_runs))
        mean_pmse = float(np.mean(pmse_runs))
        combined  = mean_wd + mean_pmse
        if combined < best_combined:
            best_combined = combined
            best_wd       = mean_wd
            best_pmse     = mean_pmse
            best_cfg      = cfg
            best_seed     = EVAL_SEEDS[int(np.argmin([w + p for w, p in zip(wd_runs, pmse_runs)]))]

    return best_wd, best_pmse, best_cfg, best_seed


def objective(trial):
    n_pretrain_epochs = trial.suggest_int('n_pretrain_epochs', 10, 20)
    train_steps       = trial.suggest_categorical('train_steps',       [100, 200, 300])
    train_horizon     = trial.suggest_categorical('train_horizon',     [40, 80, 120])
    hidden_size       = trial.suggest_categorical('hidden_size',       [64, 128])
    dropout           = trial.suggest_float('dropout',     0.0, 0.3)
    lr_pretrain       = trial.suggest_float('lr_pretrain', 1e-4, 5e-3, log=True)
    lr_rollout        = trial.suggest_float('lr_rollout',  1e-5, 5e-4, log=True)
    transition_config = trial.suggest_categorical('transition_config', ['100_7', '150_9', '200_12'])
    horizon_boost     = trial.suggest_categorical('horizon_boost',     [True, False])

    search_ahead     = int(transition_config.split('_')[0])
    threshold_sv     = float(transition_config.split('_')[1])
    threshold_scaled = threshold_sv / amoc_std
    pred_lag         = LAG + (25 if horizon_boost else 0)

    model = Predictor(
        n_features  = len(feature_cols),
        hidden_size = hidden_size,
        num_layers  = 1,
        dropout     = dropout,
        output_size = len(target_cols),
    )

    pretrain(
        X_scaled, train_cut_1, train_cut_2, model,
        feature_cols, target_cols,
        lag=LAG, horizon=1,
        n_epochs=n_pretrain_epochs, batch_size=64,
        lr=lr_pretrain, loss_fn=nn.MSELoss(),
    )

    train_rollout(
        X_scaled, train_cut_1, train_cut_2,
        0.7, threshold_scaled, search_ahead,
        model, feature_cols, target_cols,
        train_steps=train_steps, lag=LAG,
        train_horizon=train_horizon,
        loss_fn=nn.MSELoss(), lr=lr_rollout,
        seed=ROLLOUT_SEED,
    )

    single                      = SingleModel.from_trained(model, x_scaler, y_scalers)
    wd, pmse, best_cfg, best_seed = _eval_best_noise(single, pred_lag)

    trial.set_user_attr('best_k',        best_cfg['k'])
    trial.set_user_attr('best_rho',      best_cfg['rho'])
    trial.set_user_attr('best_eval_seed', best_seed)
    trial.set_user_attr('wasserstein',   wd)
    trial.set_user_attr('psd_log_mse',   pmse)

    print(f"Trial {trial.number}: epochs={n_pretrain_epochs}, steps={train_steps}, "
          f"horizon={train_horizon}, hidden={hidden_size}, config={transition_config}, "
          f"boost={horizon_boost}, lr_pre={lr_pretrain:.1e}, lr_roll={lr_rollout:.1e} | "
          f"WD={wd:.3f}  PSD={pmse:.3f}  k={best_cfg['k']}  rho={best_cfg['rho']}")

    return wd, pmse


study = optuna.create_study(
    directions     = ['minimize', 'minimize'],
    storage        = 'sqlite:///optuna_distribution_v3.db',
    study_name     = 'distribution_v3',
    load_if_exists = True,
)

study.optimize(objective, n_trials=100)

pareto = study.best_trials
print(f"\n{len(pareto)} Pareto-optimal trials:")
for t in sorted(pareto, key=lambda t: t.values[0]):
    print(f"  Trial {t.number}: WD={t.values[0]:.3f}  PSD={t.values[1]:.3f}  "
          f"seed={t.user_attrs['best_seed']}  k={t.user_attrs['best_k']}  "
          f"rho={t.user_attrs['best_rho']}  params={t.params}")
