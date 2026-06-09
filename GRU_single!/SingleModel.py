import torch
import numpy as np


def compute_covariances(X, var_names, threshold):
    on  = X['AMOC'] > threshold
    off = X['AMOC'] < threshold
    data_on  = X[on][var_names].values
    data_off = X[off][var_names].values
    return np.cov(data_on.T), np.cov(data_off.T)


def _sample_noise(AMOC, threshold, var_names, cov_on, cov_off, noise_prev, rho, k, rng):
    cov  = (cov_on if AMOC > threshold else cov_off) * k
    stds = np.sqrt(np.diag(cov))
    raw  = rng.multivariate_normal(np.zeros(len(var_names)), cov)

    noise_raw = {}
    for i, v in enumerate(var_names):
        noise_raw[v] = raw[i] if abs(raw[i]) < stds[i] else stds[i] * np.sign(raw[i])

    return {v: np.sqrt(1 - rho**2) * noise_raw[v] + rho * noise_prev[v] for v in var_names}


class SingleModel:

    @classmethod
    def from_trained(cls, model, x_scaler, y_scalers):
        instance = cls.__new__(cls)
        instance.model     = model
        instance.x_scaler  = x_scaler
        instance.y_scalers = y_scalers
        return instance

    def predict(self, X, start_idx, feature_cols, target_cols, lag, steps, threshold, rho, k, seed = None):
        
        X_init = X[feature_cols].iloc[start_idx - lag : start_idx].values
        X_w    = self.x_scaler.transform(X_init) 

        
        amoc_sc = self.y_scalers['AMOC']
        threshold_scaled = (threshold - amoc_sc.mean_[0]) / amoc_sc.scale_[0]

        
        cov_on, cov_off = compute_covariances(X, target_cols, threshold)

        noise_prev  = {v: 0.0 for v in target_cols}
        predictions = {v: []  for v in target_cols}
        rng         = np.random.default_rng(seed)

        
        tgt_in_feat = {v: feature_cols.index(v) for v in target_cols if v in feature_cols}

        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                x    = torch.tensor(X_w, dtype=torch.float32).unsqueeze(0)
                pred = self.model(x).squeeze(0).numpy()

                amoc_idx = target_cols.index('AMOC')
                noise    = _sample_noise(pred[amoc_idx], threshold_scaled,
                                         target_cols, cov_on, cov_off, noise_prev, rho, k, rng)
                pred_noisy = pred + np.array([noise[v] for v in target_cols])
                noise_prev = noise

                
                for i, v in enumerate(target_cols):
                    unscaled = self.y_scalers[v].inverse_transform([[pred_noisy[i]]])[0, 0]
                    predictions[v].append(unscaled)

              
                X_w = np.roll(X_w, -1, axis=0)
                for v, fi in tgt_in_feat.items():
                    X_w[-1, fi] = pred_noisy[target_cols.index(v)]

        return predictions
