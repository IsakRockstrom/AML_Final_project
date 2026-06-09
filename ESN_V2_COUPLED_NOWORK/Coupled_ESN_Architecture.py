import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: prevent OpenMP conflict
import torch
import numpy as np
from LT_data import torch_data
from ESN_Architecture import EchoStateNetwork


def sample_noise(amoc_scaled, threshold_scaled, cov_on,
                 cov_off, noise_prev, rho, k, clip_std=3.0):

    n_vars = cov_on.shape[0]
    cov = (cov_on if amoc_scaled > threshold_scaled else cov_off) * k
    raw = np.random.multivariate_normal(mean=np.zeros(n_vars), cov=cov)
    # Optional per-component clip. The old code clamped to +/-1 std, which removed
    # exactly the tail kicks that could drive a regime transition. Default to a much
    # looser +/-3 std; pass clip_std=None to disable entirely.
    if clip_std is not None:
        stds = np.sqrt(np.diag(cov))
        bound = clip_std * stds
        raw = np.where(np.abs(raw) < bound, raw, bound * np.sign(raw))
    noise = np.sqrt(1 - rho ** 2) * raw + rho * noise_prev

    return noise


class CoupledESN:

    def __init__(self, features, configs):

        inputs = len(features) - 1
        seeds = np.arange(69, 69 + len(features))
        self.esns = {}
        self.test_data = {}
        self.feature_pointers = {}
        self.stats = None
        for i, (feature, config, seed) in enumerate(zip(features, configs, seeds)):
            self.esns[feature] = EchoStateNetwork(inputs, **config, seed = seed)
            self.feature_pointers[feature] = i


    def fit(self, features, df, stats, washout, train_frac = 0.8):

        self.stats = stats
        for feature, esn in self.esns.items():
            input_features = features.copy()
            input_features.remove(feature)
            U_train, U_test, Y_train, Y_test = torch_data(df, input_features, feature, train_frac)
            esn.fit(U_train, Y_train, washout = washout)
            self.test_data[feature] = {"features": {"inputs": input_features, "output": feature},
                                       "data": {"U_train": U_train, "U_test": U_test, 
                                                "Y_train": Y_train, "Y_test": Y_test}}

        return self
    

    def get_residuals(self, washout):

        n = len(self.esns)
        features  = list(self.esns.keys())
        first_key = features[0]
        T = len(self.test_data[first_key]["data"]["U_train"]) - washout
        residuals = np.zeros((T, n))

        for j, feature in enumerate(features):
            U_train = self.test_data[feature]["data"]["U_train"]
            Y_train = self.test_data[feature]["data"]["Y_train"]
            pred = self.esns[feature].predict_sequence(U_train, washout=washout)
            true = Y_train[washout:]
            res_scaled = (pred - true).detach().numpy().reshape(-1)
            # Keep residuals in SCALED space. Converting back to raw units here
            # (and dividing by std again in rollout) spanned ~11 orders of magnitude
            # in variance across features, making the multivariate normal degenerate.
            residuals[:, j] = res_scaled

        return residuals
    

    def rollout(self, init_window, steps, threshold, threshold_feature, cov_on, cov_off, rho, k = 1.0):

        stats = self.stats
        threshold_scaled = (threshold - stats[threshold_feature]["mean"]) / stats[threshold_feature]["std"]

        #Reservoir warmup
        for feature, esn in self.esns.items():
            esn.reset_state()
            
        for t in range(len(init_window)):
            row = init_window[t]
            for feature, esn in self.esns.items():
                inputs = self.test_data[feature]["features"]["inputs"]
                pointer_adjusted_inputs = []
                for inp in inputs:
                    pointer_adjusted_inputs.append(row[self.feature_pointers[inp]])
                esn._step(torch.tensor(pointer_adjusted_inputs, dtype=torch.float32))

        #Prediction phase
        last = init_window[-1]
        current_scaled = {}
        for feature, esn in self.esns.items():
            inputs = self.test_data[feature]["features"]["inputs"]
            u = torch.tensor([last[self.feature_pointers[inp]] for inp in inputs], dtype=torch.float32)
            current_scaled[feature] = esn.predict_one(u).item()

        def unscale(val, feature):
            return val * stats[feature]["std"] + stats[feature]["mean"]

        results = {f: [unscale(current_scaled[f], f)] for f in self.esns}
        noise_prev = np.zeros(len(self.esns))
        feature_list = list(self.esns.keys())

        regime_flips = 0
        prev_regime  = int(current_scaled[threshold_feature] > threshold_scaled)

        for _ in range(steps):
            # count whether the regime indicator changed this step
            cur_regime = int(current_scaled[threshold_feature] > threshold_scaled)
            if cur_regime != prev_regime:
                regime_flips += 1
            prev_regime = cur_regime

            noise = sample_noise(current_scaled[threshold_feature], threshold_scaled,
                                cov_on, cov_off, noise_prev, rho, k)

            for i, feature in enumerate(feature_list):
                # residuals/noise are now in SCALED space, so add directly (no /std)
                current_scaled[feature] += noise[i]
                # state clip keeps the autoregressive map bounded; +/-5 sigma still
                # comfortably contains both AMOC regimes, so it doesn't block transitions
                current_scaled[feature] = float(np.clip(current_scaled[feature], -5.0, 5.0))

            noise_prev = noise
            next_scaled = {}
            for feature, esn in self.esns.items():
                inputs = self.test_data[feature]["features"]["inputs"]
                u = torch.tensor([current_scaled[inp] for inp in inputs], dtype=torch.float32)
                next_scaled[feature] = esn.predict_one(u).item()

            current_scaled = next_scaled

            for feature in feature_list:
                results[feature].append(unscale(current_scaled[feature], feature))

        print(f"    regime flips during rollout: {regime_flips}")
        return results
    

    def evaluate(self, washout = 0):
        results = {}
        for feature, esn in self.esns.items():
            U_test = self.test_data[feature]["data"]["U_test"]
            Y_test = self.test_data[feature]["data"]["Y_test"]
            pred = esn.predict_sequence(U_test, washout = washout)
            results[feature] = (pred, Y_test[washout:])
        return results