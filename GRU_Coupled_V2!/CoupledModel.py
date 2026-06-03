import pickle
import torch
from model_stuff import Predictor
import numpy as np


def _reconstruct_model(params, state_dict):
    model = Predictor(
        n_features=params['n_features'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
    )
    model.load_state_dict(state_dict)
    return model

def _get_scalers(scaler):
    with open(scaler, 'rb') as f:
        scalers = pickle.load(f)
    y_scaler = scalers['y_scaler']
    x_scaler = scalers['x_scaler']
    return x_scaler, y_scaler

def compute_covariances(X, models, threshold):
    on  = X['AMOC'] > threshold
    off = X['AMOC'] < threshold

    data_on  = np.column_stack([X[on][model]  for model in models])
    data_off = np.column_stack([X[off][model] for model in models])

    cov_on  = np.cov(data_on.T)
    cov_off = np.cov(data_off.T)

    return cov_on, cov_off

''' 
Sample noise from a multi variate Gaussian from the data of all model variables with:
- Separate distributions for AMOC on/off states
- noise strength scaling
- AR(1) noise smooting

Params and args:

AMOC - AMOC strength at t-1
Threshold - AMOC off < threshold < AMOC on  
models - list of model variable names
cov_on - covariance matrix for data distribution in AMOC on state
cov_off - covariance matrix for data distribution in AMOC off state

noise_prev - noise at t-1 (dict)
rho - scaling blend of noise at t with noise at t-1
k - scaling noise strength
'''
def _sample_noise(AMOC, threshold, models, cov_on, cov_off, noise_prev, rho, k):

    cov_on = cov_on * k
    cov_off = cov_off * k

    cov = cov_on if AMOC > threshold else cov_off
    stds = np.sqrt(np.diag(cov))
    noise  = np.random.multivariate_normal(mean=np.zeros(len(models)), cov=cov)

    noise_raw = {}

    for i, model in enumerate(models):
        if np.abs(noise[i]) < stds[i]:
            noise_raw[model] = noise[i]
        else:
            noise_raw[model] = stds[i] * np.sign(noise[i])

    noise = {m: (np.sqrt(1 - rho**2) * noise_raw[m]) + (noise_prev[m] * rho) for m in noise_raw}

    return noise               


class CoupledModel():

    ''' 
    Constructor loading from saved models
    '''
    def __init__(self, model_loads):
        self.models  = {}
        self.scalers = {}

        for name, load in model_loads.items():
            self.models[name]  = _reconstruct_model(load['config'], load['state_dict'])
            x_scaler, y_scaler = _get_scalers(load['scaler'])
            self.scalers[name] = {'x_scaler': x_scaler, 'y_scaler': y_scaler}

    '''
    Constructor taking model objects
    '''
    @classmethod
    def from_trained(cls, trained_models):
        instance = cls.__new__(cls)
        instance.models  = {}
        instance.scalers = {}

        for name, data in trained_models.items():
            instance.models[name]  = data['model']
            instance.scalers[name] = {'x_scaler': data['x_scaler'], 'y_scaler': data['y_scaler']}

        return instance



    def predict(self, device, X, start_idx, lag, horizon, models, features, steps, threshold, rho, k, delta = False):
    
        ''' 
        Create a dictionary with data windows containing:
        - Correct features for each model
        - Scaled by the models native scalers
        '''
        X_t = {}
        for model in models:
            X_init = X[features[model]][start_idx - (lag + horizon -1): start_idx]
            X_init_scaled = self.scalers[model]['x_scaler'].transform(X_init)
            X_init_t = torch.tensor(X_init_scaled, dtype=torch.float32)
            X_t[model] = X_init_t 

        ''' 
        Compute covariances for the given variables
        '''
        cov_on, cov_off = compute_covariances(X, models, threshold)
        ''' 
        Scale the AMOC on/off threshold such that it can be used with scaled predictions
        '''
        threshold_scaled = (threshold - self.scalers['AMOC']['y_scaler'].mean_[0]) / self.scalers['AMOC']['y_scaler'].scale_[0]
        ''' 
        A dictionary to hold time series of unscaled predictions for each model
        '''
        unscaled_predictions = {model: [] for model in models}

        init_delta = X.iloc[start_idx-1]
        unscaled_predictions_delta = {model: [init_delta[model]] for model in models}
        unscaled_deltas = {model: [] for model in models}
        ''' 
        initialise the first prev_noise as 0s as it is for the last tie step in the data
        '''
        noise_prev = {model: 0 for model in models}



        for model in models:
            self.models[model].eval()

        for s in range(steps):
            ''' 
            Each model makes one preditction based on its native window from X_t
            The prediction is unscaled by it's native scaler
            '''
            preds = {}
            for model in models:
                with torch.no_grad():
                    pred = self.models[model](X_t[model].unsqueeze(0).to(device)).cpu().numpy()
                preds[model] = pred

            ''' 
            sample noise and add it to the individual predictions
            set noise_prev to noise for next time step
            '''
            if delta:
                amoc_for_threshold = unscaled_predictions_delta['AMOC'][-1]
                noise = _sample_noise(amoc_for_threshold, threshold, models, cov_on, cov_off, noise_prev, rho, k)
            else:
                noise = _sample_noise(preds['AMOC'].item(), threshold_scaled, models, cov_on, cov_off, noise_prev, rho, k)
            
            for model in models:
                noise_scaled = noise[model] / self.scalers[model]['y_scaler'].scale_[0]
                preds[model] = preds[model] + noise_scaled


            noise_prev = noise

            ''' 
            unscale the predictions with added noise and add to the unscaled series
            '''
            for model in models:
                if delta:
                    unscaled_prediction = preds[model].item() * self.scalers[model]['y_scaler'].scale_[0]
                    unscaled_deltas[model].append(unscaled_prediction)
                    unscaled_predictions_delta[model].append(unscaled_predictions_delta[model][-1] + unscaled_prediction)
                else:
                    unscaled_prediction = self.scalers[model]['y_scaler'].inverse_transform(preds[model].reshape(-1, 1)).squeeze()
                    unscaled_predictions[model].append(unscaled_prediction)

            ''' 
            Move the windows in X_t one step forward,
            Discarding the native feature values for the first step, 
            Adding the native feature values as the last step

            If delta is true, add the last window value + the prediction at the end of the window
            '''
            for model in models:
                if delta:
                    new_row = torch.tensor(
                        [[preds[f].item() + X_t[model][-1, features[model].index(f)].item() for f in features[model]]],
                        dtype=torch.float32
                    )
                else:
                    new_row = torch.tensor([[preds[f].item() for f in features[model]]], dtype=torch.float32)
                X_t[model] = torch.cat([X_t[model][1:], new_row], dim=0) 
        
        if delta:
            return unscaled_deltas, {m: v[1:] for m, v in unscaled_predictions_delta.items()}
        return unscaled_predictions

            



        











