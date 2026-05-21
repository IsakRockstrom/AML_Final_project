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

def _sample_noise(AMOC, threshold, cov_on, cov_off):

    cov = cov_on if AMOC > threshold else cov_off
    stds = np.sqrt(np.diag(cov))
    noise  = np.random.multivariate_normal(mean=np.zeros(3), cov=cov)

    if np.abs(noise[0]) < stds[0]:
        AMOC_noise = noise[0]
    else: 
        AMOC_noise = stds[0] * np.sign(noise[0])

    if np.abs(noise[1]) < stds[1]:
        SFWF_noise = noise[1] 
    else: 
        SFWF_noise = stds[1] * np.sign(noise[1])
    
    if np.abs(noise[2]) < stds[2]:
        PD_200m_noise = noise[2]
    else: 
        PD_200m_noise = stds[2] * np.sign(noise[2])

    return AMOC_noise, SFWF_noise, PD_200m_noise        
        


class CoupledModel():

    def __init__(self, AMOC_load, SFWF_load, PD_200m_load):

        self.NN_AMOC    = _reconstruct_model(AMOC_load['config'], AMOC_load['state_dict'])
        self.NN_SFWF    = _reconstruct_model(SFWF_load['config'], SFWF_load['state_dict'])
        self.NN_PD_200m = _reconstruct_model(PD_200m_load['config'],   PD_200m_load['state_dict'])

        self.x_scaler_AMOC, self.y_scaler_AMOC = _get_scalers(AMOC_load['scaler'])
        self.x_scaler_SFWF, self.y_scaler_SFWF = _get_scalers(SFWF_load['scaler'])
        self.x_scaler_PD_200m, self.y_scaler_PD_200m = _get_scalers(PD_200m_load['scaler'])


    def predict(self, device, X, start_idx, lag, AMOC_features, SFWF_features, PD_200m_features, steps, threshold, cov_on, cov_off):

        X_init_AMOC = X[AMOC_features][start_idx - lag : start_idx]
        X_init_SFWF = X[SFWF_features][start_idx - lag : start_idx]
        X_init_PD_200m = X[PD_200m_features][start_idx - lag : start_idx]

        X_init_AMOC_scaled = self.x_scaler_AMOC.transform(X_init_AMOC)
        X_init_SFWF_scaled = self.x_scaler_SFWF.transform(X_init_SFWF)
        X_init_PD_200m_scaled = self.x_scaler_PD_200m.transform(X_init_PD_200m)

        X_init_AMOC_t = torch.tensor(X_init_AMOC_scaled, dtype=torch.float32)
        X_init_SFWF_t = torch.tensor(X_init_SFWF_scaled, dtype=torch.float32)
        X_init_PD_200m_t = torch.tensor(X_init_PD_200m_scaled, dtype=torch.float32)

        self.NN_AMOC.eval()
        self.NN_SFWF.eval()
        self.NN_PD_200m.eval()

        with torch.no_grad():
            AMOC_init = self.NN_AMOC(X_init_AMOC_t.unsqueeze(0).to(device)).cpu().numpy()
            SFWF_init = self.NN_SFWF(X_init_SFWF_t.unsqueeze(0).to(device)).cpu().numpy()
            PD_200m_init = self.NN_PD_200m(X_init_PD_200m_t.unsqueeze(0).to(device)).cpu().numpy()

        AMOC_new = torch.tensor([[SFWF_init.item(), PD_200m_init.item()]], dtype=torch.float32)
        X_AMOC_t = torch.cat([X_init_AMOC_t[1:], AMOC_new], dim=0) 
        SFWF_new = torch.tensor([[AMOC_init.item(), PD_200m_init.item()]], dtype=torch.float32)
        X_SFWF_t = torch.cat([X_init_SFWF_t[1:], SFWF_new], dim=0) 
        PD_200m_new = torch.tensor([[SFWF_init.item(), AMOC_init.item()]], dtype=torch.float32)
        X_PD_200m_t = torch.cat([X_init_PD_200m_t[1:], PD_200m_new], dim=0) 

        AMOC_init_unscaled = self.y_scaler_AMOC.inverse_transform(AMOC_init.reshape(-1, 1)).squeeze()
        SFWF_init_unscaled = self.y_scaler_SFWF.inverse_transform(SFWF_init.reshape(-1, 1)).squeeze()
        PD_200m_init_unscaled = self.y_scaler_PD_200m.inverse_transform(PD_200m_init.reshape(-1, 1)).squeeze()

        AMOC = [AMOC_init_unscaled]
        SFWF = [SFWF_init_unscaled]
        PD_200m = [PD_200m_init_unscaled]

        threshold_scaled = (threshold - self.y_scaler_AMOC.mean_[0]) / self.y_scaler_AMOC.scale_[0]

        for s in range(steps):

            with torch.no_grad():
                AMOC_s = self.NN_AMOC(X_AMOC_t.unsqueeze(0).to(device)).cpu().numpy()
                SFWF_s = self.NN_SFWF(X_SFWF_t.unsqueeze(0).to(device)).cpu().numpy()
                PD_200m_s = self.NN_PD_200m(X_PD_200m_t.unsqueeze(0).to(device)).cpu().numpy()

            AMOC_noise, SFWF_noise, PD_200m_noise = _sample_noise(AMOC_s, threshold_scaled, cov_on, cov_off)
            
            AMOC_noise_scaled   = AMOC_noise / self.y_scaler_AMOC.scale_[0]
            SFWF_noise_scaled   = SFWF_noise / self.y_scaler_SFWF.scale_[0]
            PD_200m_noise_scaled = PD_200m_noise / self.y_scaler_PD_200m.scale_[0]

            AMOC_s += AMOC_noise_scaled
            SFWF_s += SFWF_noise_scaled
            PD_200m_s += PD_200m_noise_scaled

         
            AMOC_new = torch.tensor([[SFWF_s.item(), PD_200m_s.item()]], dtype=torch.float32)
            X_AMOC_t = torch.cat([X_AMOC_t[1:], AMOC_new], dim=0) 
            SFWF_new = torch.tensor([[AMOC_s.item(), PD_200m_s.item()]], dtype=torch.float32)
            X_SFWF_t = torch.cat([X_SFWF_t[1:], SFWF_new], dim=0) 
            PD_200m_new = torch.tensor([[SFWF_s.item(), AMOC_s.item()]], dtype=torch.float32)
            X_PD_200m_t = torch.cat([X_PD_200m_t[1:], PD_200m_new], dim=0)

            AMOC_s_unscaled = self.y_scaler_AMOC.inverse_transform(AMOC_s.reshape(-1, 1)).squeeze()
            SFWF_s_unscaled = self.y_scaler_SFWF.inverse_transform(SFWF_s.reshape(-1, 1)).squeeze()
            PD_200m_s_unscaled = self.y_scaler_PD_200m.inverse_transform(PD_200m_s.reshape(-1, 1)).squeeze()

            AMOC.append(AMOC_s_unscaled)
            SFWF.append(SFWF_s_unscaled)
            PD_200m.append(PD_200m_s_unscaled)

        return AMOC, SFWF, PD_200m











