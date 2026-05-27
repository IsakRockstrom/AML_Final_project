from sklearn.mixture import GaussianMixture
import numpy as np

def objective_function(AMOC_pred, AMOC_true):

    gmm_true = GaussianMixture(n_components=2).fit(AMOC_true.reshape(-1, 1))
    gmm_pred = GaussianMixture(n_components=2).fit(AMOC_pred.reshape(-1, 1))

    order_true = np.argsort(gmm_true.means_.ravel())
    order_pred = np.argsort(gmm_pred.means_.ravel())

    params_true = np.concatenate([
        gmm_true.means_.ravel()[order_true],
        gmm_true.covariances_.ravel()[order_true],
        gmm_true.weights_[order_true]
    ])
    params_pred = np.concatenate([
        gmm_pred.means_.ravel()[order_pred],
        gmm_pred.covariances_.ravel()[order_pred],
        gmm_pred.weights_[order_pred]
    ])

    return np.sqrt(np.mean((params_true - params_pred)**2))
