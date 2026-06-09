import torch

class EchoStateNetwork:
    """
    Echo State Network with arbitrary n_inputs and n_outputs.

    Parameters
    ----------
    n_inputs        : number of input dimensions
    n_reservoir     : reservoir size
    n_outputs       : number of output dimensions
    spectral_radius : target ρ(W)
    sparsity        : fraction of zero connections in W
    leaking_rate    : α in the leaky-integrator update
    input_scaling   : multiplier on W_in
    ridge_alpha     : L2 regularisation for readout
    seed            : RNG seed
    """

    def __init__(self, n_inputs, n_reservoir, n_outputs = 1, spectral_radius = 0.9, sparsity = 0.9, 
                 leaking_rate = 0.3, input_scaling = 1.0, ridge_alpha = 1e-6, seed = 69):
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha

        torch.manual_seed(seed)
        self._build_reservoir()
        self.W_out = None


    def _build_reservoir(self):
        N = self.n_reservoir
        # Input weights: uniform [-1, 1], scaled
        self.W_in = (torch.rand(N, self.n_inputs) * 2 - 1) * self.input_scaling
        # Sparse recurrent weights
        mask = (torch.rand(N, N) > self.sparsity).float()
        W_raw = (torch.rand(N, N) * 2 - 1) * mask
        # Rescale to desired spectral radius
        rho = torch.linalg.eigvals(W_raw).abs().max().real
        self.W = W_raw * (self.spectral_radius / rho)


    def _update_state(self, x, u):
        """x(t+1) = (1-α)·x(t) + α·tanh(W·x(t) + W_in·u(t))"""
        pre = self.W @ x + self.W_in @ u
        return (1 - self.leaking_rate) * x + self.leaking_rate * torch.tanh(pre)


    def _collect_states(self, U, washout):
        """Drive reservoir with U; return states after washout."""
        x = torch.zeros(self.n_reservoir)
        states = []
        for t in range(len(U)):
            x = self._update_state(x, U[t])
            if t >= washout:
                states.append(x.clone())
        return torch.stack(states)


    def fit(self, U, Y, washout = 100):
        """Train readout via closed-form ridge regression."""
        X_res = self._collect_states(U, washout)
        # Extended state = [reservoir | raw inputs] for richer readout
        X_ext = torch.cat([X_res, U[washout:]], dim=1)
        Y_tgt = Y[washout:]
        n = X_ext.shape[1]
        A = X_ext.T @ X_ext + self.ridge_alpha * torch.eye(n)
        self.W_out = torch.linalg.solve(A, X_ext.T @ Y_tgt)
        return self


    def reset_state(self):
        self._x = torch.zeros(self.n_reservoir)


    def _step(self, u):
        self._x = self._update_state(self._x, u)
        return self._x


    def predict_one(self, u):
        x_ext = torch.cat([self._x, u])
        return x_ext @ self.W_out


    def predict_sequence(self, U, washout = 0):
        X_res = self._collect_states(U, washout)
        X_ext = torch.cat([X_res, U[washout:]], dim = 1)
        return X_ext @ self.W_out