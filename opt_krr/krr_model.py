import torch
import torch.nn as nn
from numpy.core.multiarray import _reconstruct, scalar  # Common culprits
import torch.serialization  

class KernelRidgeRegression(nn.Module):
    def __init__(self, kernel='linear', lambda_=1.0, gamma=None, degree=3, coef0=1, input_dim=1):
        super(KernelRidgeRegression, self).__init__()
        self.kernel = kernel
        self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float32), requires_grad=True)
        if gamma is None:
            gamma = torch.ones(input_dim, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma, requires_grad=True)
        self.degree = degree
        self.coef0 = coef0
        self.X_ref = None
        self.alpha_ = None
    
    def _linear_kernel(self, X, Y):
        return torch.matmul(X, Y.T)
    
    def _polynomial_kernel(self, X, Y):
        return (self.coef0 + torch.matmul(X, Y.T)) ** self.degree
    
    def _rbf_kernel(self, X, Y):
        gamma = self.gamma
        if gamma.dim() == 2:
            X_scaled = torch.matmul(X, torch.linalg.inv(torch.sqrt(gamma)))
            Y_scaled = torch.matmul(Y, torch.linalg.inv(torch.sqrt(gamma)))
        else:
            gamma_expanded = gamma.view(1, -1)
            X_scaled = X / torch.sqrt(gamma_expanded)
            Y_scaled = Y / torch.sqrt(gamma_expanded)
        K = torch.cdist(X_scaled, Y_scaled) ** 2
        return torch.exp(-K)

    def _lap_kernel(self, X, Y):
        gamma = self.gamma
        if gamma.dim() == 2:
            X_scaled = torch.matmul(X, torch.linalg.inv(gamma))
            Y_scaled = torch.matmul(Y, torch.linalg.inv(gamma))
        else:
            gamma_expanded = gamma.view(1, -1)
            X_scaled = X / gamma_expanded
            Y_scaled = Y / gamma_expanded
        K = torch.cdist(X_scaled, Y_scaled)
        return torch.exp(-K)
    
    def _kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel == 'lap':
            return self._lap_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X_ref, y_ref, solver="leastsquares"):
        self.X_ref = X_ref
        K = self._kernel_function(X_ref, X_ref)
        n = K.shape[0]
        I = torch.eye(n, device=K.device)
        if solver == "direct":
            self.alpha_ = torch.linalg.solve(K + torch.abs(self.lambda_) * I, y_ref)
        elif solver == "leastsquares":
            self.alpha_ = torch.linalg.lstsq(K + torch.abs(self.lambda_) * I, y_ref).solution
        else:
            raise ValueError(f"Unknown solver: {solver}")


    
    def predict(self, X):
        K = self._kernel_function(X, self.X_ref)
        return torch.matmul(K, self.alpha_)

    def forward(self, X):
        return self.predict(X)

    def save(self, path):
        model_data = {
            'state_dict': self.state_dict(),
            'kernel': self.kernel,
            'lambda_': self.lambda_.item(),
            'gamma': self.gamma.detach().cpu(),
            'degree': self.degree,
            'coef0': self.coef0,
            'X_ref': self.X_ref,
            'alpha_': self.alpha_
        }
        torch.save(model_data, path)

    @classmethod
    def load(cls, path, input_dim=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_data = torch.load(path, map_location=device)
        model = cls(
            kernel=model_data['kernel'],
            lambda_=model_data['lambda_'],
            gamma=torch.tensor(model_data['gamma']),
            degree=model_data['degree'],
            coef0=model_data['coef0'],
            input_dim=input_dim
        )
        model.load_state_dict(model_data['state_dict'])
        model.X_ref = model_data['X_ref']
        model.alpha_ = model_data['alpha_']
        return model
