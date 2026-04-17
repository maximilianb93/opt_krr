import torch
import torch.nn as nn
import lightning
from utils import linear_kernel, polynomial_kernel, rbf_kernel, lap_kernel

class KernelRidgeRegression(nn.Module):
    def __init__(
            self,
            X_ref=None,
            y_ref=None, 
            kernel='lap', 
            lambda_=1.0, 
            gamma=None, 
            degree=3, 
            coef0=1, 
            input_dim=1,
            loss_type="l1",
            ):
        super(KernelRidgeRegression, self).__init__()
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0

        self.register_buffer('X_ref', X_ref)
        self.register_buffer('y_ref', y_ref)
        self.register_buffer('alpha_', None)

        self.lambda_ = nn.Parameter(torch.tensor(lambda_, dtype=torch.float32), requires_grad=True)
        if gamma is None:
            gamma = torch.ones(input_dim, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma, requires_grad=True)

        # OPTIM
        self._optimizer_name = "Adam"
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}
        self.lr_scheduler_config = {}
        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"Unknown loss type: {loss_type}")
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()

    
    def _kernel_function(self, X, Y):
        if self.kernel == 'linear':
            return linear_kernel(X, Y)
        elif self.kernel == 'poly':
            return polynomial_kernel(X, Y, self.degree, self.coef0)
        elif self.kernel == 'rbf':
            return rbf_kernel(X, Y, self.gamma)
        elif self.kernel == 'lap':
            return lap_kernel(X, Y, self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, solver="leastsquares"):
        K = self._kernel_function(self.X_ref, self.X_ref)
        n = K.shape[0]
        I = torch.eye(n, device=K.device)
        if solver == "direct":
            self.alpha_ = torch.linalg.solve(K + torch.abs(self.lambda_) * I, self.y_ref)
        elif solver == "leastsquares":
            self.alpha_ = torch.linalg.lstsq(K + torch.abs(self.lambda_) * I, self.y_ref).solution
        else:
            raise ValueError(f"Unknown solver: {solver}")
    
    def predict(self, X):
        K = self._kernel_function(X, self.X_ref)
        return torch.matmul(K, self.alpha_)

    def forward(self, X):
        return self.predict(X)
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch["data"]
        y = train_batch["target"]
        # =================forward====================
        self.fit()
        y_train_pred = self.predict(x)

        # ===================loss=====================
        loss = self.loss_fn(y_train_pred, y)

        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.
        It also adds the learning rate scheduler if self.lr_scheduler_kwargs is not empty.
        The scheduler is given as a dictionary with the key 'scheduler' containing the scheduler class
        and the rest of the keys are config options for the scheduler.

        Returns
        -------
        torch.optim
            Torch optimizer
            
        dict, optional
            Learning rate scheduler configuration (if any)
        """

        # Create the optimizer from the optimizer name and kwargs
        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )
        
        # Return just the optimizer if no scheduler is defined
        if not self.lr_scheduler_kwargs:
            return optimizer
        
        # Create the scheduler from the lr_scheduler_kwargs if any
        if "scheduler" not in self.lr_scheduler_kwargs:
            raise ValueError("lr_scheduler_kwargs must include a 'scheduler' key with the scheduler class.")

        scheduler_cls = self.lr_scheduler_kwargs["scheduler"]
        scheduler_kwargs = {
            k: v for k, v in self.lr_scheduler_kwargs.items() if k != "scheduler"
        }
        lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        lr_scheduler_config = {
            "scheduler": lr_scheduler
        }

        # Add possible additional config options
        if self.lr_scheduler_config:
            if "scheduler" in self.lr_scheduler_config:
                raise ValueError("lr_scheduler_config cannot override the 'scheduler' entry.")
            lr_scheduler_config.update(self.lr_scheduler_config)
        return [optimizer], [lr_scheduler_config]

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
    def load(cls, path, input_dim=1, weights_only=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_data = torch.load(path, map_location=device, weights_only=weights_only)
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
