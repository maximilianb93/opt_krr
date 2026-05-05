import torch
import torch.nn as nn
import lightning
from opt_krr.utils import linear_kernel, polynomial_kernel, rbf_kernel, lap_kernel
from typing import List, Optional


class KernelRidgeRegression(lightning.LightningModule):
    def __init__(
        self,
        X_ref: torch.Tensor,
        y_ref: torch.Tensor,
        lambda_: torch.Tensor = None,
        gamma: torch.Tensor = None,
        kernel: str = "lap",
        degree: torch.Tensor = torch.tensor([3], dtype=torch.int32),
        coef0: torch.Tensor = torch.tensor([1.0], dtype=torch.float32),
        input_dim: torch.Tensor = torch.tensor([1], dtype=torch.int32),
        return_gradient_norm: bool = False,
        loss_type: str = "l1",
        optimizer_name: str = "Adam",
        optimizer_lr: float = 1e-2,
    ):
        super(KernelRidgeRegression, self).__init__()
        self.save_hyperparameters(ignore=["X_ref", "y_ref"])
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.return_gradient_norm = return_gradient_norm
        self.register_buffer("X_ref", X_ref)
        self.register_buffer("y_ref", y_ref)
        self.register_buffer("alpha_", None)

        self.lambda_ = nn.Parameter(
            torch.tensor(lambda_, dtype=torch.float32), requires_grad=True
        )
        if gamma is None:
            gamma = torch.ones(input_dim, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma, requires_grad=True)

        # OPTIM
        self._optimizer_name = optimizer_name
        self.optimizer_kwargs = {"lr": optimizer_lr}
        self.lr_scheduler_kwargs = {}
        self.lr_scheduler_config = {}
        if loss_type not in ["l1", "l2"]:
            raise ValueError(f"Unknown loss type: {loss_type}")
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()

        self.fit()

    def _kernel_function(self, X, Y) -> torch.Tensor:
        if self.kernel == "linear":
            return linear_kernel(X, Y)
        elif self.kernel == "poly":
            return polynomial_kernel(X, Y, self.degree, self.coef0)
        elif self.kernel == "rbf":
            return rbf_kernel(X, Y, self.gamma)
        elif self.kernel == "lap":
            return lap_kernel(X, Y, self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, solver="leastsquares") -> None:
        K = self._kernel_function(self.X_ref, self.X_ref)
        n = K.shape[0]
        I = torch.eye(n).to(self.device)
        if solver == "direct":
            self.alpha_ = torch.linalg.solve(
                K + torch.abs(self.lambda_) * I, self.y_ref
            )
        elif solver == "leastsquares":
            self.alpha_ = torch.linalg.lstsq(
                K + torch.abs(self.lambda_) * I, self.y_ref
            ).solution
        else:
            raise ValueError(f"Unknown solver: {solver}")

    def predict(self, X) -> torch.Tensor:
        K = self._kernel_function(X, self.X_ref)
        return torch.matmul(K, self.alpha_)

    def forward(self, X: torch.Tensor):
        if self.return_gradient_norm:
            X = X.requires_grad_(True)
            output = self.predict(X)

            grad_outputs = torch.jit.annotate(
                List[Optional[torch.Tensor]], [torch.ones_like(output)]
            )

            grad_list = torch.autograd.grad(
                [output], [X], grad_outputs=grad_outputs, create_graph=True
            )
            grad = grad_list[0]

            if grad is None:
                grad = torch.zeros_like(X)

            grad_norm = torch.norm(grad, p=2, dim=1, keepdim=True)
            return output, grad_norm
        else:
            return self.predict(X), None

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        x = train_batch["data"]
        y = train_batch["target"]
        # =================forward====================
        self.fit()
        y_train_pred = self.predict(x)

        # ===================loss=====================
        loss = self.loss_fn(y_train_pred, y)

        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            on_step=False,
            logger=False,
        )
        ### if end of epoch, log the kernel parameters
        if (batch_idx + 1) % len(self.trainer.datamodule.train_dataloader()) == 0:
            print(
                f"Epoch: {self.current_epoch}, Training Error: {loss.item():.4f}, lambda: {self.lambda_.item():.4f}"
            )
        return loss

    def test_step(self, test_batch, batch_idx) -> torch.Tensor:
        x = test_batch["data"]
        y = test_batch["target"]
        # =================predict====================
        y_test_pred = self.predict(x)
        # ===================loss=====================
        loss = self.loss_fn(y_test_pred, y)
        # ====================log=====================
        self.log("test_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        x = val_batch["data"]
        y = val_batch["target"]
        # =================predict====================
        y_val_pred = self.predict(x)
        # ===================loss=====================
        loss = self.loss_fn(y_val_pred, y)
        # ====================log=====================
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, on_step=False, logger=False
        )
        return loss

    def on_train_epoch_end(self):
        # ====================log=====================
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if train_loss is not None and val_loss is not None:
            self.log_dict(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

    def configure_optimizers(self) -> tuple:
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
            raise ValueError(
                "lr_scheduler_kwargs must include a 'scheduler' key with the scheduler class."
            )

        scheduler_cls = self.lr_scheduler_kwargs["scheduler"]
        scheduler_kwargs = {
            k: v for k, v in self.lr_scheduler_kwargs.items() if k != "scheduler"
        }
        lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        lr_scheduler_config = {"scheduler": lr_scheduler}

        # Add possible additional config options
        if self.lr_scheduler_config:
            if "scheduler" in self.lr_scheduler_config:
                raise ValueError(
                    "lr_scheduler_config cannot override the 'scheduler' entry."
                )
            lr_scheduler_config.update(self.lr_scheduler_config)
        return [optimizer], [lr_scheduler_config]
