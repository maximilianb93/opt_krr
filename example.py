import os
import lightning
import torch
from opt_krr.krr_model import KernelRidgeRegression
from mlcolvar.data import DictModule, DictDataset


import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Creating synthetic datasets
    torch.manual_seed(0)
    X = torch.randn(100, 3)  # 100 samples, 3 features
    y = torch.randn(100)  # 100 target values

    dataset = DictDataset(dict(data=torch.Tensor(X), target=torch.Tensor(y)))
    datamodule = DictModule(dataset, lengths=[0.4, 0.4, 0.2], shuffle=True)
    datamodule.setup()

    for batch in datamodule.val_dataloader():
        X_ref = batch["data"]
        y_ref = batch["target"]

    # One can apply q whitening transformation to the datasets:
    # mean, whitening_matrix = compute_whitening_parameters(X_train)

    # X_train_whitened = whiten_data(X_train, mean, whitening_matrix)
    # X_test_whitened = whiten_data(X_test, mean, whitening_matrix)

    # Initializing the KRR model
    input_dim = X_ref.shape[1]  # Number of features
    krr = KernelRidgeRegression(
        X_ref=X_ref,
        y_ref=y_ref,
        kernel="lap",
        lambda_=1.0,
        gamma=torch.Tensor([0.5, 0.5, 0.5]),
        input_dim=input_dim,
    )
    trainer = lightning.Trainer(max_epochs=200)
    trainer.fit(krr, datamodule=datamodule)
    trainer.test(krr, datamodule=datamodule)

    ### Deploy the model
    scripted_model = krr.to_torchscript(method="script")
    scripted_model.save("krr_scripted.pt")
