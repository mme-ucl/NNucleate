import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from NNucleate.dataset import GNNDataset
from NNucleate.models import GNNCV_mult
from NNucleate.training import train_gnn_unified, evaluate_gnn_unified


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Multi-output dataset: use all label columns (cv_col=None), select targets with 'inds'
    ds_train = GNNDataset(
        cv_file="train_cv.dat",
        traj_name="train.xtc",
        top_file="reference.pdb",
        box_length=9.283,
        rc=0.6,
        cv_col=None,
    )
    dataloader_train = DataLoader(ds_train, batch_size=64, shuffle=True)

    ds_test = GNNDataset(
        cv_file="test_cv.dat",
        traj_name="test.xyz",
        top_file="reference_13.pdb",
        box_length=9.283,
        rc=0.6,
        cv_col=None,
    )
    dataloader_test = DataLoader(ds_test, batch_size=64, shuffle=True)

    # Select which target columns to train on
    inds = [0, 1]

    # Number of nodes per frame
    n_mol = 421

    model = GNNCV_mult(len(inds), hidden_nf=10, n_layers=2, device=device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    model.to(device)

    epochs = 1
    for epoch in range(epochs):
        train_mse = train_gnn_unified(
            model,
            dataloader_train,
            n_mol=n_mol,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            cols=inds,
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {train_mse:.6f}")

    preds, labels, rmse, r2 = evaluate_gnn_unified(
        model, dataloader_train, n_mol=n_mol, device=device, cols=inds
    )
    preds_v, labels_v, rmse_v, r2_v = evaluate_gnn_unified(
        model, dataloader_test, n_mol=n_mol, device=device, cols=[1, 3]
    )

    names = ["n", "n(Q6)"]

    for i in range(len(inds)):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12, 5))
        ax1.scatter(preds[:, i], labels[:, i], s=0.4, label=f"r = {r2[i]:.4f}")
        ax1.legend()
        ax1.plot(labels[:, i], labels[:, i], color="black")
        ax1.set_title("Train " + names[inds[i]])
        ax1.set_xlabel("Prediction")
        ax1.set_ylabel("Label")

        ax2.scatter(preds_v[:, i], labels_v[:, i], s=0.4, label=f"r = {r2_v[i]:.4f}")
        ax2.legend()
        ax2.plot(labels_v[:, i], labels_v[:, i], color="black")
        ax2.set_title("Test " + names[inds[i]])
        ax2.set_xlabel("Prediction")
        ax2.set_ylabel("Label")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
