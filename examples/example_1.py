#!/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from NNucleate.dataset import GNNDataset
from NNucleate.models import initialise_weights, GNNCV
from NNucleate.training import train_gnn_unified, evaluate_gnn_unified


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    top_path = "reference.pdb"

    # Unified GNN dataset (single target via cv_col)
    ds_train = GNNDataset(
        cv_file="train_cv.dat",
        traj_name="train.xyz",
        top_file=top_path,
        box_length=9.283,
        rc=0.6,
        cv_col=1,
        root=3,
    )
    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)

    ds_val = GNNDataset(
        cv_file="cn_val.dat",
        traj_name="val.xyz",
        top_file=top_path,
        box_length=9.283,
        rc=0.6,
        cv_col=3,
        root=3,
    )
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=False)

    # Number of nodes per frame (atoms or molecules)
    n_mol = 421

    # Seed and model
    seed = 2351453
    torch.manual_seed(seed)

    model = GNNCV(hidden_nf=10, n_layers=2, device=device)
    model.apply(initialise_weights)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    epochs = 1
    for epoch in range(epochs):
        train_mse = train_gnn_unified(
            model, dl_train, n_mol=n_mol, optimizer=optimizer, loss_fn=loss_fn, device=device
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {train_mse:.6f}")

    preds, labels, rmse, r2 = evaluate_gnn_unified(model, dl_val, n_mol=n_mol, device=device)

    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final r^2: {r2:.4f}")

    plt.scatter(preds, labels, s=0.4)
    plt.plot(labels, labels, color="black")
    plt.title(f"Validation: r = {r2:.4f}")
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.show()

    torch.save(model, "model_.pt")


if __name__ == "__main__":
    main()
