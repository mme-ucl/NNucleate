#!/bin/python3

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from NNucleate.dataset import GNNDataset
from NNucleate.models import GNNCV_mult
from NNucleate.training import train_gnn_unified, test_gnn_unified, evaluate_gnn_unified
from NNucleate.committor import EAM_ase


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Use the same files as example_2.py
    # Multi-output labels: use all columns (cv_col=None), select with 'inds'
    ds_train = GNNDataset(
        cv_file="train_round_4.dat",
        traj_name="train_round_4.xyz",
        top_file="ref_500.pdb",
        box_length=1.8658222,
        rc=0.35,
        rc_extra=0.49499999999999886,  # extra edges for the force field
        cv_col=None,
    )
    dataloader_train = DataLoader(ds_train, batch_size=64, shuffle=True)

    ds_test = GNNDataset(
        cv_file="test_2D_mtd.dat",
        traj_name="test_2D_mtd.xyz",
        top_file="ref_500.pdb",
        box_length=1.8658222,
        rc=0.35,
        rc_extra=0.49499999999999886,
        cv_col=None,
    )
    dataloader_test = DataLoader(ds_test, batch_size=64, shuffle=True)

    # Select which label columns to train on
    inds = [0, 1]

    # Number of nodes per frame
    n_mol = 500

    model = GNNCV_mult(len(inds), hidden_nf=25, n_layers=1, pool_fn=torch.mean, act_fn=nn.Tanh())
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    # Committor regularization parameters
    calc = EAM_ase(potential="Cu_u3.eam")
    comm = {
        "kT": 8.617333262e-5 * 1100,  
        "calc": calc,                    # placeholder calculator
        "box_length": 1.8658222,
        "lamb": 1,
        "alpha": 0.9,
        "emax": 1e6,
        "L": 5,
    }

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8)

    conv_total, conv_mse, conv_pe, conv_test, conv_test_pe = [], [], [], [], []

    epochs = 10
    for epoch in range(epochs):
        total_loss, mse_loss, path_loss = train_gnn_unified(
            model,
            dataloader_train,
            n_mol=n_mol,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            cols=inds,
            committor=comm,
        )
        conv_total.append(total_loss)
        conv_mse.append(mse_loss)
        conv_pe.append(path_loss)
        scheduler.step()

        if epoch % 5 == 0:
            test_mse, test_pe = test_gnn_unified(
                model,
                dataloader_test,
                n_mol=n_mol,
                loss_fn=loss_fn,
                device=device,
                cols=inds,
                committor=comm,
            )
            conv_test.append(test_mse)
            conv_test_pe.append(test_pe)

    preds, labels, rmse, r2 = evaluate_gnn_unified(model, dataloader_train, n_mol=n_mol, device=device, cols=inds)
    preds_v, labels_v, rmse_v, r2_v = evaluate_gnn_unified(model, dataloader_test, n_mol=n_mol, device=device, cols=inds)


    names = ["BCC", "FCC"]
    for j in range(len(inds)):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12,5))
        ax1.scatter(preds[:,j], labels[:,j], s=0.4, label="r = %.4f" % r2[j])
        ax1.legend()
        ax1.plot(labels[:,j], labels[:,j], color="black")
        ax1.set_title("Train " + names[inds[j]])
        ax1.set_xlabel("Prediction")
        ax1.set_ylabel("Label")
        ax2.scatter(preds_v[:,j], labels_v[:,j], s=0.4, label="r = %.4f | rmse=%.4f" % (r2_v[j],rmse_v[j]))
        ax2.legend()
        ax2.plot(labels_v[:,j], labels_v[:,j], color="black")
        ax2.set_title("Test " + names[inds[j]])
        ax2.set_xlabel("Prediction")
        ax2.set_ylabel("Label")
        plt.tight_layout()
        plt.savefig("Model_PTM_%s.png" % (names[j]), dpi=750)
        plt.close()

    np.savetxt("Model_pe_conv.dat", np.array(conv_total))
    np.savetxt("Model_pe_mse_conv.dat", np.array(conv_mse))
    np.savetxt("Model_pe_path_conv.dat", np.array(conv_pe))
    np.savetxt("Model_pe_test_conv.dat", np.array(conv_test))
    np.savetxt("Model_pe_test_path_conv.dat", np.array(conv_test_pe))
    torch.save(model, "Model_pe.pt")


if __name__ == "__main__":
    main()