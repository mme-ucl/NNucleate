import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from NNucleate.dataset import GNNTrajectory_mult
from NNucleate.models import GNNCV_mult
from NNucleate.training import train_gnn_mult, evaluate_model_gnn_mult

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


ds_train = GNNTrajectory_mult("train_cv.dat",  "train.xtc", "reference.pdb", 9.283, 0.6)
dataloader_train = DataLoader(ds_train, batch_size=64, shuffle=True)

ds_test = GNNTrajectory_mult("test_cv.dat",  "test.xyz", "reference_13.pdb", 9.283, 0.6)
dataloader_test = DataLoader(ds_test, batch_size=64, shuffle=True)

inds = [0, 1]
model_2D = GNNCV_mult(len(inds), hidden_nf=10, n_layers=2, device=device)
optimizer = optim.Adam(model_2D.parameters(), lr=5e-4)
loss = nn.MSELoss()
model_2D.to(device)

epochs = 1

for epoch in range(epochs):
    train_mse = train_gnn_mult(model_2D, dataloader_train, 421, optimizer, loss, device, inds)
    if epoch % 10 == 0:
        print("Epoch %d: %f" % (epoch, train_mse))

preds, labels, rmse, r2 = evaluate_model_gnn_mult(model_2D, dataloader_train, 421, device, inds)
preds_v, labels_v, rmse_v, r2_v = evaluate_model_gnn_mult(model_2D, dataloader_test, 421, device, [1, 3])

names = ["n", "n(Q6)"]

for i in range(len(inds)):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12,5))
    ax1.scatter(preds[:,i], labels[:,i], s=0.4, label="r = %.4f" % r2[i])
    ax1.legend()
    ax1.plot(labels[:,i], labels[:,i], color="black")
    ax1.set_title("Train " + names[inds[i]])
    ax1.set_xlabel("Prediction")
    ax1.set_ylabel("Label")
    ax2.scatter(preds_v[:,i], labels_v[:,i], s=0.4, label="r = %.4f" % r2_v[i])
    ax2.legend()
    ax2.plot(labels_v[:,i], labels_v[:,i], color="black")
    ax2.set_title("Test " + names[inds[i]])
    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Label")
    plt.tight_layout()
    plt.show()

