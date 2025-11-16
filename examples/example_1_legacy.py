#!/bin/python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from NNucleate.dataset import GNNTrajectory
from NNucleate.models import initialise_weights, GNNCV
from NNucleate.training import evaluate_model_gnn, early_stopping_gnn, train_gnn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

top_path="reference.pdb"
set_train = GNNTrajectory("train_cv.dat", "train.xyz", top_path, 1, 9.283, 0.6,root=3)
dl_train = DataLoader(set_train, batch_size=64, shuffle=True)

val_cv_path = "cn_val.dat"
val_traj_path = "val.xyz"
ds_GNN_val = GNNTrajectory(val_cv_path, val_traj_path, top_path, 3, 9.283, 0.6, root=3)
dl_GNN_val = DataLoader(ds_GNN_val, batch_size=64, shuffle=False)

n_at = 421

seed = 2351453
torch.manual_seed(seed)

model = GNNCV(hidden_nf=10, n_layers=2, device=device)
model.apply(initialise_weights)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss = nn.MSELoss()

epochs = 1
for epoch in epochs:
    train_mse = train_gnn(model, dl_train, 421, optimizer, loss, device)
    if epoch % 10 == 0:
        print("Epoch %d: %f" % (epoch, train_mse))  
#model, train_conv, test_conv = early_stopping_gnn(model, dl_train, dl_GNN_val, n_at, optimizer, loss, device, test_freq=10)
preds, label, rmse, r2 = evaluate_model_gnn(model, dl_GNN_val, n_at, device)

print("Final RMSE: %.4f" % rmse)
print("Final $r^2$: %.4f" % r2)

#plt.plot(train_conv)
#plt.plot(test_conv)

#plt.yscale("log")
#plt.xlabel("epochs")
#plt.ylabel("RMSE")  
#plt.show()

plt.scatter(preds, label, s=0.4)
plt.plot(label, label, color="black")
plt.title("Train: r = %.4f" % r2)   
plt.xlabel("Prediction")
plt.ylabel("Label")
plt.show()

torch.save(model, "model_.pt")
