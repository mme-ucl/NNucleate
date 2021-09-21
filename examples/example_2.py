from torch.optim import optimizer
from NNucleate.trainig import test, train, NNCV, CVTrajectory, NdistTrajectory, KNNTrajectory
from NNucleate.data_augmentaion import transform_frame_to_knn_list, transform_frame_to_ndist_list
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import math

# Train a model on the trajectory 13 with 5000 dist and 5 neighbours
# Get a cpu or GPU for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Necessary files
traj_path = "example_files/dump.3Dcolloid.13.xyz"
cv_path = "example_files/cn.13.dat"
top_path = "example_files/reference_13.pdb"

# Start with n_dist
n_dist = 5000
mod_fun = lambda x: transform_frame_to_ndist_list(n_dist, x, [92.83, 92.83, 92.83, 90.0, 90.0, 90.0]) 
ds = CVTrajectory(cv_path, traj_path, top_path, 3, 92.83, transform = mod_fun)
train_set, test_set = torch.utils.data.random_split(ds, [int(len(ds)*0.8)+1, int(len(ds)*0.2)])

# Create the wrappers
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

epochs = 20
learning_rate = 1e-4
loss_fn = nn.MSELoss()
n_at = 421

# The model
model_1 = NNCV(n_dist, 512, 256, 128).to(device) 
optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

train_errors = []
test_errors = []

for t in range(epochs):
    print(f"Epoch {t+1}\n ------------------------------")
    tr = train(train_dataloader, model_1, loss_fn, optimizer, device, 500)
    train_errors.append(tr)
    te = test(test_dataloader, model_1, loss_fn, device)
    test_errors.append(te)

plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()

# k-NN
k = 5
mod_fun = lambda x: transform_frame_to_knn_list(k, x, [92.83, 92.83, 92.83])
ds = CVTrajectory(cv_path, traj_path, top_path, 3, 92.83, transform = mod_fun)
train_set, test_set = torch.utils.data.random_split(ds, [int(len(ds)*0.8)+1, int(len(ds)*0.2)])

# Create the wrappers
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

# The model
model_2 = NNCV(int(k*n_at/2)+1, 512, 256, 128).to(device) 
optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

train_errors = []
test_errors = []

for t in range(epochs):
    print(f"Epoch {t+1}\n ------------------------------")
    tr = train(train_dataloader, model_2, loss_fn, optimizer, device, 500)
    train_errors.append(tr)
    te = test(test_dataloader, model_2, loss_fn, device)
    test_errors.append(te)

plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()

###############

# The above works but is very slow since the kNN lists are recalculated every epoch
# For memeory intensive tasks, like training with all distances the above is more suitable
# Alternatively they can be done in preprocessing

################

# Train a model on the trajectory 13 with 5000 dist and 5 neighbours
# Get a cpu or GPU for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Necessary files
traj_path = "example_files/dump.3Dcolloid.13.xyz"
cv_path = "example_files/cn.13.dat"
top_path = "example_files/reference_13.pdb"

# Start with n_dist
n_dist = 500
# The critical difference
ds = NdistTrajectory(cv_path, traj_path, top_path, 3, [92.83, 92.83, 92.83], n_dist)
train_set, test_set = torch.utils.data.random_split(ds, [int(len(ds)*0.8)+1, int(len(ds)*0.2)])

# Create the wrappers
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

epochs = 20
learning_rate = 1e-4
loss_fn = nn.MSELoss()
n_at = 421

# The model
model_1 = NNCV(n_dist, 512, 256, 128).to(device) 
optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

train_errors = []
test_errors = []

for t in range(epochs):
    print(f"Epoch {t+1}\n ------------------------------")
    tr = train(train_dataloader, model_1, loss_fn, optimizer, device, 500)
    train_errors.append(tr)
    te = test(test_dataloader, model_1, loss_fn, device)
    test_errors.append(te)

plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()


########

# Train a model on the trajectory 13 with 5000 dist and 5 neighbours
# Get a cpu or GPU for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Necessary files
traj_path = "example_files/dump.3Dcolloid.13.xyz"
cv_path = "example_files/cn.13.dat"
top_path = "example_files/reference_13.pdb"

k = 5
# The critical difference
ds = KNNTrajectory(cv_path, traj_path, top_path, 3, [92.83, 92.83, 92.83], k)
train_set, test_set = torch.utils.data.random_split(ds, [int(len(ds)*0.8)+1, int(len(ds)*0.2)])

# Create the wrappers
train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

epochs = 20
learning_rate = 1e-4
loss_fn = nn.MSELoss()
n_at = 421

# The model
model_2 = NNCV(int(math.ceil(n_at*k/2)), 512, 256, 128).to(device) 
optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

train_errors = []
test_errors = []

for t in range(epochs):
    print(f"Epoch {t+1}\n ------------------------------")
    tr = train(train_dataloader, model_2, loss_fn, optimizer, device, 500)
    train_errors.append(tr)
    te = test(test_dataloader, model_2, loss_fn, device)
    test_errors.append(te)

plt.plot(train_errors, label="Train")
plt.plot(test_errors, label="Test")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()