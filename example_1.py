import NNucleate
from NNucleate.trainig import test, train_perm, NNCV, CVTrajectory
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time
import numpy as np

# Train multiple models with train perm and plot performance against n_trans
# Get a cpu or GPU for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


ns = [1, 2, 5]

traj_path = "example_files/dump.3Dcolloid.13.xyz"
cv_path = "example_files/cn.13.dat"
top_path = "example_files/reference_13.pdb"
val_path = "example_files/even_shuffled_6k.xtc"
val_cv_path = "example_files/even_shuffled_6k_cv.dat"

ds = CVTrajectory(cv_path, traj_path, top_path, 3, 92.83)
ds_val = CVTrajectory(val_cv_path, val_path, top_path, 1, 92.83)
train_set, test_set = torch.utils.data.random_split(ds, [int(len(ds)*0.8)+1, int(len(ds)*0.2)])

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=64, shuffle=True)

epochs = 200
learning_rate = 1e-4
loss_fn = nn.MSELoss()
n_at = 421

cost = []
performance = []
trainings = []
tests = []
vals = []

for n in ns:
    t1 = time()
    model = NNCV(n_at*3, 512, 256, 128).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    train_errors = []
    test_errors = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n ------------------------------")
        tr = train_perm(train_dataloader, model, loss_fn, optimizer, n, device, 50000)
        train_errors.append(tr)
        #scheduler.step()
        te = test(test_dataloader, model, loss_fn, device)
        test_errors.append(te)

    t2 = time() - t1
    cost.append(t2)
    trainings.append(train_errors[-1])
    tests.append(test_errors[-1])
    print("---Validation---")
    val = test(val_loader, model, loss_fn, device)
    performance.append(val)

    plt.plot(train_errors, label="Train")
    plt.plot(test_errors, label="Test")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("figs/lr_%d.png" % n)
    plt.show()

    print(" n = %d Done!" % n)

print("--- cost ---")
print(cost)
print("--- performance ---")
print(performance)
print("--- trainings ---")
print(trainings)
print("--- test ---")
print(tests)

cost = np.array(cost)/3600

fig, ax1 = plt.subplots()

ax1.plot(ns, trainings, label="Train Error")
ax1.plot(ns, tests, label="Test Error")
ax1.plot(ns, performance, label="Val. Error")
ax1.legend()
ax1.set_xlabel("Number of Permutations")
ax1.set_ylabel("RMSE")
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
color = 'tab:purple'
ax2.set_ylabel("GPU hours", color=color)
ax2.plot(ns, cost, color=color, linestyle="dashed")
ax2.tick_params(axis="y", labelcolor=color)
fig.tight_layout()
plt.savefig("example_files/comp_graph.png", dpi=500)


######################
# Testing grounds
#####################

traj_path = "example_files/dump.3Dcolloid.13.xyz"
cv_path = "example_files/cn.13.dat"
top_path = "example_files/reference_13.pdb"
val_path = "example_files/even_shuffled_6k.xtc"
val_cv_path = "example_files/even_shuffled_6k_cv.dat"

ds = CVTrajectory(cv_path, traj_path, top_path, 3, 92.83)
import mdtraj as md
configs = md.load(traj_path, top=top_path)

class CVTraj:

    def __init__(self, cv_file, traj_name, top_file, cv_col, box_length, transform=None):
        """Instantiates a dataset from a trajectory file in xtc/xyz format and a text file containing the nucleation CVs (Assumes cubic cell)

        Args:
            cv_file (str): Path to text file structured in columns containing the CVs
            traj_name (str): Path to the trajectory in .xtc or .xyz file format
            top_file (str): Path to the topology file in .pdb file format
            cv_col (int): Indicates the column in which the desired CV is written in the CV file (0 indexing)
            box_length (float). Length of the cubic cell
            transform (function, optional): A function to be applied to the configuration before returning e.g. to_dist(). 
        """
        self.cv_labels = np.loadtxt(cv_file)[:, cv_col]
        self.configs = md.load(traj_name, top=top_file)
        self.length = box_length
        # Option to transform the configs before returning
        self.transform = transform

ds = CVTraj(cv_path, traj_path, top_path, 3, 92.83)

ds = CVTrajectory(cv_path, traj_path, top_path, 3, 92.83)