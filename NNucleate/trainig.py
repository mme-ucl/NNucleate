from .utils import pbc
from .data_augmentaion import transform_traj_to_knn_list, transform_traj_to_ndist_list
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation as R


class CVTrajectory(Dataset):

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
        self.length = box_length
        self.configs = pbc(md.load(traj_name, top=top_file), self.length).xyz

        # Option to transform the configs before returning
        self.transform = transform

    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = self.configs[idx]
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()
        # if transformation functions are set they are applied to label and image

        if self.transform:
            config = torch.tensor(self.transform(config)).float()
        else:
            config = torch.tensor(config).float()

        return config/self.length, label


class KNNTrajectory(Dataset):
    def __init__(self, cv_file, traj_name, top_file, cv_col, box, k):
        """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the sorted distances of all atoms to their k nearest neighbours.

        Args:
            cv_file (str): Path to the cv file.
            traj_name (str): Path to the trajectory file (.xtc/.xyz)
            top_file (str): Path to the topology file (.pdb)
            cv_col (int): Gives the colimn in which the CV of interest is stored
            box (list): List of box vectors (cubic).
            k (int): Number of neighbours to consider.
        """

        
        self.cv_labels = np.loadtxt(cv_file)[:, cv_col]
        self.box = box
        self.configs = transform_traj_to_knn_list(k, pbc(md.load(traj_name, top=top_file), self.length).xyz, box)


    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = self.configs[idx]
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()
        # if transformation functions are set they are applied to label and image

        if self.transform:
            config = torch.tensor(self.transform(config)).float()
        else:
            config = torch.tensor(config).float()

        return config/self.box[0], label

class NdistTrajectory(Dataset):
    def __init__(self, cv_file, traj_name, top_file, cv_col, box, n_dist):
        """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the n_dist sorted distances.

        Args:
            cv_file (str): Path to the cv file.
            traj_name (str): Path to the trajectory file (.xtc/.xyz)
            top_file (str): Path to the topology file (.pdb)
            cv_col (int): Gives the colimn in which the CV of interest is stored
            box (list): List of box vectors (cubic).
            n_dist (int): Number of distances to consider.
        """

        
        self.cv_labels = np.loadtxt(cv_file)[:, cv_col]
        self.box = box
        self.configs = transform_traj_to_ndist_list(n_dist, pbc(md.load(traj_name, top=top_file), self.box[0]).xyz, box)


    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = self.configs[idx]
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()
        # if transformation functions are set they are applied to label and image

        if self.transform:
            config = torch.tensor(self.transform(config)).float()
        else:
            config = torch.tensor(config).float()

        return config/self.box[0], label


# Model
class NNCV(nn.Module):
    def __init__(self, insize, l1, l2=0, l3=0):
        """Instantiates an NN for approximating CVs. Supported are architectures with up to 3 layers.

        Args:
            insize (int): Size of the input layer
            l1 (int): Size of dense layer 1
            l2 (int, optional): Size of dense layer 2. Defaults to 0.
            l3 (int, optional): Size of dense layer 3. Defaults to 0.
        """
        super(NNCV, self).__init__()
        self.flatten = nn.Flatten()
        # defines the structure
        if l2 > 0:
            if l3 > 0:
                self.sig_stack = nn.Sequential(
                    nn.Linear(insize, l1),
                    nn.Sigmoid(),
                    nn.Linear(l1, l2),
                    nn.Sigmoid(),
                    nn.Linear(l2, l3),
                    nn.Sigmoid(),
                    nn.Linear(l3, 1)
                )
            else:
                self.sig_stack = nn.Sequential(
                    nn.Linear(insize, l1),
                    nn.Sigmoid(),
                    nn.Linear(l1, l2),
                    nn.Sigmoid(),
                    nn.Linear(l2, 1)
                )
        else:
            self.sig_stack = nn.Sequential(
                nn.Linear(insize, l1),
                nn.Sigmoid(),
                nn.Linear(l3, 1)
            )

    def forward(self, x):
        # defines the application of the network to data
        # NEVER call forward directly
        # Only say model(x)
        x = self.flatten(x)
        label = self.sig_stack(x)
        return label


def train(dataloader, model, loss_fn, optimizer, device, print_batch=1000000):
    """Performs one training epoch for a NNCV.

    Args:
        dataloader (Dataloader): Wrappper for the training set
        model (NNCV): The network to be trained
        loss_fn (function): Pytorch loss to be used during training
        optimizer (function): Pytorch optimizer to be used during training
        device (dvice): Pytorch device to run the calculation on. Supports CPU and GPU (cuda)
        print_batch (int, optional): Set to recieve printed updates on the lost every print_batch batches. Defaults to 1000000.

    Returns:
        float: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


def train_perm(dataloader, model, loss_fn, optimizer, n_trans, device, print_batch=1000000):
    """Performs one training epoch for a NNCV but the loss for each batch is not just calculated on one reference structure but a set of n_trans permutated versions of that structure.

    Args:
        dataloader (Dataloader): Wrapper around the training data
        model (NNCV): The model that is to be trained
        loss_fn (function): Pytorch loss function to be used for the training
        optimizer (function): Pytorch optimizer to be used during training
        n_trans (int): Number of permutated structures used for the loss calculations
        device (device): Pytorch device to run the calculations on. Supports CPU and GPU (cuda)
        print_batch (int, optional): Set to recieve printed updates on the loss every print_batches batches. Defaults to 1000000.

    Returns:
        float: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        for i in range(n_trans):
            # Shuffle the tensors in the batch
            pred = model(X[:, torch.randperm(X.size()[1])])
            loss += loss_fn(pred.flatten(), y)

        loss /= n_trans
        # print(loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


def test(dataloader, model, loss_fn, device):
    """Calculates the current average test set loss.

    Args:
        dataloader (dataloader): Dataloader loading the test set
        model (NNCV): model that is being trained
        loss_fn (function): Pytorch loss function

    Returns:
        float: Current avg. test set loss
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.flatten(), y).item()

    test_loss /= num_batches
    print(f"Avg. Test loss: {test_loss:>8f} \n")
    return test_loss
