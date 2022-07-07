from .utils import pbc
from .data_augmentaion import transform_traj_to_knn_list, transform_traj_to_ndist_list
import torch
from torch.utils.data import Dataset
import numpy as np
import mdtraj as md


class CVTrajectory(Dataset):

    def __init__(self, cv_file, traj_name, top_file, cv_col, box_length, transform=None, start=0, stop=-1, stride=1, root=1):
        """Instantiates a dataset from a trajectory file in xtc/xyz format and a text file containing the nucleation CVs (Assumes cubic cell)
        WARNING: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å

        Args:
            cv_file (str): Path to text file structured in columns containing the CVs
            traj_name (str): Path to the trajectory in .xtc or .xyz file format
            top_file (str): Path to the topology file in .pdb file format
            cv_col (int): Indicates the column in which the desired CV is written in the CV file (0 indexing)
            box_length (float). Length of the cubic cell
            transform (function, optional): A function to be applied to the configuration before returning e.g. to_dist().
            start (int, optional): Starting frame of the trajectory
            stop (int, optional): The last file of the trajectory that is read
            stride (int, optional): The stride with which the trajectory frames are read
            root (int, optional): Allows for the loading of the n-th root of the CV data (to compress the numerical range) 
        """

        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col]**(1/root)
        self.length = box_length
        self.configs = pbc(md.load(traj_name, top=top_file), self.length)[start:stop:stride]

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
    def __init__(self, cv_file, traj_name, top_file, cv_col, box_length, k, start=0, stop=-1, stride=1, root=1):
        """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the sorted distances of all atoms to their k nearest neighbours.
        WARNING: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å

        Args:
            cv_file (str): Path to the cv file.
            traj_name (str): Path to the trajectory file (.xtc/.xyz)
            top_file (str): Path to the topology file (.pdb)
            cv_col (int): Gives the colimn in which the CV of interest is stored
            box_length (float): Length of the cubic box
            k (int): Number of neighbours to consider.
            start (int, optional): Starting frame of the trajectory
            stop (int, optional): The last file of the trajectory that is read
            stride (int, optional): The stride with which the trajectory frames are read
            root (int, optional): Allows for the loading of the n-th root of the CV data (to compress the numerical range) 
        """

        
        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col]**(1/root)
        self.box_length = box_length
        self.configs = transform_traj_to_knn_list(k, pbc(md.load(traj_name, top=top_file), self.box_length)[start:stop:stride].xyz, box_length)


    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = torch.tensor(self.configs[idx]).float()
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()

        return config/self.box[0], label


class NdistTrajectory(Dataset):
    def __init__(self, cv_file, traj_name, top_file, cv_col, box_length, n_dist, start=0, stop=-1, stride=1, root=1):
        """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the n_dist sorted distances.

        Args:
            cv_file (str): Path to the cv file.
            traj_name (str): Path to the trajectory file (.xtc/.xyz)
            top_file (str): Path to the topology file (.pdb)
            cv_col (int): Gives the colimn in which the CV of interest is stored
            box_length (float): Length of the cubic box.
            n_dist (int): Number of distances to consider.
            start (int, optional): Starting frame of the trajectory
            stop (int, optional): The last file of the trajectory that is read
            stride (int, optional): The stride with which the trajectory frames are read
            root (int, optional): Allows for the loading of the n-th root of the CV data (to compress the numerical range) 
        """

        
        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col]**(1/root)
        self.box_length = box_length
        self.configs = transform_traj_to_ndist_list(n_dist, pbc(md.load(traj_name, top=top_file), self.box_length)[start:stop:stride].xyz, box_length)


    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = torch.tensor(self.configs[idx]).float()
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()

        return config/self.box[0], label



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
