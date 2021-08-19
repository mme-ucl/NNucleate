from utils import pbc
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation as R

class CVTrajectory(Dataset):

    def __init__(self, cv_file, traj_name, top_file, cv_col, box, transform=None):
        """Instantiates a dataset from a trajectory file in xtc/xyz format and a text file containing the nucleation CVs

        Args:
            cv_file (str): Path to text file structured in columns containing the CVs
            traj_name (str): Path to the trajectory in .xtc or .xyz file format
            top_file (str): Path to the topology file in .pdb file format
            cv_col (int): Indicates the column in which the desired CV is written in the CV file (0 indexing)
            transform (function, optional): A function to be applied to the configuration before returning e.g. to_dist(). 
        """
        self.cv_labels = np.loadtxt(cv_file)[:, cv_col]
        self.configs = pbc(md.load(traj_name, top=top_file), box).xyz
        # Option to transform the configs before returning
        self.transform = transform


    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = torch.tensor(self.configs[idx]).float()
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()
        # if transformation functions are set they are applied to label and image

        if self.transform:
            config = self.transform(config)

        return config, label


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

def train_loop(dataloader, model, loss_fn, optimizer, device, n_trans=0, transform=None):
    """Performs one training epoch for an NNCV. The loss is not just evaluated on the trajectory frames but on n_trans transformed ones.

    Args:
        dataloader (dataloader): The dataloader loading the training set
        model (NNCV): The NNCV model to train
        loss_fn (function): Pytorch loss function to be used
        optimizer (function): Pytorch optimizer to be used
        n_trans (int): Number of transformations performed on each frame for loss calculation
        transform (str, optional): Describes the type of transformation (Either Permutation or Rotation). Defaults to "Permutation".

    Raises:
        ValueError: Raised if unknown transformation is given.

    Returns:
        float: The final training loss
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        if transform == "Permutation":
            for i in range(n_trans):
                # Shuffle the tensors in the batch
                pred = model(X[:, torch.randperm(X.size()[1])])
                loss += loss_fn(pred.flatten(), y)

        elif transform == "Rotation":
            for i in range(n_trans):
                quat = md.utils.uniform_quaternion()
                rot = R.from_quat(quat)
                X = [rot.apply(x) for x in X]
                pred = model(X)
                loss += loss_fn(pred.flatten(), y)

        elif transform:
            raise ValueError("This function currently only supports transform= \"Permutation\" or transform = \"Rotation\"")
        
        loss /= n_trans
        #print(loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

            
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