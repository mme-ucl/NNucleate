from torch.utils.data import Dataset
import numpy as np
import mdtraj as md
import torch
from torch import nn
from .utils import pbc, get_rc_edges
from .data_augmentaion import transform_traj_to_knn_list, transform_traj_to_ndist_list


class CVTrajectory(Dataset):
    def __init__(
        self,
        cv_file,
        traj_name,
        top_file,
        cv_col,
        box_length,
        transform=None,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
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

        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col] ** (1 / root)
        self.length = box_length
        self.configs = pbc(md.load(traj_name, top=top_file), self.length)[
            start:stop:stride
        ]

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

        return config / self.length, label


class KNNTrajectory(Dataset):
    def __init__(
        self,
        cv_file,
        traj_name,
        top_file,
        cv_col,
        box_length,
        k,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
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

        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col] ** (1 / root)
        self.box_length = box_length
        self.configs = transform_traj_to_knn_list(
            k,
            pbc(md.load(traj_name, top=top_file), self.box_length)[
                start:stop:stride
            ].xyz,
            box_length,
        )

    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = torch.tensor(self.configs[idx]).float()
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()

        return config / self.box_length, label


class NdistTrajectory(Dataset):
    def __init__(
        self,
        cv_file,
        traj_name,
        top_file,
        cv_col,
        box_length,
        n_dist,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
        """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the n_dist sorted distances.
        WARNING: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å

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

        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col] ** (1 / root)
        self.box_length = box_length
        self.configs = transform_traj_to_ndist_list(
            n_dist,
            pbc(md.load(traj_name, top=top_file), self.box_length)[
                start:stop:stride
            ].xyz,
            box_length,
        )

    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = torch.tensor(self.configs[idx]).float()
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()

        return config / self.box_length, label


class GNNTrajectory(Dataset):
    def __init__(
        self,
        cv_file,
        traj_name,
        top_file,
        cv_col,
        box_length,
        rc,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
        """Generates a dataset from a trajectory in .xtc/.xyz format for the training of a GNN. 

        Args:
            cv_file (str): Path to the cv file.
            traj_name (str): Path to the trajectory file (.xtc/.xyz)
            top_file (str): Path to the topology file (.pdb)
            cv_col (int): Gives the colimn in which the CV of interest is stored
            box_length (float): Length of the cubic box.
            rc (float): Cut-off radius for the construction of the graph
            start (int, optional): Starting frame of the trajectory. Defaults to 0.
            stop (int, optional): The last file of the trajectory that is read. Defaults to -1.
            stride (int, optional): The stride with which the trajectory frames are read. Defaults to 1.
            root (int, optional): Allows for the loading of the n-th root of the CV data (to compress the numerical range). Defaults to 1. 
        """

        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col] ** (1 / root)
        self.length = box_length
        traj = pbc(md.load(traj_name, top=top_file), self.length)[start:stop:stride]
        vecs = np.zeros((len(traj), 3, 3))
        for i in range(len(traj)):
            vecs[i, 0] = np.array([self.length, 0, 0])
            vecs[i, 1] = np.array([0, self.length, 0])
            vecs[i, 2] = np.array([0, 0, self.length])
        traj.unitcell_vectors = vecs
        traj = pbc(traj, self.length)
        self.rows, self.cols = get_rc_edges(rc, traj)
        self.max_l = np.max([len(r) for r in self.rows])
        print(self.max_l)
        self.configs = traj.xyz

    def __len__(self):
        # Returns the length of the dataset
        return len(self.cv_labels)

    def __getitem__(self, idx):
        # Gets a configuration from the given index
        config = self.configs[idx]
        # Label is read from the numpy array
        label = torch.tensor(self.cv_labels[idx]).float()
        # if transformation functions are set they are applied to label and image
        config = torch.tensor(config).float()
        rows = self.rows[idx]
        cols = self.cols[idx]
        rows = nn.functional.pad(rows, [0, self.max_l - len(rows)], value=-1)
        cols = nn.functional.pad(cols, [0, self.max_l - len(cols)], value=-1)
        return config / self.length, label, rows, cols
