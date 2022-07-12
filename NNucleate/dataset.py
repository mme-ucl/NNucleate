from torch.utils.data import Dataset
import numpy as np
import mdtraj as md
import torch
from torch import nn
from .utils import pbc, get_rc_edges
from .data_augmentation import transform_traj_to_knn_list, transform_traj_to_ndist_list


class CVTrajectory(Dataset):
    """
    Instantiates a dataset from a trajectory file in xtc/xyz format and a text file containing the nucleation CVs (Assumes cubic cell)
    
    .. warning:: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å.

    :param cv_file: Path to text file structured in columns containing the CVs.
    :type cv_file: str
    :param traj_name: Path to the trajectory in .xtc or .xyz file format.
    :type traj_name: str
    :param top_file: Path to the topology file in .pdb file format.
    :type top_file: str
    :param cv_col: Indicates the column in which the desired CV is written in the CV file (0 indexing).
    :type cv_col: int
    :param box_length: Length of the cubic cell.
    :type box_length: float
    :param transform: A function to be applied to the configuration before returning e.g. to_dist(), defaults to None.
    :type transform: function, optional
    :param start: Starting frame of the trajectory, defaults to 0.
    :type start: int, optional
    :param stop: The last file of the trajectory that is read, defaults to -1.
    :type stop: int, optional
    :param stride: The stride with which the trajectory frames are read, defaults to 1.
    :type stride: int, optional
    :param root: Allows for the loading of the n-th root of the CV data (to compress the numerical range), defaults to 1.
    :type root: int, optional
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        cv_col: int,
        box_length: float,
        transform=None,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):

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
    """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the sorted distances of all atoms to their k nearest neighbours.

    .. warning:: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å.

    :param cv_file: Path to the cv file.
    :type cv_file: str
    :param traj_name: Path to the trajectory file (.xtc/.xyz).
    :type traj_name: str
    :param top_file: Path to the topology file (.pdb).
    :type top_file: str
    :param cv_col: Gives the colimn in which the CV of interest is stored.
    :type cv_col: int
    :param box_length: Length of the cubic box.
    :type box_length: float
    :param k: Number of neighbours to consider.
    :type k: int
    :param start: Starting frame of the trajectory, defaults to 0.
    :type start: int, optional
    :param stop: The last file of the trajectory that is read, defaults to -1.
    :type stop: int, optional
    :param stride: The stride with which the trajectory frames are read, defaults to 1.
    :type stride: int, optional
    :param root: Allows for the loading of the n-th root of the CV data (to compress the numerical range), defaults to 1.
    :type root: int, optional
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        cv_col: int,
        box_length: float,
        k: int,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):

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
    """Generates a dataset from a trajectory in .xtc/xyz format. 
        The trajectory frames are represented via the n_dist sorted distances.
    
    .. warning:: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å.

    :param cv_file: Path to the cv file.
    :type cv_file: str
    :param traj_name: Path to the trajectory file (.xtc/.xyz).
    :type traj_name: str
    :param top_file: Path to the topology file (.pdb).
    :type top_file: str
    :param cv_col: Gives the colimn in which the CV of interest is stored.
    :type cv_col: int
    :param box_length: Length of the cubic box.
    :type box_length:float
    :param n_dist: Number of distances to consider.
    :type n_dist: int
    :param start: Starting frame of the trajectory, defaults to 0.
    :type start: int, optional
    :param stop: The last file of the trajectory that is read, defaults to -1.
    :type stop: int, optional
    :param stride: The stride with which the trajectory frames are read, defaults to 1.
    :type stride: int, optional
    :param root: Allows for the loading of the n-th root of the CV data (to compress the numerical range), defaults to 1.
    :type root: int, optional
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        cv_col: int,
        box_length: float,
        n_dist: int,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):

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
    """Generates a dataset from a trajectory in .xtc/.xyz format for the training of a GNN. 
    .. warning:: For .xtc give the boxlength in nm and for .xyz give the boxlength in Å.

    :param cv_file: Path to the cv file.
    :type cv_file: str
    :param traj_name: Path to the trajectory file (.xtc/.xyz).
    :type traj_name: str
    :param top_file: Path to the topology file (.pdb).
    :type top_file: str
    :param cv_col: Gives the colimn in which the CV of interest is stored.
    :type cv_col: int
    :param box_length: Length of the cubic box.
    :type box_length: float
    :param rc: Cut-off radius for the construction of the graph.
    :type rc: float
    :param start: Starting frame of the trajectory, defaults to 0.
    :type start: int, optional
    :param stop: The last file of the trajectory that is rea, defaults to -1.
    :type stop: int, optional
    :param stride: The stride with which the trajectory frames are read, defaults to 1.
    :type stride: int, optional
    :param root: Allows for the loading of the n-th root of the CV data (to compress the numerical range), defaults to 1.
    :type root: int, optional
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        cv_col: int,
        box_length: float,
        rc: float,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):

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
