import torch
from time import time
from copy import deepcopy
import numpy as np
from NNucleate.models import GNNCV, NNCV
from torch.utils.data import DataLoader
from typing import Callable
from scipy.spatial.transform import Rotation as R
import mdtraj as md

from NNucleate.utils import pbc_config


def train_linear(
    model_t: NNCV,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: Callable,
    device: str,
    print_batch=1000000,
) -> float:
    """Performs one training epoch for a NNCV.

    :param model_t: The network to be trained.
    :type model_t: NNCV
    :param dataloader: Wrappper for the training set.
    :type dataloader: torch.utils.data.Dataloader
    :param loss_fn: Pytorch loss to be used during training.
    :type loss_fn: torch.nn._Loss
    :param optimizer: Pytorch optimizer to be used during training.
    :type optimizer: torch.optim
    :param device: Pytorch device to run the calculation on. Supports CPU and GPU (cuda).
    :type device: str
    :param print_batch: Set to recieve printed updates on the lost every print_batch batches, defaults to 1000000.
    :type print_batch: int, optional
    :return: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    :rtype: float
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model_t(X)
        loss = loss_fn(pred.flatten(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


def train_gnn(
    model: GNNCV,
    loader: DataLoader,
    n_mol: int,
    optimizer: Callable,
    loss: Callable,
    device: str,
    n_at=1,
) -> float:
    """Function to perform one epoch of a GNN training.

    :param model: Graph-based model_t to be trained.
    :type model: GNNCV
    :param loader: Wrapper around a GNNTrajectory dataset.
    :type loader: torch.utils.data.Dataloader
    :param n_at: Number of nodes per frame.
    :type n_at: int
    :param optimizer: The optimizer object for the training.
    :type optimizer: torch.optim
    :param loss: Loss function for the training.
    :type loss: torch.nn._Loss
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Return the average loss over the epoch.
    :rtype: float
    """
    res = {"loss": 0, "counter": 0}
    for batch, (X, y, r, c) in enumerate(loader):
        model.train()
        optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)

        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        label = y.to(device)

        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)
        l = loss(pred, label)
        l.backward()
        optimizer.step()

        res["loss"] += l.item() * batch_size
        res["counter"] += batch_size

    return res["loss"] / res["counter"]


def train_gnn_mult(
    model: GNNCV,
    loader: DataLoader,
    n_mol: int,
    optimizer,
    loss,
    device: str,
    cols: list,
    n_at=1,
) -> float:
    """Function to perform one epoch of a GNN with multidimensional output training.

    :param model: Graph-based model_t to be trained.
    :type model: GNNCV
    :param loader: Wrapper around a GNNTrajectory dataset.
    :type loader: torch.utils.data.Dataloader
    :param n_at: Number of nodes per frame.
    :type n_at: int
    :param optimizer: The optimizer object for the training.
    :type optimizer: torch.optim
    :param loss: Loss function for the training.
    :type loss: torch.nn._Loss
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param cols: List of column indices representing the CVs the model is learning from the dataset.
    :type cols: list
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Return the average loss over the epoch.
    :rtype: float
    """
    res = {"loss": 0, "counter": 0}
    for batch, (X, y, r, c) in enumerate(loader):
        model.train()
        optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)

        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        label = y[:,cols].to(device)

        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)

        l = loss(pred, label)
        l.backward()
        optimizer.step()

        res["loss"] += l.item() * batch_size
        res["counter"] += batch_size

    return res["loss"] / res["counter"]

def train_gnn_e(
    model,
    loader: DataLoader,
    n_mol: int,
    optimizer,
    loss,
    cols,
    device: str,
    n_at=1,
) -> float:
    """Function to perform one epoch of a GNN training.

    :param model: Graph-based model_t to be trained.
    :type model: GNNCV
    :param loader: Wrapper around a GNNTrajectory dataset.
    :type loader: torch.utils.data.Dataloader
    :param n_at: Number of nodes per frame.
    :type n_at: int
    :param optimizer: The optimizer object for the training.
    :type optimizer: torch.optim
    :param loss: Loss function for the training.
    :type loss: torch.nn._Loss
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Return the average loss over the epoch.
    :rtype: float
    """
    resi = {"loss": 0, "counter": 0}
    for batch, (X, y, r, c, re, ce) in enumerate(loader):
        model.train()
        optimizer.zero_grad()
        batch_size = len(X)
        X.requires_grad = True
        ap = X.view(-1, 3 * n_at).to(device)
        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        label = y[:,cols].to(device)
        pred = model(x=ap, edges=edges, n_nodes=n_mol)
        #print(path_loss)
        l = loss(pred, label)
        l.backward()
        optimizer.step()

        resi["loss"] += l.item() * batch_size
        resi["counter"] += batch_size

    return  resi["loss"] / resi["counter"]


def train_perm(
    model_t: NNCV,
    dataloader: DataLoader,
    optimizer: Callable,
    loss_fn: Callable,
    n_trans: int,
    device: str,
    print_batch=1000000,
) -> float:
    """Performs one training epoch for a NNCV but the loss for each batch is not just calculated on one reference structure but a set of n_trans permutated versions of that structure.

    :param dataloader: Wrapper around a GNNTrajectory dataset.
    :type dataloader: torch.utils.data.Dataloader
    :param optimizer: The optimizer object for the training.
    :type optimizer: torch.optim
    :param loss_fn: Loss function for the training.
    :type loss_fn: torch.nn._Loss
    :param n_trans: Number of permutated structures used for the loss calculations.
    :type n_trans: int
    :param device: Pytorch device to run the calculations on. Supports CPU and GPU (cuda).
    :type device: str
    :param print_batch: Set to recieve printed updates on the loss every print_batches batches, defaults to 1000000.
    :type print_batch: int, optional
    :return: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    :rtype: float
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model_t(X)
        loss = loss_fn(pred.flatten(), y)

        for i in range(n_trans):
            # Shuffle the tensors in the batch
            pred = model_t(X[:, torch.randperm(X.size()[1])])
            loss += loss_fn(pred.flatten(), y)

        loss /= n_trans

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


def train_rot(
    model_t: NNCV,
    dataloader: DataLoader,
    optimizer: Callable,
    loss_fn: Callable,
    n_trans: int,
    device: str,
    print_batch=1000000,
) -> float:
    """Performs one training epoch for a NNCV but the loss for each batch is not just calculated on one reference structure but a set of n_trans rotated versions of that structure.

    :param dataloader: Wrapper around a GNNTrajectory dataset.
    :type dataloader: torch.utils.data.Dataloader
    :param optimizer: The optimizer object for the training.
    :type optimizer: torch.optim
    :param loss_fn: Loss function for the training.
    :type loss_fn: torch.nn._Loss
    :param n_trans: Number of rotated structures used for the loss calculations.
    :type n_trans: int
    :param device: Pytorch device to run the calculations on. Supports CPU and GPU (cuda).
    :type device: str
    :param print_batch: Set to recieve printed updates on the loss every print_batches batches, defaults to 1000000.
    :type print_batch: int, optional
    :return: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    :rtype: float
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model_t(X)
        loss = loss_fn(pred.flatten(), y)

        for i in range(n_trans):
            # Shuffle the tensors in the batch
            quat = md.utils.uniform_quaternion()
            rot = R.from_quat(quat)
            X_rot = torch.tensor([pbc_config(rot.apply(x), 1.0) for x in X]).float()
            pred = model_t(X_rot)
            loss += loss_fn(pred.flatten(), y)

        loss /= n_trans

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


from committor_regularisation import e_path_loss

def train_gnn_comm_reg(
    model: GNNCV,
    loader: DataLoader,
    n_mol: int,
    optimizer,
    loss,
    cols,
    device: str,
    kT: float,
    calc,
    box_length: float,
    lamb=1,
    n_at=1,
    alpha=0.9,
    emax=10,
    L=5
) -> float:
    """Performs one training epoch for a GNN with Committor Regularisation.

    :param model: The model to be trained
    :type model: GNNCV
    :param loader: Wrapper around a GNNTrajectory dataset.
    :type loader: DataLoader
    :param n_mol: Number of nodes per frame.
    :type n_mol: int
    :param optimizer: The optimizer object for the training.
    :type optimizer: torch.optim.Optimizer
    :param loss: Loss function for the training.
    :type loss: torch.nn._Loss
    :param cols: Columns to be used from the dataset.
    :type cols: list
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param kT: kT value for the CR Loss
    :type kT: float
    :param calc: Calculator object for energy calculations.
    :type calc: _type_
    :param box_length: Box length for periodic boundary conditions.
    :type box_length: float
    :param lamb: Regularization parameter for mixing CR loss and default loss, defaults to 1
    :type lamb: int, optional
    :param n_at: Number of atoms per node, defaults to 1
    :type n_at: int, optional
    :param alpha: Decay parameter for the CR Loss, defaults to 0.9
    :type alpha: float, optional
    :param emax: Maximum energy threshold for the CR Loss, defaults to 10
    :type emax: int, optional
    :param L: Number of steps taking in the path loss evaluation, defaults to 5
    :type L: int, optional
    :return: Return the average total loss over the epoch.
    :rtype: float
    """
    resi = {"loss": 0, "path_loss":0, "total_loss":0, "counter": 0}
    for batch, (X, y, r, c, re, ce) in enumerate(loader):
        model.train()
        optimizer.zero_grad()
        batch_size = len(X)
        X.requires_grad = True
        ap = X.view(-1, 3 * n_at).to(device)
        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]

        row_new_e = []
        col_new_e = []
        for i in range(0, len(r)):
            row_new_e.append(re[i][re[i] >= 0] + n_mol * (i))
            col_new_e.append(ce[i][ce[i] >= 0] + n_mol * (i))

        row_new_e = torch.cat([ro for ro in row_new_e])
        col_new_e = torch.cat([co for co in col_new_e])

        if row_new_e[0] >= n_mol - 1:
            row_new_e -= n_mol
            col_new_e -= n_mol

        edges_e = [row_new_e.long().to(device), col_new_e.long().to(device)]

        label = y[:,cols].to(device)
        beta = 1/(kT)
        pred = model(x=ap, edges=edges, n_nodes=n_mol)

        path_loss = torch.mean(e_path_loss(model, calc, ap, edges, edges_e, n_mol, box_length, batch_size, beta, alpha, emax, L, device))
        #print(path_loss)
        l = loss(pred, label)
        resi["loss"] += l.item() * batch_size
        resi["path_loss"] += torch.mean(path_loss).item() * batch_size

        #print(comm_reg)
        l += lamb*torch.mean(path_loss)

        l.backward()
        optimizer.step()

        resi["total_loss"] += l.item() * batch_size
        resi["counter"] += batch_size

    return resi["total_loss"] / resi["counter"], resi["loss"] / resi["counter"], resi["path_loss"] / resi["counter"]


def test_linear(
    model_t: NNCV, dataloader: DataLoader, loss_fn: Callable, device: str
) -> float:
    """Calculates the current average test set loss.

    :param model_t: Model that is being trained.
    :type model_t: NNCV
    :param dataloader: Dataloader loading the test set.
    :type dataloader: torch.utils.data.Dataloader
    :param loss_fn: Pytorch loss function.
    :type loss_fn: torch.nn._Loss
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :return: Return the validation loss.
    :rtype: float
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_t.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model_t(X)
            test_loss += loss_fn(pred.flatten(), y).item()

    test_loss /= num_batches
    print(f"Avg. Test loss: {test_loss:>8f} \n")
    return test_loss


def test_gnn(
    model: GNNCV, loader: DataLoader, n_mol: int, loss_l1: Callable, device: str, n_at=1
) -> float:
    """Evaluate the test/validation error of a graph based model_t on a validation set. 

    :param model: Graph-based model_t to be trained.
    :type model: GNNCV
    :param loader: Wrapper around a GNNTrajectory dataset.
    :type loader: torch.utils.data.Dataloader
    :param n_mol: Number of nodes per frame.
    :type n_mol: int
    :param loss_l1: Loss function for the training.
    :type loss_l1: torch.nn._Loss
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Return the average loss over the epoch.
    :rtype: float
    """
    res = {"loss": 0, "counter": 0}
    for batch, (X, y, r, c) in enumerate(loader):
        model.eval()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)

        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])
        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        label = y.to(device)
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)
        # print(label, pred)
        loss = loss_l1(pred, label)

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size

    return res["loss"] / res["counter"]


def test_gnn_mult(
    model: GNNCV, loader: DataLoader, n_mol: int, loss_l1, device: str, cols: list, n_at=1
) -> float:
    """Evaluate the test/validation error of a graph based model with multidimensional output on a test set.

    :param model: Graph-based model_t to be trained.
    :type model: GNNCV
    :param loader: Wrapper around a GNNTrajectory dataset.
    :type loader: torch.utils.data.Dataloader
    :param n_mol: Number of nodes per frame.
    :type n_mol: int
    :param loss_l1: Loss function for the training.
    :type loss_l1: torch.nn._Loss
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param cols: List of column indices representing the CVs the model is learning from the dataset.
    :type cols: list
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Return the average loss over the epoch.
    :rtype: float
    """
    res = {"loss": 0, "counter": 0}
    for batch, (X, y, r, c) in enumerate(loader):
        model.eval()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)

        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])
        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        label = y[:,cols].to(device)
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)
        loss = loss_l1(pred, label)

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size

    return res["loss"] / res["counter"]


def test_gnn_e(
    model: GNNCV, loader: DataLoader, n_mol: int, loss_l1, cols, device: str, kT: float, calc, boxlength: float,
    n_at=1,
    alpha=0.9,
    emax=10,
    L=5

    ):
    """Evaluate the test/validation error and CR/Committor Path Loss for a GNN

    :param model: Model to be evaluated
    :type model: GNNCV
    :param loader: dataloader containing the test frames
    :type loader: DataLoader
    :param n_mol: Number of molecules/number of nodes in the model
    :type n_mol: int
    :param loss_l1: Loss function to be used for the test error
    :type loss_l1: torch.nn._Loss
    :param cols: List of column indices representing the target CVs in the dataset
    :type cols: List[int]
    :param device: Device to perform the computation on
    :type device: str
    :param kT: kT value for the CR Loss
    :type kT: float
    :param calc: Calculator object for evaluating the energy of a frame
    :type calc: Object
    :param boxlength: Box length for periodic boundary conditions
    :type boxlength: float
    :param n_at: Number of atoms per molecule, defaults to 1
    :type n_at: int, optional
    :param alpha: Decay factor for the path loss, defaults to 0.9
    :type alpha: float, optional
    :param emax: Maximum error value for the path loss, defaults to 10
    :type emax: int, optional
    :param L: Number of steps for the path loss, defaults to 5
    :type L: int, optional
    :return: Tuple containing the average loss and average path loss over the epoch
    :rtype: Tuple[float, float]
    """
    res = {"loss": 0, "counter": 0, "path_loss": 0}
    for batch, (X, y, r, c, re, ce) in enumerate(loader):
        model.eval()
        batch_size = len(X)

        X.requires_grad = True
        atom_positions = X.view(-1, 3 * n_at).to(device)

        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])
        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        
        row_new_e = []
        col_new_e = []
        for i in range(0, len(r)):
            row_new_e.append(re[i][re[i] >= 0] + n_mol * (i))
            col_new_e.append(ce[i][ce[i] >= 0] + n_mol * (i))

        row_new_e = torch.cat([ro for ro in row_new_e])
        col_new_e = torch.cat([co for co in col_new_e])

        if row_new_e[0] >= n_mol - 1:
            row_new_e -= n_mol
            col_new_e -= n_mol

        edges_e = [row_new_e.long().to(device), col_new_e.long().to(device)]

        label = y[:,cols].to(device)
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)
        # print(label, pred)
        loss = loss_l1(pred, label)

        beta = 1/(kT)

        path_loss = torch.mean(e_path_loss(model, calc, atom_positions, edges, edges_e, n_mol, boxlength, batch_size, beta, alpha, emax, L, device))
        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size
        res["path_loss"] += path_loss.item() * batch_size

    return res["loss"] / res["counter"], res["path_loss"] / res["counter"]



def evaluate_model_gnn(
    model: GNNCV, dataloader: DataLoader, n_mol: int, device: str, n_at=1
) -> tuple:
    """Helper function that evaluates a model on a training set and calculates some properies for the generation of performance scatter plots.

    :param model: The model that is to be evaluated.
    :type model: GNNCV
    :param dataloader: Wrapper around the dataset that the model is supposed to be evaluated on.
    :type dataloader: torch.utils.data.Dataloader
    :param n_mol: Number of nodes in the graph of each frame. (Number of atoms or molecules)
    :type n_mol: int
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Returns the prediction of the model on each frame, the corresponding true values, the root mean square error of the predictions and the r2 correlation coefficient.
    :rtype: List of float, List of float, float, float
    """
    preds = []
    ys = []
    for batch, (X, y, r, c) in enumerate(dataloader):
        model.eval()
        # optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)
        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)
        [ys.append(ref.item()) for ref in y]
        [preds.append(pre.item()) for pre in pred]

    rmse = np.mean((np.array(preds) - np.array(ys)) ** 2) ** 0.5
    r2 = np.corrcoef(preds, ys)[0, 1]

    return preds, ys, rmse, r2


def evaluate_model_gnn_mult(
    model: GNNCV, dataloader: DataLoader, n_mol: int, device: str, cols: list, n_at=1
) -> tuple:
    """Helper function that evaluates a model on a training set and calculates some properies for the generation of performance scatter plots.

    :param model: The model that is to be evaluated.
    :type model: GNNCV
    :param dataloader: Wrapper around the dataset that the model is supposed to be evaluated on.
    :type dataloader: torch.utils.data.Dataloader
    :param n_mol: Number of nodes in the graph of each frame. (Number of atoms or molecules)
    :type n_mol: int
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param cols: List of column indices representing the CVs the model is learning from the dataset.
    :type cols: list
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Returns the prediction of the model on each frame, the corresponding true values, the root mean square errors of the predictions and the r2 correlation coefficients.
    :rtype: List of List of float, List of List of float, List of float, List of float
    """
    preds = []
    ys = []
    for batch, (X, y, r, c) in enumerate(dataloader):
        model.eval()
        # optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)
        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)

        [ys.append(ref.cpu().detach().numpy()) for ref in y[:, cols]]
        [preds.append(pre.cpu().detach().numpy()) for pre in pred]

    preds = np.array(preds)
    ys = np.array(ys)

    rmse_1 = np.mean((preds - ys) ** 2, axis=0) ** 0.5
    r2 = []
    for i in range(len(cols)):
        r2.append(np.corrcoef(preds[:,i], ys[:,i])[0, 1])

    return preds, ys, rmse_1, r2


def evaluate_model_gnn_mult_e(
    model: GNNCV, dataloader: DataLoader, n_mol: int, device: str, cols: list, n_at=1
) -> tuple:
    """Helper function that evaluates a model on an energy training set and calculates some properies for the generation of performance scatter plots.

    :param model: The model that is to be evaluated.
    :type model: GNNCV
    :param dataloader: Wrapper around the dataset that the model is supposed to be evaluated on.
    :type dataloader: torch.utils.data.Dataloader
    :param n_mol: Number of nodes in the graph of each frame. (Number of atoms or molecules)
    :type n_mol: int
    :param device: Device that the training is performed on. (Required for GPU compatibility)
    :type device: str
    :param cols: List of column indices representing the CVs the model is learning from the dataset.
    :type cols: list
    :param n_at: Number of atoms per molecule.
    :type n_at: int, optional
    :return: Returns the prediction of the model on each frame, the corresponding true values, the root mean square errors of the predictions and the r2 correlation coefficients.
    :rtype: List of List of float, List of List of float, List of float, List of float
    """
    preds = []
    ys = []
    for batch, (X, y, r, c, re, ce) in enumerate(dataloader):
        model.eval()
        # optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)
        row_new = []
        col_new = []
        for i in range(0, len(r)):
            row_new.append(r[i][r[i] >= 0] + n_mol * (i))
            col_new.append(c[i][c[i] >= 0] + n_mol * (i))

        row_new = torch.cat([ro for ro in row_new])
        col_new = torch.cat([co for co in col_new])

        if row_new[0] >= n_mol - 1:
            row_new -= n_mol
            col_new -= n_mol

        edges = [row_new.long().to(device), col_new.long().to(device)]
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)

        [ys.append(ref.cpu().detach().numpy()) for ref in y[:, cols]]
        [preds.append(pre.cpu().detach().numpy()) for pre in pred]

    preds = np.array(preds)
    ys = np.array(ys)

    rmse_1 = np.mean((preds - ys) ** 2, axis=0) ** 0.5
    r2 = []
    for i in range(len(cols)):
        r2.append(np.corrcoef(preds[:,i], ys[:,i])[0, 1])

    return preds, ys, rmse_1, r2