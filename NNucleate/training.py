import torch
from time import time
from copy import deepcopy
import numpy as np
from NNucleate.models import GNNCV, NNCV
from torch.utils.data import DataLoader
from typing import Callable
from scipy.spatial.transform import Rotation as R
import mdtraj as md

from NNucleate.utils import pbc_config, flatten_graph_edges, select_labels

# ==========================
# Unified NN (linear) training/testing/evaluation
# ==========================

def train_nn(
    model: NNCV,
    dataloader: DataLoader,
    loss_fn: callable,
    optimizer: callable,
    device: str,
    print_batch: int = 1000000,
    augment: str = None,  # None | 'permute' | 'rotate'
    n_trans: int = 1,
) -> float:
    """One training epoch for an NN model with optional augmentation.

    augment='permute' reproduces train_perm; augment='rotate' reproduces train_rot.
    """
    size = len(dataloader.dataset)
    last_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        if augment == 'permute' and n_trans > 0:
            for _ in range(n_trans):
                pred_perm = model(X[:, torch.randperm(X.size(1))])
                loss += loss_fn(pred_perm.flatten(), y)
            loss /= n_trans
        elif augment == 'rotate' and n_trans > 0:
            for _ in range(n_trans):
                quat = md.utils.uniform_quaternion()
                rot = R.from_quat(quat)
                # Inputs are normalized by box length in datasets; keep 1.0 as in original
                X_rot = torch.tensor([pbc_config(rot.apply(x.cpu().numpy()), 1.0) for x in X]).float().to(device)
                pred_rot = model(X_rot)
                loss += loss_fn(pred_rot.flatten(), y)
            loss /= n_trans

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if batch % print_batch == 0:
            current = batch * len(X)
            print(f"loss: {last_loss:>7f} [{current:>5d}/{size:>5d}]")

    return last_loss


def test_nn(model: NNCV, dataloader: DataLoader, loss_fn: callable, device: str) -> float:
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.flatten(), y).item()
    test_loss /= max(1, num_batches)
    print(f"Avg. Test loss: {test_loss:>8f} \n")
    return test_loss


def evaluate_nn(model: NNCV, dataloader: DataLoader, device: str):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X).flatten().cpu().numpy()
            preds.extend(pred.tolist())
            ys.extend(y.cpu().numpy().tolist())
    preds_arr = np.array(preds)
    ys_arr = np.array(ys)
    rmse = np.mean((preds_arr - ys_arr) ** 2) ** 0.5
    r2 = np.corrcoef(preds_arr, ys_arr)[0, 1]
    return preds, ys, rmse, r2


# ==========================
# Unified GNN training/testing/evaluation
# ==========================

try:
    # Prefer package-relative import
    from .committor_regularisation import e_path_loss
except Exception:
    # Fallback to original local import path for compatibility
    from committor_regularisation import e_path_loss


def train_gnn_unified(
    model: GNNCV,
    loader: DataLoader,
    n_mol: int,
    optimizer: callable,
    loss_fn: callable,
    device: str,
    *,
    n_at: int = 1,
    cols: list = None,
    committor: dict = None,  # {kT, calc, box_length, lamb, alpha, emax, L}
) -> float | tuple:
    """One training epoch for a GNN model.

    - cols: select multi-output columns if provided (replaces train_gnn_mult, train_gnn_e)
    - committor: if provided and loader yields extra edges (re, ce), add path loss as in train_gnn_comm_reg
    Returns avg loss; if committor provided, returns (avg_total_loss, avg_pred_loss, avg_path_loss)
    """
    res = {"loss": 0.0, "path_loss": 0.0, "total_loss": 0.0, "counter": 0}
    use_comm = committor is not None
    for batch in loader:
        # Support batches with or without extra edges
        if len(batch) == 4:
            X, y, r, c = batch
            re = ce = None
        elif len(batch) == 6:
            X, y, r, c, re, ce = batch
        else:
            raise ValueError("Unexpected batch structure for GNN loader.")

        model.train()
        optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3 * n_at).to(device)

        edges = flatten_graph_edges(r, c, n_mol)
        edges = [edges[0].to(device), edges[1].to(device)]

        label = flatten_graph_edges(y, cols).to(device)
        pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)

        l_pred = loss_fn(pred, label)

        if use_comm:
            if re is None or ce is None:
                raise ValueError("Committor regularisation requires dataset to provide extra edges (re, ce).")
            edges_e = flatten_graph_edges(re, ce, n_mol)
            edges_e = [edges_e[0].to(device), edges_e[1].to(device)]

            beta = 1.0 / committor["kT"]
            path_l = torch.mean(
                e_path_loss(
                    model,
                    committor["calc"],
                    atom_positions,
                    edges,
                    edges_e,
                    n_mol,
                    committor["box_length"],
                    batch_size,
                    beta,
                    committor.get("alpha", 0.9),
                    committor.get("emax", 10),
                    committor.get("L", 5),
                    device,
                )
            )
            total_l = l_pred + committor.get("lamb", 1) * path_l
            total_l.backward()
            optimizer.step()

            res["loss"] += l_pred.item() * batch_size
            res["path_loss"] += float(path_l.item()) * batch_size
            res["total_loss"] += float(total_l.item()) * batch_size
            res["counter"] += batch_size
        else:
            l_pred.backward()
            optimizer.step()
            res["loss"] += l_pred.item() * batch_size
            res["counter"] += batch_size

    if use_comm:
        return (
            res["total_loss"] / res["counter"],
            res["loss"] / res["counter"],
            res["path_loss"] / res["counter"],
        )
    return res["loss"] / res["counter"]


def test_gnn_unified(
    model: GNNCV,
    loader: DataLoader,
    n_mol: int,
    loss_fn: callable,
    device: str,
    *,
    n_at: int = 1,
    cols: list = None,
    committor: dict = None,
):
    """Evaluate GNN model on a validation/test set.

    Returns avg loss; if committor provided, also returns avg path loss.
    """
    res = {"loss": 0.0, "counter": 0, "path_loss": 0.0}
    use_comm = committor is not None
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                X, y, r, c = batch
                re = ce = None
            elif len(batch) == 6:
                X, y, r, c, re, ce = batch
            else:
                raise ValueError("Unexpected batch structure for GNN loader.")

            batch_size = len(X)
            atom_positions = X.view(-1, 3 * n_at).to(device)
            edges = flatten_graph_edges(r, c, n_mol)
            edges = [edges[0].to(device), edges[1].to(device)]
            label = select_labels(y, cols).to(device)

            pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)
            l = loss_fn(pred, label)
            res["loss"] += float(l.item()) * batch_size
            res["counter"] += batch_size

            if use_comm:
                if re is None or ce is None:
                    raise ValueError("Committor evaluation requires dataset to provide extra edges (re, ce).")
                edges_e = flatten_graph_edges(re, ce, n_mol)
                edges_e = [edges_e[0].to(device), edges_e[1].to(device)]
                beta = 1.0 / committor["kT"]
                pl = torch.mean(
                    e_path_loss(
                        model,
                        committor["calc"],
                        atom_positions,
                        edges,
                        edges_e,
                        n_mol,
                        committor["box_length"],
                        batch_size,
                        beta,
                        committor.get("alpha", 0.9),
                        committor.get("emax", 10),
                        committor.get("L", 5),
                        device,
                    )
                )
                res["path_loss"] += float(pl.item()) * batch_size

    if use_comm:
        return res["loss"] / res["counter"], res["path_loss"] / res["counter"]
    return res["loss"] / res["counter"]


def evaluate_gnn_unified(
    model: GNNCV,
    dataloader: DataLoader,
    n_mol: int,
    device: str,
    *,
    n_at: int = 1,
    cols: list = None,
):
    """Evaluate GNN and return predictions, targets, RMSE, and R^2.

    For single target: rmse, r2 are scalars; for multi-target (cols provided): arrays per target.
    """
    preds = []
    ys = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                X, y, r, c = batch
            elif len(batch) == 6:
                X, y, r, c, _, _ = batch
            else:
                raise ValueError("Unexpected batch structure for GNN loader.")
            atom_positions = X.view(-1, 3 * n_at).to(device)
            edges = flatten_graph_edges(r, c, n_mol)
            edges = [edges[0].to(device), edges[1].to(device)]
            pred = model(x=atom_positions, edges=edges, n_nodes=n_mol)

            if cols is None:
                ys.extend([ref.item() for ref in y])
                preds.extend([pre.item() for pre in pred])
            else:
                ys.extend(y[:, cols].cpu().detach().numpy())
                preds.extend(pred.cpu().detach().numpy())

    preds = np.array(preds)
    ys = np.array(ys)

    if cols is None:
        rmse = np.mean((preds - ys) ** 2) ** 0.5
        r2 = np.corrcoef(preds, ys)[0, 1]
        return preds.tolist(), ys.tolist(), rmse, r2
    else:
        rmse = np.mean((preds - ys) ** 2, axis=0) ** 0.5
        r2 = []
        for i in range(preds.shape[1]):
            r2.append(np.corrcoef(preds[:, i], ys[:, i])[0, 1])
        return preds, ys, rmse, r2


def train_linear(model_t: NNCV, dataloader: DataLoader, loss_fn: Callable, optimizer: Callable, device: str, print_batch=1000000) -> float:
    """Backward-compatible wrapper; uses train_nn without augmentation."""
    return train_nn(model_t, dataloader, loss_fn, optimizer, device, print_batch=print_batch, augment=None)


def train_gnn(model: GNNCV, loader: DataLoader, n_mol: int, optimizer: Callable, loss: Callable, device: str, n_at=1) -> float:
    """Backward-compatible wrapper: single-output atom-graph training."""
    return train_gnn_unified(model, loader, n_mol, optimizer, loss, device, n_at=n_at, cols=None, committor=None)


def train_gnn_mult(model: GNNCV, loader: DataLoader, n_mol: int, optimizer, loss, device: str, cols: list, n_at=1) -> float:
    """Backward-compatible wrapper: multi-output training using selected columns."""
    return train_gnn_unified(model, loader, n_mol, optimizer, loss, device, n_at=n_at, cols=cols, committor=None)

def train_gnn_e(model, loader: DataLoader, n_mol: int, optimizer, loss, cols, device: str, n_at=1) -> float:
    """Backward-compatible wrapper: energy dataset training (multi-output columns)."""
    return train_gnn_unified(model, loader, n_mol, optimizer, loss, device, n_at=n_at, cols=cols, committor=None)


def train_perm(model_t: NNCV, dataloader: DataLoader, optimizer: Callable, loss_fn: Callable, n_trans: int, device: str, print_batch=1000000) -> float:
    """Backward-compatible wrapper; uses train_nn with augment='permute'."""
    return train_nn(model_t, dataloader, loss_fn, optimizer, device, print_batch=print_batch, augment='permute', n_trans=n_trans)


def train_rot(model_t: NNCV, dataloader: DataLoader, optimizer: Callable, loss_fn: Callable, n_trans: int, device: str, print_batch=1000000) -> float:
    """Backward-compatible wrapper; uses train_nn with augment='rotate'."""
    return train_nn(model_t, dataloader, loss_fn, optimizer, device, print_batch=print_batch, augment='rotate', n_trans=n_trans)


def train_gnn_comm_reg(model: GNNCV, loader: DataLoader, n_mol: int, optimizer, loss, cols, device: str, kT: float, calc, box_length: float, lamb=1, n_at=1, alpha=0.9, emax=10, L=5) -> float:
    """Backward-compatible wrapper: committor-regularized GNN training."""
    comm = {
        "kT": kT,
        "calc": calc,
        "box_length": box_length,
        "lamb": lamb,
        "alpha": alpha,
        "emax": emax,
        "L": L,
    }
    return train_gnn_unified(model, loader, n_mol, optimizer, loss, device, n_at=n_at, cols=cols, committor=comm)


def test_linear(model_t: NNCV, dataloader: DataLoader, loss_fn: Callable, device: str) -> float:
    """Backward-compatible wrapper; uses test_nn."""
    return test_nn(model_t, dataloader, loss_fn, device)


def test_gnn(model: GNNCV, loader: DataLoader, n_mol: int, loss_l1: Callable, device: str, n_at=1) -> float:
    """Backward-compatible wrapper: single-output atom-graph testing."""
    return test_gnn_unified(model, loader, n_mol, loss_l1, device, n_at=n_at, cols=None, committor=None)


def test_gnn_mult(model: GNNCV, loader: DataLoader, n_mol: int, loss_l1, device: str, cols: list, n_at=1) -> float:
    """Backward-compatible wrapper: multi-output testing."""
    return test_gnn_unified(model, loader, n_mol, loss_l1, device, n_at=n_at, cols=cols, committor=None)


def test_gnn_e(model: GNNCV, loader: DataLoader, n_mol: int, loss_l1, cols, device: str, kT: float, calc, boxlength: float, n_at=1, alpha=0.9, emax=10, L=5):
    """Backward-compatible wrapper: evaluation with committor path loss reporting."""
    comm = {
        "kT": kT,
        "calc": calc,
        "box_length": boxlength,
        "alpha": alpha,
        "emax": emax,
        "L": L,
    }
    return test_gnn_unified(model, loader, n_mol, loss_l1, device, n_at=n_at, cols=cols, committor=comm)



def evaluate_model_gnn(model: GNNCV, dataloader: DataLoader, n_mol: int, device: str, n_at=1) -> tuple:
    """Backward-compatible wrapper: single-output evaluation."""
    preds, ys, rmse, r2 = evaluate_gnn_unified(model, dataloader, n_mol, device, n_at=n_at, cols=None)
    return preds, ys, rmse, r2


def evaluate_model_gnn_mult(model: GNNCV, dataloader: DataLoader, n_mol: int, device: str, cols: list, n_at=1) -> tuple:
    """Backward-compatible wrapper: multi-output evaluation."""
    return evaluate_gnn_unified(model, dataloader, n_mol, device, n_at=n_at, cols=cols)


def evaluate_model_gnn_mult_e(model: GNNCV, dataloader: DataLoader, n_mol: int, device: str, cols: list, n_at=1) -> tuple:
    """Backward-compatible wrapper: energy multi-output evaluation (same as multi)."""
    return evaluate_gnn_unified(model, dataloader, n_mol, device, n_at=n_at, cols=cols)