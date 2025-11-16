from torch.utils.data import Dataset
import numpy as np
import mdtraj as md
import torch
from torch import nn
from .utils import pbc, get_rc_edges, get_mol_edges
from .data_augmentation import transform_traj_to_knn_list, transform_traj_to_ndist_list


class NNDataset(Dataset):
    """Unified dataset for non-graph (NN) models.

    Supports three representations:
    - mode="raw": raw atomic positions (optionally transformed)
    - mode="knn": per-atom sorted distances to k nearest neighbors
    - mode="ndist": globally sorted n smallest inter-atomic distances

    All modes return (features, label) with features as torch.float32.

    Parameters mirror prior specialized classes to preserve functionality.
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        cv_col: int,
        box_length: float,
        mode: str = "raw",
        *,
    k=None,
    n_dist=None,
        transform=None,
        start: int = 0,
        stop: int = -1,
        stride: int = 1,
        root: int = 1,
        normalize: bool = True,
    ):
        if mode not in {"raw", "knn", "ndist"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'raw', 'knn', or 'ndist'.")

        self.mode = mode
        self.box_length = box_length
        self.transform = transform if mode == "raw" else None
        self.normalize = normalize

        self.cv_labels = np.loadtxt(cv_file)[start:stop:stride, cv_col] ** (1 / root)

        # Load and wrap trajectory in PBC with cubic box of given length
        traj = pbc(md.load(traj_name, top=top_file), self.box_length)[start:stop:stride]

        if mode == "raw":
            # Store raw coordinates; per-item transform is applied in __getitem__
            self.configs = traj.xyz
        elif mode == "knn":
            if k is None:
                raise ValueError("'k' must be provided for mode='knn'.")
            self.configs = transform_traj_to_knn_list(
                k,
                traj.xyz,
                self.box_length,
            )
        elif mode == "ndist":
            if n_dist is None:
                raise ValueError("'n_dist' must be provided for mode='ndist'.")
            self.configs = transform_traj_to_ndist_list(
                n_dist,
                traj.xyz,
                self.box_length,
            )

    def __len__(self):
        return len(self.cv_labels)

    def __getitem__(self, idx):
        x = self.configs[idx]
        if self.mode == "raw" and self.transform is not None:
            x = self.transform(x)
        x = torch.tensor(x).float()
        y = torch.tensor(self.cv_labels[idx]).float()

        if self.normalize:
            x = x / self.box_length

        return x, y


class GNNDataset(Dataset):
    """Unified dataset for GNN models with flexible edge construction.

    Supports:
    - Atom graph via cutoff radius rc (previous GNNTrajectory)
    - Multi-target labels when cv_col=None is set(previous GNNTrajectory_mult)
    - Molecular COM graph via (n_mol, n_at) and rc (previous GNNMolecularTrajectory)
    - Optional secondary neighbor graph for energy evaluations with rc_extra (previous GNNTrajectory_energy)

    Returns either:
    - (positions, labels, rows, cols) when rc_extra is None, or
    - (positions, labels, rows, cols, rows_e, cols_e) when rc_extra is set.
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        box_length: float,
        *,
    rc: float = None,
        start: int = 0,
        stop: int = -1,
        stride: int = 1,
        root: int = 1,
    cv_col: int | None = 0,
        graph_mode: str = "atom",  # "atom" or "molecule"
    n_mol: int = None,
    n_at: int = None,
    rc_extra: float = None,
        normalize_positions: bool = True,
    ):
        if graph_mode not in {"atom", "molecule"}:
            raise ValueError("graph_mode must be 'atom' or 'molecule'.")
        if rc is None:
            raise ValueError("'rc' (cutoff radius) must be provided.")

        self.length = box_length
        self.graph_mode = graph_mode
        self.normalize_positions = normalize_positions
        self.rc_extra = rc_extra

        # Load labels: single column or all columns
        labels = np.loadtxt(cv_file)[start:stop:stride, :]
        if cv_col is None:
            self.cv_labels = labels ** (1 / root)
        else:
            self.cv_labels = labels[:, cv_col] ** (1 / root)

        # Load trajectory and set cubic unitcell vectors (required for PBC edge builds)
        traj = pbc(md.load(traj_name, top=top_file), self.length)[start:stop:stride]
        vecs = np.zeros((len(traj), 3, 3))
        for i in range(len(traj)):
            vecs[i, 0] = np.array([self.length, 0, 0])
            vecs[i, 1] = np.array([0, self.length, 0])
            vecs[i, 2] = np.array([0, 0, self.length])
        traj.unitcell_vectors = vecs
        traj = pbc(traj, self.length)

        # Build primary edges
        if self.graph_mode == "atom":
            rows, cols = get_rc_edges(rc, traj)
        else:
            if n_mol is None or n_at is None:
                raise ValueError("'n_mol' and 'n_at' must be provided for graph_mode='molecule'.")
            rows, cols = get_mol_edges(rc, traj, n_mol, n_at, self.length)

        self.rows = rows
        self.cols = cols
        self.max_l = int(np.max([len(r) for r in self.rows])) if len(self.rows) > 0 else 0

        # Optional extra edges (mirrors the edge type used for primary graph)
        self.rows_e = None
        self.cols_e = None
        self.max_le = 0
        if rc_extra is not None:
            if self.graph_mode == "atom":
                rows_e, cols_e = get_rc_edges(rc_extra, traj)
            else:
                rows_e, cols_e = get_mol_edges(rc_extra, traj, n_mol, n_at, self.length)
            self.rows_e = rows_e
            self.cols_e = cols_e
            self.max_le = int(np.max([len(r) for r in self.rows_e])) if len(self.rows_e) > 0 else 0

        self.configs = traj.xyz

    def __len__(self):
        return len(self.cv_labels)

    def _pad_index_tensor(self, t, target_len: int):
        # Ensure tensor and pad with -1 to a fixed length per dataset instance
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.long)
        if t.dtype != torch.long:
            t = t.long()
        return nn.functional.pad(t, [0, target_len - len(t)], value=-1)

    def __getitem__(self, idx):
        config = torch.tensor(self.configs[idx]).float()
        label = torch.tensor(self.cv_labels[idx]).float()

        rows = self._pad_index_tensor(self.rows[idx], self.max_l)
        cols = self._pad_index_tensor(self.cols[idx], self.max_l)

        if self.normalize_positions:
            config_out = config / self.length
        else:
            config_out = config

        if self.rows_e is not None and self.cols_e is not None:
            rows_e = self._pad_index_tensor(self.rows_e[idx], self.max_le)
            cols_e = self._pad_index_tensor(self.cols_e[idx], self.max_le)
            return config_out, label, rows, cols, rows_e, cols_e

        return config_out, label, rows, cols


class CVTrajectory(Dataset):
    """Wrapper for backward compatibility; use NNDataset(mode='raw')."""

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
        self._base = NNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            cv_col=cv_col,
            box_length=box_length,
            mode="raw",
            transform=transform,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            normalize=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


class KNNTrajectory(Dataset):
    """Wrapper for backward compatibility; use NNDataset(mode='knn')."""

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
        self._base = NNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            cv_col=cv_col,
            box_length=box_length,
            mode="knn",
            k=k,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            normalize=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


class NdistTrajectory(Dataset):
    """Wrapper for backward compatibility; use NNDataset(mode='ndist')."""

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
        self._base = NNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            cv_col=cv_col,
            box_length=box_length,
            mode="ndist",
            n_dist=n_dist,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            normalize=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


class GNNTrajectory(Dataset):
    """Wrapper for backward compatibility; uses GNNDataset with atom graph and single-label output."""

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
        self._base = GNNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            box_length=box_length,
            rc=rc,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            cv_col=cv_col,
            graph_mode="atom",
            normalize_positions=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


class GNNTrajectory_mult(Dataset):
    """Wrapper for backward compatibility; uses GNNDataset with atom graph and multi-label output."""

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        box_length: float,
        rc: float,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
        self._base = GNNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            box_length=box_length,
            rc=rc,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            cv_col=None,  # all columns (multi-target)
            graph_mode="atom",
            normalize_positions=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]


class GNNMolecularTrajectory(Dataset):
    """Wrapper for backward compatibility; uses GNNDataset with molecule graph (COM-based)."""

    def __init__(
        self,
        cv_file,
        traj_name,
        top_file,
        cv_col,
        box_length,
        rc,
        n_mol,
        n_at,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
        self._base = GNNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            box_length=box_length,
            rc=rc,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            cv_col=cv_col,
            graph_mode="molecule",
            n_mol=n_mol,
            n_at=n_at,
            normalize_positions=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]

class GNNTrajectory_energy(Dataset):
    """Wrapper for backward compatibility; uses GNNDataset with atom graph, multi-label output, and extra edges.

    Note: keeps the original behavior of NOT normalizing positions by box length.
    """

    def __init__(
        self,
        cv_file: str,
        traj_name: str,
        top_file: str,
        box_length: float,
        rc: float,
        rc_e: float,
        start=0,
        stop=-1,
        stride=1,
        root=1,
    ):
        self._base = GNNDataset(
            cv_file=cv_file,
            traj_name=traj_name,
            top_file=top_file,
            box_length=box_length,
            rc=rc,
            start=start,
            stop=stop,
            stride=stride,
            root=root,
            cv_col=None,  # energy dataset used multi-column labels
            graph_mode="atom",
            rc_extra=rc_e,
            normalize_positions=False,  # keep original behavior
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        return self._base[idx]
