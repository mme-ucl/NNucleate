import heapq
import itertools
from scipy.spatial import cKDTree
import numpy as np
import mdtraj as md
import torch
from scipy.spatial.transform import Rotation as R


def pbc(trajectory: md.Trajectory, box_length: float) -> md.Trajectory:
    """Centers an mdtraj Trajectory around the centre of a cubic box with the given box length and wraps all atoms into the box.

    :param trajectory: The trajectory that is to be modified, i.e. contains the configurations that shall be wrapped back into the simulation box.
    :type trajectory: mdtraj.trajectory
    :param box_length: Length of the cubic box which shall contain all the positions.
    :type box_length: float
    :return: Returns a trajectory object obeying PBC according to the given box length.
    :rtype: mdtraj.trajectory
    """
    # function defining PBC
    def transform(x):
        while x >= box_length:
            x -= box_length
        while x <= 0:
            x += box_length
        return x

    # Prepare the function for 2D mapping
    func = np.vectorize(transform)

    for i in range(trajectory.n_frames):
        # map the function on all coordinates
        trajectory.xyz[i] = func(trajectory.xyz[i])

    return trajectory


def pbc_config(config: np.ndarray, box_length: float) -> md.Trajectory:
    """Wraps all atoms in a given configuration into the box.

    :param config: The trajectory that is to be modified, i.e. contains the configurations that shall be wrapped back into the simulation box.
    :type np.ndarray: mdtraj.trajectory
    :param box_length: Length of the cubic box which shall contain all the positions.
    :type box_length: float
    :return: Returns a trajectory object obeying PBC according to the given box length.
    :rtype: np.ndarray
    """
    # function defining PBC
    def transform(x):
        while x >= box_length:
            x -= box_length
        while x <= 0:
            x += box_length
        return x

    # Prepare the function for 2D mapping
    func = np.vectorize(transform)

    return func(config)


def rotate_trajs(trajectories: np.ndarray) -> np.ndarray:
    """Rotates each frame in the given trajectories according to a random quaternion.

    :param trajectories: A list of mdtraj.trajectory objects to be modified.
    :type trajectories: list of md.trajectory
    :return: Returns a list of trajectories, the frames of which have been randomly rotated and wrapped back into the box.
    :rtype: list of md.trajectory
    """
    for t in trajectories:
        for i in range(t.n_frames):
            quat = md.utils.uniform_quaternion()
            rot = R.from_quat(quat)
            t.xyz[i] = rot.apply(t.xyz[i])

    return trajectories


# GNN utils
def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Function that sums the segments of a matrix. Each row has a non-unique ID and all rows with the same ID are summed such that a matrix with the number of rows equal to the number of unique IDs is obtained.

    :param data: A tensor that contains the data that is to be summed.
    :type data: torch.tensor
    :param segment_ids: An array that has the same number of entries as data has rows which indicates which rows shall be summed.
    :type segment_ids: torch.tensor
    :param num_segments: This is the number of unique IDs, i.e. the dimensionality of the resulting tensor.
    :type num_segments: int
    :return: Returns a tensor shaped num_segments x data.size(1) containing all the segment sums.
    :rtype: torch.Tensor
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def get_rc_edges(rc: float, traj: md.Trajectory) -> list:
    """Returns the edges of the graph constructed by interpreting the atoms in the trajectory as nodes that are connected to all other nodes within a distance of rc.

    :param rc: Cut-off radius for the graph construction.
    :type rc: float
    :param traj: The trajectory for which the graphs shall be constructed.
    :type traj: md.trajectory
    :return: A list containing two tensors which represent the adjacency matrix of the graph.
    :rtype: list of torch.tensor
    """
    n_at = len(traj.xyz[0])
    n_frames = len(traj)

    row_list = []
    col_list = []
    for i in range(n_frames):
        frame_row = []
        frame_col = []
        dist = md.compute_neighborlist(traj, rc, frame=i)
        for j in range(n_at):
            for k in range(len(dist[j])):
                frame_row.append(j)
                frame_col.append(dist[j][k])
        assert len(frame_row) == len(frame_col)
        row_list.append(torch.Tensor(frame_row))
        col_list.append(torch.Tensor(frame_col))

    return row_list, col_list


def com(xyz: np.ndarray) -> list:
    """Calculates the centre of mass of a set of coordinates.

    :param xyz: Array containing the list of 3-dimensional coordinates.
    :type xyz: np.ndarray
    :return: A list of the calculated centres of mass.  
    :rtype: list of float
    """
    coms = np.zeros((len(xyz), len(xyz[0]), 3))
    coms[:, :, 0] = np.mean(xyz[:, :, ::3], axis=2)
    coms[:, :, 1] = np.mean(xyz[:, :, 1::3], axis=2)
    coms[:, :, 2] = np.mean(xyz[:, :, 2::3], axis=2)

    return coms


def get_mol_edges(
    rc: float, traj: md.Trajectory, n_mol: int, n_at: int, box: float
) -> list:
    """Generate the edges for a neighbourlist graph based on the COMs of the given molecules.

    :param rc: Cut off radius for the neighbourlist graph.
    :type rc: float
    :param traj: The Trajectory containing the frames.
    :type traj: md.Trajectory
    :param n_mol: Number of molecules per frame.
    :type n_mol: int
    :param n_at: Number of atoms per molecule.
    :type n_at: int
    :param box: Length of the cubic box
    :type box: float
    :return: A list containing two tensors which represent the adjacency matrix of the graph.
    :rtype: list of torch.tensor
    """

    traj_mol = traj.xyz.reshape((len(traj), n_mol, n_at * 3))

    t = md.Topology()
    for i in range(n_mol):
        t.add_atom("suc", "H", traj.topology.residue(0))

    xyz = com(traj_mol)
    traj2 = md.Trajectory(xyz, t)

    vecs = np.zeros((len(traj2), 3, 3))
    for i in range(len(traj2)):
        vecs[i, 0] = np.array([box, 0, 0])
        vecs[i, 1] = np.array([0, box, 0])
        vecs[i, 2] = np.array([0, 0, box])
    traj2.unitcell_vectors = vecs

    return get_rc_edges(rc, traj2)


# ==========================
# Unified helpers
# ==========================

def flatten_graph_edges(r_batch, c_batch, n_mol: int):
    """Flatten per-sample edge lists (padded with -1) into a single edge index for the batch.

    Mirrors the original logic including the post-adjustment by -n_mol when needed.
    """
    row_new = []
    col_new = []
    for i in range(0, len(r_batch)):
        row_new.append(r_batch[i][r_batch[i] >= 0] + n_mol * (i))
        col_new.append(c_batch[i][c_batch[i] >= 0] + n_mol * (i))

    row_new = torch.cat([ro for ro in row_new]) if len(row_new) > 0 else torch.tensor([], dtype=torch.long)
    col_new = torch.cat([co for co in col_new]) if len(col_new) > 0 else torch.tensor([], dtype=torch.long)

    if row_new.numel() > 0 and row_new[0] >= n_mol - 1:
        row_new -= n_mol
        col_new -= n_mol

    return [row_new.long(), col_new.long()]


def select_labels(y: torch.Tensor, cols):
    """Select target columns if provided, else return y as-is."""
    if cols is None:
        return y
    return y[:, cols]

# A wrapper around scipy.spatial.kdtree to implement periodic boundary
# conditions
#
# !!!!Written by Patrick Varilly, 6 Jul 2012!!!
# "https://github.com/patvarilly/periodic_kdtree"
# Released under the scipy license


def _gen_relevant_images(x, bounds, distance_upper_bound):
    # Map x onto the canonical unit cell, then produce the relevant
    # mirror images
    real_x = x - np.where(bounds > 0.0, np.floor(x / bounds) * bounds, 0.0)
    m = len(x)

    xs_to_try = [real_x]
    for i in range(m):
        if bounds[i] > 0.0:
            disp = np.zeros(m)
            disp[i] = bounds[i]

            if distance_upper_bound == np.inf:
                xs_to_try = list(
                    itertools.chain.from_iterable(
                        (_ + disp, _, _ - disp) for _ in xs_to_try
                    )
                )
            else:
                extra_xs = []

                # Point near lower boundary, include image on upper side
                if abs(real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ + disp for _ in xs_to_try)

                # Point near upper boundary, include image on lower side
                if abs(bounds[i] - real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ - disp for _ in xs_to_try)

                xs_to_try.extend(extra_xs)

    return xs_to_try


class PeriodicCKDTree(cKDTree):
    """
    A wrapper around scipy.spatial.kdtree to implement periodic boundary conditions

    !!!!Written by Patrick Varilly, 6 Jul 2012!!!
    "https://github.com/patvarilly/periodic_kdtree"
    Released under the scipy license

    Cython kd-tree for quick nearest-neighbor lookup with periodic boundaries
    See scipy.spatial.ckdtree for details on kd-trees.
    Searches with periodic boundaries are implemented by mapping all
    initial data points to one canonical periodic image, building an
    ordinary kd-tree with these points, then querying this kd-tree multiple
    times, if necessary, with all the relevant periodic images of the
    query point.
    Note that to ensure that no two distinct images of the same point
    appear in the results, it is essential to restrict the maximum
    distance between a query point and a data point to half the smallest
    box dimension.
    Construct a kd-tree.

    :param bounds: Size of the periodic box along each spatial dimension.  A
        negative or zero size for dimension k means that space is not
        periodic along k.
    :type bounds: array_like, shape (k,)
    :param data: The n data points of dimension mto be indexed. This array is 
        not copied unless this is necessary to produce a contiguous 
        array of doubles, and so modifying this data will result in 
        bogus results.
    :type data: array-like, shape (n,m)
    :param leafsize: The number of points at which the algorithm switches over to
        brute-force, defaults to 10.
    :type leafsize: int, optional
    """

    def __init__(self, bounds: np.ndarray, data: np.ndarray, leafsize=10):
        # Map all points to canonical periodic image
        self.bounds = np.array(bounds)
        self.real_data = np.asarray(data)
        wrapped_data = self.real_data - np.where(
            bounds > 0.0, (np.floor(self.real_data / bounds) * bounds), 0.0
        )

        # Calculate maximum distance_upper_bound
        self.max_distance_upper_bound = np.min(
            np.where(self.bounds > 0, 0.5 * self.bounds, np.inf)
        )

        # Set up underlying kd-tree
        super(PeriodicCKDTree, self).__init__(wrapped_data, leafsize)

    # Ideally, KDTree and cKDTree would expose identical query and __query
    # interfaces.  But they don't, and cKDTree.__query is also inaccessible
    # from Python.  We do our best here to cope.
    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points
        #
        # A slight complication: k could be "None", which means "return
        # all neighbors within the given distance_upper_bound".

        # Cap distance_upper_bound
        distance_upper_bound = np.min(
            [distance_upper_bound, self.max_distance_upper_bound]
        )

        # Run queries over all relevant images of x
        hits_list = []
        for real_x in _gen_relevant_images(x, self.bounds, distance_upper_bound):
            d, i = super(PeriodicCKDTree, self).query(
                real_x, k, eps, p, distance_upper_bound
            )
            if k > 1:
                hits_list.append(list(zip(d, i)))
            else:
                hits_list.append([(d, i)])

        # Now merge results
        if k > 1:
            return heapq.nsmallest(k, itertools.chain(*hits_list))
        elif k == 1:
            return [min(itertools.chain(*hits_list))]
        else:
            raise ValueError("Invalid k in periodic_kdtree._KDTree__query")

    def query(
        self, x: np.ndarray, k=1, eps=0, p=2, distance_upper_bound=np.inf
    ) -> np.ndarray:
        """Query the kd-tree for nearest neighbors.

        :param x: An array of points to query.
        :type x: array_like, last dimension self.m
        :param k: The number of nearest neighbors to return, defaults to 1
        :type k: int, optional.
        :param eps: Return approximate nearest neighbors; the kth returned value 
            is guaranteed to be no further than (1+eps) times the 
            distance to the real k-th nearest neighbor, defaults to 0.
        :type eps: int, optional
        :param p: Which Minkowski p-norm to use. 
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance, defaults to 2.
        :type p: int, optional
        :param distance_upper_bound: Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point, defaults to np.inf.
        :type distance_upper_bound: float, optional
        :return: The distances to the nearest neighbors. 
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        :rtype: array of floats
        :return: The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.
        :rtype: ndarray of ints
        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError(
                "x must consist of vectors of length %d but has shape %s"
                % (self.m, np.shape(x))
            )
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k > 1:
                dd = np.empty(retshape + (k,), dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape + (k,), dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None"
                )
            for c in np.ndindex(retshape):
                hits = self.__query(
                    x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound
                )
                if k > 1:
                    for j in range(len(hits)):
                        dd[c + (j,)], ii[c + (j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(
                x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound
            )
            if k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError(
                    "Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None"
                )

    # Ideally, KDTree and cKDTree would expose identical __query_ball_point
    # interfaces.  But they don't, and cKDTree.__query_ball_point is also
    # inaccessible from Python.  We do our best here to cope.
    def __query_ball_point(self, x, r, p=2.0, eps=0):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points

        # Cap r
        r = min(r, self.max_distance_upper_bound)

        # Run queries over all relevant images of x
        results = []
        for real_x in _gen_relevant_images(x, self.bounds, r):
            results.extend(
                super(PeriodicCKDTree, self).query_ball_point(real_x, r, p, eps)
            )
        return results

    def query_ball_point(self, x: np.ndarray, r: float, p=2.0, eps=0) -> np.ndarray:
        """Find all points within distance r of point(s) x.
        Notes: If you have many points whose neighbors you want to find, you may
        save substantial amounts of time by putting them in a
        PeriodicCKDTree and using query_ball_tree.

        :param x: The point or points to search for neighbors of.
        :type x: array_like, shape tuple + (self.m,)
        :param r: The radius of points to return.
        :type r: float
        :param p: Which Minkowski p-norm to use.  Should be in the range [1, inf], defaults to 2.0.
        :type p: float, optional
        :param eps: Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``, defaults to 0.
        :type eps: int, optional
        :return: If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.
        :rtype: list or array of lists
        """
        x = np.asarray(x).astype(np.float)
        if x.shape[-1] != self.m:
            raise ValueError(
                "Searching for a %d-dimensional point in a "
                "%d-dimensional KDTree" % (x.shape[-1], self.m)
            )
        if len(x.shape) == 1:
            return self.__query_ball_point(x, r, p, eps)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self.__query_ball_point(x[c], r, p, eps)
            return result
