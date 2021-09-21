import numpy as np
import math
from copy import deepcopy
from .utils import pbc, rotate_trajs, PeriodicCKDTree
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis.distances import self_distance_array
import math

def augment_evenly(n,trajname, topology, cvname, savename, box, n_min=0, col=3, bins=25, n_max=math.inf):
    """Takes in a trajectory and adds degenerate rotated frames such that the resulting trajectory represents and even histogram.
       Writes a new trajectory and CV file.

    Args:
        n (int): Height of the target histogram
        trajname (str): Path to the trajectory file (.xtc or .xyz)
        topology (str): Path to the topology file (.pdb)
        cvname (str): Path to the CV file. Text file with CVs organised in four columns
        savename (str): String without file ending under which the final CV and traj will be saved
        box (float): Box length for applying PBC
        n_min (int, optional): The minimum number of frames to add per frame. Defaults to 0.
        col (int, optional): The column in the CV file from which to read the CV (0 indexing). Defaults to 3.
        bins (int, optional): Number of bins in the target histogram. Defaults to 25.
        n_max (int, optional): Maximal height of a histogram column. Defaults to math.inf.
    """

    # Load the trajectory and cvs
    traj = md.load(trajname, top=topology)
    cvs = np.loadtxt(cvname)
    traj = pbc(traj, box)
    
    # Create the starting histogram
    counts, bins = np.histogram(cvs[:, col], bins=bins)

    # Calculate the number of degenerate frames to add per frame 
    rot_per_frame = []
    for i in range(len(counts)):
        # Add either so many that the resulting histogram is n high or the minimum amount
        amount = min(math.ceil(max( (n - counts[i])/counts[i], n_min )), n_max)
        rot_per_frame.append(amount)

    copy = deepcopy(traj)
    cvs_long = cvs.tolist()

    # So for each different number of rot per frame i.e. for each bin
    # Get the mask for all frames in the bin
    # make a list of copies for these frames with rot_per_frame[i] copies
    # join the copies together and append to the super copy list
    for i in range(len(rot_per_frame)):
        # mask for frames in bin
        m1 = cvs[:, col] >= bins[i]
        m2 = cvs[:, col] < bins[i+1]
        mask = m1 & m2
        c = 0
        for m in mask:
            if m:
                c +=1

        # make the copies
        sub_copies = []

        for j in range(rot_per_frame[i]):
            sub_copies.append(deepcopy(traj[mask]))
            cvs_long = cvs_long + cvs[mask].tolist()
          
        sub_copies = rotate_trajs(sub_copies)  
        for sub_copy in sub_copies:
            copy = copy.join(sub_copy)

        
    
    # center the trajectory and apply PBC
    copy = pbc(copy, box)
    
    # save the cvs and trajectory
    ###################################################
    # WARNING: number of coloumns in CV file hard coded
    ###################################################
    with open(savename + '_cv' + ".dat", 'w') as f:
        for item in cvs_long:
            f.write("%f %f %f %f\n" % (item[0], item[1], item[2], item[3]))
    
    copy.save_xtc(savename + ".xtc")
    return


def transform_frame_to_ndist_list(n_dist, traj, box):
    """Transform the the cartesian coordinates of a given trajectory frame into a sorted list of the n_dist shortest distances in the system

    Args:
        n_dist (int): Number of distances to include (max: n*(n-1)/2)
        traj (array): List of list of coordinates to transform
        box (list): List of the box vectors and angles

    Returns:
        Array: Array of shape n_atoms x n_dists
    """

    dist_frames = np.sort(self_distance_array(traj, box))[:n_dist]

    return dist_frames 

def transform_traj_to_ndist_list(n_dist, traj, box):
    """Transform the cartesian coordinates of a given trajectory into a sorted list of the n_dist shortest distances in the system

    Args:
        n_dist (int): Number of distances to include (max: n*(n-1)/2)
        traj (list): Trajectory that is to be transformed.
        box (list): List of the box vectors

    Returns:
        list: Array of shape n_frames x n_atoms x n_dists
    """

    box = [box[0], box[1], box[2], 90.0, 90.0, 90.0]
    n_at = len(traj[0])
    n_frames = len(traj)
    dist_frames = np.ones(shape=(len(traj), int((n_at*(n_at - 1))/2 )))
    target = np.zeros( (int(n_at*(n_at-1)/2),))

    for i in range(n_frames):
        dist_frames[i] = np.sort(self_distance_array(traj, box, result=target))[:n_dist]
    
    return dist_frames



def transform_frame_to_knn_list(k, traj, box):
    """Transforms the cartesian representation of a given trajectory frame to a list of sorted distances including the distance of each atom to its k nearest neighbours. This guarantees symmetry invariances but at significant cost and risk of kinks in the CV space.

    Args:
        k (int): Number of neighbours to consider for each atom
        traj (array): List of coordinates to be transformed
        box (list): List of box vectors

    Returns:
        Array: Returns an array of shape n_atoms x k*n_atoms/2
    """
    n_at = len(traj)
    box = np.array(box)

    d, j = PeriodicCKDTree(box, traj).query(traj, k=k+1)
    d = d.flatten()
    d.sort()
    result = d[n_at:][::2]
    return result

def transform_traj_to_knn_list(k, traj, box):

    n_at = len(traj[0])
    box = np.array(box)
    n_frames = len(traj)
    result = np.zeros(shape=(n_frames, int(math.ceil(n_at*k/2))))
    
    for i in range(n_frames):
        T = PeriodicCKDTree(box, traj[i])
        d, j = T.query(traj[i], k=k+1)
        d = d.flatten()
        d.sort()
        result[i] = d[n_at:][::2]
    return result