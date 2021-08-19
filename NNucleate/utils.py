import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation as R

def pbc(trajectory, box_length):
    """Centers an mdtraj Trajectory around the centre of a cubic box with the given box length and wraps all atoms into the box.

    Args:
        trajectory (mdtraj.trajectory): The trajectory that is to be modified
        box_length (float): Length of the target cubic box

    Returns:
        mdtraj.trajectory: A trajectory object obeying PBC according to the given box length
    """
    # function defining PBC
    def transform(x):
        while(x >= box_length):
            x -= box_length
        while(x <= 0):
            x += box_length
        return x

    # Prepare the function for 2D mapping
    func = np.vectorize(transform)

    for i in range(trajectory.n_frames):
        # map the function on all coordinates
        trajectory.xyz[i] = func(trajectory.xyz[i])

    return trajectory


def rotate_trajs(trajectories):
    """Rotates each frame in the given trajectories according to a random quaternion

    Args:
        trajectories (list of md.trajectory): list of mdtraj.trajectory objects to be modified

    Returns:
        list of md.trajectory: Returns the randomly rotated frames
    """
    # 
    # Rotates each frame in the given trajectories according to a random quaternion
    # Parameters:
    #     trajectories: list of mdtraj.Trajectory objects
    # Returns:
    #     List of modified trajectory objects
    # 
    for t in trajectories:
        for i in range(t.n_frames):
            quat = md.utils.uniform_quaternion()
            rot = R.from_quat(quat)
            t.xyz[i] = rot.apply(t.xyz[i])
    
    return trajectories

