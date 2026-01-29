"""
Functions to determine robot velocity based on current image 
"""
from tkinter import N

# from kornia.utils import vec_like
from motion import *
import numpy as np
from typing import List, Union
import torch

from pathlib import Path
TARGET_PATH = Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000050_crop.png')
REF_PATH = Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000061_crop.png')

def get_error_vec_K(K_sample_mkpts0: torch.Tensor, 
                    K_sample_mkpts1: torch.Tensor) -> np.ndarray:
    """
    returns an error vector given the observed features
    shape : (n x 2)
    
    Supplementary notes:
    --------------------
    Eq.(1) in Chaumette et al.
    Note the target image is now constantly a constantly changing video feed.
    """
    K_mkpts0, K_mkpts1 = np.array(K_sample_mkpts0.cpu()), np.array(K_sample_mkpts1.cpu())
    # error = K_mkpts1 - K_mkpts0
    error = np.array([K_mkpts0[i][j] - K_mkpts1[i][j] for i in range(3) for j in range(2)])
    return error

def sample_points(source_mkpts:torch.Tensor, target_mkpts:torch.Tensor, K=3):
    """ Uniformly sample K indices from the source and target keypoints.
    
    returns: torch.Tensor """
    import random
    K_sampled_indices = random.sample(range(0, len(source_mkpts)), K)

    K_sample_mkpts0 = source_mkpts[K_sampled_indices]
    K_sample_mkpts1 = target_mkpts[K_sampled_indices]

    return K_sample_mkpts0, K_sample_mkpts1

def get_velocity_K_points(
    K_sample_mkpts0: Union[List[List[float]], np.ndarray], K_sample_mkpts1: Union[List[List[float]], np.ndarray],
    depth_buffer: Union[List[List[float]], np.ndarray], K: int = 3
) -> np.ndarray:
    """ gets the velocity vector given the points and the depth buffer
    NOTE: we have 8 features and only 6 dof. In this case we naively take the first three points.

    args :
        K_sample_mkpts0 : sampled source image keypoints matched with the target image
        K_sample_mkpts1 : sampled target image keypoints matched with the source image
        depth_buffer : a pixel-wise map for depth values.
        K : number of points to compute Jacobian

    returns:
        Velocity vector : [linear, angular] ; (6,)
    
    Supplementary notes:
    --------------------
    The reason to use the first-three points is that each point contributes 2 DoF [Chaumette et al.].
    By using 3 points, we now have k = 6.
    This means the Interaction Matrix (aka. Jacobian) becomes a full-rank square matrix.
    This allows us to use the true inverse instead of the Moore-Penrose Psuedo-inverse.
    """
    error : np.ndarray = get_error_vec_K(K_sample_mkpts0, K_sample_mkpts1)

    J = np.vstack(
        [
            jacobian(  ## int(.) rounds to pixel pos
                X=int(K_sample_mkpts0[i][0]),
                Y=int(K_sample_mkpts0[i][1]),
                # Z=(depth_buffer[int(K_sample_mkpts0[i][0])][int(K_sample_mkpts0[i][1])]),  # Get depth at x,y # TODO: fix bug
                Z=(depth_buffer[int(K_sample_mkpts0[i][1])][int(K_sample_mkpts0[i][0])]),  # switched X, Y -> which one is correct?
            )
            for i in range(K) 
        ]
    )

    # compute Penrose P-Inv
    J_pinv = np.linalg.pinv(J)

    # 6D velocity vector : [linear, angular]
    vel = -LAMBDA * np.matmul(J_pinv, error)
    return vel


def update_perpendicular_velocity(vel: np.ndarray, K_sample_mkpts0: Union[List[List[float]], np.ndarray], 
                                K_sample_mkpts1: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Computes a corrective linear motion in the perpendicular plane
    This design is inspired by RoboTAP

    Returns a unit velocity
    """
    assert vel.shape == (6,), "velocity object is not a 6d ndarray"
    
    # Convert inputs to numpy arrays
    mkpts0 = np.array(K_sample_mkpts0.cpu())
    mkpts1 = np.array(K_sample_mkpts1.cpu())

    # Calculate displacement vectors between corresponding points
    displacements = mkpts1 - mkpts0  # [horizontal, vertical]

    # Average displacement vectors into single command velocity in the perpendicular plane
    mean_displacement = np.mean(displacements, axis=0)  # Shape: (2,)
    
    # x (left-right), z (top-down)
    vel[0] += mean_displacement[0]
    vel[2] += (-1) * mean_displacement[1]

    # flip direction to move camera (robot) relative to scene
    vel[0] *= (-1)
    vel[2] *= (-1)

    # Normalize velocity vector to unit length
    vel_magnitude = np.linalg.norm(vel)

    assert vel_magnitude < 1e-15, "Perpendicular velocity magnitude is almost zero"
    vel = vel / vel_magnitude
    
    return vel

def update_forward_velocity(vel: np.ndarray):
    """ Not Implemented.
        Currently we just return the passed object"""
    assert vel.shape == (6,), "velocity object is not a 6d ndarray"
    return vel


def update_angular_velocity(vel: np.ndarray):
    """ Not Implemented.
        Currently we just return the passed object"""
    assert vel.shape == (6,), "velocity object is not a 6d ndarray"
    return vel


def scale_velocity(velocity: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Scale each velocity component by coefficients from config file.
    
    Args:
        velocity: Velocity vector with 6 components [vx, vy, vz, wx, wy, wz]
                 Can also accept 3 components [vx, vy, vz] for translational only
        config: Configuration dictionary containing scale factors
                If None, loads from default config.yaml
    
    Returns:
        Scaled velocity vector with same shape as input
    """
    
    scaled_velocity = velocity.copy()
    
    # Scale translational components (x, y, z)
    if len(velocity) >= 3:
        scaled_velocity[0] *= config.get('x_scale', 1.0)
        scaled_velocity[1] *= config.get('y_scale', 1.0)
        scaled_velocity[2] *= config.get('z_scale', 1.0)
    
    # Scale rotational components (rx, ry, rz) if present
    if len(velocity) >= 6:
        scaled_velocity[3] *= config.get('rx_scale', 1.0)
        scaled_velocity[4] *= config.get('ry_scale', 1.0)
        scaled_velocity[5] *= config.get('rz_scale', 1.0)
    
    return scaled_velocity
