"""
Functions to determine robot velocity based on current image 
"""
from tkinter import N
from motion import *  # append original methods
import numpy as np
from typing import List, Union

from pathlib import Path
TARGET_PATH = Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000050_crop.png')
REF_PATH = Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000061_crop.png')


def get_error_vec_K(K_sample_mkpts0: Union[List[List[float]], np.ndarray], 
                    K_sample_mkpts1: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    returns an error vector given the observed corner features
    shape : (n x 2)
    
    Supplementary notes:
    --------------------
    Eq.(1) in Chaumette et al.
    Note the target image is now constantly a constantly changing video feed.
    """
    K_mkpts0, K_mkpts1 = np.array(K_sample_mkpts0), np.array(K_sample_mkpts1)
    # error = K_mkpts1 - K_mkpts0
    error = np.array([K_mkpts0[i][j] - K_mkpts1[i][j] for i in range(3) for j in range(2)])
    return error

def sample_points(source_mkpts, target_mkpts, K=3):
    """ Uniformly sample K indices from the source and target keypoints."""
    import random
    K_sampled_indices = random.sample(range(0, len(source_mkpts)), K)

    K_sample_mkpts0 = [source_mkpts[idx] for idx in K_sampled_indices]
    K_sample_mkpts1 = [target_mkpts[idx] for idx in K_sampled_indices]

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
                Z=(depth_buffer[int(K_sample_mkpts0[i][0])][int(K_sample_mkpts0[i][1])]),  # Get depth at x,y
            )
            for i in range(K) 
        ]
    )

    # compute Penrose P-Inv
    J_pinv = np.linalg.pinv(J)

    # 6D velocity vector : [linear, angular]
    vel = -LAMBDA * np.matmul(J_pinv, error)
    return vel
