"""
Functions to determine robot velocity based on current image 
"""
from tkinter import N

# from kornia.utils import vec_like
from motion import *
import numpy as np
from typing import List, Union, Optional
import torch
import cv2

from pathlib import Path
TARGET_PATH = Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000050_crop.png')
REF_PATH = Path('/home/khw/Documents/6dpose/LightGlue/myassets/frame_000061_crop.png')

def get_error_vec_Ksample(K_sample_mkpts0: torch.Tensor, 
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

    assert vel_magnitude > 1e-15, "Perpendicular velocity magnitude is almost zero"
    vel = vel / vel_magnitude
    
    return vel

def update_forward_velocity(vel: np.ndarray, config, curr_robot_pos = None, goal_pos = None, d0 = None, i=None, dt=None):
    """
    A forward velocity controller that increases / decreases speed as a function of distance to the object.
    
    Two options:
    * Use the current displacement between the robot position and the goal position
    * Use an initial distance as parameter, then change speed based on a given dt
    """
    assert vel.shape == (6,), "velocity object is not a 6d ndarray"

    _gain = config.get('K', 20.0)

    use_d0_method = False
    if not use_d0_method:
        # print("Currently : curr_robot_pos = {curr_robot_pos}, goal_pos = {goal_pos}")
        assert curr_robot_pos is not None and goal_pos is not None, f"If not using d0 formula, then you must pass current robot position and the goal position"
        _displacement = goal_pos - curr_robot_pos

        _distance_to_goal = np.linalg.norm(_displacement)

        custom_dist_bool = True
        if custom_dist_bool:
            _distance_to_goal -= 1.35  # stop before the image

        _forward_velocity = _gain * _distance_to_goal

        print(f"Forward velocity: {_forward_velocity:.4f}, Distance to goal: {_distance_to_goal:.4f}")

    else:
        #   WARNING : Implemented but not tested.
        #   This formulation assumes the true trajectory and optimal trajectory are sufficiently similar
        assert type(dt) is not None and d0 is not None and i is not None, "If using d0 formula, then you must pass d0, dt and iteration number"
        time = dt * i
        _forward_velocity = _gain * d0 * np.exp( (-1) * _gain * time)

    vel[1] += _forward_velocity

    return vel

def update_angular_velocity(vel: np.ndarray, 
                            K_sample_mkpts0: Union[List[List[float]], np.ndarray, torch.Tensor],
                            K_sample_mkpts1: Union[List[List[float]], np.ndarray, torch.Tensor],
                            K: Optional[np.ndarray] = None,
                            lambda_gain: float = 1.0) -> np.ndarray:
    """
    Homography-based angular velocity control that rotates the robot.
    
    Implements the following equations:
    - H_tilde = K^-1 * H * K  (normalized homography)
    - H = R + (t * n^T) / d   (homography decomposition)
    - R_e = R                  (extract rotation)
    - omega = Log(R_e)         (matrix logarithm)
    - Omega = -lambda * omega  (angular velocity)
    
    Args:
        vel: Velocity vector (6,) - angular components [3:6] will be updated
        K_sample_mkpts0: Source image keypoints (N, 2)
        K_sample_mkpts1: Target image keypoints (N, 2)
        K: Camera intrinsic matrix (3, 3). If None, uses default based on image config
        lambda_gain: Gain parameter lambda (default: 1.0)
    
    Returns:
        Updated velocity vector with angular components set
    """
    assert vel.shape == (6,), "velocity object is not a 6d ndarray"
    
    # Convert inputs to numpy arrays
    if isinstance(K_sample_mkpts0, torch.Tensor):
        mkpts0 = np.array(K_sample_mkpts0.cpu())
    else:
        mkpts0 = np.array(K_sample_mkpts0)
    
    if isinstance(K_sample_mkpts1, torch.Tensor):
        mkpts1 = np.array(K_sample_mkpts1.cpu())
    else:
        mkpts1 = np.array(K_sample_mkpts1)
    
    # Ensure keypoints are in the right shape for cv2.findHomography
    # cv2.findHomography expects (N, 1, 2) shape
    if mkpts0.ndim == 2 and mkpts0.shape[1] == 2:
        mkpts0_reshaped = mkpts0.reshape(-1, 1, 2)
    else:
        mkpts0_reshaped = mkpts0
    
    if mkpts1.ndim == 2 and mkpts1.shape[1] == 2:
        mkpts1_reshaped = mkpts1.reshape(-1, 1, 2)
    else:
        mkpts1_reshaped = mkpts1
    
    # Need at least 4 points for homography estimation
    if len(mkpts0_reshaped) < 4:
        print(f"Warning: Only {len(mkpts0_reshaped)} points available, need at least 4 for homography. Skipping angular velocity update.")
        return vel
    
    # Compute homography H from keypoint matches
    H, mask = cv2.findHomography(mkpts0_reshaped, mkpts1_reshaped, 
                                  method=cv2.RANSAC, 
                                  ransacReprojThreshold=5.0)
    
    if H is None:
        print("Warning: Homography estimation failed. Skipping angular velocity update.")
        return vel
    
    # Get or construct camera intrinsic matrix K
    if K is None:
        from image import get_image_config
        img_config = get_image_config()
        width = img_config['width']
        height = img_config['height']
        fov_rad = np.radians(img_config['fov'])
        fx = fy = (width / 2.0) / np.tan(fov_rad / 2.0)
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
    
    # Compute normalized homography: H_tilde = K^-1 * H * K
    K_inv = np.linalg.inv(K)
    H_tilde = K_inv @ H @ K
    
    # Decompose homography to extract rotation R
    # Note: cv2.decomposeHomographyMat expects the original H and camera matrix K
    # Since H_tilde is normalized, we decompose H directly (rotation is the same)
    # Alternatively, we could decompose H_tilde with identity matrix, but decomposing H is more standard
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    
    if num_solutions == 0:
        print("Warning: Homography decomposition failed. Skipping angular velocity update.")
        return vel
    
    # Use the first solution from homography decomposition
    # OpenCV uses standard camera frame: x=right, y=down, z=forward (optical axis)
    # PyBullet uses: x=left-right, y=forward, z=up
    # We need to transform from CV frame to PyBullet frame
    
    # Transformation matrix from OpenCV camera frame to PyBullet frame
    R_cv_to_pb = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, -1, 0]], dtype=np.float64)
    
    # Use first rotation solution from openCV.
    # This is a good enough heuristic since opencv sorts its solutions
    R_e_cv = rotations[0]
    
    # Transform rotation matrix from CV frame to PyBullet frame
    R_e = R_cv_to_pb @ R_e_cv @ R_cv_to_pb.T
    
    # Compute matrix logarithm: omega = Log(R_e)
    # For SO(3), the matrix logarithm gives the axis-angle representation
    # cv2.Rodrigues converts rotation matrix to axis-angle (which is the log map)
    rodrigues_vec, _ = cv2.Rodrigues(R_e)
    omega = rodrigues_vec.flatten()  # Shape: (3,) - now in PyBullet frame
    
    # Compute angular velocity: Omega = -lambda * omega
    Omega = -lambda_gain * omega
    
    # Update angular velocity components [wx, wy, wz] in the velocity vector
    # PyBullet convention: [wx, wy, wz] = [rotation around x, rotation around y, rotation around z]
    vel[3:6] = Omega
    
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

def get_ibvs_velocity(
    src_kpts: Union[List[List[float]], np.ndarray, torch.Tensor],
    tgt_kpts: Union[List[List[float]], np.ndarray, torch.Tensor],
    depth_buffer: np.ndarray,
    img_config: dict,
    lambda_gain: float = 1.0
) -> np.ndarray:
    """
    Computes camera velocity using Chaumette's classical IBVS control law.
    v = -lambda * L_s^+ * (s - s*)
    
    Args:
        src_kpts: Current image keypoints (N, 2) in pixels
        tgt_kpts: Target image keypoints (N, 2) in pixels
        depth_buffer: Depth buffer from PyBullet (H, W)
        img_config: Image configuration dictionary
        lambda_gain: Control gain
        
    Returns:
        velocity: 6D velocity vector [vx, vy, vz, wx, wy, wz]
    """
    # Convert inputs to numpy arrays
    if isinstance(src_kpts, torch.Tensor):
        src_kpts = np.array(src_kpts.cpu())
    if isinstance(tgt_kpts, torch.Tensor):
        tgt_kpts = np.array(tgt_kpts.cpu())
        
    # Get camera intrinsics
    width = img_config['width']
    height = img_config['height']
    fov_rad = np.radians(img_config['fov'])
    # Assuming square pixels and principal point at center
    fx = fy = (width / 2.0) / np.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    
    # PyBullet depth buffer conversion to true depth
    near = img_config['near_val']
    far = img_config['far_val']
    
    # Interaction matrix and error vector
    L_s_list = []
    error_list = []
    
    for i in range(len(src_kpts)):
        u, v = src_kpts[i]
        u_star, v_star = tgt_kpts[i]
        
        # Convert to normalized coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy
        
        x_star = (u_star - cx) / fx
        y_star = (v_star - cy) / fy
        
        # Get depth at current point
        # Clamp coordinates to image bounds
        u_idx = int(np.clip(u, 0, width - 1))
        v_idx = int(np.clip(v, 0, height - 1))
        
        # PyBullet depth buffer is (height, width)
        d_buffer_val = depth_buffer[v_idx, u_idx]
        
        # Convert depth buffer to linear depth Z
        # Z = far * near / (far - (far - near) * depth_buffer)
        Z = far * near / (far - (far - near) * d_buffer_val)
        
        # Construct interaction matrix for this point (2x6)
        # L_s = [[-1/Z, 0, x/Z, xy, -(1+x^2), y],
        #        [0, -1/Z, y/Z, 1+y^2, -xy, -x]]
        
        # Note: PyBullet camera frame vs OpenCV camera frame
        # OpenCV: x right, y down, z forward
        # PyBullet: x right, y up, z backward (OpenGL convention) ??
        # Actually PyBullet getCameraImage view matrix usually follows OpenGL:
        # Camera looks down -Z. X is right, Y is up.
        # But we usually want velocity in the robot/camera frame.
        # Let's assume standard IBVS frame (Z forward).
        # If the simulation uses a different frame, we might need coordinate transformation.
        # Based on existing code, it seems we might need to be careful.
        # Existing `jacobian` uses standard formula.
        
        L_point = np.array([
            [-1/Z, 0, x/Z, x*y, -(1+x**2), y],
            [0, -1/Z, y/Z, 1+y**2, -x*y, -x]
        ])
        
        L_s_list.append(L_point)
        
        # Error: s - s*
        error_list.append([x - x_star, y - y_star])
        
    # Stack interaction matrices: (2N, 6)
    L_s = np.vstack(L_s_list)
    
    # Stack error vectors: (2N, 1)
    error = np.array(error_list).flatten()
    
    # Compute pseudo-inverse
    # Use damped pseudo-inverse or standard pinv
    L_s_pinv = np.linalg.pinv(L_s)
    
    # Control law: v = -lambda * L_s^+ * e
    vel = -lambda_gain * (L_s_pinv @ error)
    
    # The computed velocity is in the camera frame (OpenCV convention: x right, y down, z forward)
    # We need to convert this to the robot frame used in simulation.
    # In `update_pos_and_orn`:
    # del_pos = np.matmul(transform, [*velocity[:3], 1])
    # It seems `velocity` is expected in the camera frame (local frame).
    # But we need to check if the camera frame matches the IBVS frame.
    # IBVS frame: X right, Y down, Z forward.
    # PyBullet camera setup in `capture_camera_image`:
    # init_camera_vector = (0, 1, 0) # y axis (forward?)
    # init_up_vector = (0, 0, 1) # z axis (up?)
    # This suggests Robot Frame: Y is forward, Z is up, X is right.
    # Camera Frame (View Matrix): usually -Z is forward in OpenGL.
    
    # Let's look at `convert_to_transformation_matrix`.
    # It constructs a transform from robot pose.
    # `update_pos_and_orn` transforms velocity from "local" to "world".
    
    # If we assume the velocity we compute is in the "Optical" frame (Z forward, Y down, X right),
    # and the robot moves in (Y forward, Z up, X right).
    # Optical -> Robot:
    # Z_opt (forward) -> Y_rob (forward)
    # Y_opt (down) -> -Z_rob (down)
    # X_opt (right) -> X_rob (right)
    
    # So v_rob = [v_opt_x, v_opt_z, -v_opt_y, w_opt_x, w_opt_z, -w_opt_y]
    
    vx, vy, vz, wx, wy, wz = vel
    
    # Remap to Robot Frame (assuming standard PyBullet robot frame)
    # This is a guess based on typical setups.
    # However, `update_perpendicular_velocity` does:
    # vel[0] += mean_displacement[0] (x)
    # vel[2] += (-1) * mean_displacement[1] (z)
    # This suggests X is X, and Z is Y-image-axis (vertical).
    # So Image Y corresponds to Robot Z (inverted).
    # Image X corresponds to Robot X.
    # Image Depth (Z) corresponds to Robot Y (forward).
    
    # So:
    # v_rob_x = v_opt_x
    # v_rob_y = v_opt_z
    # v_rob_z = -v_opt_y
    
    # w_rob_x = w_opt_x
    # w_rob_y = w_opt_z
    # w_rob_z = -w_opt_y
    
    vel_robot = np.array([vx, vz, -vy, wx, wz, -wy])
    
    return vel_robot
